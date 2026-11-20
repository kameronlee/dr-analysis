"""Experiment E (secondary, external): DRAC training images, segmenter-derived graphs.

Readiness gates (all must pass, otherwise a readiness report is written and
NO group statistic is computed): corrected label release flag, physical
geometry, human segmentation QC (qc.py), segmented masks for every training
image, pipeline features, and no hidden-label test rows.

Analyses per segmenter, reported per dataset (raw FI values are never pooled
with OCTA-500): Spearman trend with grade, permutation Jonckheere-Terpstra
(ties = 1/2), and grouped repeated CV AUROC (baseline vs baseline + FI) for
any-DR (grade >= 1) and PDR-vs-rest. Grouping is image-level because DRAC has
no subject IDs; outputs say so and do not claim patient independence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..manifest import grouping_level, is_synthetic, resolve_input
from ..provenance import SYNTHETIC_BANNER, read_json, sha256_file, write_json
from ..qc import graph_review_identity, qc_status
from ..seeds import rng_for
from ..stats import cv as cvmod
from ..stats.effects import jonckheere_terpstra, spearman
from .common import ReadinessError, header, write_results

FEATURES = ["FI_adaptive_betweenness", "FI_random", "G10_adaptive_betweenness",
            "qc_adaptive_betweenness", "vessel_density", "loop_density_per_mm2", "fractal_dimension"]


P_VALUE_ROLE = ("exploratory_uncorrected: external Experiment E trend/association p-values are "
                "not members of any prespecified Holm family (stats.holm_families)")


def external_faz_spec(cfg: dict) -> dict:
    """FAZ handling for external (DRAC) graphs: the configured fixed disk."""
    return dict(cfg["exp_e"]["faz"], mode="disk")


def qc_binding(cfg: dict, frame: pd.DataFrame, segmenter: str, pipeline_root,
               root=None) -> tuple[dict, list[str]]:
    """What the QC sheet must match: cohort, analysed mask hashes and run identities.

    ``frame`` is the analysed *segmented* manifest (training rows). Every
    analysed image's pipeline provenance must name the same mask file hash. The
    pipeline outputs must come from the current per-image code, and the graph
    identity uses the configuration and FAZ handling the pipeline actually used.
    """
    from ..pipeline import pipeline_code_fingerprint, safe_name

    root = Path.cwd() if root is None else Path(root)
    pipeline_root = Path(pipeline_root)
    reasons: list[str] = []
    ids = frame.image_id.astype(str).tolist()
    seg_col = frame.get("segmentation_config_sha256")
    seg_ids = sorted(set(seg_col)) if seg_col is not None else []
    if len(seg_ids) != 1 or not seg_ids[0]:
        reasons.append("analysed manifest is not a single bound segmentation run (needs one non-empty "
                       f"segmentation_config_sha256; found {seg_ids[:3]}); re-run `segment`")
    if "segmenter" in frame and set(frame.segmenter.astype(str)) != {segmenter}:
        reasons.append(f"analysed manifest segmenter {sorted(set(frame.segmenter))} is not {segmenter}")
    masks: dict[str, str] = {}
    for r in frame.itertuples():
        try:
            masks[str(r.image_id)] = sha256_file(resolve_input(r.vessel_mask_path, root))
        except (OSError, ValueError) as exc:
            reasons.append(f"analysed mask unreadable for {r.image_id}: {exc}")
    run_p, cfg_p = pipeline_root / "run.json", pipeline_root / "config_used.json"
    faz_spec = external_faz_spec(cfg)
    graph_cfg = cfg
    if run_p.exists() and cfg_p.exists():
        run = read_json(run_p)
        graph_cfg = read_json(cfg_p)
        faz_spec = run.get("faz_spec") or faz_spec
        code = (run.get("code_fingerprint") or {}).get("sha256")
        if code != pipeline_code_fingerprint()["sha256"]:
            reasons.append("pipeline outputs were produced by different per-image code than the "
                           "current checkout (code fingerprint mismatch); re-run `process`")
    else:
        reasons.append(f"pipeline run record missing ({run_p} / {cfg_p})")
    feats_p = pipeline_root / "features.csv"
    analysed = pd.read_csv(feats_p, usecols=["image_id"]).image_id.astype(str) if feats_p.exists() else []
    stale = []
    for iid in analysed:
        if iid not in masks:
            continue
        prov_p = pipeline_root / "images" / safe_name(iid) / "provenance.json"
        vessel_h = (read_json(prov_p).get("input_sha256") or {}).get("vessel") if prov_p.exists() else None
        if vessel_h != masks[iid]:
            stale.append(iid)
    if stale:
        reasons.append(f"pipeline features were computed from other masks than the analysed "
                       f"segmentation run for {len(stale)} images (e.g. {stale[:5]}); re-run `process`")
    img = dict(zip(frame.image_id.astype(str), frame.image_sha256)) if "image_sha256" in frame else {}
    binding = {"cohort_ids": set(ids), "mask_sha256": masks, "image_sha256": img,
               "segmentation_config_sha256": seg_ids[0] if len(seg_ids) == 1 else None,
               "graph_review_sha256": graph_review_identity(graph_cfg, faz_spec)["graph_review_sha256"]}
    return binding, reasons


def readiness(cfg: dict, frame: pd.DataFrame, segmenter: str, qc_csv, pipeline_root,
              root=None) -> tuple[list[str], dict]:
    e = cfg["exp_e"]
    reasons = []
    if not e["labels_corrected_version"]:
        reasons.append("exp_e.labels_corrected_version is not true (corrected DRAC labels required)")
    if e["pixel_size_um"] is None:
        reasons.append("DRAC physical scale not configured (pixel-unit pipeline not implemented)")
    if (frame.split.str.lower() == "test").any():
        reasons.append("hidden-label test rows present in the Experiment E input")
    feats = Path(pipeline_root) / "features.csv"
    if not feats.exists():
        reasons.append(f"no pipeline features for segmenter {segmenter}: {feats}")
    train = frame[frame.split.str.lower() == "train"] if "split" in frame else frame
    binding, bind_reasons = qc_binding(cfg, train, segmenter, pipeline_root, root)
    reasons += bind_reasons
    qc = qc_status(qc_csv, segmenter, int(e["min_qc_audited"]), float(e["max_qc_fail_fraction"]),
                   allow_synthetic_reviewer=is_synthetic(frame), binding=binding)
    reasons += qc["reasons"]
    return reasons, qc


def run(cfg: dict, frame: pd.DataFrame, segmenter: str, qc_csv, pipeline_root, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = frame[frame.split.str.lower() == "train"] if "split" in frame else frame
    reasons, qc = readiness(cfg, frame, segmenter, qc_csv, pipeline_root)
    if reasons:
        write_json(out_dir / "readiness.json", {"experiment": "E", "segmenter": segmenter,
                                                "ready": False, "reasons": reasons, "qc": qc})
        raise ReadinessError(reasons)
    feats = pd.read_csv(Path(pipeline_root) / "features.csv")
    cov = frame[["image_id", "group_key", "grade_num"]].reset_index(drop=True)
    table = cov.merge(feats, on="image_id", how="inner")
    excluded_qc = table.image_id.isin(qc["failed_image_ids"])
    table = table[~excluded_qc].copy()
    table["any_dr"] = (table.grade_num >= 1).astype(int)
    table["pdr"] = (table.grade_num == 2).astype(int)
    e, st = cfg["exp_e"], cfg["stats"]
    seed = cfg["project"]["base_seed"]
    trend = []
    for f in FEATURES:
        if f not in table:
            continue
        jt = jonckheere_terpstra(table[f].to_numpy(float), table.grade_num.to_numpy(),
                                 int(e["n_permutations"]), rng_for(seed, "exp_e_jt", segmenter, f))
        trend.append({"feature": f, "spearman": spearman(table[f], table.grade_num), "jt": jt,
                      "median_by_grade": table.groupby("grade_num")[f].median().to_dict(),
                      "p_value_role": P_VALUE_ROLE})
    base = [c for c in e["baseline"] if c in table]
    dropped = {}
    for c in list(base):
        v = table[c].to_numpy(float)
        if np.nanstd(v) == 0:
            base.remove(c)
            dropped[c] = "constant in this dataset (e.g. fixed-disk FAZ); dropped from design"
    fi = st["fi_column"]
    cvres = {}
    for outcome in ("any_dr", "pdr"):
        y = table[outcome]
        if y.nunique() < 2 or min((y == 1).sum(), (y == 0).sum()) < int(e["cv"]["n_splits"]):
            cvres[outcome] = {"status": "not_run", "reason": "too few cases per class"}
            continue
        preds, folds, issues = cvmod.repeated_cv(table.reset_index(drop=True), outcome, "group_key",
                                                 {"baseline": base, "baseline_plus_FI": base + [fi]},
                                                 int(e["cv"]["n_splits"]), int(e["cv"]["n_repeats"]),
                                                 float(e["cv"]["C"]), seed, f"exp_e_{segmenter}_{outcome}")
        preds.to_csv(out_dir / f"exp_e_oof_{outcome}.csv", index=False)
        aucs = cvmod.auc_by_repeat(preds)
        cvres[outcome] = {"status": "ok", "mean_auroc": aucs.groupby("model").auroc.mean().to_dict(),
                          "paired_delta": cvmod.paired_delta_auc(preds, "baseline", "baseline_plus_FI",
                                                                 int(st["n_boot"]), seed,
                                                                 f"exp_e_{segmenter}_{outcome}"),
                          "fit_issues": issues}
    info = {"data_provenance": SYNTHETIC_BANNER if is_synthetic(frame) else "real",
            "grouping_level": grouping_level(frame), "n_analysed": int(len(table)),
            "n_excluded_qc_fail": int(excluded_qc.sum())}
    res = header(cfg, "E", "secondary external (H1 external)", info, None)
    res.update(segmenter=segmenter, qc=qc, trend=trend, cv=cvres, baseline_used=base,
               p_value_roles={"spearman.p, jt.p_increasing, jt.p_two_sided": P_VALUE_ROLE,
                              "cv": "no p-values (AUROC and paired ΔAUROC CI are estimates)"},
               baseline_dropped=dropped,
               limitations=[("Image-level grouping: DRAC provides no patient IDs; patient overlap "
                             "between folds cannot be excluded."),
                            "No age/sex available for adjustment.",
                            ("Segmenter-derived graphs, not reference labels; FI values are not "
                             "comparable in absolute terms with OCTA-500 and are never pooled.")])
    write_results(out_dir, res)
    return res
