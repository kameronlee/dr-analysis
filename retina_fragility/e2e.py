"""End-to-end software validation on SYNTHETIC fixtures (never clinical evidence).

Chain: fixtures -> manifest validation -> frozen calibration -> per-image
pipeline -> Experiments B (primary code path), A, C, D -> Holm summary ->
Frangi segmentation of a synthetic 'external' fixture -> QC gate -> E ->
(optional torch) U-Net train/eval/predict and embeddings -> F -> ablations.
Also records the real-data readiness failures (DRAC QC gate, OCTA-500 absence).
Every output directory is labelled SYNTHETIC - NOT CLINICAL.
"""
from __future__ import annotations

import copy
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .config import config_hash
from .provenance import SYNTHETIC_BANNER, write_json


def _override(cfg: dict, quick: bool) -> dict:
    c = copy.deepcopy(cfg)
    if quick:
        c["calibration"]["n_per_fov"] = {"3": 3, "6": 3}
        c["stress"]["random_repetitions"] = 10
        c["stats"]["n_boot"] = 200
        c["stats"]["min_events_per_class"] = 5
        c["stats"]["cv"]["n_repeats"] = 3
        c["exp_c"]["repetitions"] = 5
        c["exp_e"]["n_permutations"] = 500
        c["exp_e"]["cv"]["n_repeats"] = 3
        c["exp_e"]["min_qc_audited"] = 10
        c["segmentation"]["unet"]["epochs"] = 2
        c["ablations"] = {"step_frac": [0.02], "d_percentile": [75.0], "crop_center_mm": [3.0],
                          "seg_threshold_scale": [1.1], "coverage_geometry": ["centerline"]}
    else:
        c["segmentation"]["unet"]["epochs"] = 8
        c["exp_e"]["min_qc_audited"] = 20
    c["calibration"]["frozen_file"] = "UNUSED_SEE_E2E_OUT"
    return c


def _step(log: list, name: str, fn):
    t0 = time.monotonic()
    try:
        val = fn()
        log.append({"step": name, "status": "ok", "seconds": round(time.monotonic() - t0, 1)})
        return val
    except Exception as exc:
        log.append({"step": name, "status": "failed", "error": repr(exc),
                    "traceback": traceback.format_exc(), "seconds": round(time.monotonic() - t0, 1)})
        return None


def run_synthetic_e2e(out: str | Path, cfg: dict, quick: bool = False, n_workers: int = 8) -> dict:
    from . import ablations, embeddings
    from .calibration import compute_calibration, freeze_calibration, select_calibration_ids
    from .experiments import exp_a, exp_b, exp_c, exp_d, exp_e, exp_f, secondary_summary
    from .experiments.common import load_analysis_table
    from .io_masks import read_intensity
    from .manifest import analysis_rows, load_manifest, validate_manifest
    from .pipeline import run_manifest
    from .segmentation import run as segrun
    from .segmentation.unet import torch_status
    from .synthetic import SYNTHETIC_DATASET, make_fixture_dataset

    out = Path(out).resolve()
    root = Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    (out / "SYNTHETIC_NOT_CLINICAL.txt").write_text(
        "Every file below this directory derives from SYNTHETIC fixtures (software validation only).\n")
    cfg = _override(cfg, quick)
    cfg["project"]["n_workers"] = int(n_workers)
    log: list = []
    summary: dict = {"data_provenance": SYNTHETIC_BANNER, "quick": quick, "config_sha256": config_hash(cfg)}
    n_norm, n_dr = (12, 8) if quick else (22, 12)
    fx = out / "fixtures"
    _step(log, "make_fixtures", lambda: make_fixture_dataset(fx, n_norm, n_dr, seed=11, n_seeds=150,
                                                             dropout_patches=2))
    m = load_manifest(fx / "manifest.csv", cfg)
    rep = _step(log, "validate_manifest", lambda: validate_manifest(m, cfg, root))
    summary["manifest_valid"] = rep and rep["all_inputs_valid"]
    valid = {r["image_id"] for r in rep["rows"] if r["valid_inputs"]} if rep else set()

    def calibrate():
        ids, _ = select_calibration_ids(m, cfg, SYNTHETIC_DATASET, valid)
        rec = compute_calibration(m, ids, cfg, root, SYNTHETIC_DATASET, cfg["ablations"].get("d_percentile", []))
        return freeze_calibration(rec, out / "calibration.json")

    cal = _step(log, "calibrate", calibrate)
    summary["calibration"] = {"n": len(cal["calibration_ids"]), "d_um": cal["d_um"]} if cal else None
    rows, _ = analysis_rows(m, cal["calibration_ids"], dataset=SYNTHETIC_DATASET)
    run = _step(log, "pipeline", lambda: run_manifest(rows, cfg, root, out / "pipeline", cal["d_um"],
                                                      calibration_record=cal))
    summary["pipeline"] = {k: run[k] for k in ("n_ok", "n_excluded", "exclusion_reasons")} if run else None
    table, info = load_analysis_table(out / "pipeline", m, cal, SYNTHETIC_DATASET)
    from .experiments.exp0_dropout import overlay_sheet, patch_dropout

    cfg_nov = copy.deepcopy(cfg)
    cfg_nov["pipeline"]["overlays"] = False
    dd = _step(log, "exp0_patch_dropout", lambda: patch_dropout(
        rows, cfg_nov, root, cal["d_um"], out / "exp0_patch_dropout", n_images=4 if quick else 10))
    if dd is not None:
        summary["exp0_patch_dropout_median_delta"] = {c: float(dd[c].median()) for c in dd if c.startswith("delta_")}
    _step(log, "overlay_sheet", lambda: overlay_sheet(out / "pipeline", table.image_id, out / "overlay_qc_queue",
                                                      20, cfg["project"]["base_seed"]))
    rb = _step(log, "exp_b", lambda: exp_b.run(table, info, cfg, out / "exp_b", cal))
    if rb:
        summary["exp_b_primary"] = {k: rb["primary"].get(k) for k in
                                    ("status", "failure_reasons", "n", "lr_statistic", "p_value",
                                     "odds_ratio_added")}
    _step(log, "exp_a", lambda: exp_a.run(table, info, cfg, out / "pipeline", out / "exp_a", cal))
    rc = _step(log, "exp_c", lambda: exp_c.run(table, info, cfg, out / "pipeline", out / "exp_c", cal,
                                                n_workers))
    summary["exp_c"] = rc["summary"] if rc else None
    _step(log, "exp_d", lambda: exp_d.run(table, info, cfg, out / "exp_d", cal))
    hs = _step(log, "holm", lambda: secondary_summary.run(
        cfg, {k: out / f"exp_{k.lower()}" for k in "ACD"}, out / "secondary_holm.json"))
    summary["holm"] = hs["families"] if hs else None

    # ---- synthetic external dataset (DRAC-like grades) for E/F code paths
    ext = m.copy()
    ext["dataset"] = "SYNTHETIC-DRAC-FIXTURE"
    ext = ext[~ext.image_id.isin(cal["calibration_ids"])]
    dr_idx = np.flatnonzero(ext.disease.eq("DR").to_numpy())
    grade = np.where(ext.disease.eq("DR"), 1, 0)
    grade[dr_idx[::2]] = 2
    ext["grade"] = grade.astype(str)
    ext["disease"] = pd.Series(grade, index=ext.index).map({0: "non_dr", 1: "npdr", 2: "pdr"})
    ext["subject_id"] = ""
    ext["age"], ext["sex"] = "", ""
    cols = ["image_id", "dataset", "subject_id", "image_path", "vessel_mask_path", "faz_mask_path",
            "large_vessel_mask_path", "fov_mm", "pixel_size_um", "disease", "grade", "age", "sex", "split",
            "seg_split", "calibration", "qc_audited"]
    ext_dir = out / "external"
    ext_dir.mkdir(exist_ok=True)
    ext[cols].to_csv(ext_dir / "manifest.csv", index=False)
    em = load_manifest(ext_dir / "manifest.csv", cfg)
    seg_manifest = _step(log, "frangi_segment", lambda: segrun.frangi_manifest(em, cfg, root, ext_dir / "frangi"))
    ecfg_faz = dict(cfg["exp_e"]["faz"], mode="disk")
    if seg_manifest:
        sm = load_manifest(seg_manifest, cfg)
        cfg_nc = copy.deepcopy(cfg)
        cfg_nc["stress"]["compute_coverage"] = False
        _step(log, "frangi_pipeline", lambda: run_manifest(sm, cfg_nc, root, ext_dir / "frangi_pipeline",
                                                           None, ecfg_faz))
        # Hash-bound review sheet via the real export path, then automated fixture
        # decisions that are explicitly non-human. They are accepted only for synthetic data.
        from .experiments.exp_e import external_faz_spec
        from .qc_export import export_review

        sel = pd.DataFrame({"image_id": sm.image_id.head(int(cfg["exp_e"]["min_qc_audited"]) + 2)})
        sel.to_csv(ext_dir / "qc_selection.csv", index=False)
        exp_ = _step(log, "qc_export", lambda: export_review(ext_dir / "frangi", ext_dir / "qc_selection.csv",
                                                             cfg, root, ext_dir / "qc_review",
                                                             external_faz_spec(cfg)))
        if exp_:
            qc = pd.read_csv(exp_["qc_sheet"], dtype=str, keep_default_na=False)
            qc["reviewer"], qc["review_date"] = "SYNTHETIC_FIXTURE_NOT_HUMAN", "2026-09-29"
            qc["decision"] = "pass"
            qc.to_csv(ext_dir / "qc_synthetic.csv", index=False)
        re_ = _step(log, "exp_e_frangi", lambda: exp_e.run(cfg, sm, "frangi", ext_dir / "qc_synthetic.csv",
                                                          ext_dir / "frangi_pipeline", ext_dir / "exp_e_frangi"))
        summary["exp_e_frangi_cv"] = re_["cv"] if re_ else None
        # the same gate must refuse the REAL DRAC queue (no human QC recorded yet)
        real_qc = Path("data/manifests/drac_qc_selection.csv")
        if real_qc.exists():
            from .qc import qc_status

            summary["real_drac_qc_gate"] = qc_status(real_qc, "frangi", 30,
                                                     float(cfg["exp_e"]["max_qc_fail_fraction"]))["reasons"]

    ts = torch_status()
    summary["torch"] = ts
    if ts["available"]:
        model_dir = out / "unet_model"
        meta = _step(log, "unet_train", lambda: segrun.unet_train_manifest(
            m, cfg, root, cal["calibration_ids"], model_dir, patches_per_epoch=16 if quick else 64))
        if meta:
            summary["unet_train"] = {"n_train": meta["n_train"], "loss_first_last":
                                     [meta["history"][0]["loss"], meta["history"][-1]["loss"]]}
            evalm = m[m.seg_split.isin(["val", "test"])]
            pm = _step(log, "unet_predict_octa_fixture", lambda: segrun.unet_predict_manifest(
                evalm, cfg, root, model_dir, out / "unet_eval", splits=("train",)))
            if pm:
                d = _step(log, "unet_dice", lambda: segrun.evaluate_against_reference(
                    m, cfg, root, pm, cal["calibration_ids"], meta))
                if d is not None:
                    d.to_csv(out / "unet_eval" / "dice.csv", index=False)
                    summary["unet_dice_median"] = float(d.dice.median()) if len(d) else None
            leak = _step(log, "unet_leakage_guard_expected_failure", lambda: segrun.unet_predict_manifest(
                m[m.seg_split.eq("train") & ~m.image_id.isin(cal["calibration_ids"])].head(2), cfg, root,
                model_dir, out / "unet_leak", splits=("train",)))
            summary["unet_leakage_guard_raised"] = leak is None
        emb = _step(log, "embeddings_resnet50_RANDOM_INIT_TEST_ONLY", lambda: embeddings.extract(
            [read_intensity(p)[0] for p in em.image_path], "resnet50", cfg["embeddings"], _random_init_for_tests=True))
        if emb is not None and seg_manifest:
            X, emeta = emb
            sm = load_manifest(seg_manifest, cfg)
            feats = pd.read_csv(ext_dir / "frangi_pipeline" / "features.csv")
            t = sm[["image_id", "group_key", "grade_num"]].merge(feats, on="image_id")
            rf = _step(log, "exp_f_plumbing", lambda: exp_f.run(
                cfg, t, np.array(em.image_id.tolist()), X, emeta, ext_dir / "exp_f",
                {"data_provenance": SYNTHETIC_BANNER, "grouping_level": "image"}))
            summary["exp_f"] = rf["delta_b_minus_a"] if rf else None
    abl = _step(log, "ablations", lambda: ablations.run(m, SYNTHETIC_DATASET, cal, cfg, root, out / "ablations",
                                                         n_workers=n_workers))
    if abl is not None:
        summary["ablations"] = abl.drop(columns=[c for c in ("traceback",) if c in abl]).to_dict("records")
    # Keep the report compact: per-image ablation outputs and U-Net weights are
    # regenerable; their run-level summaries and hashes are kept.
    import shutil

    pruned = []
    for d in sorted((out / "ablations").glob("*/pipeline/images")):
        shutil.rmtree(d)
        pruned.append(str(d.relative_to(out)))
    w = out / "unet_model" / "model.pt"
    if w.exists():
        w.unlink()
        pruned.append(str(w.relative_to(out)) + " (sha256 in meta.json)")
    summary["pruned_after_run"] = pruned
    summary["steps"] = log
    summary["n_failed_steps"] = sum(1 for s in log if s["status"] == "failed"
                                    and "expected_failure" not in s["step"])
    write_json(out / "e2e_summary.json", summary)
    return {k: summary[k] for k in ("data_provenance", "n_failed_steps", "exp_b_primary")
            if k in summary} | {"steps": [(s["step"], s["status"]) for s in log]}
