"""OCTA-500 data contract: analysis manifest, readiness audit, label checks and graph QC.

Decisions implemented here are recorded before any OCTA-500 outcome in
``docs/OCTA500_PREANALYSIS_NOTE_2026-09-30.md``:

* ``image_path`` is the configured primary projection (paper B5, OCTA max
  projection ILM-OPL), on whose grid the labels were drawn.
* ``vessel_mask_path`` / ``large_vessel_mask_path`` are the released
  GT_Capillary / GT_LargeVessel files; the loaders form their union in memory,
  so no derived mask is written and every path is a verified original.
* ``subject_id`` is the released subject number (one eye per subject, paper
  section 3.1). ``split`` is the label-availability split (``public``: every
  OCTA-500 disease label is released); ``seg_split`` is the official
  segmentation split. The two are never conflated.
* Rows are never dropped. Disease labels outside ``data.disease_map`` stay in
  the manifest; the readiness audit gives every row's eligibility per use.

Nothing in this module computes FI, stress curves or group comparisons.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import analysis_rows, parse_bool, resolve_input
from .provenance import sha256_file

CATALOG_REQUIRED = ("source_subject_id", "image_id", "dataset", "fov_mm", "pixel_size_um", "disease",
                    "age", "sex", "eye", "metadata_path", "metadata_sha256")
EYES = ("OD", "OS")
ALIGNMENT_VARIANTS = ("identity", "transpose", "fliplr", "flipud")


class Octa500Error(ValueError):
    pass


def seg_split_for(source_subject_id: str, cfg: dict) -> str:
    """Official segmentation split of a released subject number (inclusive ranges)."""
    number = int(str(source_subject_id))
    hits = [name for name, spans in cfg["octa500"]["seg_split_ranges"].items()
            for lo, hi in spans if int(lo) <= number <= int(hi)]
    if len(hits) != 1:
        raise Octa500Error(f"subject {source_subject_id} is in {len(hits)} official split ranges")
    return hits[0]


def _file_columns(cfg: dict) -> dict[str, str]:
    """Manifest role -> catalog column prefix, for every source file the manifest names."""
    o = cfg["octa500"]
    cols = {"image": o["primary_projection"], "vessel_mask": o["vessel_label"],
            "faz_mask": o["faz_label"]}
    if o["large_vessel_label"]:
        cols["large_vessel_mask"] = o["large_vessel_label"]
    for name in o["context_projections"]:
        cols[f"context_{name}"] = name
    return cols


def _calibration_problems(calibration: dict, frame: pd.DataFrame, cfg: dict) -> list[str]:
    """The frozen record must belong to exactly these rows and these source files."""
    problems = []
    if calibration.get("dataset") != cfg["data"]["primary_dataset"]:
        problems.append(f"calibration dataset {calibration.get('dataset')!r} is not the primary dataset")
    idx = frame.set_index("image_id")
    for iid in calibration["calibration_ids"]:
        if iid not in idx.index:
            problems.append(f"{iid}: calibration ID absent from the catalog")
            continue
        r = idx.loc[iid]
        if cfg["data"]["disease_map"].get(r["disease"]) != cfg["calibration"]["eligible_group"]:
            problems.append(f"{iid}: calibration eye is not in the eligible group")
        stored = (calibration.get("input_sha256") or {}).get(iid, {})
        for key, col in (("vessel", "vessel_mask_sha256"), ("faz", "faz_mask_sha256"),
                         ("large_vessel", "large_vessel_mask_sha256")):
            if key in stored and stored[key] != r.get(col):
                problems.append(f"{iid}: calibration {key} hash differs from the verified source file")
    return problems


def build_manifest(catalog: pd.DataFrame, cfg: dict, root: str | Path,
                   calibration: dict | None = None,
                   verify_hashes: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (analysis manifest, per-row readiness audit) for every catalog row.

    Each referenced file must exist and, with ``verify_hashes``, match the
    extraction-verified SHA256 in the catalog. Problems are recorded per row in
    ``input_errors``; callers must refuse to write a manifest with any error
    (``manifest_errors``). Missing demographics are reported, never filled.
    """
    missing = [c for c in CATALOG_REQUIRED if c not in catalog.columns]
    files = _file_columns(cfg)
    missing += [f"{p}_{s}" for p in files.values() for s in ("path", "sha256")
                if f"{p}_{s}" not in catalog.columns]
    if missing:
        raise Octa500Error(f"source catalog lacks columns {sorted(set(missing))}")
    cat = catalog.astype(str).apply(lambda s: s.str.strip())
    cat = cat.replace({"nan": ""})
    if cat.image_id.duplicated().any() or cat.source_subject_id.duplicated().any():
        raise Octa500Error("catalog image/subject IDs must be unique (one eye per subject)")
    dataset = cfg["data"]["primary_dataset"]
    geometry = cfg["data"]["expected_geometry"]
    tol = float(cfg["data"]["pixel_size_rel_tolerance"])
    sex_map = cfg["data"]["sex_map"]
    o = cfg["octa500"]
    rows, audit = [], []
    for r in cat.to_dict("records"):
        errors, warnings = [], []
        sid = r["source_subject_id"]
        if not sid.isdigit():
            errors.append(f"non-numeric source subject ID {sid!r}")
        if r["image_id"] != f"{dataset}_{sid}":
            errors.append(f"image_id {r['image_id']!r} does not encode subject {sid!r}")
        if r["dataset"] != dataset:
            errors.append(f"dataset {r['dataset']!r} is not {dataset!r}")
        try:
            seg_split = seg_split_for(sid, cfg)
        except (Octa500Error, ValueError) as exc:
            seg_split = ""
            errors.append(str(exc))
        fov_key = f"{float(r['fov_mm']):g}" if r["fov_mm"] else ""
        if fov_key not in geometry:
            errors.append(f"FOV {r['fov_mm']!r} has no configured geometry")
        elif not np.isclose(float(r["pixel_size_um"]), geometry[fov_key]["pixel_size_um"], rtol=tol, atol=0):
            errors.append("catalog pixel size differs from the configured geometry")
        for field in ("age", "sex", "eye", "disease"):
            if r[field] == "":
                warnings.append(f"missing {field} (not imputed)")
        if r["sex"] and r["sex"] not in sex_map:
            errors.append(f"sex {r['sex']!r} has no data.sex_map entry")
        if r["eye"] and r["eye"] not in EYES:
            errors.append(f"eye {r['eye']!r} is not one of {EYES}")
        verified = {}
        for role, prefix in {**files, "metadata": "metadata"}.items():
            rel, expected = r[f"{prefix}_path"], r[f"{prefix}_sha256"]
            try:
                path = resolve_input(rel, root)
                if not path.is_file():
                    raise FileNotFoundError(rel)
                if verify_hashes and sha256_file(path) != expected:
                    raise ValueError("SHA256 differs from the extraction-verified catalog value")
                verified[role] = (rel, expected)
            except (OSError, ValueError) as exc:
                errors.append(f"{role} ({rel}): {exc}")
                verified[role] = (rel, expected)
        group = cfg["data"]["disease_map"].get(r["disease"])
        row = {
            "image_id": r["image_id"], "dataset": dataset, "subject_id": sid,
            "source_subject_id": sid, "subject_id_source": "released subject number; one eye per subject "
                                                          "(Li et al. 2024, section 3.1)",
            "image_path": verified["image"][0], "image_projection": o["primary_projection"],
            "vessel_mask_path": verified["vessel_mask"][0],
            "large_vessel_mask_path": verified.get("large_vessel_mask", ("", ""))[0],
            "faz_mask_path": verified["faz_mask"][0],
            "fov_mm": r["fov_mm"], "pixel_size_um": r["pixel_size_um"],
            "disease": r["disease"], "grade": "", "age": r["age"], "sex": r["sex"], "eye": r["eye"],
            "split": o["label_split"], "seg_split": seg_split,
            "calibration": "false", "qc_audited": "false",
            "image_sha256": verified["image"][1], "vessel_mask_sha256": verified["vessel_mask"][1],
            "large_vessel_mask_sha256": verified.get("large_vessel_mask", ("", ""))[1],
            "faz_mask_sha256": verified["faz_mask"][1],
            **{f"{name}_path": verified[f"context_{name}"][0] for name in o["context_projections"]},
            **{f"{name}_sha256": verified[f"context_{name}"][1] for name in o["context_projections"]},
            "metadata_path": r["metadata_path"], "metadata_sha256": r["metadata_sha256"],
        }
        rows.append(row)
        audit.append({"image_id": r["image_id"], "source_subject_id": sid, "fov_mm": r["fov_mm"],
                      "seg_split": seg_split, "disease": r["disease"],
                      "analysis_group": group or "",
                      "source_checks_passed": not errors, "hashes_verified": bool(verify_hashes),
                      "input_errors": "; ".join(errors), "warnings": "; ".join(warnings)})
    manifest = pd.DataFrame(rows)
    if calibration is not None:
        problems = _calibration_problems(calibration, manifest, cfg)
        if problems:
            raise Octa500Error("frozen calibration does not match this manifest: " + "; ".join(problems[:10]))
        manifest.loc[manifest.image_id.isin(calibration["calibration_ids"]), "calibration"] = "true"
    audit = pd.DataFrame(audit)
    return manifest, eligibility_audit(audit, manifest, cfg)


def eligibility_audit(audit: pd.DataFrame, manifest: pd.DataFrame, cfg: dict,
                      validation: dict | None = None) -> pd.DataFrame:
    """Add explicit per-use eligibility and reasons for every row (nothing is dropped).

    ``validation`` is ``manifest.validate_manifest`` output; without it input
    decoding is recorded as not yet checked and nothing is marked ready.
    """
    a = audit.drop(columns=[c for c in audit.columns if c.startswith(("eligible_", "reasons_"))
                            or c in ("calibration", "valid_inputs", "validation_errors",
                                     "validation_warnings", "inputs_decoded")]).copy()
    cal = manifest.set_index("image_id").calibration.map(lambda v: parse_bool(v, "calibration"))
    a["calibration"] = a.image_id.map(cal).fillna(False).astype(bool)
    if validation is None:
        a["inputs_decoded"] = False
        a["valid_inputs"] = False
        a["validation_errors"] = "not yet validated"
        a["validation_warnings"] = ""
    else:
        by_id = {r["image_id"]: r for r in validation["rows"]}
        a["inputs_decoded"] = a.image_id.isin(by_id)
        a["valid_inputs"] = a.image_id.map(lambda i: bool(by_id.get(i, {}).get("valid_inputs", False)))
        a["validation_errors"] = a.image_id.map(lambda i: "; ".join(by_id.get(i, {}).get("errors", ["absent"])))
        a["validation_warnings"] = a.image_id.map(lambda i: "; ".join(by_id.get(i, {}).get("warnings", [])))
    group_ok = a.analysis_group.isin(["normal", "dr"])
    base = []
    for r in a.itertuples():
        reasons = []
        if r.input_errors:
            reasons.append("source_file_error")
        if not r.valid_inputs:
            reasons.append("inputs_not_validated_or_invalid")
        base.append(reasons)
    uses = {
        "primary_analysis": lambda r, g: ([] if g else [f"disease_not_in_primary_analysis:{r.disease}"])
        + (["calibration"] if r.calibration else []),
        "calibration_pool": lambda r, g: ([] if r.analysis_group == cfg["calibration"]["eligible_group"]
                                          else ["not_calibration_eligible_group"]),
        "exp_c_donor_pool": lambda r, g: ([] if r.analysis_group == "normal" else ["not_normal"])
        + (["calibration"] if r.calibration else []),
        "unet_training": lambda r, g: ([] if r.seg_split == "train" else ["not_official_seg_train"])
        + (["calibration"] if r.calibration else []),
        "segmentation_evaluation": lambda r, g: ([] if r.seg_split in ("val", "test")
                                                 else ["official_seg_train"])
        + (["calibration"] if r.calibration else []),
    }
    for use, rule in uses.items():
        reasons = [b + rule(r, g) for b, r, g in zip(base, a.itertuples(), group_ok)]
        a[f"eligible_{use}"] = [not x for x in reasons]
        a[f"reasons_{use}"] = [";".join(x) for x in reasons]
    return a


def manifest_errors(audit: pd.DataFrame) -> list[str]:
    bad = audit[audit.input_errors.astype(str).ne("")]
    return [f"{r.image_id}: {r.input_errors}" for r in bad.itertuples()]


# ------------------------------------------------------------ label/projection checks
def auroc_levels(values: np.ndarray, positive: np.ndarray) -> float:
    """Exact Mann-Whitney AUROC P(pos > neg) + 0.5 P(tie), over discrete intensity levels."""
    values = np.asarray(values).ravel()
    positive = np.asarray(positive, bool).ravel()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    levels, inverse = np.unique(values, return_inverse=True)
    pos = np.bincount(inverse[positive], minlength=len(levels)).astype(float)
    neg = np.bincount(inverse[~positive], minlength=len(levels)).astype(float)
    below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    return float((pos * (below + 0.5 * neg)).sum() / (n_pos * n_neg))


def _variant(mask: np.ndarray, name: str) -> np.ndarray:
    return {"identity": mask, "transpose": mask.T, "fliplr": mask[:, ::-1],
            "flipud": mask[::-1, :]}[name]


def alignment_scores(intensities: dict[str, np.ndarray], vessel: np.ndarray) -> dict:
    """AUROC of each projection's intensity for the vessel mask in each orientation."""
    out = {}
    for proj, img in intensities.items():
        if img.shape != vessel.shape:
            raise Octa500Error(f"{proj} shape {img.shape} differs from mask {vessel.shape}")
        for v in ALIGNMENT_VARIANTS:
            out[f"auroc_{proj}_{v}"] = auroc_levels(img, _variant(vessel, v))
    return out


def summarize_alignment(table: pd.DataFrame, cfg: dict) -> dict:
    """Apply the prespecified D1 criteria (no disease labels involved)."""
    o = cfg["octa500"]
    projections = [o["primary_projection"], *o["context_projections"]]
    medians = {p: float(table[f"auroc_{p}_identity"].median()) for p in projections}
    primary = o["primary_projection"]
    others = [f"auroc_{primary}_{v}" for v in ALIGNMENT_VARIANTS if v != "identity"]
    beats = (table[f"auroc_{primary}_identity"].values[:, None] > table[others].values).all(axis=1)
    frac = float(beats.mean())
    best = max(medians, key=medians.get)
    need = float(o["alignment_min_identity_fraction"])
    return {"median_identity_auroc_by_projection": medians,
            "median_auroc_by_variant_primary": {v: float(table[f"auroc_{primary}_{v}"].median())
                                                for v in ALIGNMENT_VARIANTS},
            "highest_median_projection": best,
            "primary_identity_beats_all_variants_fraction": frac,
            "criterion_primary_highest": best == primary,
            "criterion_identity_fraction": frac >= need,
            "required_identity_fraction": need,
            "passed": bool(best == primary and frac >= need), "n_images": len(table)}


def cavf_consistency(capillary: np.ndarray, large: np.ndarray, faz: np.ndarray, artery: np.ndarray,
                     vein: np.ndarray, cavf: np.ndarray) -> dict:
    """Pixel counts relating GT_CAVF values to the individual released masks."""
    cap_only = capillary & ~large
    return {
        "cavf1_xor_capillary_minus_large_px": int(np.count_nonzero((cavf == 1) ^ cap_only)),
        "cavf1_xor_capillary_minus_large_minus_faz_px": int(np.count_nonzero((cavf == 1) ^ (cap_only & ~faz))),
        "artery_outside_large_px": int(np.count_nonzero(artery & ~large)),
        "vein_outside_large_px": int(np.count_nonzero(vein & ~large)),
        "large_not_artery_or_vein_px": int(np.count_nonzero(large & ~artery & ~vein)),
        "cavf0_inside_large_px": int(np.count_nonzero((cavf == 0) & large)),
        "cavf0_inside_capillary_px": int(np.count_nonzero((cavf == 0) & capillary)),
    }


# ------------------------------------------------------------------------ graph QC
def graph_qc_key(base_seed: int, purpose: str, image_id: str) -> str:
    return hashlib.sha256(f"{int(base_seed)}|{purpose}|{image_id}".encode()).hexdigest()


def select_graph_qc(manifest: pd.DataFrame, cfg: dict, calibration_ids) -> pd.DataFrame:
    """Outcome-blind QC selection from the primary cohort (DR or NORMAL, non-calibration).

    ``manifest`` is a loaded manifest (``manifest.load_manifest``). Eligibility
    is used; group membership is not (no stratification by disease).
    """
    rows, _ = analysis_rows(manifest, calibration_ids, dataset=cfg["data"]["primary_dataset"])
    purpose = cfg["qc"]["octa500_graph_selection_purpose"]
    seed = cfg["project"]["base_seed"]
    picks = []
    for fov, n in cfg["qc"]["octa500_graph_n_per_fov"].items():
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise Octa500Error("qc.octa500_graph_n_per_fov values must be positive integers")
        pool = rows[rows.fov_mm.map(lambda f: f"{float(f):g}") == str(fov)].copy()
        pool["selection_key"] = pool.image_id.map(lambda i: graph_qc_key(seed, purpose, i))
        pool = pool.sort_values("selection_key")
        if len(pool) < n:
            raise Octa500Error(f"FOV {fov}: {len(pool)} eligible QC images, {n} required")
        picks.append(pool.head(n)[["image_id", "fov_mm", "selection_key"]])
    out = pd.concat(picks, ignore_index=True)
    out.insert(0, "review_id", [f"octa500-graph-{i:02d}" for i in range(1, len(out) + 1)])
    return out


QC_SHEET_COLUMNS = ("review_id", "image_id", "fov_mm", "image_sha256", "vessel_mask_sha256",
                    "large_vessel_mask_sha256", "faz_mask_sha256", "graph_sha256", "graph_review_sha256",
                    "pipeline_code_sha256", "reviewer", "review_date", "decision", "fail_reason", "notes")


def _hstack(images: list[np.ndarray], gap: int = 4) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    parts = []
    for n, img in enumerate(images):
        if n:
            parts.append(np.full((h, gap, 3), 255, np.uint8))
        parts.append(img)
    return np.concatenate(parts, axis=1)


def export_graph_qc(manifest: pd.DataFrame, selection: pd.DataFrame, cfg: dict, root: str | Path,
                    out_dir: str | Path, calibration: dict) -> dict:
    """Write native-resolution source/label/graph review images for the QC selection.

    Graphs come from ``pipeline.build_graph`` on the manifest's union vessel
    mask with the reference FAZ, i.e. exactly the analysis graph before stress
    testing. Stress tests, FI and biomarkers are never computed. Every input is
    re-hashed against the manifest; the review sheet binds decisions to those
    hashes, the graph file, the graph-construction identity and the pipeline
    code fingerprint. Reviewer fields are left blank.
    """
    from PIL import Image

    from . import viz
    from .config import config_hash
    from .io_masks import read_intensity
    from .pipeline import build_graph, load_row_masks, pipeline_code_fingerprint
    from .provenance import now_utc, sha256_strings, write_json
    from .qc import graph_review_identity

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise Octa500Error(f"{out_dir} exists and is not empty; QC exports are never overwritten")
    cal_ids = set(calibration["calibration_ids"])
    leak = sorted(set(selection.image_id) & cal_ids)
    if leak:
        raise Octa500Error(f"calibration eyes selected for graph QC: {leak}")
    faz_spec = {"mode": "reference"}
    gid = graph_review_identity(cfg, faz_spec)
    code = pipeline_code_fingerprint()
    o = cfg["octa500"]
    idx = manifest.set_index("image_id", drop=False)
    records, sheet = [], []
    for sel in selection.itertuples():
        r = idx.loc[sel.image_id].to_dict()
        checks = {"image": ("image_path", "image_sha256"),
                  "vessel_mask": ("vessel_mask_path", "vessel_mask_sha256"),
                  "large_vessel_mask": ("large_vessel_mask_path", "large_vessel_mask_sha256"),
                  "faz_mask": ("faz_mask_path", "faz_mask_sha256"),
                  **{f"context_{p}": (f"{p}_path", f"{p}_sha256") for p in o["context_projections"]}}
        hashes = {}
        for role, (pcol, hcol) in checks.items():
            if not str(r.get(pcol, "")):
                continue
            h = sha256_file(resolve_input(r[pcol], root))
            if h != r[hcol]:
                raise Octa500Error(f"{sel.image_id}: {role} hash differs from the manifest")
            hashes[role] = h
        masks = load_row_masks(r, cfg, root, faz_spec)
        if masks["intensity_note"] != "ok":
            raise Octa500Error(f"{sel.image_id}: primary projection unusable ({masks['intensity_note']})")
        px = masks["pixel_size_um"]
        g = build_graph(masks["vessel"], masks["faz"], px, cfg,
                        meta={"image_id": sel.image_id, "dataset": r["dataset"], "faz": faz_spec})
        clean = g["clean"]
        item = out_dir / sel.review_id
        files = viz.qc_composites(masks["intensity"], masks["vessel"], clean if clean.n_edges else None,
                                  item, synthetic=False)
        rgb = np.asarray(Image.open(files["intensity"]).convert("RGB"))
        faz_rgb = viz._blend(rgb, masks["faz"], (60, 140, 255), 0.5)
        Image.fromarray(faz_rgb).save(item / "qc_faz_on_intensity.png")
        panel = [rgb, np.asarray(Image.open(files["mask_on_intensity"]).convert("RGB")),
                 np.asarray(Image.open(files["graph_on_intensity"]).convert("RGB")), faz_rgb]
        Image.fromarray(_hstack(panel)).save(item / "review_panel.png")
        context = {}
        for proj in o["context_projections"]:
            img, _ = read_intensity(resolve_input(r[f"{proj}_path"], root))
            Image.fromarray(viz._to_rgb8(img)).save(item / f"context_{proj}_intensity.png")
            Image.fromarray(viz.graph_overlay_rgb(img, clean)).save(item / f"context_{proj}_graph.png")
            context[proj] = [f"context_{proj}_intensity.png", f"context_{proj}_graph.png"]
        clean.save(item / "graph.json")
        area = float(g["analysis"].sum()) * (px / 1000.0) ** 2
        rec = {"review_id": sel.review_id, "image_id": sel.image_id, "fov_mm": float(r["fov_mm"]),
               "pixel_size_um": px, "primary_projection": o["primary_projection"],
               "vessel_mask": "GT_Capillary OR GT_LargeVessel (union formed in memory)",
               "faz_mask": "GT_FAZ (reference)", "input_sha256": hashes,
               "graph_sha256": sha256_file(item / "graph.json"),
               "graph_review_sha256": gid["graph_review_sha256"],
               "pipeline_code_sha256": code["sha256"], "graph_stats": clean.stats(area),
               "analysis_area_mm2": area, "cleanup_audit": g["audit"],
               "images": {**{k: Path(v).name for k, v in files.items()},
                          "faz_on_intensity": "qc_faz_on_intensity.png", "panel": "review_panel.png",
                          "context": context},
               "status": "review_pending", "fi_computed": False, "stress_tests_run": False,
               "human_qc_completed": False, "disease_label_written": False}
        write_json(item / "provenance.json", rec)
        records.append(rec)
        sheet.append({"review_id": sel.review_id, "image_id": sel.image_id, "fov_mm": r["fov_mm"],
                      "image_sha256": hashes["image"], "vessel_mask_sha256": hashes["vessel_mask"],
                      "large_vessel_mask_sha256": hashes.get("large_vessel_mask", ""),
                      "faz_mask_sha256": hashes["faz_mask"], "graph_sha256": rec["graph_sha256"],
                      "graph_review_sha256": gid["graph_review_sha256"],
                      "pipeline_code_sha256": code["sha256"], "reviewer": "", "review_date": "",
                      "decision": "", "fail_reason": "", "notes": ""})
    pd.DataFrame(sheet, columns=list(QC_SHEET_COLUMNS)).to_csv(out_dir / "qc_sheet_octa500_graph.csv",
                                                               index=False)
    selection.to_csv(out_dir / "selection.csv", index=False)
    _write_index(out_dir, records, o)
    summary = {
        "created_utc": now_utc(), "n_images": len(records),
        "n_per_fov": {str(k): int(v) for k, v in selection.fov_mm.map(lambda f: f"{float(f):g}")
                      .value_counts().sort_index().items()},
        "selection_rule": ("primary cohort (DR or NORMAL, non-calibration, via manifest.analysis_rows); "
                           "per FOV ascending sha256(base_seed|purpose|image_id); no disease stratification"),
        "selection_purpose": cfg["qc"]["octa500_graph_selection_purpose"],
        "calibration_ids_sha256": calibration["calibration_ids_sha256"],
        "calibration_overlap": [], "graph_review_sha256": gid["graph_review_sha256"],
        "graph_review_identity": gid["identity"], "pipeline_code_sha256": code["sha256"],
        "config_sha256": config_hash(cfg), "selection_ids_sha256": sha256_strings(selection.image_id),
        "primary_projection": o["primary_projection"], "context_projections": o["context_projections"],
        "fi_computed": False, "stress_tests_run": False, "clinical_statistics_computed": False,
        "human_qc_completed": False, "disease_labels_in_outputs": False,
        "records": [{k: v for k, v in rec.items() if k != "cleanup_audit"} for rec in records],
        "note": ("Review images of the analysis graph before stress testing. Automated or automated "
                 "inspection is not human QC; reviewer fields are blank."),
    }
    write_json(out_dir / "review_inventory.json", summary)
    return summary


def _write_index(out_dir: Path, records: list[dict], o: dict) -> None:
    import html
    import json

    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             ("<title>OCTA-500 graph review</title><style>body{font:15px system-ui;margin:20px;"
              "background:#f4f5f7;color:#1c2430}section{background:#fff;border:1px solid #ccd3dc;"
              "margin:0 0 20px;padding:12px}img{max-width:100%;height:auto;image-rendering:pixelated}"
              ".ctx{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}"
              "figure{margin:0}figcaption{font-size:13px;padding:4px 0}</style>"),
             "<h1>OCTA-500 graph review (pre-outcome)</h1><p>Panel, left to right: primary projection "
             + html.escape(o["primary_projection"]) + "; vessel union (red); cleaned graph (edges green, "
             "node clusters yellow); reference FAZ (blue). Context projections below. Disease labels "
             "are not shown. No FI or stress test was computed. Nothing here is a QC decision.</p>"]
    for rec in records:
        s = rec["graph_stats"]
        parts.append(f"<section id='{html.escape(rec['review_id'])}'><h2>{html.escape(rec['review_id'])}"
                     f" · {rec['fov_mm']:g} mm</h2><p>{s['n_nodes']} nodes, {s['n_edges']} edges, "
                     f"{s['n_components']} components</p>")
        parts.append(f"<img src='{rec['review_id']}/review_panel.png' alt='review panel'>")
        parts.append("<div class='ctx'>")
        for proj, (plain, graph) in rec["images"]["context"].items():
            for name, label in ((plain, proj), (graph, proj + " + graph")):
                parts.append(f"<figure><figcaption>{html.escape(label)}</figcaption>"
                             f"<img src='{rec['review_id']}/{name}' alt='{html.escape(label)}'></figure>")
        parts.append("</div></section>")
    parts.append("<script type='application/json' id='records'>"
                 + json.dumps([r["review_id"] for r in records]).replace("<", "\\u003c")
                 + "</script></html>")
    (out_dir / "index.html").write_text("\n".join(parts))
