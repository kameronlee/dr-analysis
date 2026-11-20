"""Shared loading of analysis tables and output headers for experiments A-F."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import config_hash
from ..manifest import analysis_rows, assert_no_calibration, grouping_level, is_synthetic
from ..provenance import SYNTHETIC_BANNER, environment, now_utc, read_json, write_json


class ReadinessError(RuntimeError):
    """Raised with a precise list of unmet prerequisites; no results are produced."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


def load_analysis_table(pipeline_root: str | Path, manifest: pd.DataFrame, calibration: dict | None,
                        dataset: str, groups=("normal", "dr"),
                        require_calibration: bool = True,
                        expected_d_um: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Merge per-image features with manifest covariates for eligible (non-calibration) rows.

    The pipeline's ``run.json`` must name the same calibration IDs and the
    expected coverage distance: the frozen d, or ``expected_d_um`` for a
    labelled d-percentile ablation.
    """
    pipeline_root = Path(pipeline_root)
    reasons = []
    if calibration is None and require_calibration:
        reasons.append("no frozen calibration record (calibration IDs must be excluded first)")
    fpath = pipeline_root / "features.csv"
    if not fpath.exists():
        reasons.append(f"missing pipeline features: {fpath}")
    run_p = pipeline_root / "run.json"
    if calibration is not None and run_p.exists():
        run = read_json(run_p)
        ids_h = run.get("calibration_ids_sha256")
        if ids_h is not None and ids_h != calibration.get("calibration_ids_sha256"):
            reasons.append("pipeline was processed with a different frozen calibration (ID hash differs)")
        d_run = run.get("d_um")
        d_want = calibration.get("d_um", np.nan) if expected_d_um is None else expected_d_um
        if d_run is not None and not np.isclose(d_run, d_want, rtol=1e-9, atol=0):
            reasons.append(f"pipeline coverage used d = {d_run} µm, expected {d_want} µm "
                           "(frozen calibration unless a d-percentile ablation)")
    if reasons:
        raise ReadinessError(reasons)
    cal_ids = list(calibration["calibration_ids"]) if calibration else []
    rows, audit = analysis_rows(manifest, cal_ids, dataset=dataset, groups=tuple(groups),
                                require_calibration_present=bool(calibration))
    feats = pd.read_csv(fpath)
    feats["image_id"] = feats["image_id"].astype(str)
    excl_path = pipeline_root / "exclusions.csv"
    excl = pd.read_csv(excl_path) if excl_path.exists() else pd.DataFrame(columns=["image_id", "reason"])
    cov = rows[["image_id", "dataset", "group_key", "subject_id_assumed", "analysis_group",
                "fov_mm", "pixel_size_um", "age", "sex_male", "grade_num", "split"]].reset_index(drop=True)
    table = cov.merge(feats, on="image_id", how="inner", suffixes=("", "_feat"))
    missing = sorted(set(cov.image_id) - set(table.image_id))
    miss_reason = excl.set_index("image_id").reason.to_dict() if len(excl) else {}
    table = table.copy()
    table["y"] = (table.analysis_group == "dr").astype(int)
    table["fov_6mm"] = (table.fov_mm == 6.0).astype(float)
    assert_no_calibration(table.image_id, cal_ids, "analysis table")
    info = {"n_manifest_rows": int(len(manifest)), "n_analysis_rows": int(len(rows)),
            "n_with_features": int(len(table)),
            "rows_without_features": [{"image_id": i, "pipeline_reason": miss_reason.get(i, "not_processed")}
                                      for i in missing],
            "manifest_exclusions": audit[~audit.included].reasons.value_counts().to_dict(),
            "n_calibration_excluded": int(audit.reasons.str.contains("calibration").sum()),
            "grouping_level": grouping_level(rows) if len(rows) else None,
            "data_provenance": SYNTHETIC_BANNER if is_synthetic(manifest) else "real"}
    return table, info


def header(cfg: dict, experiment: str, role: str, info: dict, calibration: dict | None) -> dict:
    return {"experiment": experiment, "role": role, "created_utc": now_utc(),
            "data_provenance": info.get("data_provenance"),
            "grouping_level": info.get("grouping_level"),
            "config_sha256": config_hash(cfg),
            "calibration_ids_sha256": (calibration or {}).get("calibration_ids_sha256"),
            "calibration_d_um": (calibration or {}).get("d_um"),
            "analysis_table_audit": info, "environment": environment(),
            "interpretation_boundary": ("Cross-sectional structural association only; not a "
                                        "flow model, progression predictor or grader.")}


def write_results(out_dir: str | Path, results: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if results.get("data_provenance") == SYNTHETIC_BANNER:
        (out / "SYNTHETIC_NOT_CLINICAL.txt").write_text(
            "These outputs were computed from SYNTHETIC fixtures. They validate software only "
            "and are NOT clinical findings.\n")
    write_json(out / "results.json", results)
    return out / "results.json"


def finite(x) -> np.ndarray:
    return np.isfinite(np.asarray(x, float))
