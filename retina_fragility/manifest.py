"""Manifest schema, audited input checks and explicit analysis eligibility.

Relative file paths resolve against the project root supplied by the caller,
never against the manifest's own directory. Subject IDs stay missing when
unknown; a separate grouping key records the image-level assumption.

Safeguards (resume_review.md A3):
* numeric fields are converted explicitly; invalid values are errors;
* sex is coded through the explicit ``data.sex_map``; unmapped values are errors;
* a ``split == test`` row carrying a grade or disease label is an error
  (public DRAC test labels are hidden, so a value means contamination);
* frozen calibration IDs must exist in the manifest and are excluded from
  every analysis row set.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .io_masks import load_reference_masks, read_plane

REQUIRED_COLUMNS = (
    "image_id", "dataset", "subject_id", "image_path", "vessel_mask_path",
    "faz_mask_path", "fov_mm", "pixel_size_um", "disease", "age", "sex",
)
# ``split`` is the label-availability split of the release: ``test`` means hidden
# labels (DRAC). ``seg_split`` is the segmentation-development split (e.g. the
# OCTA-500 official train/val/test for vessel segmentation, whose disease labels
# are all public). The two must not be conflated.
OPTIONAL_DEFAULTS = (("large_vessel_mask_path", ""), ("split", ""), ("seg_split", ""),
                     ("grade", ""), ("calibration", "false"), ("qc_audited", "false"))
SYNTHETIC_PREFIX = "SYNTHETIC"


class ManifestError(ValueError):
    pass


def parse_bool(value: object, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", "", "none", "nan"}:
        return False
    raise ManifestError(f"Invalid boolean in {field}: {value!r}")


def _to_float(series: pd.Series, field: str) -> pd.Series:
    blank = series.eq("")
    out = pd.to_numeric(series.where(~blank, None), errors="coerce")
    bad = out.isna() & ~blank
    if bad.any():
        raise ManifestError(f"Non-numeric {field} for: {series.index[bad].tolist()[:5]}")
    return out.astype(float)


def is_synthetic(frame: pd.DataFrame) -> bool:
    return bool(frame["dataset"].astype(str).str.upper().str.startswith(SYNTHETIC_PREFIX).any())


def load_manifest(path: str | Path, cfg: dict) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ManifestError(f"Missing manifest columns: {sorted(missing)}")
    if frame.empty:
        raise ManifestError("Manifest has no rows")
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    if frame.image_id.eq("").any() or frame.image_id.duplicated().any():
        raise ManifestError("image_id must be nonempty and globally unique")
    if frame.dataset.eq("").any():
        raise ManifestError("dataset must be supplied for every image")
    for name, default in OPTIONAL_DEFAULTS:
        if name not in frame:
            frame[name] = default
    frame.index = frame.image_id.values
    for name in ["calibration", "qc_audited"]:
        frame[name] = frame[name].map(lambda value, n=name: parse_bool(value, n))
    # Public test labels must not be present (DRAC test-label isolation).
    test = frame.split.str.lower().eq("test")
    leaked = test & (frame.grade.ne("") | frame.disease.ne(""))
    if leaked.any():
        raise ManifestError("Test-split rows carry grade/disease labels (label leakage): "
                            f"{frame.image_id[leaked].tolist()[:5]}")
    for name in ["fov_mm", "pixel_size_um", "age"]:
        frame[name] = _to_float(frame[name], name)
    frame["grade_num"] = _to_float(frame["grade"], "grade")
    sex_map = cfg["data"]["sex_map"]
    unknown = sorted(set(frame.sex[frame.sex.ne("")]) - set(sex_map))
    if unknown:
        raise ManifestError(f"Sex values without an explicit data.sex_map entry: {unknown}")
    frame["sex_male"] = frame.sex.map(lambda v: float(sex_map[v]) if v else np.nan)
    mapping = cfg["data"]["disease_map"]
    # Case/label changes are not inferred. Explicit mappings must describe the release.
    frame["analysis_group"] = frame.disease.map(mapping)
    frame["subject_id_assumed"] = frame.subject_id.eq("")
    frame["group_key"] = frame.subject_id
    if cfg["data"]["subject_id_policy"] == "image_is_subject":
        assumed = frame.subject_id_assumed
        frame.loc[assumed, "group_key"] = frame.loc[assumed, "image_id"]
    elif frame.subject_id_assumed.any():
        raise ManifestError("data.subject_id_policy=require but subject IDs are missing")
    return frame


def grouping_level(frame: pd.DataFrame) -> str:
    if frame["subject_id_assumed"].all():
        return "image (subject IDs unavailable; independence of images unverified)"
    if frame["subject_id_assumed"].any():
        return "mixed (subject IDs where provided, image IDs otherwise)"
    return "subject"


def resolve_input(value: object, root: str | Path) -> Path:
    text = str(value).strip()
    if not text:
        raise ManifestError("Missing input path")
    path = Path(text)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def validate_manifest(
    frame: pd.DataFrame, cfg: dict, root: str | Path, *, reference_masks: bool = True,
) -> dict:
    """Decode every required input and return per-image failures without exclusions.

    This is a readiness check, not permission to drop failed images silently.
    Missing demographic covariates are reported; experiment-specific complete
    case decisions must later be saved in each statistical analysis audit.
    """
    rows = []
    geometry = cfg["data"]["expected_geometry"]
    tolerance = cfg["data"]["pixel_size_rel_tolerance"]
    for row in frame.to_dict("records"):
        errors, warnings = [], []
        decoded = None
        try:
            image = read_plane(resolve_input(row["image_path"], root))
            shape = image.shape
        except (OSError, ValueError) as exc:
            errors.append(f"image: {exc}")
            shape = None
        if reference_masks:
            try:
                large = row.get("large_vessel_mask_path", "")
                decoded = load_reference_masks(
                    resolve_input(row["vessel_mask_path"], root),
                    resolve_input(row["faz_mask_path"], root),
                    cfg["data"]["label_maps"],
                    resolve_input(large, root) if large else None,
                )
                if shape is not None and decoded[0].shape != shape:
                    errors.append("Image and reference mask dimensions differ")
                if not decoded[0].any():
                    errors.append("Reference vessel mask is empty")
                if not decoded[1].any():
                    warnings.append("Reference FAZ mask is empty; verify annotation convention")
            except (OSError, ValueError) as exc:
                errors.append(f"reference_masks: {exc}")
        try:
            fov, pixel_size = float(row["fov_mm"]), float(row["pixel_size_um"])
            if not np.isfinite([fov, pixel_size]).all() or min(fov, pixel_size) <= 0:
                raise ValueError("Physical scale must be finite and positive")
            if shape is not None:
                if not np.allclose(np.array(shape) * pixel_size, fov * 1000,
                                   rtol=tolerance, atol=0):
                    errors.append("Physical pixel size, FOV and image dimensions disagree")
                # OCTA-500's expected dimensions do not constrain other datasets.
                if str(row["dataset"]).upper().replace("-", "") == "OCTA500":
                    expected = geometry.get(f"{fov:g}")
                    if expected is None:
                        errors.append("OCTA-500 FOV is absent from expected_geometry")
                    elif list(shape) != list(expected["shape"]) or not np.isclose(
                        pixel_size, expected["pixel_size_um"], rtol=tolerance, atol=0,
                    ):
                        errors.append("OCTA-500 dimensions/scale disagree with configured geometry")
        except (ValueError, TypeError) as exc:
            errors.append(f"physical_scale: {exc}")
        if not row["subject_id"]:
            if cfg["data"]["subject_id_policy"] == "require":
                errors.append("Subject ID required but missing")
            else:
                warnings.append("Unknown subject ID: group_key uses image_id; independence unverified")
        for covariate in ["age", "sex"]:
            value = row[covariate]
            if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
                warnings.append(f"Missing {covariate}; primary complete-case analysis cannot use this row")
        if pd.isna(row.get("analysis_group")) and str(row.get("split")).lower() != "test":
            warnings.append("Disease label has no configured primary-analysis mapping")
        rows.append({"image_id": row["image_id"], "valid_inputs": not errors,
                     "errors": errors, "warnings": warnings,
                     "shape": list(shape) if shape is not None else None,
                     "reference_masks_checked": reference_masks})
    return {"n_images": len(rows), "n_valid_inputs": sum(r["valid_inputs"] for r in rows),
            "all_inputs_valid": all(r["valid_inputs"] for r in rows), "rows": rows,
            "note": "Input validity does not establish segmentation quality or analysis eligibility."}


def analysis_rows(
    frame: pd.DataFrame,
    calibration_ids: Iterable[str],
    *,
    dataset: str,
    groups: tuple[str, ...] = ("normal", "dr"),
    require_calibration_present: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude calibration and unlabeled/test rows, returning a complete audit.

    Callers must pass the IDs from the frozen calibration record; calibration
    flags in the manifest are additionally honored, never used to override it.
    A frozen ID missing from the manifest means the calibration record does not
    belong to this manifest and raises ``ManifestError``.
    """
    frozen = set(str(i) for i in calibration_ids)
    if require_calibration_present:
        absent = sorted(frozen - set(frame.image_id))
        if absent:
            raise ManifestError(f"Frozen calibration IDs absent from manifest: {absent[:5]}")
    selected, audit = [], []
    for index, row in frame.iterrows():
        reasons = []
        if row.dataset != dataset:
            reasons.append("different_dataset")
        if row.image_id in frozen or parse_bool(row.calibration, "calibration"):
            reasons.append("calibration")
        if str(row.split).lower() == "test":
            reasons.append("held_out_test")
        if row.analysis_group not in groups:
            reasons.append("disease_not_in_analysis")
        if not reasons:
            selected.append(index)
        audit.append({"image_id": row.image_id, "included": not reasons,
                      "reasons": ";".join(reasons)})
    return frame.loc[selected].copy(), pd.DataFrame(audit)


def assert_no_calibration(ids: Iterable[str], calibration_ids: Iterable[str], where: str) -> None:
    """Guard used on every experiment output table."""
    leak = sorted(set(map(str, ids)) & set(map(str, calibration_ids)))
    if leak:
        raise ManifestError(f"Calibration images leaked into {where}: {leak[:5]}")
