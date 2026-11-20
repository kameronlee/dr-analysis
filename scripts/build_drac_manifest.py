"""Inventory the official DRAC grading release without computing disease features.

Run from project root: .venv/bin/python -m scripts.build_drac_manifest
Test grades and unavailable subject/demographic metadata remain missing.
Physical scale is sourced from the published dataset paper, via config.yaml.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from retina_fragility.config import load_config, config_hash

ROOT = Path(__file__).resolve().parents[1]


def inspect_image(path: Path) -> dict:
    raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        image.load()  # force decoding, not just a valid PNG header
        if image.mode != "L":
            raise ValueError(f"Expected grayscale release image: {path}")
        array = np.asarray(image)
        shape = array.shape
        content_hash = hashlib.sha256(str(shape).encode() + array.tobytes()).hexdigest()
        return {"image_path": str(path.relative_to(ROOT)), "filename": path.name,
                "height_px": shape[0], "width_px": shape[1], "mode": image.mode,
                "image_sha256": raw_hash, "pixel_sha256": content_hash}


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    ecfg = cfg["exp_e"]
    geometry_verified = all(ecfg.get(k) is not None for k in ["pixel_size_um", "fov_mm", "geometry_source"])
    folder = ROOT / "data/raw/drac/C. Diabetic Retinopathy Grading/C. Diabetic Retinopathy Grading"
    labels_path = folder / "2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv"
    labels = pd.read_csv(labels_path)
    if list(labels.columns) != ["image name", "DR grade"]:
        raise ValueError("Unexpected official grading-label schema")
    if labels["image name"].duplicated().any() or not labels["DR grade"].isin([0, 1, 2]).all():
        raise ValueError("Nonunique filenames or invalid official grades")
    corrected_107 = labels.loc[labels["image name"].eq("107.png"), "DR grade"].tolist()
    if corrected_107 != [0]:
        raise ValueError("Known official 107.png label correction is absent")
    lookup = labels.set_index("image name")["DR grade"].to_dict()
    rows = []
    for split, dirname in [("train", "a. Training Set"), ("test", "b. Testing Set")]:
        paths = sorted((folder / "1. Original Images" / dirname).glob("*.png"))
        if not paths:
            raise ValueError(f"No {split} images")
        if split == "train" and set(lookup) != {p.name for p in paths}:
            raise ValueError("Training labels and image filenames are not one-to-one")
        with ThreadPoolExecutor(max_workers=4) as pool:
            inspected = list(pool.map(inspect_image, paths))
        for item in inspected:
            if geometry_verified:
                expected_um = ecfg["fov_mm"] * 1000
                for axis in ["height_px", "width_px"]:
                    if not np.isclose(item[axis] * ecfg["pixel_size_um"], expected_um, rtol=0.005):
                        raise ValueError("DRAC image dimensions disagree with configured source-backed geometry")
            grade = lookup[item["filename"]] if split == "train" else None
            rows.append({"image_id": f"DRAC_C_{split}_{Path(item['filename']).stem}",
                         "dataset": "DRAC", "subject_id": None,
                         "subject_id_status": "not_provided_in_public_release",
                         "split": split, "grade": grade,
                         "disease": {0: "non_dr", 1: "npdr", 2: "pdr"}.get(grade),
                         "age": None, "sex": None, "fov_mm": ecfg.get("fov_mm"),
                         "pixel_size_um": ecfg.get("pixel_size_um"),
                         "physical_scale_status": "verified_from_published_dataset_paper" if geometry_verified else "pending_primary_source_verification",
                         "geometry_source": ecfg.get("geometry_source"),
                         "vessel_mask_path": None, "large_vessel_mask_path": None,
                         "faz_mask_path": None, "calibration": False,
                         "qc_audited": False, "qc_status": "pending",
                         "labels_release": "zenodo:10280359",
                         "label_correction_107_verified": True, **item})
        print(f"Decoded and hashed {len(paths)} {split} images", flush=True)
    frame = pd.DataFrame(rows)
    frame["grade"] = frame["grade"].astype("Int64")
    assert frame.image_id.is_unique
    assert frame.loc[frame.split.eq("test"), "grade"].isna().all()
    out = ROOT / "data/manifests"
    manifest = out / "drac_manifest.csv"
    if manifest.exists():
        old = pd.read_csv(manifest)
        if old.get("qc_audited", pd.Series(dtype=bool)).fillna(False).astype(str).str.lower().eq("true").any():
            raise RuntimeError("Manifest contains completed human QC; refuse to overwrite it")
    frame.to_csv(manifest, index=False)
    duplicates = frame.loc[frame.pixel_sha256.duplicated(keep=False)].sort_values("pixel_sha256")
    duplicates[["image_id", "split", "filename", "pixel_sha256"]].to_csv(out / "drac_exact_duplicates.csv", index=False)
    cross_split = [g.image_id.tolist() for _, g in duplicates.groupby("pixel_sha256") if g.split.nunique() > 1]

    # Deterministic random ordering depends only on image ID and prespecified seed.
    # No grade is consulted to select the review images.
    train = frame[frame.split.eq("train")].copy()
    train["selection_hash"] = train.image_id.map(lambda iid: hashlib.sha256(
        f"{cfg['project']['base_seed']}|DRAC_QC|{iid}".encode()).hexdigest())
    selected = train.sort_values("selection_hash").head(cfg["exp_e"]["min_qc_audited"])
    qc_path = out / "drac_qc_selection.csv"
    if not qc_path.exists():
        qc = selected[["image_id", "image_path", "image_sha256", "selection_hash"]].copy()
        for col in ["reviewer", "review_date", "frangi_pass", "unet_pass", "exclusion_reason", "notes"]:
            qc[col] = ""
        qc.to_csv(qc_path, index=False)
    counts = train.grade.value_counts().sort_index()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config_hash": config_hash(cfg),
        "source": "https://zenodo.org/records/10280359",
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "training_images": int(len(train)), "test_images": int(frame.split.eq("test").sum()),
        "training_grade_counts": {str(k): int(v) for k, v in counts.items()},
        "dimensions": sorted({f"{r.height_px}x{r.width_px}" for r in frame.itertuples()}),
        "known_label_correction_verified": {"image": "107.png", "grade": 0},
        "exact_duplicate_image_count": len(duplicates), "exact_duplicates_cross_split": cross_split,
        "qc_selected": len(selected), "qc_completed": 0,
        "subject_ids_available": False, "physical_scale_verified": geometry_verified,
        "fov_mm": ecfg.get("fov_mm"), "pixel_size_um": ecfg.get("pixel_size_um"),
        "geometry_source": ecfg.get("geometry_source"),
        "clinical_analysis_ready": False,
        "readiness_reasons": ["vessel segmentation not available", "human segmentation QC pending"] +
                              ([] if geometry_verified else ["physical scale needs verification"]),
        "notes": ["Test grades remain missing and cannot be used in local training or validation.",
                  "Image filenames are task-local; do not join grading, lesion, and quality tasks by filename alone.",
                  "Task A annotates lesions, not the full vessel network.",
                  "The final DRAC paper explicitly reports that metadata needed to count eyes and patients are unavailable. Any CV without such metadata is image-level, not proven patient-independent.",
                  "Exact duplicate checks cannot establish independence of different images from one patient."]}
    (out / "drac_inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
