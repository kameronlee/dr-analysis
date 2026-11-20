"""Summarize the reference and dilated mask on the same study eyes. Run from the project root."""
import json
from pathlib import Path

import pandas as pd

manifest = pd.read_csv("data/manifests/octa500_manifest.csv").set_index("image_id")
assert manifest.index.is_unique
eligible = manifest[manifest["disease"].isin(["DR", "NORMAL"]) & ~manifest["calibration"]]
assert len(eligible) == eligible["subject_id"].nunique() == 295
folders = {
    "reference": Path("outputs/exploratory_20261009/pipeline"),
    "dilated": Path("outputs/exploratory_20261116_step0/perturbation/dilate1/pipeline"),
}
frames = {
    key: pd.read_csv(folder / "features.csv").set_index("image_id").loc[eligible.index]
    for key, folder in folders.items()
}
assert all(frame.index.equals(eligible.index) for frame in frames.values())
assert frames["reference"]["analysis_area_px"].equals(frames["dilated"]["analysis_area_px"])

# Vessel density = fraction of analysis pixels covered by the union vessel mask.
# n_endpoints_post = degree-one nodes on the entire cleaned graph (all locations).
fields = ["vessel_density", "n_endpoints_post", "total_length_mm_post"]
report = {}
for field, eyes in eligible.groupby("fov_mm"):
    reference = frames["reference"].loc[eyes.index, fields]
    dilated = frames["dilated"].loc[eyes.index, fields]
    assert reference.notna().all().all() and dilated.notna().all().all()
    assert (reference > 0).all().all()
    report[str(int(field)) + "mm"] = {
        "eyes": len(eyes),
        "dr_eyes": int(eyes["disease"].eq("DR").sum()),
        "reference_median": reference.median().to_dict(),
        "dilated_median": dilated.median().to_dict(),
        "median_within_eye_relative_change_percent": (100 * (dilated / reference - 1)).median().to_dict(),
        "median_within_eye_area_fraction_change_percentage_points":
            float(100 * (dilated["vessel_density"] - reference["vessel_density"]).median()),
    }
print(json.dumps(report, indent=2, allow_nan=False))
