"""Remeasure the within-eye change in interior endpoints per mm² for a one-pixel dilation.

Run from the Diabetic project root using its CPU Python environment:
python paired_interior_dilation.py --root . --output paired_interior_dilation.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm


REFERENCE = Path("outputs/exploratory_20261009/pipeline")
AUDIT = Path("outputs/exploratory_20261116_step0")
DILATED = AUDIT / "perturbation/dilate1/pipeline"
LENGTH = "graph_length_density_mm_per_mm2"


def remeasure(root):
    sys.path.insert(0, str(root))
    from retina_fragility.config import load_config
    from retina_fragility.io_masks import read_mask
    from scripts.endnode_mechanism import location_parts
    from scripts.step0_null_controls import graph_path

    cfg = load_config(root / "config.yaml")
    manifest = pd.read_csv(root / cfg["data"]["manifest"]).set_index("image_id", verify_integrity=True)
    keep = manifest.disease.isin(["DR", "NORMAL"]) & ~manifest.calibration
    cohort = manifest.loc[keep].sort_index()
    if len(cohort) != 295 or cohort.subject_id.nunique() != len(cohort):
        raise ValueError("Eligible cohort changed")
    ref_parts = pd.read_csv(root / AUDIT / "endnode_mechanism/end_node_parts.csv").set_index("image_id", verify_integrity=True)
    sources = {name: pd.read_csv(root / folder / "features.csv").set_index("image_id", verify_integrity=True)
               for name, folder in [("reference", REFERENCE), ("dilated", DILATED)]}
    digest = hashlib.sha256()
    rows = []
    for eye, case in cohort.iterrows():
        faz = read_mask(root / case.faz_mask_path, cfg["data"]["label_maps"]["faz"])
        digest.update(eye.encode() + faz.tobytes())
        primary = sources["reference"].loc[eye]
        area = primary.analysis_area_mm2
        if not np.isclose(area, primary.analysis_area_px * (case.pixel_size_um / 1000) ** 2):
            raise ValueError(f"Invalid pixel scale for {eye}")
        row = dict(image_id=eye, fov_mm=case.fov_mm, disease=case.disease,
                   dr=int(case.disease == "DR"), reference_length=primary[LENGTH],
                   pixel_size_um=case.pixel_size_um)
        for condition, folder in [("reference", REFERENCE), ("dilated", DILATED)]:
            feature = sources[condition].loc[eye]
            if feature.analysis_area_px != primary.analysis_area_px or feature.analysis_area_mm2 != area:
                raise ValueError(f"Analysis area changed for {eye}/{condition}")
            raw = graph_path(root / folder, eye).read_bytes()
            digest.update(condition.encode() + hashlib.sha256(raw).digest())
            g = json.loads(raw)
            if g["meta"]["image_id"] != eye or tuple(g["shape"]) != faz.shape:
                raise ValueError(f"Invalid graph identity or size for {eye}/{condition}")
            parts = location_parts(g["nodes"], g["edges"], g["shape"], faz)
            if sum(parts.values()) != feature.n_endpoints_post:
                raise ValueError(f"Endpoint partition failed for {eye}/{condition}")
            row[condition] = parts["end_loc_interior"] / area
        if not np.isclose(row["reference"], ref_parts.at[eye, "end_loc_interior"], atol=1e-10, rtol=0):
            raise ValueError(f"Reference interior density differs from archived measurement for {eye}")
        rows.append(row)
    data = pd.DataFrame(rows)
    data["change"] = data.dilated - data.reference
    return data, digest.hexdigest()


def contrast(outcome, covariates, frame):
    x = sm.add_constant(frame[covariates], has_constant="add")
    fit = sm.OLS(frame[outcome], x, missing="raise").fit(cov_type="HC3")
    lo, hi = fit.conf_int().loc["dr"]
    return {"dr_minus_healthy": float(fit.params["dr"]), "hc3_ci95": [float(lo), float(hi)]}


def summarize(group):
    # The design matrix is kept identical in all three fits at each field.
    ans = {}
    for name, terms in [("observed", ["dr"]), ("at_common_original_length", ["dr", "reference_length"])]:
        estimates = {var: contrast(var, terms, group) for var in ["reference", "dilated", "change"]}
        diff = estimates["dilated"]["dr_minus_healthy"] - estimates["reference"]["dr_minus_healthy"]
        if not np.isclose(diff, estimates["change"]["dr_minus_healthy"], atol=1e-10, rtol=0):
            raise ValueError("Mismatch in difference between contrasts")
        ans[name] = estimates
    ans["by_diagnosis"] = {}
    for diagnosis, subset in group.groupby("disease"):
        ans["by_diagnosis"][diagnosis] = {
            "n": len(subset),
            "mean_per_mm2": subset[["reference", "dilated", "change"]].mean().to_dict(),
            "median_per_mm2": subset[["reference", "dilated", "change"]].median().to_dict(),
        }
    ans["overall_median_per_mm2"] = group[["reference", "dilated", "change"]].median().to_dict()
    ans["rank_correlation"] = float(spearmanr(group.reference, group.dilated).statistic)
    sizes = group.pixel_size_um.unique()
    if len(sizes) != 1:
        raise ValueError("Pixel size differs within field")
    ans["one_pixel_um"] = float(sizes[0])
    return ans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, digest = remeasure(args.root.resolve())
    record = {
        "analysis": "Post hoc reviewer-requested check of one-pixel capillary-mask dilation and interior endpoints. All intervals nominal.",
        "unit": "Count of degree-one nodes on the cleaned graph more than five pixels from both FAZ and image border, divided by original analysis area in mm² (identical under both masks).",
        "change": "Dilated minus original endpoint density for the same eye. The interval for the between-diagnosis difference in changes uses HC3 on one pairwise change per participant.",
        "models": "Separate OLS regressions within each field in endpoints/mm², with or without the original vessel length density. Adjusted fit uses one common length slope within field for each outcome; estimates are model summaries and are sensitive to lack of support and slope heterogeneity. Never substitute the dilated length density in these fits.",
        "cohort": len(data),
        "ordered_graph_and_faz_mask_sha256": digest,
        "fields_mm": {str(int(fov)): summarize(group) for fov, group in data.groupby("fov_mm")},
    }
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
