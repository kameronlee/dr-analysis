"""Reproduce the referee's post hoc check of Eq. 4, using archived original graphs.

Example from the Diabetic project root:
.deps/venv-batch-cpu-2026-10-01/bin/python -B endpoint_sensitivity.py --root . --output results.json
The archive owner may use another interpreter with numpy, pandas, scipy, patsy, statsmodels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.stats import norm
import statsmodels.formula.api as smf


FILES = {
    "manifest": "data/manifests/octa500_manifest.csv",
    "features": "outputs/exploratory_20261009/pipeline/features.csv",
    "endpoints": "outputs/exploratory_20261116_step0/endnode_mechanism/end_node_parts.csv",
    "quality": "outputs/exploratory_20261116_step0/quality_specificity/quality_proxies.csv",
}
QUALITY = ["q_signal", "q_cnr", "q_sharp", "q_stripe"]
QUANTILES = [0.25, 0.5, 0.75]


def estimate(fit, contrast):
    weights = np.array([contrast.get(name, 0) for name in fit.params.index])
    estimate = float(weights @ fit.params)
    se = float(np.sqrt(weights @ fit.cov_params() @ weights))
    distance = norm.ppf(.975) * se
    return {"estimate": estimate, "hc3_ci95": [estimate - distance, estimate + distance]}


def ols(formula, data):
    return smf.ols(formula, data).fit(cov_type="HC3")


def load_data(root):
    manifest = pd.read_csv(root / FILES["manifest"])
    valid = manifest.disease.isin(["DR", "NORMAL"]) & ~manifest.calibration
    data = manifest.loc[valid, ["image_id", "fov_mm", "disease"]].copy()
    features = pd.read_csv(root / FILES["features"], usecols=["image_id", "graph_length_density_mm_per_mm2"])
    features = features.rename(columns={"graph_length_density_mm_per_mm2": "length"})
    data = data.merge(features, on="image_id", validate="one_to_one")
    ends = pd.read_csv(root / FILES["endpoints"], usecols=["image_id", "end_loc_interior"])
    quality = pd.read_csv(root / FILES["quality"], usecols=["image_id", *QUALITY])
    for table in [ends, quality]:
        data = data.merge(table, on="image_id", validate="one_to_one")
    assert len(data) == 295 and not data.isna().any().any()
    data["dr"] = data.disease.eq("DR").astype(int)
    data["wide"] = data.fov_mm.eq(6).astype(int)
    for col in ["end_loc_interior", *QUALITY]:
        groups = data.groupby("fov_mm")[col]
        data[col + "_z"] = (data[col] - groups.transform("mean")) / groups.transform("std")
    data["z"] = data["end_loc_interior_z"]
    data["lc"] = data.length - data.groupby("fov_mm").length.transform("mean")
    data["joint"] = False
    for _, field in data.groupby("fov_mm"):
        for dr, group in field.groupby("dr"):
            axes = ["length", "q_cnr_z"]
            other = field.loc[field.dr.ne(dr), axes].to_numpy()
            hull = Delaunay(other)
            data.loc[group.index, "joint"] = hull.find_simplex(group[axes].to_numpy()) >= 0
    return data


def by_field(field):
    fit = ols("z ~ dr * lc", field)
    # Model one common smooth trend in vessel length and retain a diagnosis-specific linear departure.
    # cr() uses a natural cubic spline. Set its two interior knots to the observed field tertiles.
    # Centering removes its constant basis term. Boundary knots are at the field minimum and maximum.
    knots = tuple(float(v) for v in field.lc.quantile([1/3, 2/3]))
    spline = f"cr(lc, knots={knots}, constraints='center')"
    curved = ols(f"z ~ dr + {spline} + dr:lc", field)
    qlist = []
    for percentile in QUANTILES:
        density = float(field.length.quantile(percentile))
        centered = density - field.length.mean()
        nearby = field.loc[field.length.sub(density).abs().le(1)].groupby("dr").size()
        qlist.append({
            "field_pooled_percentile": percentile,
            "length_mm_per_mm2": density,
            "dr_minus_healthy_sd": estimate(fit, {"dr": 1, "dr:lc": centered}),
            "eyes_within_one_length_unit": {
                "healthy": int(nearby.get(0, 0)), "DR": int(nearby.get(1, 0)),
            },
        })
    return {
        "length_range_by_diagnosis_mm_per_mm2": {
            str(name): list(map(float, group.length.agg(["min","max"])))
            for name, group in field.groupby("disease")
        },
        "full_sample_slopes_sd_per_one_length_unit": {
            "healthy": estimate(fit, {"lc": 1}),
            "DR": estimate(fit, {"lc": 1, "dr:lc": 1}),
            "DR_minus_healthy": estimate(fit, {"dr:lc": 1}),
        },
        "full_sample_linear_contrasts": qlist,
        "natural_spline_sensitivity": {
            "interior_knots_mm_per_mm2": [float(v + field.length.mean()) for v in knots],
            "DR_minus_healthy_slope_beyond_common_curve": estimate(curved, {"dr:lc": 1}),
        },
        "joint_support_n": {
            str(name): {"included": int(group.joint.sum()), "all": len(group)}
            for name, group in field.groupby("disease")
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    d = load_data(args.root)
    qterms = " + ".join(q + "_z" for q in QUALITY)
    base = ols("z ~ dr + wide + lc", d)
    full = ols("z ~ dr * wide * lc", d)
    qfit = ols("z ~ dr + wide + lc + " + qterms, d)
    qfit_joint = ols("z ~ dr + wide + lc + " + qterms, d.loc[d.joint])
    tests = full.wald_test("dr:lc = 0, dr:wide:lc = 0", scalar=True)
    result = {
        "status": "Reviewer-initiated, post hoc exploratory model diagnostics. All intervals nominal.",
        "inputs_relative_to_project_root": {key: {"file": value, "sha256": hashlib.sha256((args.root/value).read_bytes()).hexdigest()} for key, value in FILES.items()},
        "outcome_units": "Interior endpoints per square mm, standardized by the original across-diagnosis mean and sample SD within each scan field. These units are held fixed for every fit, even the restricted sample.",
        "length_units": "millimeters of traced vessel per square millimeter of analysis area",
        "diagnosis_contrast": "DR minus healthy at the same length and field",
        "quartiles": "Original all-eye within-field empirical quartiles of cleaned vessel length density, irrespective of diagnosis (pandas linear interpolation). The nearby counts use plus or minus 1 mm/mm2 as a descriptive window.",
        "joint_support": "Retain eyes lying within the opposite diagnosis group's 2-D convex hull of traced vessel length density and contrast (q_cnr) within the same field. This rule can retain sparsely supported points. It does not balance other covariates.",
        "pooled_models": {
            "article_common_slope_dr": estimate(base, {"dr": 1}),
            "unsplit_sample_quality_adjusted_dr": estimate(qfit, {"dr": 1}),
            "joint_hull_quality_adjusted_dr": estimate(qfit_joint, {"dr": 1}),
            "joint_hull_n": {"DR": int(d.loc[d.joint,"dr"].sum()), "healthy": int((d.joint & d.dr.eq(0)).sum())},
            "disease_by_length_heterogeneity": {"robust_joint_wald": float(tests.statistic), "df": 2, "p": float(tests.pvalue), "model": "z ~ dr * wide * lc; joint HC3 Wald test of dr:lc = dr:wide:lc = 0"}
        },
        "fields": {str(int(fov)): by_field(field) for fov, field in d.groupby("fov_mm")},
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
