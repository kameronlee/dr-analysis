"""Refresh inputs for the cohort figures from the original analysis archive.

The separate export_retinal_tracing.py script rebuilds the image and overlays.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[3]
SOURCE_FIGURES = ROOT / "paper_materials/figure_data"
PIPE = ROOT / "outputs/exploratory_20261009/pipeline"
INDEX = "FI_adaptive_betweenness"


def export_fragility():
    cohort = pd.read_csv(ROOT / "outputs/exploratory_20261116_step0/saved/per_eye.csv")
    inputs = pd.read_csv(PIPE / "features.csv", usecols=["image_id", "vessel_density", INDEX])
    eyes = cohort.merge(inputs, validate="one_to_one", on="image_id")
    assert len(eyes) == 295 and eyes.y.sum() == 64
    assert np.allclose(eyes.FI, eyes[INDEX])
    assert eyes[[INDEX, "vessel_density"]].notna().all().all()

    grid = np.linspace(0, .30, 31)
    curves = []
    for eye in eyes.itertuples(index=False):
        saved = pd.read_csv(PIPE / "images" / eye.image_id / "curves.csv")
        curve = saved.loc[saved.order == "adaptive_betweenness"].sort_values("step")
        assert curve.q_edge.iloc[0] == 0 and curve.q_edge.iloc[-1] == 1
        assert np.isclose(1 - 2 * curve.S.iloc[1:].mean(), eye.FI_adaptive_betweenness)
        curves.append(np.interp(grid, curve.q_edge, curve.S))
    low, median, high = np.quantile(curves, [.25, .5, .75], axis=0)
    pd.DataFrame({"edges_removed_fraction": grid, "q25": low, "median": median, "q75": high}).to_csv(
        DEST / "fragility_curve.csv", index=False)
    eyes[["fov_mm", "vessel_density", INDEX]].to_csv(DEST / "fragility_points.csv", index=False)


def export_metric_audit():
    audit = pd.read_csv(SOURCE_FIGURES / "F_res_2_metric_audit.csv", dtype=str, keep_default_na=False)
    base = audit[audit.estimate == "pooled"].set_index("metric")
    quality = audit[audit.estimate == "pooled, quality set A"].set_index("metric")
    rank = audit[audit.panel == "b"].set_index("metric")
    values = pd.DataFrame({"row": base.row, "difference": base.x, "lo": base.ci95_lo_hc3,
                           "hi": base.ci95_hi_hc3, "quality_difference": quality.x,
                           "quality_lo": quality.ci95_lo_hc3, "quality_hi": quality.ci95_hi_hc3,
                           "min_rank": rank.x})
    assert len(values) == 11 and values.index.is_unique
    values.to_csv(DEST / "metric_audit.csv")


def export_endpoint_overlap():
    manifest = pd.read_csv(ROOT / "data/manifests/octa500_manifest.csv")
    eligible = manifest.disease.isin(["NORMAL", "DR"]) & ~manifest.calibration
    eyes = manifest.loc[eligible, ["image_id", "fov_mm", "disease"]]
    lengths = pd.read_csv(PIPE / "features.csv",
                          usecols=["image_id", "analysis_area_mm2", "graph_length_density_mm_per_mm2"])
    parts = pd.read_csv(ROOT / "outputs/exploratory_20261116_step0/endnode_mechanism/end_node_parts.csv",
                        usecols=["image_id", "end_loc_interior"])
    eyes = eyes.merge(lengths, on="image_id", validate="one_to_one")
    eyes = eyes.merge(parts, on="image_id", validate="one_to_one")
    assert eyes.groupby(["fov_mm", "disease"]).size().to_dict() == {
        (3, "DR"): 29, (3, "NORMAL"): 150, (6, "DR"): 35, (6, "NORMAL"): 81,
    }
    # The archive stores interior endpoints per mm², already divided by analysis area.
    counts = eyes.end_loc_interior * eyes.analysis_area_mm2
    assert np.isfinite(eyes.graph_length_density_mm_per_mm2).all()
    assert np.allclose(counts, counts.round(), rtol=0, atol=1e-7)
    values = eyes.rename(columns={"graph_length_density_mm_per_mm2": "length_mm_per_mm2",
                                  "end_loc_interior": "interior_endpoints_per_mm2"})
    values[["fov_mm", "disease", "length_mm_per_mm2", "interior_endpoints_per_mm2"]].to_csv(
        DEST / "endpoint_overlap.csv", index=False)


if __name__ == "__main__":
    export_fragility()
    export_metric_audit()
    export_endpoint_overlap()
