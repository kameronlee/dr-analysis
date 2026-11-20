"""Remeasure cleaned interior endpoints in the frozen OCTA-500 graphs.

Run from the project root with the project's CPU environment:
    .deps/venv-batch-cpu-2026-10-01/bin/python -B \
        paper_materials/W3PHIAI-27/analysis/interior_perturbation.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from retina_fragility.config import load_config
from retina_fragility.io_masks import read_mask
from scripts import metric_audit, step0_null_controls
from scripts.endnode_mechanism import location_parts

ARCHIVE = ROOT / "outputs/exploratory_20261116_step0"
CONDITIONS = ("original", "dropout1", "boundary10", "dilate1")
LENGTH = "graph_length_density_mm_per_mm2"
OUTPUT = HERE / "interior_perturbation_results.json"


def main():
    cfg = load_config(ROOT / "config.yaml")
    cohort, _ = step0_null_controls.load_table(cfg)
    cohort = cohort.set_index("image_id", verify_integrity=True).sort_index()
    manifest = pd.read_csv(ROOT / cfg["data"]["manifest"]).set_index("image_id", verify_integrity=True)
    archived = pd.read_csv(ARCHIVE / "endnode_mechanism/end_node_parts.csv").set_index("image_id", verify_integrity=True)
    original = ROOT / step0_null_controls.SRC_RUN / "pipeline"
    features, graphs, missing = {}, {}, {}
    common = set(cohort.index)
    for condition in CONDITIONS:
        folder = original if condition == "original" else ARCHIVE / "perturbation" / condition / "pipeline"
        features[condition] = pd.read_csv(folder / "features.csv").set_index("image_id", verify_integrity=True)
        graphs[condition] = {eye: step0_null_controls.graph_path(folder, eye) for eye in cohort.index}
        available = {eye for eye, path in graphs[condition].items() if path.is_file() and eye in features[condition].index}
        if not available:
            raise ValueError(f"No paired graphs and features under {condition}")
        missing[condition] = len(cohort) - len(available)
        common.intersection_update(available)
    analysis = cohort.loc[sorted(common), ["y", "fov_mm", "analysis_area_mm2"]].rename(columns={"y": "dr"})
    analysis["fov_6mm"] = (analysis.fov_mm == 6).astype(int)
    digest = hashlib.sha256()
    for eye, row in analysis.iterrows():
        faz = read_mask(ROOT / manifest.at[eye, "faz_mask_path"], cfg["data"]["label_maps"]["faz"])
        digest.update(eye.encode() + faz.tobytes())
        for condition in CONDITIONS:
            saved = features[condition].loc[eye]
            payload = graphs[condition][eye].read_bytes()
            digest.update(condition.encode() + hashlib.sha256(payload).digest())
            graph = json.loads(payload)
            if graph["meta"]["image_id"] != eye or tuple(graph["shape"]) != faz.shape:
                raise ValueError(f"Graph identity or shape mismatch at {eye}/{condition}")
            if not np.isclose(saved.analysis_area_mm2, row.analysis_area_mm2, rtol=1e-12):
                raise ValueError(f"Analysis area mismatch at {eye}/{condition}")
            counts = location_parts(graph["nodes"], graph["edges"], graph["shape"], faz)
            if sum(counts.values()) != saved.n_endpoints_post:
                raise ValueError(f"Endpoint total mismatch at {eye}/{condition}")
            analysis.loc[eye, condition] = counts["end_loc_interior"] / row.analysis_area_mm2
    difference = np.abs(analysis.original - archived.loc[analysis.index, "end_loc_interior"])
    if not np.allclose(difference, 0, atol=1e-10):
        raise ValueError("Original interior densities disagree with archived decomposition")
    results = {}
    for condition in CONDITIONS:
        data = analysis.assign(
            interior_z=metric_audit.zscore_within_fov(analysis, condition),
            density=features[condition].loc[analysis.index, LENGTH],
            coverage_percent=100 * features[condition].loc[analysis.index, "vessel_density"],
            paired_change=analysis[condition] - analysis.original,
        )
        model = metric_audit.coef(data, "interior_z", ["dr", "density", "fov_6mm"])
        if model["status"] != "ok" or model["n"] != len(data):
            raise ValueError(f"Incomplete model for {condition}")
        results[condition] = {
            "n": model["n"], "dr_coefficient_sd": model["coef"], "hc3_ci95": model["ci"],
            "by_field": {
                str(int(fov)): {
                    "median_coverage_percent": float(group.coverage_percent.median()),
                    "median_paired_change_interior_per_mm2": float(group.paired_change.median()),
                }
                for fov, group in data.groupby("fov_mm")
            },
            "spearman_by_fov_mm": {
                str(int(fov)): float(spearmanr(group.original, group[condition]).statistic)
                for fov, group in data.groupby("fov_mm")
            },
        }
    totals = analysis.groupby(["fov_mm", "dr"]).size()
    summary = {
        "n_eligible": len(cohort), "n_common": len(analysis), "missing_graph_or_features": missing,
        "cohort": {f"{int(fov)}mm_{'dr' if dr else 'healthy'}": int(n) for (fov, dr), n in totals.items()},
        "original_max_abs_discrepancy_vs_archive_per_mm2": float(difference.max()),
        "endpoint_partition_checks_passed": len(analysis) * len(CONDITIONS),
        "ordered_graph_and_faz_mask_sha256": digest.hexdigest(),
        "plan_sha256": hashlib.sha256((HERE / "interior_perturbation_plan.md").read_bytes()).hexdigest(),
        "conditions": results,
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
