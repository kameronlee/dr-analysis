"""Post hoc sensitivity to a common 15-µm dilation cutoff on each native pixel grid.

Run on a compute node from the full project using its CPU Python environment.
"""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr


RADIUS_UM = 15.0
ARCHIVE = Path("outputs/exploratory_20261116_step0")
DILATION = ARCHIVE / "perturbation/dilate1"
REF_ROOT = Path("outputs/exploratory_20261009/pipeline")
MANIFEST = Path("data/manifests/octa500_manifest.csv")
PARTS = ARCHIVE / "endnode_mechanism/end_node_parts.csv"


def disk(pixel_um):
    r = int(np.floor(RADIUS_UM / pixel_um))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return np.hypot(yy, xx) * pixel_um <= RADIUS_UM


def require_close(actual, expected, label):
    if not np.isclose(actual, expected, atol=1e-9, rtol=1e-10):
        raise ValueError(f"{label}: {actual} vs {expected}")


def graph_measures(mask, faz, pitch, cfg):
    from retina_fragility.pipeline import build_graph
    from scripts.endnode_mechanism import location_parts

    built = build_graph(mask, faz, pitch, cfg)
    clean = built["clean"]
    area_pixels = int(built["analysis"].sum())
    area = area_pixels * (pitch / 1000) ** 2
    payload = clean.to_dict()
    counts = location_parts(payload["nodes"], payload["edges"], payload["shape"], faz)
    measures = {
        "area_mm2": area,
        "coverage": np.count_nonzero(mask & built["analysis"]) / area_pixels,
        "length_mm_per_mm2": clean.total_length() / 1000 / area,
        "interior_count": counts["end_loc_interior"],
        "all_count": sum(counts.values()),
        "interior_per_mm2": counts["end_loc_interior"] / area,
    }
    return measures, payload


def compare_archive(root, eye, folder, measures, payload, archived_features):
    saved = json.loads((root / folder / "images" / eye / "graph.json").read_text())
    if not all(payload[k] == saved[k] for k in ["nodes", "edges", "shape"]):
        raise ValueError(f"{eye} rebuilt graph differs from {folder}")
    for k, source in [("area_mm2", "analysis_area_mm2"), ("coverage", "vessel_density"),
                      ("length_mm_per_mm2", "graph_length_density_mm_per_mm2"),
                      ("all_count", "n_endpoints_post")]:
        require_close(measures[k], archived_features[source], f"{eye}/{folder}/{k}")


def measure_one(task):
    from retina_fragility.io_masks import read_mask

    row, ref, cross_saved, original_interior, calibration, root, cfg = task
    eye, pitch = row["image_id"], float(row["pixel_size_um"])
    maps = cfg["data"]["label_maps"]
    cap = read_mask(root / row["vessel_mask_path"], maps["vessel"])
    large = read_mask(root / row["large_vessel_mask_path"], maps["large_vessel"])
    faz = read_mask(root / row["faz_mask_path"], maps["faz"])
    if not cap.shape == large.shape == faz.shape:
        raise ValueError(f"{eye}: mask dimensions differ")
    cross = ndi.generate_binary_structure(2, 1)
    cap_cross = ndi.binary_dilation(cap, structure=cross)
    archived_mask = read_mask(root / DILATION / "masks" / f"{eye}.png", maps["vessel"])
    if not np.array_equal(cap_cross, archived_mask):
        raise ValueError(f"{eye}: four-neighbor dilation disagrees with the archived capillary mask")
    footprint = disk(pitch)
    cap_physical = ndi.binary_dilation(cap, structure=footprint)
    if int(row["fov_mm"]) == 6 and not np.array_equal(cap_cross, cap_physical):
        raise ValueError(f"{eye}: the 6-mm kernel should reproduce the four-neighbor dilation")
    measured, graph = graph_measures(cap_physical | large, faz, pitch, cfg)
    require_close(measured["area_mm2"], ref["analysis_area_mm2"], f"{eye}: changed denominator")
    if np.array_equal(footprint, cross):
        compare_archive(root, eye, DILATION / "pipeline", measured, graph, cross_saved)
    elif calibration:
        control, control_graph = graph_measures(cap_cross | large, faz, pitch, cfg)
        compare_archive(root, eye, DILATION / "pipeline", control, control_graph, cross_saved)
    if calibration:
        native, native_graph = graph_measures(cap | large, faz, pitch, cfg)
        compare_archive(root, eye, REF_ROOT, native, native_graph, ref)
        require_close(native["interior_per_mm2"], original_interior, f"{eye}: reference endpoints")
    return {
        "eye": eye, "field": int(row["fov_mm"]), "diagnosis": row["disease"],
        "reference": original_interior, "dilated": measured["interior_per_mm2"],
        "reference_coverage": ref["vessel_density"], "dilated_coverage": measured["coverage"],
        "reference_length": ref["graph_length_density_mm_per_mm2"],
        "dilated_length": measured["length_mm_per_mm2"],
    }


def summarize(group):
    return {
        "eyes": len(group), "dr": int(group.diagnosis.eq("DR").sum()),
        "median_reference_interior_per_mm2": float(group.reference.median()),
        "median_dilated_interior_per_mm2": float(group.dilated.median()),
        "median_paired_change_interior_per_mm2": float((group.dilated - group.reference).median()),
        "median_reference_coverage_percent": float(100 * group.reference_coverage.median()),
        "median_dilated_coverage_percent": float(100 * group.dilated_coverage.median()),
        "median_paired_change_length_mm_per_mm2": float((group.dilated_length - group.reference_length).median()),
        "spearman_original_vs_15um_interiors": float(spearmanr(group.reference, group.dilated).statistic),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from retina_fragility.config import load_config

    cfg = load_config(root / "config.yaml")
    raw = pd.read_csv(root / MANIFEST)
    eligible = raw.disease.isin(["DR", "NORMAL"]) & ~raw.calibration
    eyes = raw.loc[eligible].sort_values("image_id")
    if len(eyes) != 295 or eyes.subject_id.nunique() != len(eyes):
        raise ValueError("Unexpected analysis cohort")
    calibration = set(eyes.groupby(["fov_mm", "disease"]).head(1).image_id)
    frames = [pd.read_csv(root / name / "features.csv").set_index("image_id", verify_integrity=True)
              for name in [REF_ROOT, DILATION / "pipeline"]]
    original = pd.read_csv(root / PARTS).set_index("image_id", verify_integrity=True)["end_loc_interior"]
    tasks = [(r, frames[0].loc[r["image_id"]].to_dict(), frames[1].loc[r["image_id"]].to_dict(),
              float(original.loc[r["image_id"]]), r["image_id"] in calibration, root, cfg)
             for r in eyes.to_dict(orient="records")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for i, result in enumerate(pool.map(measure_one, tasks), 1):
            rows.append(result)
            if i == 1 or i % 25 == 0:
                print(f"Processed {i}/{len(tasks)}", flush=True)
    data = pd.DataFrame(rows)
    summary = {
        "status": "Reviewer-requested post hoc sensitivity. Neither a repeat scan nor an independent tracing.",
        "radius_um": RADIUS_UM,
        "operation": "Include native pixel centers at Euclidean physical distance at most 15 micrometers from an original capillary-mask pixel, then add the original large-vessel mask.",
        "endpoint_rule": "Rebuild and clean graphs with the archived pipeline; retain degree-one nodes more than five native pixels from FAZ and image edge. Use the original per-eye analysis area.",
        "paired_change": "Each eye's 15-micrometer measurement minus its original measurement; medians taken after differencing.",
        "kernel_by_field": {
            str(int(f)): {"pitch_um": (p := float(g.pixel_size_um.iloc[0])), "grid": disk(p).astype(int).tolist(),
                          "max_included_center_distance_um": float(max(np.hypot(y, x) * p for y in range(-2,3) for x in range(-2,3) if np.hypot(y,x) * p <= RADIUS_UM))}
            for f, g in eyes.groupby("fov_mm")
        },
        "validation": {
            "cross_dilated_masks_pixel_identical_to_archive": len(eyes),
            "original_graphs_checked_against_archive": sorted(calibration),
            "cross_dilation_graphs_checked_all_six_mm": int(eyes.fov_mm.eq(6).sum()),
            "cross_dilation_graphs_checked_three_mm": sorted(eyes.loc[eyes.fov_mm.eq(3) & eyes.image_id.isin(calibration),"image_id"]),
            "graph_checks": "Exact node records, edge records and shapes, plus spot checks of analysis area, coverage, length density and total endpoint count vs feature CSV.",
        },
        "by_field": {str(f): summarize(g) for f, g in data.groupby("field")},
        "input_sha256": {str(p): hashlib.sha256((root / p).read_bytes()).hexdigest()
                         for p in [MANIFEST, Path("config.yaml"), PARTS, REF_ROOT / "features.csv",
                                   DILATION / "pipeline/features.csv"]},
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(summary["by_field"], flush=True)


if __name__ == "__main__":
    main()
