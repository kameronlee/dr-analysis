"""Extract the retinal tracing figure from one released OCTA projection and two archived cleaned graphs."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from retina_fragility.graph.extract import skeletonize_mask
from retina_fragility.graph.model import VesselGraph
from retina_fragility.viz import graph_pixel_masks
from scripts.endnode_mechanism import BAND_PX, location_parts

REFERENCE = ROOT / "outputs/exploratory_20261009/pipeline"
WIDENED = ROOT / "outputs/exploratory_20261116_step0/perturbation/dilate1"
# A local window selected to illustrate a join between two nearby reference tips.
TOP, LEFT, SIDE = 106, 124, 25
WINDOW = np.s_[TOP:TOP + SIDE, LEFT:LEFT + SIDE]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def extract_tracing(path, mask, faz, eye):
    graph = VesselGraph.load(path)
    if graph.meta["image_id"] != eye or graph.shape != mask.shape:
        raise ValueError("The frozen graph and masks must share an eye and coordinate frame")
    edge_pixels, node_pixels = graph_pixel_masks(graph, mask.shape)
    retained = edge_pixels | node_pixels
    raw_skeleton = skeletonize_mask(mask)
    if (retained & ~raw_skeleton).any():
        raise ValueError("Archived graph pixels disagree with the source skeleton")
    degrees = graph.degrees()
    faz_distance = ndimage.distance_transform_edt(~faz)
    height, width = mask.shape
    endpoints = []
    for node in graph.nodes.values():
        y, x = node.y, node.x
        if (degrees[node.id] == 1 and min(y, x, height - 1 - y, width - 1 - x) > BAND_PX
                and faz_distance[round(y), round(x)] > BAND_PX):
            endpoints.append((x, y))
    record = json.loads(path.read_text())
    total = location_parts(record["nodes"], record["edges"], graph.shape, faz)["end_loc_interior"]
    if total != len(endpoints):
        raise ValueError("The independently counted graph endpoints disagree")
    local_ends = np.asarray(endpoints).reshape(-1, 2) - (LEFT, TOP)
    visible = np.all((local_ends >= -0.5) & (local_ends < SIDE - 0.5), axis=1)
    return retained[WINDOW], local_ends[visible], total


def main():
    key = pd.read_csv(ROOT / "outputs/expert_review_2026-11-17/key_NOT_FOR_REVIEWER.csv",
                      usecols=["part", "fov_mm", "image_id", "label"])
    packet = key.loc[(key.part == "A") & (key.fov_mm == 3)].sort_values("label")
    eye = packet.iloc[0].image_id
    manifest = pd.read_csv(ROOT / "data/manifests/octa500_manifest.csv").set_index("image_id")
    row = manifest.loc[eye]
    paths = {name: ROOT / row[column] for name, column in [
        ("scan", "image_path"), ("capillary", "vessel_mask_path"),
        ("large_vessel", "large_vessel_mask_path"), ("faz", "faz_mask_path")]}
    paths.update(dilated_capillary=WIDENED / "masks" / f"{eye}.png",
                 reference_graph=REFERENCE / "images" / eye / "graph.json",
                 dilated_graph=WIDENED / "pipeline/images" / eye / "graph.json")
    scan = read_pixels(paths["scan"])
    capillary, large, faz, dilated = [
        read_pixels(paths[name]) > 0 for name in ("capillary", "large_vessel", "faz", "dilated_capillary")]
    assert np.array_equal(dilated, ndimage.binary_dilation(capillary, ndimage.generate_binary_structure(2, 1)))
    bundle = dict(scan=scan, pixel_size_um=float(row.pixel_size_um), top=TOP, left=LEFT)
    counts = {}
    for key, mask in [("reference", capillary | large), ("dilated", dilated | large)]:
        centerline, ends, total = extract_tracing(paths[f"{key}_graph"], mask, faz, eye)
        bundle[f"{key}_mask"] = mask[WINDOW]
        bundle[f"{key}_centerline"] = centerline
        bundle[f"{key}_endpoints"] = ends
        counts[key] = {"crop_interior_endpoints": len(ends), "full_graph_interior_endpoints": total}

    # Confirm that the two indicated tips lie in different local eight-connected
    # mask components before widening and the same local component afterward.
    pair = np.array([[117, 138], [118, 135]]) - (TOP, LEFT)
    structures = {}
    for key in counts:
        components = ndimage.label(bundle[f"{key}_mask"], structure=np.ones((3, 3), int))[0]
        labels = components[pair[:, 0], pair[:, 1]]
        assert np.all(labels > 0)
        structures[key] = bool(labels[0] == labels[1])
    assert all(any((bundle["reference_endpoints"] == xy).all(axis=1)) for xy in pair[:, ::-1])
    assert structures == {"reference": False, "dilated": True}
    assert counts["reference"]["crop_interior_endpoints"] == 3 and counts["dilated"]["crop_interior_endpoints"] == 0
    np.savez_compressed(HERE / "retinal_tracing.npz", **bundle)
    receipt = dict(
        dataset="OCTA-500", dataset_url="https://ieee-dataport.org/open-access/octa-500",
        primary_publication="https://doi.org/10.1016/j.media.2024.103092",
        license="https://creativecommons.org/licenses/by/4.0/", eye=eye,
        selection="First 3-mm Part A entry by blinded packet order. The crop deliberately illustrates a local mask join; diagnosis was not used.",
        projection="OCTA(ILM_OPL)", pixel_size_um=float(row.pixel_size_um),
        zero_based_crop=dict(top=TOP, left=LEFT, height=SIDE, width=SIDE),
        example_tip_pair_full_scan_row_column=(pair + (TOP, LEFT)).tolist(),
        pair_connected_in_crop_mask_eight_neighbors=structures, counts=counts,
        display="Unaltered grayscale projection, magnification without smoothing, union-mask outline, retained centerline pixels (including junction clusters), and degree-one interior nodes from whole-field graphs.",
        sha256_by_project_path={str(path.relative_to(ROOT)): digest(path) for path in paths.values()},
    )
    (HERE / "retinal_tracing_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
