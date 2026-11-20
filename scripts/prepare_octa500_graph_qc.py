"""Export provisional graph overlays without selecting a projection or running FI."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from retina_fragility.config import config_hash, dump_config, load_config
from retina_fragility.graph.cleanup import clean_graph
from retina_fragility.graph.extract import graph_from_mask
from retina_fragility.io_masks import load_reference_masks, make_analysis_mask, write_mask_png
from retina_fragility.provenance import now_utc, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/octa500_graph_qc/provisional_20260930")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")
    catalog = root / "data/manifests/octa500_source_catalog.csv"
    frame = pd.read_csv(catalog)
    frame["selection_key"] = frame.image_id.map(lambda i: hashlib.sha256(
        f"{cfg['project']['base_seed']}|OCTA_GRAPH_QC|{i}".encode()).hexdigest())
    selected = []
    for fov, n in cfg["qc"]["octa500_graph_n_per_fov"].items():
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError("Graph review count must be a positive integer")
        pool = frame[frame.fov_mm.eq(float(fov))].sort_values("selection_key")
        if len(pool) < n:
            raise ValueError("Too few source images for the configured review selection")
        selected.extend(pool.head(n).to_dict("records"))
    out = (root / args.out).resolve()
    if root not in out.parents:
        raise ValueError("Outputs must stay inside the project")
    out.mkdir(parents=True, exist_ok=False)
    dump_config(cfg, out / "config.snapshot.yaml")
    source_hashes = {str(p.relative_to(root)): sha256_file(p)
                     for p in sorted((root / "retina_fragility").rglob("*.py"))}
    source_hashes[str(Path(__file__).resolve().relative_to(root))] = sha256_file(__file__)
    write_json(out / "source_hashes_at_start.json", source_hashes)
    pd.DataFrame([{k: r[k] for k in ["image_id", "fov_mm", "selection_key"]}
                  for r in selected]).to_csv(out / "selection.csv", index=False)
    records = []
    for index, row in enumerate(selected, 1):
        review_id = f"graph-review-{index:02d}"
        item = out / review_id
        item.mkdir()
        input_hashes = {}
        for name in ["capillary", "large_vessel", "faz", "octa_full", "octa_ilm_opl", "octa_opl_bm"]:
            digest = sha256_file(root / row[name + "_path"])
            if digest != row[name + "_sha256"]:
                raise ValueError(f"Input changed: {row['image_id']} {name}")
            input_hashes[name] = digest
        vessel, faz = load_reference_masks(
            root / row["capillary_path"], root / row["faz_path"],
            cfg["data"]["label_maps"], root / row["large_vessel_path"])
        scale = float(row["pixel_size_um"])
        analysis = make_analysis_mask(vessel.shape, scale, faz, cfg["analysis_mask"])
        area = float(analysis.sum()) * (scale / 1000) ** 2
        raw, _ = graph_from_mask(vessel, scale,
            force_2x2=cfg["graph"]["force_2x2_blocks_as_junctions"])
        graph, cleanup_audit = clean_graph(raw, cfg["graph"], faz=faz, area_mm2=area)
        graph.save(item / "graph.json")
        write_mask_png(item / "provisional_vessel_union.png", vessel)
        write_mask_png(item / "faz.png", faz)
        edge_pixels = np.zeros_like(vessel)
        for eid in graph.edges:
            pixels = np.asarray(graph.edge_pixels(eid, include_nodes=True), dtype=int)
            if len(pixels):
                edge_pixels[pixels[:, 0], pixels[:, 1]] = True
        node_pixels = np.zeros_like(vessel)
        for node in graph.nodes.values():
            y, x = int(round(node.y)), int(round(node.x))
            if 0 <= y < vessel.shape[0] and 0 <= x < vessel.shape[1]:
                node_pixels[y, x] = True
        for name in ["octa_full", "octa_ilm_opl", "octa_opl_bm"]:
            with Image.open(root / row[name + "_path"]) as image:
                gray = np.array(image)
            if gray.shape != vessel.shape or gray.dtype != np.uint8:
                raise ValueError("Unexpected source-image encoding")
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            mask_overlay = rgb.copy()
            mask_overlay[vessel] = (0.45 * rgb[vessel] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
            graph_overlay = rgb.copy()
            graph_overlay[edge_pixels] = [50, 255, 80]
            graph_overlay[node_pixels] = [255, 215, 40]
            Image.fromarray(gray).save(item / (name + "_source.png"))
            Image.fromarray(mask_overlay).save(item / (name + "_mask.png"))
            Image.fromarray(graph_overlay).save(item / (name + "_graph.png"))
        record = {
            "review_id": review_id, "image_id": row["image_id"], "fov_mm": row["fov_mm"],
            "pixel_size_um": scale, "status": "provisional_review_pending",
            "input_sha256": input_hashes, "graph_sha256": sha256_file(item / "graph.json"),
            "graph_stats": graph.stats(area), "cleanup_audit": cleanup_audit,
            "review_directory": str(item.relative_to(root)),
            "mask_construction": "Provided capillary OR large-vessel mask; provisional pending source/provenance review",
            "clinical_statistics_computed": False, "human_qc_completed": False,
        }
        write_json(item / "provenance.json", record)
        records.append(record)
        print(f"Prepared {review_id}: FOV {row['fov_mm']} mm; {graph.n_nodes} nodes, {graph.n_edges} edges", flush=True)
    pd.DataFrame([{"review_id": r["review_id"], "image_id": r["image_id"],
                   "fov_mm": r["fov_mm"], "reviewer": "", "reviewed_at": "", "notes": ""}
                  for r in records]).to_csv(out / "review_queue.csv", index=False)
    write_json(out / "preparation.json", {
        "created_utc": now_utc(), "n_selected": len(records), "config_sha256": config_hash(cfg),
        "catalog_sha256": sha256_file(catalog), "records": records,
        "primary_projection_selected": False, "clinical_statistics_computed": False,
        "human_qc_completed": False, "known_integration_fixes_pending": True,
        "note": "Provisional graph geometry for review on all three OCTA projections. No FI/stress testing. Rebuild if review changes relevant source or parameters.",
    })
    print(f"Completed {len(records)} provisional graph exports; review is pending.", flush=True)


if __name__ == "__main__":
    main()
