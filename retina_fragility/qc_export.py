"""Blinded, hash-bound segmentation review packages (no FI, no statistics, no decisions).

For each selected image of a segmentation run this writes, under
``<out>/review/<safe_name>/``:

    qc_intensity.png               the OCTA scan as segmented (native resolution)
    qc_mask_on_intensity.png       segmentation mask over the scan
    qc_graph_on_intensity.png      cleaned graph over the scan (edges green, nodes yellow)
    diagnostic_mask_only_graph.png graph over the binary mask (label/graph agreement only)
    graph.json, provenance.json    graph and hashes of image, mask and graph

The graph is built with the pipeline's own ``build_graph`` (same cleanup and
FAZ handling as the analysis). Stress tests are never run. The hash-bound
review sheet (``qc.make_qc_sheet``) is written next to the packages with blank
decisions. Grades and disease labels are never read or written.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import viz
from .io_masks import faz_disk_mask, read_intensity, read_mask
from .manifest import SYNTHETIC_PREFIX, resolve_input
from .pipeline import build_graph, check_output_names
from .provenance import SYNTHETIC_BANNER, now_utc, sha256_file, write_json
from .qc import graph_review_identity, load_segmentation_run, make_qc_sheet


def export_review(segmented_dir: str | Path, selection_csv: str | Path, cfg: dict, root: str | Path,
                  out_dir: str | Path, faz_spec: dict) -> dict:
    out_dir = Path(out_dir)
    run, man = load_segmentation_run(segmented_dir)
    sel = pd.read_csv(selection_csv, dtype=str, keep_default_na=False)
    ids = sel.image_id.str.strip().tolist()
    names = check_output_names(ids)
    man = man.set_index("image_id")
    missing = sorted(set(ids) - set(man.index))
    if missing:
        raise ValueError(f"selected images are not in segmentation run {segmented_dir}: {missing[:10]}")
    if faz_spec.get("mode") != "disk":
        raise ValueError("review export supports the external-data fixed-disk FAZ only")
    graph_id = graph_review_identity(cfg, faz_spec)
    records, review_dirs = [], {}
    for iid in ids:
        m = man.loc[iid]
        synthetic = str(m["dataset"]).upper().startswith(SYNTHETIC_PREFIX)
        ipath = resolve_input(m["image_path"], root)
        mpath = resolve_input(m["vessel_mask_path"], root)
        img, _ = read_intensity(ipath)
        vessel = read_mask(mpath, {"mode": "binary"})
        px = float(m["pixel_size_um"])
        faz = faz_disk_mask(vessel.shape, float(faz_spec["radius_px"]), faz_spec.get("center", "image_center"))
        g = build_graph(vessel, faz, px, cfg, meta={"image_id": iid, "dataset": m["dataset"],
                                                    "faz": dict(faz_spec)})
        item = out_dir / "review" / names[iid]
        files = viz.qc_composites(img, vessel, g["clean"] if g["clean"].n_edges else None, item, synthetic)
        g["clean"].save(item / "graph.json")
        rec = {"image_id": iid, "segmenter": run["segmenter"], "status": "review_pending",
               "segmentation_config_sha256": run["segmentation_config_sha256"],
               "graph_review_sha256": graph_id["graph_review_sha256"],
               "image_sha256": sha256_file(ipath), "mask_sha256": sha256_file(mpath),
               "graph_sha256": sha256_file(item / "graph.json"),
               "mask_path": str(m["vessel_mask_path"]), "review_directory": str(item),
               "n_nodes": g["clean"].n_nodes, "n_edges": g["clean"].n_edges,
               "images": files, "cleanup_audit": g["audit"],
               "data_provenance": SYNTHETIC_BANNER if synthetic else "real"}
        write_json(item / "provenance.json", rec)
        review_dirs[iid] = str(item)
        records.append(rec)
    sheet_path = out_dir / f"qc_sheet_{run['segmenter']}.csv"
    make_qc_sheet(segmented_dir, selection_csv, cfg, root, faz_spec, sheet_path, review_dirs)
    synthetic_any = any(r["data_provenance"] == SYNTHETIC_BANNER for r in records)
    summary = {"created_utc": now_utc(), "segmented_dir": str(segmented_dir),
               "segmenter": run["segmenter"], "segmentation_config_sha256": run["segmentation_config_sha256"],
               "graph_review_sha256": graph_id["graph_review_sha256"],
               "selection_csv": str(selection_csv), "selection_sha256": sha256_file(selection_csv),
               "n_images": len(records), "qc_sheet": str(sheet_path),
               "human_qc_completed": False, "fi_computed": False, "clinical_statistics_computed": False,
               "data_provenance": SYNTHETIC_BANNER if synthetic_any else "real",
               "records": [{k: v for k, v in r.items() if k != "cleanup_audit"} for r in records],
               "note": ("Native-resolution review images of exactly the masks named by mask_sha256. "
                        "Automated or automated previews are not human QC.")}
    write_json(out_dir / "review_inventory.json", summary)
    return summary
