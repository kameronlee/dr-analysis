"""Experiment 0 extensions: patch-dropout sanity check and visual-QC overlay sheet.

Patch dropout: for each selected healthy, non-calibration image, delete
``n_patches`` vessel-free disks of radius ``radius_um`` at seeded random
positions inside the analysis area, re-run the per-image pipeline, and record
paired changes (ΔL0, ΔFI, ΔP0, ...). No direction is asserted: FI need not
increase after every deletion. The check shows how the index responds to
localised capillary loss. On real data it is pending OCTA-500 access.

Overlay sheet: a deterministic (hash-ordered) selection of images whose
overlays are copied into one folder with a blank reviewer CSV. It is a queue,
not evidence of review.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..io_masks import make_analysis_mask
from ..manifest import SYNTHETIC_PREFIX
from ..pipeline import load_row_masks, process_masks, safe_name
from ..provenance import SYNTHETIC_BANNER, write_json
from ..seeds import rng_for

KEYS = ("L0_um", "m0_edges", "FI_adaptive_betweenness", "FI_random", "G10_adaptive_betweenness",
        "P0", "FIperf_adaptive_betweenness", "vessel_density", "loop_density_per_mm2")


def _pick(ids, n, seed, purpose):
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}|{purpose}|{i}".encode()).hexdigest())[:n]


def patch_dropout(rows: pd.DataFrame, cfg: dict, root, d_um: float | None, out_dir, n_images: int = 10,
                  n_patches: int = 3, radius_um: float = 250.0) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["project"]["base_seed"]
    healthy = rows[rows.analysis_group == "normal"]
    chosen = _pick(healthy.image_id.tolist(), n_images, seed, "exp0_dropout")
    recs = []
    synthetic = bool(rows.dataset.astype(str).str.upper().str.startswith(SYNTHETIC_PREFIX).any())
    for iid in chosen:
        row = healthy.set_index("image_id").loc[iid].to_dict() | {"image_id": iid}
        masks = load_row_masks(row, cfg, root, {"mode": "reference"})
        v, f, px = masks["vessel"], masks["faz"], masks["pixel_size_um"]
        am = make_analysis_mask(v.shape, px, f, cfg["analysis_mask"])
        rng = rng_for(seed, "exp0_dropout_positions", iid)
        cand = np.argwhere(am)
        v2 = v.copy()
        yy, xx = np.mgrid[:v.shape[0], :v.shape[1]]
        centres = cand[rng.choice(len(cand), size=n_patches, replace=False)]
        for cy, cx in centres:
            v2[np.hypot(yy - cy, xx - cx) <= radius_um / px] = False
        before = process_masks(iid, v, f, px, cfg, d_um, synthetic)
        after = process_masks(iid, v2, f, px, cfg, d_um, synthetic)
        rec = {"image_id": iid, "status_before": before["status"], "status_after": after["status"],
               "n_patches": n_patches, "radius_um": radius_um,
               "removed_vessel_px": int(v.sum() - v2.sum())}
        for k in KEYS:
            b = before.get("features", {}).get(k, np.nan)
            a = after.get("features", {}).get(k, np.nan)
            rec[f"{k}_before"], rec[f"{k}_after"], rec[f"delta_{k}"] = b, a, a - b
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(out_dir / "patch_dropout.csv", index=False)
    write_json(out_dir / "patch_dropout.json", {
        "data_provenance": SYNTHETIC_BANNER if synthetic else "real", "n_images": len(df),
        "median_delta": {k: float(df[f"delta_{k}"].median()) for k in KEYS if len(df)},
        "note": "Descriptive response to localised vessel deletion; no direction is assumed."})
    return df


def overlay_sheet(pipeline_root, image_ids, out_dir, n: int = 20, seed: int = 0) -> pd.DataFrame:
    pipeline_root, out_dir = Path(pipeline_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = _pick(list(image_ids), n, seed, "overlay_qc")
    rows = []
    for iid in chosen:
        src = pipeline_root / "images" / safe_name(iid)
        # graph over the OCTA scan (native resolution) first; *_on_mask.png files are
        # mask-only diagnostics and cannot show alignment with the scan
        files = (sorted(src.glob("qc_graph_on_intensity.png")) + sorted(src.glob("overlay_*.png"))
                 + sorted(src.glob("curves.png")))
        for fpath in files:
            shutil.copy2(fpath, out_dir / f"{safe_name(iid)}__{fpath.name}")
        rows.append({"image_id": iid, "n_figures": len(files),
                     "intensity_background": (src / "qc_graph_on_intensity.png").exists(),
                     "reviewer": "", "review_date": "", "graph_ok": "", "notes": ""})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "overlay_review.csv", index=False)
    return df
