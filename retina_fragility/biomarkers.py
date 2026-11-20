"""Standard per-image OCTA biomarkers, before and after graph cleanup.

All areas are the analysis area (image minus border band minus FAZ) unless the
name says otherwise. Densities are per mm² of analysis area.

* vessel_density: vessel pixels inside the analysis mask / analysis pixels.
* skeleton_length_density_mm_per_mm2: pre-clean m-adjacency skeleton length
  inside the analysis mask.
* graph_length_density_mm_per_mm2: post-clean total edge length L0 / analysis
  area. Edges between the 1-px graph border rule and the 50-µm analysis band
  count towards L0 (review A1).
* faz_area_mm2, faz_circularity (4πA/P², largest FAZ component, skimage
  perimeter). An empty FAZ gives area 0 and circularity NaN.
* fractal_dimension: box counting on the skeleton inside the analysis mask,
  dyadic box sizes 2..min(H, W)/4 px, OLS slope of log N vs log(1/s).
* junction/end density pre and post clean; loop_density_per_mm2 = cyclomatic
  number (e - v + c) of the cleaned graph per mm².
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import perimeter

from .graph.extract import skeleton_length
from .graph.model import VesselGraph


def pixel_area_mm2(pixel_size_um: float) -> float:
    return (float(pixel_size_um) / 1000.0) ** 2


def faz_metrics(faz: np.ndarray | None, pixel_size_um: float) -> dict:
    if faz is None:
        return {"faz_area_mm2": np.nan, "faz_circularity": np.nan, "faz_n_components": 0}
    faz = np.asarray(faz, bool)
    lab, n = ndi.label(faz)
    area_px = int(faz.sum())
    out = {"faz_area_mm2": area_px * pixel_area_mm2(pixel_size_um), "faz_n_components": int(n)}
    if n == 0:
        out["faz_circularity"] = np.nan
        return out
    sizes = ndi.sum(faz, lab, np.arange(1, n + 1))
    big = lab == (int(np.argmax(sizes)) + 1)
    per = perimeter(big)
    out["faz_circularity"] = float(4 * math.pi * big.sum() / per ** 2) if per > 0 else np.nan
    return out


def box_count_dimension(binary: np.ndarray, min_size: int = 2) -> dict:
    b = np.asarray(binary, bool)
    H, W = b.shape
    sizes = []
    s = min_size
    while s <= min(H, W) / 4:
        sizes.append(s)
        s *= 2
    if len(sizes) < 2 or not b.any():
        return {"fractal_dimension": np.nan, "fd_n_scales": len(sizes)}
    counts = []
    for s in sizes:
        ph, pw = (-H) % s, (-W) % s
        p = np.pad(b, ((0, ph), (0, pw)))
        blocks = p.reshape(p.shape[0] // s, s, p.shape[1] // s, s).any(axis=(1, 3))
        counts.append(int(blocks.sum()))
    x = np.log(1.0 / np.array(sizes, float))
    y = np.log(np.array(counts, float))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else np.nan
    return {"fractal_dimension": float(slope), "fd_n_scales": len(sizes), "fd_r2": r2,
            "fd_box_sizes_px": sizes}


def compute_biomarkers(vessel: np.ndarray, faz: np.ndarray | None, analysis: np.ndarray,
                       skel: np.ndarray, raw_graph: VesselGraph, clean_graph: VesselGraph,
                       pixel_size_um: float) -> dict:
    vessel = np.asarray(vessel, bool)
    analysis = np.asarray(analysis, bool)
    skel = np.asarray(skel, bool)
    pa = pixel_area_mm2(pixel_size_um)
    n_area = int(analysis.sum())
    area_mm2 = n_area * pa
    out: dict = {"analysis_area_mm2": area_mm2, "analysis_area_px": n_area,
                 "image_area_mm2": vessel.size * pa, "pixel_size_um": float(pixel_size_um)}
    out["vessel_density"] = float((vessel & analysis).sum() / n_area) if n_area else np.nan
    out["vessel_density_full_image"] = float(vessel.mean())
    sk_in = skel & analysis
    out["skeleton_length_mm_pre"] = skeleton_length(sk_in, pixel_size_um) / 1000.0
    out["skeleton_length_density_mm_per_mm2"] = (out["skeleton_length_mm_pre"] / area_mm2
                                                 if area_mm2 else np.nan)
    out.update(faz_metrics(faz, pixel_size_um))
    fd = box_count_dimension(sk_in)
    out["fractal_dimension"] = fd["fractal_dimension"]
    out["fd_r2"] = fd.get("fd_r2", np.nan)
    pre = raw_graph.stats(area_mm2 if area_mm2 else None)
    post = clean_graph.stats(area_mm2 if area_mm2 else None)
    for tag, st in (("pre", pre), ("post", post)):
        for k in ("n_nodes", "n_edges", "n_components", "cyclomatic_number", "n_junctions",
                  "n_endpoints", "n_self_loops", "n_parallel_extra_edges"):
            out[f"{k}_{tag}"] = st[k]
        out[f"total_length_mm_{tag}"] = st["total_length"] / 1000.0
        out[f"junction_density_per_mm2_{tag}"] = st.get("junction_density_per_mm2", np.nan)
        out[f"end_density_per_mm2_{tag}"] = st.get("end_density_per_mm2", np.nan)
    out["graph_length_density_mm_per_mm2"] = (post["total_length"] / 1000.0 / area_mm2
                                              if area_mm2 else np.nan)
    out["loop_density_per_mm2"] = post["cyclomatic_number"] / area_mm2 if area_mm2 else np.nan
    out["loop_density_per_mm2_pre"] = pre["cyclomatic_number"] / area_mm2 if area_mm2 else np.nan
    return out
