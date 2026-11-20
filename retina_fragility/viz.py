"""Overlays and curves with physical units (axes in mm, values in µm / mm²).

Graph overlays are drawn on the OCTA intensity image whenever it is available,
so a reviewer can check that extracted paths follow vessels in the original
scan. A binary-mask background can only show agreement with the mask itself;
figures drawn on it are named ``*_on_mask.png`` and titled as mask-only
diagnostics. ``qc_composites`` writes native-resolution RGB images (no
resampling) for segmentation/graph QC.

Figures derived from synthetic inputs carry the SYNTHETIC - NOT CLINICAL banner.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from .graph.model import VesselGraph


def edge_value_image(graph: VesselGraph, values: dict[int, float], thicken_px: int = 1) -> np.ndarray:
    """Float image with each edge's pixels set to its value (max where edges overlap)."""
    H, W = graph.shape
    img = np.full((H, W), -np.inf)
    for eid, val in values.items():
        if eid not in graph.edges or not np.isfinite(val):
            continue
        for (y, x) in graph.edge_pixels(eid, include_nodes=True):
            img[y, x] = max(img[y, x], float(val))
    if thicken_px > 0:
        img = ndi.grey_dilation(img, size=(2 * thicken_px + 1, 2 * thicken_px + 1))
    img[~np.isfinite(img)] = np.nan
    return img


def _extent_mm(shape, pixel_size_um):
    H, W = shape
    s = pixel_size_um / 1000.0
    return (0, W * s, H * s, 0)


def overlay(vessel: np.ndarray, graph: VesselGraph, values: dict[int, float], out: str | Path,
            title: str, cbar_label: str, pixel_size_um: float, synthetic: bool,
            cmap: str = "viridis", analysis: np.ndarray | None = None,
            removed_pixels: np.ndarray | None = None, intensity: np.ndarray | None = None) -> None:
    """Edge values over the OCTA ``intensity`` image (or, if None, over the binary mask).

    With an intensity background the vessel-mask outline is drawn as a thin
    contour, so image, label and graph can be compared in one panel.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    ext = _extent_mm(vessel.shape, pixel_size_um)
    if intensity is not None:
        if np.shape(intensity) != np.shape(vessel):
            raise ValueError("intensity image and vessel mask shapes differ")
        ax.imshow(np.asarray(intensity, float), cmap="gray", vmin=0, vmax=1, extent=ext,
                  interpolation="nearest")
        ax.contour(np.asarray(vessel, float), levels=[0.5], colors="tab:cyan", linewidths=0.3,
                   extent=(ext[0], ext[1], ext[3], ext[2]), origin="upper")
    else:
        base = np.where(np.asarray(vessel, bool), 0.8, 1.0)
        ax.imshow(base, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
        title = f"{title}\n[MASK-ONLY DIAGNOSTIC: background is the binary label, not the scan]"
    if analysis is not None:
        ax.contour(np.asarray(analysis, float), levels=[0.5], colors="tab:blue", linewidths=0.6,
                   extent=(ext[0], ext[1], ext[3], ext[2]), origin="upper")
    if removed_pixels is not None and np.any(removed_pixels):
        red = np.zeros(vessel.shape + (4,))
        red[np.asarray(removed_pixels, bool)] = (0.85, 0.1, 0.1, 0.9)
        ax.imshow(red, extent=ext, interpolation="nearest")
    img = edge_value_image(graph, values)
    im = ax.imshow(img, cmap=cmap, extent=ext, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(cbar_label)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(("SYNTHETIC - NOT CLINICAL\n" if synthetic else "") + title, fontsize=9)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _to_rgb8(intensity: np.ndarray) -> np.ndarray:
    g = np.clip(np.asarray(intensity, float), 0.0, 1.0)
    g8 = np.round(g * 255.0).astype(np.uint8)
    return np.repeat(g8[:, :, None], 3, axis=2)


def _blend(rgb: np.ndarray, where: np.ndarray, colour, alpha: float) -> np.ndarray:
    out = rgb.astype(float)
    out[where] = (1 - alpha) * out[where] + alpha * np.asarray(colour, float)
    return np.round(out).astype(np.uint8)


def graph_pixel_masks(graph: VesselGraph | None, shape) -> tuple[np.ndarray, np.ndarray]:
    """Boolean images of edge interior pixels and node-cluster pixels."""
    edge_px = np.zeros(shape, bool)
    node_px = np.zeros(shape, bool)
    if graph is not None:
        for eid in graph.edges:
            px = np.asarray(graph.edge_pixels(eid, include_nodes=False), dtype=int).reshape(-1, 2)
            edge_px[px[:, 0], px[:, 1]] = True
        for node in graph.nodes.values():
            px = np.asarray(getattr(node, "pixels", []) or [], dtype=int).reshape(-1, 2)
            node_px[px[:, 0], px[:, 1]] = True
    return edge_px, node_px


def graph_overlay_rgb(intensity: np.ndarray, graph: VesselGraph | None) -> np.ndarray:
    """Native-resolution RGB: edges green, node clusters yellow, over the intensity image."""
    rgb = _to_rgb8(intensity)
    edge_px, node_px = graph_pixel_masks(graph, rgb.shape[:2])
    g = _blend(rgb, edge_px, (50, 255, 80), 1.0)
    return _blend(g, node_px, (255, 230, 0), 1.0)


def qc_composites(intensity: np.ndarray, vessel: np.ndarray, graph: VesselGraph | None,
                  out_dir: str | Path, synthetic: bool) -> dict[str, str]:
    """Native-resolution QC images (one output pixel per input pixel).

    * ``qc_intensity.png``: the OCTA intensity image as analysed (nominal-range
      scaling only, no per-image stretching).
    * ``qc_mask_on_intensity.png``: vessel mask in red over the intensity image.
    * ``qc_graph_on_intensity.png``: cleaned-graph edge pixels (incl. node
      clusters) in green over the intensity image; node-cluster pixels in yellow.
    * ``diagnostic_mask_only_graph.png``: graph over the binary mask. It shows
      label/graph agreement only and cannot establish alignment with the scan.

    Synthetic inputs get a ``SYNTHETIC_NOT_CLINICAL.txt`` marker next to the images.
    """
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vessel = np.asarray(vessel, bool)
    if np.shape(intensity) != vessel.shape:
        raise ValueError("intensity image and vessel mask shapes differ")
    rgb = _to_rgb8(intensity)
    files = {"intensity": out_dir / "qc_intensity.png",
             "mask_on_intensity": out_dir / "qc_mask_on_intensity.png",
             "graph_on_intensity": out_dir / "qc_graph_on_intensity.png",
             "mask_only_diagnostic": out_dir / "diagnostic_mask_only_graph.png"}
    Image.fromarray(rgb).save(files["intensity"])
    Image.fromarray(_blend(rgb, vessel, (255, 60, 60), 0.55)).save(files["mask_on_intensity"])
    edge_px, node_px = graph_pixel_masks(graph, vessel.shape)
    Image.fromarray(graph_overlay_rgb(intensity, graph)).save(files["graph_on_intensity"])
    mask_bg = np.where(vessel, 200, 255).astype(np.uint8)
    m = np.repeat(mask_bg[:, :, None], 3, axis=2)
    m = _blend(m, edge_px, (0, 110, 0), 1.0)
    m = _blend(m, node_px, (200, 120, 0), 1.0)
    Image.fromarray(m).save(files["mask_only_diagnostic"])
    if synthetic:
        (out_dir / "SYNTHETIC_NOT_CLINICAL.txt").write_text(
            "These QC images derive from SYNTHETIC fixtures (software validation only).\n")
    return {k: str(v) for k, v in files.items()}


def plot_curves(curves: dict, out: str | Path, title: str, synthetic: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for name, c in curves.items():
        axes[0].plot(c.q_edge, c.S, label=name.replace("_", " "))
        if c.P is not None:
            axes[1].plot(c.q_edge, c.P / c.P[0] if c.P[0] > 0 else c.P, label=name.replace("_", " "))
    axes[0].set(xlabel="fraction of edges removed q", ylabel="S(q): largest-component length share",
                xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="fraction of edges removed q", ylabel="P(q)/P0 coverage", xlim=(0, 1))
    axes[0].legend(fontsize=7)
    fig.suptitle(("SYNTHETIC - NOT CLINICAL: " if synthetic else "") + title, fontsize=9)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_group_curves(grid: np.ndarray, groups: dict, out: str | Path, title: str,
                      synthetic: bool, ylabel: str = "S(q)") -> None:
    """groups: name -> dict(mean, lo, hi) arrays on ``grid``."""
    fig, ax = plt.subplots(figsize=(5.2, 4), constrained_layout=True)
    for name, g in groups.items():
        line, = ax.plot(grid, g["mean"], label=f"{name} (n={g.get('n', '?')})")
        ax.fill_between(grid, g["lo"], g["hi"], color=line.get_color(), alpha=0.25)
    ax.set(xlabel="fraction of edges removed q", ylabel=ylabel, xlim=(0, 1))
    ax.legend(fontsize=7)
    ax.set_title(("SYNTHETIC - NOT CLINICAL\n" if synthetic else "") + title, fontsize=7)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
