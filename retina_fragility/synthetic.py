"""Synthetic graphs and mask fixtures (Experiment 0 and software tests).

Everything produced here is SYNTHETIC - NOT CLINICAL.
"""

from __future__ import annotations

import math

import numpy as np
from skimage.draw import disk as draw_disk
from skimage.draw import line as draw_line

from .graph.model import VesselGraph, graph_from_edge_list


def grid_graph(n: int, length: float = 1.0, width: float = 1.0) -> VesselGraph:
    """n x n square lattice (n^2 nodes, 2n(n-1) edges); edge ids in raster order."""
    edges = []
    for r in range(n):
        for c in range(n):
            i = r * n + c
            if c + 1 < n:
                edges.append((i, i + 1, length))
            if r + 1 < n:
                edges.append((i, i + n, length))
    return graph_from_edge_list(n * n, edges, [width] * len(edges))


def binary_tree_graph(depth: int, length: float = 1.0, width: float = 1.0) -> VesselGraph:
    """Complete binary tree with ``depth`` levels below the root (2^(depth+1)-1 nodes)."""
    n = 2 ** (depth + 1) - 1
    edges = [((i - 1) // 2, i, length) for i in range(1, n)]
    return graph_from_edge_list(n, edges, [width] * len(edges))


# --------------------------------------------------------------- mask fixtures
def _thick_line(mask, y0, x0, y1, x1, radius):
    rr, cc = draw_line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
    H, W = mask.shape
    for y, x in zip(rr, cc):
        if radius <= 0:
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] = True
        else:
            dy, dx = draw_disk((y, x), radius + 0.5, shape=mask.shape)
            mask[dy, dx] = True


def path_mask(shape=(64, 64), radius=1) -> np.ndarray:
    m = np.zeros(shape, bool)
    _thick_line(m, shape[0] // 2, 8, shape[0] // 2, shape[1] - 9, radius)
    return m


def loop_mask(shape=(64, 64), r=18, radius=1) -> np.ndarray:
    """An isolated ring (one cycle, no junction)."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    d = np.hypot(yy - shape[0] / 2, xx - shape[1] / 2)
    return np.abs(d - r) <= radius + 0.5


def grid_mask(n_lines=5, spacing=20, margin=10, radius=1) -> np.ndarray:
    size = margin * 2 + spacing * (n_lines - 1) + 1
    m = np.zeros((size, size), bool)
    for i in range(n_lines):
        p = margin + i * spacing
        _thick_line(m, p, margin, p, size - margin - 1, radius)
        _thick_line(m, margin, p, size - margin - 1, p, radius)
    return m


def tree_mask(shape=(128, 128), depth=4, radius=1) -> np.ndarray:
    m = np.zeros(shape, bool)

    def rec(y, x, ang, ln, d):
        y1 = y + ln * math.sin(ang)
        x1 = x + ln * math.cos(ang)
        _thick_line(m, y, x, y1, x1, radius)
        if d > 0:
            rec(y1, x1, ang - 0.5, ln * 0.7, d - 1)
            rec(y1, x1, ang + 0.5, ln * 0.7, d - 1)

    rec(shape[0] - 10, shape[1] / 2, -math.pi / 2, shape[0] * 0.3, depth)
    return m


def random_network_mask(shape=(304, 304), n_seeds=60, k=3, radius=1, seed=0,
                        faz_radius=25) -> tuple[np.ndarray, np.ndarray]:
    """Random planar-ish capillary network: seed points connected to k nearest neighbours.

    Returns (vessel mask, FAZ mask). FAZ is a central disk kept vessel-free.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    pts = np.column_stack([rng.uniform(4, H - 5, n_seeds), rng.uniform(4, W - 5, n_seeds)])
    c = np.array([H / 2, W / 2])
    pts = pts[np.hypot(*(pts - c).T) > faz_radius + 8]
    m = np.zeros(shape, bool)
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    for i in range(len(pts)):
        for j in np.argsort(d[i])[1:k + 1]:
            if j > i or i not in np.argsort(d[j])[1:k + 1]:
                _thick_line(m, pts[i, 0], pts[i, 1], pts[j, 0], pts[j, 1], radius)
    yy, xx = np.mgrid[:H, :W]
    faz = np.hypot(yy - c[0], xx - c[1]) <= faz_radius
    m &= ~faz
    return m, faz


def patch_dropout(mask: np.ndarray, center, radius: float) -> np.ndarray:
    """Delete vessels inside a disk (synthetic capillary dropout)."""
    out = mask.copy()
    rr, cc = draw_disk(center, radius, shape=mask.shape)
    out[rr, cc] = False
    return out


def rescale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour upsampling by an integer factor (for physical-scaling tests)."""
    return np.kron(mask.astype(np.uint8), np.ones((factor, factor), np.uint8)).astype(bool)


# ------------------------------------------------------ synthetic fixture dataset
SYNTHETIC_DATASET = "SYNTHETIC-OCTA500-FIXTURE"


def synthetic_intensity(vessel: np.ndarray, seed: int, noise: float = 0.08) -> np.ndarray:
    """OCTA-like projection: blurred bright vessels on dark background plus noise (uint8)."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    img = ndi.gaussian_filter(vessel.astype(float), 0.8) * 0.75 + 0.1
    img += rng.normal(0, noise, vessel.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def make_fixture_dataset(out_dir, n_normal_per_fov: int = 14, n_dr_per_fov: int = 8,
                         seed: int = 0, fovs=(("3", (304, 304), 3000.0 / 304.0),
                                              ("6", (400, 400), 15.0)),
                         n_seeds: int = 110, dropout_patches: int = 4) -> pd.DataFrame:  # noqa: F821
    """Write SYNTHETIC mask fixtures and a manifest. NOT CLINICAL DATA.

    'DR' fixtures are the same generator plus random vessel-free patches
    (synthetic capillary dropout). Ages/sexes are random. The manifest dataset
    is ``SYNTHETIC-OCTA500-FIXTURE`` so every downstream output is labelled
    synthetic.
    """
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    out = Path(out_dir)
    for sub in ("images", "vessel", "faz"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    k = 0
    for fov, shape, px in fovs:
        for group, n in (("NORMAL", n_normal_per_fov), ("DR", n_dr_per_fov)):
            for i in range(n):
                s = int(rng.integers(0, 2**31 - 1))
                faz_r = int(round(float(rng.uniform(180.0, 330.0)) / px))  # FAZ radius 180-330 µm
                vessel, faz = random_network_mask(shape, n_seeds=n_seeds, k=3, radius=1,
                                                  seed=s, faz_radius=faz_r)
                if group == "DR":
                    r2 = np.random.default_rng(s + 1)
                    for _ in range(dropout_patches):
                        c = (r2.uniform(0.15, 0.85) * shape[0], r2.uniform(0.15, 0.85) * shape[1])
                        vessel = patch_dropout(vessel, c, r2.uniform(0.06, 0.12) * shape[0])
                iid = f"SYN{fov}_{group}_{i:03d}"
                Image.fromarray(synthetic_intensity(vessel, s)).save(out / "images" / f"{iid}.png")
                Image.fromarray((vessel * 255).astype(np.uint8)).save(out / "vessel" / f"{iid}.png")
                Image.fromarray((faz * 255).astype(np.uint8)).save(out / "faz" / f"{iid}.png")
                rows.append({"image_id": iid, "dataset": SYNTHETIC_DATASET, "subject_id": "",
                             "image_path": str(out / "images" / f"{iid}.png"),
                             "vessel_mask_path": str(out / "vessel" / f"{iid}.png"),
                             "large_vessel_mask_path": "",
                             "faz_mask_path": str(out / "faz" / f"{iid}.png"),
                             "fov_mm": float(fov), "pixel_size_um": px, "disease": group,
                             "grade": "", "age": int(rng.integers(30, 80)),
                             "sex": "M" if rng.random() < 0.5 else "F",
                             "split": "train",
                             "seg_split": ("train", "train", "train", "val", "test")[k % 5],
                             "calibration": False, "qc_audited": False})
                k += 1
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "manifest.csv", index=False)
    (out / "README.txt").write_text(
        "SYNTHETIC - NOT CLINICAL. Generated by retina_fragility.synthetic.make_fixture_dataset "
        f"(seed={seed}). For software validation only.\n")
    return frame
