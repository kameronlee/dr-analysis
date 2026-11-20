"""Coverage bookkeeping: analysis pixels within distance d of remaining edges.

Each edge owns a pixel set, and covers the analysis pixels within Euclidean
distance <= d of that set. A per-pixel reference count of remaining covering
edges makes removal exact: a pixel shared by two edges (e.g. a common
endpoint) stays covered until both go.

Two edge pixel-set geometries exist (resume_review.md B1):

* ``centerline``: interior path pixels plus attachment node-cluster pixels.
* ``vessel_territory`` (default): the vessel-mask pixels whose nearest *raw*
  skeleton pixel belongs to the edge's centreline set. This matches the
  calibration distance d, which is measured from tissue to vessel pixels, not
  to centrelines. Pixels nearest to skeleton parts removed by cleanup belong
  to no edge.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from .model import VesselGraph


def centerline_pixel_sets(graph: VesselGraph) -> dict[int, np.ndarray]:
    """edge id -> flat pixel indices of interior path + attachment node clusters."""
    if graph.shape is None:
        raise ValueError("pixel sets require pixel geometry")
    H, W = graph.shape
    out = {}
    for eid in sorted(graph.edges):
        px = graph.edge_pixels(eid, include_nodes=True)
        if px:
            arr = np.array(px, dtype=np.int64)
            out[eid] = np.unique(arr[:, 0] * W + arr[:, 1])
        else:
            out[eid] = np.zeros(0, dtype=np.int64)
    return out


def vessel_territories(graph: VesselGraph, raw_skeleton: np.ndarray,
                       vessel_mask: np.ndarray) -> dict[int, np.ndarray]:
    """edge id -> flat indices of vessel pixels whose nearest raw-skeleton pixel it owns.

    Node-cluster pixels are owned by every incident edge, so territory around a
    junction is shared and reference-counted. Each edge's centreline pixels are
    always part of its territory.
    """
    if graph.shape is None:
        raise ValueError("territories require pixel geometry")
    skel = np.asarray(raw_skeleton, dtype=bool)
    vessel = np.asarray(vessel_mask, dtype=bool)
    if skel.shape != tuple(graph.shape) or vessel.shape != skel.shape:
        raise ValueError("skeleton / vessel mask shape does not match graph")
    H, W = skel.shape
    centre = centerline_pixel_sets(graph)
    owners: dict[int, list[int]] = {}
    for eid, pix in centre.items():
        for p in pix.tolist():
            owners.setdefault(p, []).append(eid)
    terr: dict[int, list[np.ndarray]] = {eid: [pix] for eid, pix in centre.items()}
    if skel.any() and vessel.any():
        _, (iy, ix) = ndi.distance_transform_edt(~skel, return_indices=True)
        vflat = np.flatnonzero(vessel.ravel())
        nearest = (iy.ravel()[vflat] * W + ix.ravel()[vflat]).astype(np.int64)
        order = np.argsort(nearest, kind="stable")
        nearest_sorted = nearest[order]
        vsorted = vflat[order]
        uniq, start = np.unique(nearest_sorted, return_index=True)
        stop = np.append(start[1:], len(nearest_sorted))
        for sp, a, b in zip(uniq.tolist(), start.tolist(), stop.tolist()):
            eids = owners.get(sp)
            if eids is None:
                continue
            block = vsorted[a:b]
            for eid in eids:
                terr[eid].append(block)
    return {eid: np.unique(np.concatenate(parts)) for eid, parts in terr.items()}


class CoverageTracker:
    def __init__(self, graph: VesselGraph, analysis_mask: np.ndarray, d_um: float | None = None,
                 d_px: float | None = None, edge_pixel_sets: dict[int, np.ndarray] | None = None):
        if graph.shape is None:
            raise ValueError("coverage requires pixel geometry")
        am = np.asarray(analysis_mask, dtype=bool)
        if am.shape != tuple(graph.shape):
            raise ValueError("analysis mask shape does not match graph shape")
        if d_px is None:
            if d_um is None or graph.pixel_size_um is None:
                raise ValueError("coverage needs d in px, or d in um plus pixel size")
            d_px = float(d_um) / float(graph.pixel_size_um)
        if not (d_px >= 0 and math.isfinite(d_px)):
            raise ValueError("d must be finite and non-negative")
        self.d_px = float(d_px)
        self.shape = am.shape
        self.analysis_mask = am
        self.area_px = int(am.sum())
        self.pixel_area_mm2 = ((graph.pixel_size_um / 1000.0) ** 2
                               if graph.pixel_size_um is not None else None)
        if edge_pixel_sets is None:
            edge_pixel_sets = centerline_pixel_sets(graph)
        c = graph.compact()
        self.edge_ids = c["edge_ids"]
        H, W = am.shape
        r = int(math.ceil(self.d_px))
        self.cover: list[np.ndarray] = []
        for eid in self.edge_ids.tolist():
            flat = np.asarray(edge_pixel_sets.get(eid, np.zeros(0, np.int64)), dtype=np.int64)
            if flat.size == 0:
                self.cover.append(np.zeros(0, dtype=np.int64))
                continue
            ys, xs = flat // W, flat % W
            y0, y1 = max(0, ys.min() - r - 1), min(H, ys.max() + r + 2)
            x0, x1 = max(0, xs.min() - r - 1), min(W, xs.max() + r + 2)
            win = np.ones((y1 - y0, x1 - x0), dtype=bool)
            win[ys - y0, xs - x0] = False
            dist = ndi.distance_transform_edt(win)
            sel = (dist <= self.d_px + 1e-9) & am[y0:y1, x0:x1]
            wy, wx = np.nonzero(sel)
            self.cover.append(((wy + y0) * W + (wx + x0)).astype(np.int64))
        self.reset()

    def reset(self) -> None:
        self.count = np.zeros(self.shape[0] * self.shape[1], dtype=np.int32)
        for idx in self.cover:
            self.count[idx] += 1
        self.n_covered = int((self.count > 0).sum())

    def remove(self, edge_positions) -> None:
        """Remove edges by *compact position* (index into compact arrays)."""
        for i in np.atleast_1d(edge_positions).tolist():
            idx = self.cover[i]
            if idx.size == 0:
                continue
            self.count[idx] -= 1
            self.n_covered -= int((self.count[idx] == 0).sum())

    def fraction(self) -> float:
        return self.n_covered / self.area_px if self.area_px > 0 else float("nan")

    def unique_area_px(self) -> np.ndarray:
        """Per edge: analysis pixels covered by this edge only (on the current state)."""
        return np.array([int((self.count[idx] == 1).sum()) for idx in self.cover])

    def covered_mask(self) -> np.ndarray:
        return (self.count > 0).reshape(self.shape)


def empty_coverage_fraction(analysis_mask: np.ndarray) -> float:
    """Coverage of an empty graph: 0 if there is analysis area, NaN otherwise."""
    a = int(np.asarray(analysis_mask, bool).sum())
    return 0.0 if a > 0 else float("nan")
