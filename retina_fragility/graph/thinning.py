"""Density-matched random thinning of healthy graphs (Experiment C, H4).

Whole segments are removed in a fixed random order; no edge is added or
rewired. After removal the topology-only cleanup steps are re-applied
(fragment rule, isolated nodes, degree-2 contraction; review B3). Length
density after that cleanup is non-increasing in the removal prefix k, so the
nearest feasible prefix is found by binary search.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .cleanup import contract_degree2, drop_small_fragments
from .model import VesselGraph


def cleaned_length(n: int, eu: np.ndarray, ev: np.ndarray, length: np.ndarray, active: np.ndarray,
                   min_fragment: float) -> float:
    """Total length of active edges in components with total length >= min_fragment."""
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return 0.0
    a = coo_matrix((np.ones(idx.size), (eu[idx], ev[idx])), shape=(n, n))
    _, lab = connected_components(a, directed=False)
    comp_len = np.bincount(lab[eu[idx]], weights=length[idx])
    if min_fragment <= 0:
        return float(comp_len.sum())
    return float(comp_len[comp_len >= min_fragment].sum())


def nearest_prefix(perm: np.ndarray, n: int, eu, ev, length, target_density: float,
                   min_fragment: float, area_mm2: float) -> tuple[int, float]:
    """Prefix length k of ``perm`` whose post-cleanup density is nearest the target.

    Density is mm per mm² (lengths in µm). Ties prefer the smaller k.
    """
    eu, ev, length = np.asarray(eu), np.asarray(ev), np.asarray(length, float)
    m = len(length)

    def dens(k: int) -> float:
        active = np.ones(m, bool)
        active[perm[:k]] = False
        return cleaned_length(n, eu, ev, length, active, min_fragment) / 1000.0 / area_mm2

    d0 = dens(0)
    if d0 <= target_density:
        return 0, d0
    lo, hi = 0, m  # dens(lo) > target, dens(m) = 0 <= target
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if dens(mid) <= target_density:
            hi = mid
        else:
            lo = mid
    d_lo, d_hi = dens(lo), dens(hi)
    if abs(d_lo - target_density) <= abs(d_hi - target_density):
        return lo, d_lo
    return hi, d_hi


def thinned_graph(graph: VesselGraph, removed_edge_ids, min_fragment: float) -> VesselGraph:
    g = graph.subgraph_without(removed_edge_ids)
    drop_small_fragments(g, min_fragment)
    g.drop_isolated_nodes()
    contract_degree2(g)
    g.drop_isolated_nodes()
    return g.renumber()
