"""Fixed, ordered, audited graph cleanup (review section 3.2).

Order: border -> FAZ -> contract -> spurs -> contract -> fragments -> isolated.
Each stage logs graph statistics so pre-/post-clean information is preserved.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .model import Edge, VesselGraph


def _edge_all_pixels(g: VesselGraph, e: Edge) -> list[tuple[int, int]]:
    return list(e.path) + list(g.nodes[e.u].pixels) + list(g.nodes[e.v].pixels)


def drop_border_edges(g: VesselGraph, band_px: int) -> list[int]:
    if band_px <= 0 or g.shape is None:
        return []
    H, W = g.shape
    rm = []
    for eid in sorted(g.edges):
        for (y, x) in _edge_all_pixels(g, g.edges[eid]):
            if y < band_px or x < band_px or y >= H - band_px or x >= W - band_px:
                rm.append(eid)
                break
    for eid in rm:
        del g.edges[eid]
    return rm


def drop_faz_edges(g: VesselGraph, faz: np.ndarray | None, inside_fraction: float) -> list[int]:
    if faz is None:
        return []
    faz = np.asarray(faz, dtype=bool)
    rm = []
    for eid in sorted(g.edges):
        e = g.edges[eid]
        px = e.path if e.path else _edge_all_pixels(g, e)
        frac = float(np.mean([faz[p] for p in px]))
        if frac >= inside_fraction:
            rm.append(eid)
    for eid in rm:
        del g.edges[eid]
    return rm


def _oriented(e: Edge, toward: int) -> list[tuple[int, int]]:
    """Path of e oriented so that it ends at node ``toward``."""
    return list(e.path) if e.v == toward else list(reversed(e.path))


def contract_degree2(g: VesselGraph) -> int:
    """Merge the two edges at every degree-2 node (not self-loops). Returns #contractions."""
    n_done = 0
    incmap: dict[int, set[int]] = {n: set() for n in g.nodes}
    for eid, e in g.edges.items():
        incmap[e.u].add(eid)
        incmap[e.v].add(eid)
    for nid in sorted(list(g.nodes)):
        if nid not in g.nodes:
            continue
        inc = sorted(incmap[nid])
        # degree counts self-loops twice
        deg = sum(2 if g.edges[eid].u == g.edges[eid].v else 1 for eid in inc)
        if deg != 2 or len(inc) != 2:
            continue  # includes the isolated loop case (one self-loop edge)
        a, b = g.edges[inc[0]], g.edges[inc[1]]
        x = a.u if a.v == nid else a.v
        y = b.u if b.v == nid else b.v
        node_px = sorted(g.nodes[nid].pixels)
        path_a = _oriented(a, nid)
        path_b = list(reversed(_oriented(b, nid)))  # now starts at nid, ends at y
        path = path_a + node_px + path_b
        n_w = a.n_width_px + b.n_width_px
        width = (a.width * a.n_width_px + b.width * b.n_width_px) / n_w
        new = Edge(min(a.id, b.id), x, y, a.length + b.length, width, path, n_w,
                   sorted(set(a.source_ids) | set(b.source_ids)))
        for old in (a, b):
            del g.edges[old.id]
            incmap[old.u].discard(old.id)
            incmap[old.v].discard(old.id)
        del g.nodes[nid]
        del incmap[nid]
        g.add_edge(new)
        incmap[x].add(new.id)
        incmap[y].add(new.id)
        n_done += 1
    return n_done


def prune_spurs(g: VesselGraph, factor: float, passes: int = 1) -> list[int]:
    """Drop terminal edges shorter than ``factor`` x local width (simultaneous per pass).

    Terminal = one endpoint degree 1, the other degree >= 3. Local width = max
    width of the other edges incident to the junction end.
    """
    removed: list[int] = []
    if factor <= 0:
        return removed
    for _ in range(max(0, int(passes))):
        inc = g.incidence()
        deg = {n: len(v) for n, v in inc.items()}
        rm = []
        for eid in sorted(g.edges):
            e = g.edges[eid]
            if e.u == e.v:
                continue
            if deg[e.u] == 1 and deg[e.v] >= 3:
                j = e.v
            elif deg[e.v] == 1 and deg[e.u] >= 3:
                j = e.u
            else:
                continue
            others = [g.edges[o].width for o in inc[j] if o != eid]
            local_w = max(others) if others else e.width
            if e.length < factor * local_w:
                rm.append(eid)
        if not rm:
            break
        for eid in rm:
            del g.edges[eid]
        removed += rm
        g.drop_isolated_nodes()
        contract_degree2(g)
    return removed


def drop_small_fragments(g: VesselGraph, min_length: float) -> list[int]:
    rm = []
    if min_length <= 0:
        return rm
    for nodes, edges in g.components():
        tot = sum(g.edges[e].length for e in edges)
        if tot < min_length:
            rm += edges
    for eid in rm:
        del g.edges[eid]
    return rm


def clean_graph(raw: VesselGraph, gcfg: dict, faz: np.ndarray | None = None,
                area_mm2: float | None = None) -> tuple[VesselGraph, list[dict]]:
    """Apply the fixed cleanup. ``raw`` is not modified. Returns (clean graph, audit)."""
    g = copy.deepcopy(raw)
    audit: list[dict] = []

    def log(stage: str, **info):
        st = g.stats(area_mm2)
        st.update(info)
        st["stage"] = stage
        audit.append(st)

    log("0_raw")
    rm = drop_border_edges(g, int(gcfg.get("border_band_px", 1)))
    log("1_border", removed_edges=len(rm))
    rm = drop_faz_edges(g, faz, float(gcfg.get("faz_inside_fraction", 0.5)))
    log("2_faz", removed_edges=len(rm))
    iso = g.drop_isolated_nodes()
    nc = contract_degree2(g)
    log("3_contract", removed_isolated_nodes=iso, contractions=nc)
    rm = prune_spurs(g, float(gcfg.get("spur_length_factor", 2.0)),
                     int(gcfg.get("spur_prune_passes", 1)))
    log("4_spurs", removed_edges=len(rm))
    iso = g.drop_isolated_nodes()
    nc = contract_degree2(g)
    log("5_contract", removed_isolated_nodes=iso, contractions=nc)
    if g.units == "um":
        thr = float(gcfg.get("min_fragment_length_um", 50.0))
    else:
        thr_px = gcfg.get("min_fragment_length_px")
        if thr_px is None:
            raise ValueError("pixel-unit graph requires graph.min_fragment_length_px "
                             "(physical threshold cannot be applied without pixel size)")
        thr = float(thr_px)
    rm = drop_small_fragments(g, thr)
    log("6_fragments", removed_edges=len(rm), threshold=thr)
    iso = g.drop_isolated_nodes()
    g = g.renumber()
    log("7_final", removed_isolated_nodes=iso)
    for st in audit:
        for k, v in list(st.items()):
            if isinstance(v, float) and not math.isfinite(v):
                st[k] = None
    return g, audit
