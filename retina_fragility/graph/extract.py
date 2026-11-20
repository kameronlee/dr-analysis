"""Skeleton -> multigraph by m-adjacency path tracing.

See docs/reviews/scientific_review.md section 3.1 for the rules.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .model import Edge, Node, VesselGraph

# fixed neighbour order: 4-neighbours first, then diagonals (deterministic tracing)
OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise ValueError("mask must be 2-D")
    return skeletonize(m)


def m_adjacency(skel: np.ndarray) -> np.ndarray:
    """Boolean array (8, H, W): nb[k, y, x] is True if (y,x) is m-adjacent to (y,x)+OFFSETS[k].

    A diagonal neighbour is dropped when the two pixels share a 4-neighbour in
    the skeleton (mixed adjacency, Gonzalez & Woods).
    """
    s = np.asarray(skel, dtype=bool)
    H, W = s.shape
    p = np.pad(s, 1)
    nb = np.zeros((8, H, W), dtype=bool)
    for k, (dy, dx) in enumerate(OFFSETS):
        shifted = p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        a = s & shifted
        if dy != 0 and dx != 0:
            via_v = p[1 + dy:1 + dy + H, 1:1 + W]  # (y+dy, x)
            via_h = p[1:1 + H, 1 + dx:1 + dx + W]  # (y, x+dx)
            a &= ~(via_v | via_h)
        nb[k] = a
    return nb


def trace_graph(skel: np.ndarray, pixel_size_um: float | None = None,
                vessel_mask: np.ndarray | None = None, force_2x2: bool = True,
                meta: dict | None = None) -> VesselGraph:
    """Trace a 2-D skeleton into a VesselGraph.

    Lengths: Euclidean pixel steps from the attachment pixel of the start node
    through the interior path to the attachment pixel of the end node, times
    ``pixel_size_um`` (or px if unknown). Distances inside node clusters are
    not counted. Width: 2 * mean(EDT of ``vessel_mask`` along interior path).
    """
    s = np.asarray(skel, dtype=bool)
    H, W = s.shape
    scale = float(pixel_size_um) if pixel_size_um is not None else 1.0
    nb = m_adjacency(s)
    deg = nb.sum(axis=0) * s
    if vessel_mask is not None:
        edt = ndi.distance_transform_edt(np.asarray(vessel_mask, dtype=bool))
    else:
        edt = ndi.distance_transform_edt(s)  # = 1 on skeleton pixels

    junction = s & (deg >= 3)
    if force_2x2:
        blk = s[:-1, :-1] & s[1:, :-1] & s[:-1, 1:] & s[1:, 1:]
        forced = np.zeros_like(s)
        forced[:-1, :-1] |= blk
        forced[1:, :-1] |= blk
        forced[:-1, 1:] |= blk
        forced[1:, 1:] |= blk
        junction |= forced
    # junction pixels strictly closer than 2 px (Euclidean 1 or sqrt2) -> one cluster
    jlab, nj = ndi.label(junction, structure=np.ones((3, 3), dtype=int))
    endpoint = s & (deg == 1) & ~junction
    isolated = s & (deg == 0) & ~junction

    # collect node pixel groups, order nodes by first raster pixel
    groups: list[tuple[int, list[tuple[int, int]], str]] = []
    if nj:
        objs = ndi.find_objects(jlab)
        for lab in range(1, nj + 1):
            sl = objs[lab - 1]
            ys, xs = np.nonzero(jlab[sl] == lab)
            pix = sorted((int(y + sl[0].start), int(x + sl[1].start)) for y, x in zip(ys, xs))
            groups.append((pix[0][0] * W + pix[0][1], pix, "junction"))
    for kind, arr in (("endpoint", endpoint), ("isolated", isolated)):
        ys, xs = np.nonzero(arr)
        for y, x in zip(ys.tolist(), xs.tolist()):
            groups.append((y * W + x, [(y, x)], kind))
    groups.sort(key=lambda t: t[0])

    g = VesselGraph((H, W), pixel_size_um, meta)
    node_of = -np.ones((H, W), dtype=np.int64)
    for nid, (_, pix, kind) in enumerate(groups):
        ys = np.array([p[0] for p in pix], float)
        xs = np.array([p[1] for p in pix], float)
        g.add_node(Node(nid, float(ys.mean()), float(xs.mean()), pix, kind))
        for (y, x) in pix:
            node_of[y, x] = nid

    visited = np.zeros((H, W), dtype=bool)
    done_direct: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    next_eid = 0

    def neighbours(y: int, x: int) -> list[tuple[int, int]]:
        return [(y + dy, x + dx) for k, (dy, dx) in enumerate(OFFSETS) if nb[k, y, x]]

    def step(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def make_edge(u: int, v: int, start: tuple[int, int], path: list[tuple[int, int]],
                  end: tuple[int, int]) -> None:
        nonlocal next_eid
        seq = [start] + path + [end]
        length_px = sum(step(seq[i], seq[i + 1]) for i in range(len(seq) - 1))
        wpix = path if path else [start, end]
        mean_edt = float(np.mean([edt[p] for p in wpix]))
        g.add_edge(Edge(next_eid, u, v, length_px * scale, 2.0 * mean_edt * scale,
                        list(path), len(wpix), [next_eid]))
        next_eid += 1

    def walk(start: tuple[int, int], first: tuple[int, int]) -> tuple[list, tuple[int, int]]:
        path = [first]
        visited[first] = True
        prev, cur = start, first
        while True:
            nxt = [q for q in neighbours(*cur) if q != prev]
            if not nxt:  # cannot happen for degree-2 interior pixels
                raise RuntimeError(f"dead end while tracing at {cur}")
            # prefer a node pixel if reachable (closes the edge)
            node_nxt = [q for q in nxt if node_of[q] >= 0]
            q = node_nxt[0] if node_nxt else nxt[0]
            if node_of[q] >= 0:
                return path, q
            if visited[q]:
                raise RuntimeError(f"revisited interior pixel {q}")
            visited[q] = True
            path.append(q)
            prev, cur = cur, q

    for nid in range(len(groups)):
        for p in g.nodes[nid].pixels:
            for q in neighbours(*p):
                if node_of[q] == nid:
                    continue
                if node_of[q] >= 0:
                    key = (min(p, q), max(p, q))
                    if key in done_direct:
                        continue
                    done_direct.add(key)
                    make_edge(nid, int(node_of[q]), p, [], q)
                    continue
                if visited[q]:
                    continue
                path, end = walk(p, q)
                make_edge(nid, int(node_of[end]), p, path, end)

    # isolated loops: remaining unvisited skeleton pixels are all degree 2
    rest = s & ~visited & (node_of < 0)
    ys, xs = np.nonzero(rest)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x] or node_of[y, x] >= 0:
            continue
        nid = len(g.nodes)
        g.add_node(Node(nid, float(y), float(x), [(y, x)], "loop_anchor"))
        node_of[y, x] = nid
        nbrs = neighbours(y, x)
        if len(nbrs) != 2:
            raise RuntimeError(f"loop anchor at {(y, x)} has degree {len(nbrs)}")
        path, end = walk((y, x), nbrs[0])
        make_edge(nid, int(node_of[end]), (y, x), path, end)
    n_absorbed = absorb_spurious_self_loops(g)
    g.meta["n_spurious_self_loops_absorbed"] = n_absorbed
    return g


def encloses_background(pixels: list[tuple[int, int]]) -> bool:
    """True if the pixel set encloses at least one background pixel.

    Holes are background regions not 4-connected to the outside, which is the
    dual of 8/m-connected foreground.
    """
    ys = np.array([p[0] for p in pixels])
    xs = np.array([p[1] for p in pixels])
    m = np.zeros((ys.max() - ys.min() + 3, xs.max() - xs.min() + 3), dtype=bool)
    m[ys - ys.min() + 1, xs - xs.min() + 1] = True
    return int(ndi.binary_fill_holes(m).sum()) > int(m.sum())


def absorb_spurious_self_loops(g: VesselGraph) -> int:
    """Merge self-loops that enclose no background pixel into their node cluster.

    A pixel lying in the corner between two diagonal pixels of one junction
    cluster has m-degree 2 and would otherwise be traced as a 1-pixel self-loop.
    It is not a vessel loop (resume_review.md A4). A genuine loop must
    enclose at least one background pixel.
    """
    n = 0
    for eid in sorted(g.edges):
        e = g.edges[eid]
        if e.u != e.v:
            continue
        node = g.nodes[e.u]
        pix = list(node.pixels) + list(e.path)
        if encloses_background(pix):
            continue
        node.pixels = sorted(set(pix))
        node.y = float(np.mean([p[0] for p in node.pixels]))
        node.x = float(np.mean([p[1] for p in node.pixels]))
        del g.edges[eid]
        n += 1
    return n


def graph_from_mask(mask: np.ndarray, pixel_size_um: float | None, force_2x2: bool = True,
                    meta: dict | None = None) -> tuple[VesselGraph, np.ndarray]:
    skel = skeletonize_mask(mask)
    g = trace_graph(skel, pixel_size_um, vessel_mask=mask, force_2x2=force_2x2, meta=meta)
    return g, skel


def skeleton_length(skel: np.ndarray, pixel_size_um: float | None = None) -> float:
    """Total m-adjacency step length of a skeleton (each adjacency counted once)."""
    nb = m_adjacency(skel)
    tot = 0.0
    for k, (dy, dx) in enumerate(OFFSETS):
        w = math.sqrt(2.0) if dy and dx else 1.0
        tot += w * nb[k].sum()
    return 0.5 * tot * (pixel_size_um if pixel_size_um is not None else 1.0)
