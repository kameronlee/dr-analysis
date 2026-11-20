"""Exact length-weighted edge betweenness on undirected multigraphs.

Definition (path multiplicity): shortest paths are sequences of *edges*;
two parallel edges of equal length yield two distinct shortest paths. Each
unordered node pair (s, t), s != t, contributes (#shortest s-t paths through e)
/ (#shortest s-t paths) to edge e. Self-loops never lie on shortest paths.

igraph implements exactly this and is the production backend. NetworkX's
MultiGraph ``edge_betweenness_centrality`` collapses parallel edges and is a
*different* definition; it agrees with ours on simple graphs only. The
pure-Python ``reference_edge_betweenness`` is validated against NetworkX on
simple graphs and against brute-force path enumeration on multigraphs.
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np

REL_TOL = 1e-12


def _active_arrays(eu, ev, length, active):
    idx = np.flatnonzero(active) if active is not None else np.arange(len(eu))
    return idx, np.asarray(eu)[idx], np.asarray(ev)[idx], np.asarray(length, float)[idx]


def igraph_edge_betweenness(n: int, eu, ev, length, active=None) -> np.ndarray:
    """Return betweenness for all edges (0 for inactive ones)."""
    import igraph as ig

    m = len(eu)
    out = np.zeros(m, dtype=float)
    idx, u, v, w = _active_arrays(eu, ev, length, active)
    if len(idx) == 0:
        return out
    if np.any(w <= 0):
        raise ValueError("edge lengths must be strictly positive")
    g = ig.Graph(n=int(n), edges=list(zip(u.tolist(), v.tolist())), directed=False)
    b = np.asarray(g.edge_betweenness(directed=False, weights=w.tolist()), dtype=float)
    out[idx] = b
    return out


def _dijkstra_sigma(n, adj, s):
    dist = np.full(n, np.inf)
    sigma = np.zeros(n)
    preds: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    dist[s] = 0.0
    sigma[s] = 1.0
    order = []
    done = np.zeros(n, dtype=bool)
    heap = [(0.0, s)]
    while heap:
        _, v = heapq.heappop(heap)
        if done[v]:
            continue  # stale heap entry
        done[v] = True
        order.append(v)
        for (w, e, le) in adj[v]:
            if done[w]:
                continue
            nd = dist[v] + le
            tol = REL_TOL * max(nd, dist[w] if np.isfinite(dist[w]) else 0.0, 1.0)
            if nd < dist[w] - tol:
                dist[w] = nd
                sigma[w] = sigma[v]
                preds[w] = [(v, e)]
                heapq.heappush(heap, (nd, w))
            elif abs(nd - dist[w]) <= tol:
                sigma[w] += sigma[v]
                preds[w].append((v, e))
    return dist, sigma, preds, order


def _adjacency(n, eu, ev, length, idx):
    adj: list[list[tuple[int, int, float]]] = [[] for _ in range(n)]
    for e, a, b, le in zip(idx.tolist(), eu.tolist(), ev.tolist(), length.tolist()):
        if a == b:
            continue
        adj[a].append((b, e, le))
        adj[b].append((a, e, le))
    return adj


def reference_edge_betweenness(n: int, eu, ev, length, active=None) -> np.ndarray:
    """Pure-Python Brandes (weighted, multigraph-aware, path multiplicity)."""
    m = len(eu)
    out = np.zeros(m, dtype=float)
    idx, u, v, w = _active_arrays(eu, ev, length, active)
    if len(idx) == 0:
        return out
    if np.any(w <= 0):
        raise ValueError("edge lengths must be strictly positive")
    adj = _adjacency(n, u, v, w, idx)
    for s in range(n):
        if not adj[s]:
            continue
        dist, sigma, preds, order = _dijkstra_sigma(n, adj, s)
        delta = np.zeros(n)
        for t in reversed(order):
            for (p, e) in preds[t]:
                c = sigma[p] / sigma[t] * (1.0 + delta[t])
                out[e] += c
                delta[p] += c
    return out / 2.0  # each unordered pair counted from both ends


def brute_force_edge_betweenness(n: int, eu, ev, length) -> np.ndarray:
    """Enumerate all shortest edge-paths for every unordered pair (tiny graphs only)."""
    m = len(eu)
    idx = np.arange(m)
    adj = _adjacency(n, np.asarray(eu), np.asarray(ev), np.asarray(length, float), idx)
    out = np.zeros(m)
    for s, t in itertools.combinations(range(n), 2):
        dist, _, preds, _ = _dijkstra_sigma(n, adj, s)
        if not np.isfinite(dist[t]):
            continue
        paths: list[list[int]] = []

        def back(node, acc):
            if node == s:
                paths.append(acc)
                return
            for (p, e) in preds[node]:
                back(p, acc + [e])

        back(t, [])
        for pth in paths:
            for e in pth:
                out[e] += 1.0 / len(paths)
    return out


def edge_betweenness(n: int, eu, ev, length, active=None, backend: str = "igraph") -> np.ndarray:
    if backend == "igraph":
        return igraph_edge_betweenness(n, eu, ev, length, active)
    if backend == "reference":
        return reference_edge_betweenness(n, eu, ev, length, active)
    raise ValueError(f"unknown betweenness backend {backend}")
