"""Per-edge fragility map quantities on the intact graph (review section 5).

* removed_length: length of the edge itself.
* cutoff_length: if the edge is a bridge of its own component, the total
  length on the smaller side after removal; otherwise 0. Parallel edges and
  self-loops are never bridges. Initially disconnected graphs are handled per
  component.
* delta_S: S0 - S(G - e), change of the global largest-component share.
"""

from __future__ import annotations

import numpy as np

from .model import VesselGraph


def bridges_and_sides(n: int, eu, ev, length):
    """Tarjan bridges keyed by edge id (multigraph-correct), with side lengths.

    Returns (is_bridge[m], child_side_length[m], comp_of_node[n], comp_total[#comp]).
    """
    eu = np.asarray(eu)
    ev = np.asarray(ev)
    length = np.asarray(length, float)
    m = len(eu)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for e in range(m):
        a, b = int(eu[e]), int(ev[e])
        if a != b:
            adj[a].append((b, e))
            adj[b].append((a, e))
    disc = -np.ones(n, dtype=np.int64)
    low = np.zeros(n, dtype=np.int64)
    parent = -np.ones(n, dtype=np.int64)
    parent_edge = -np.ones(n, dtype=np.int64)
    comp = -np.ones(n, dtype=np.int64)
    order: list[int] = []
    is_bridge = np.zeros(m, dtype=bool)
    timer = 0
    ncomp = 0
    for root in range(n):
        if disc[root] >= 0:
            continue
        disc[root] = low[root] = timer
        timer += 1
        comp[root] = ncomp
        order.append(root)
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            v, pe, it = stack[-1]
            advanced = False
            for (w, e) in it:
                if e == pe:
                    continue
                if disc[w] < 0:
                    disc[w] = low[w] = timer
                    timer += 1
                    parent[w] = v
                    parent_edge[w] = e
                    comp[w] = ncomp
                    order.append(w)
                    stack.append((w, e, iter(adj[w])))
                    advanced = True
                    break
                low[v] = min(low[v], disc[w])
            if not advanced:
                stack.pop()
                if pe >= 0:
                    p = parent[v]
                    low[p] = min(low[p], low[v])
                    if low[v] > disc[p]:
                        is_bridge[pe] = True
        ncomp += 1
    # assign every edge to its endpoint discovered later; self-loops to their node
    assigned = np.zeros(n)
    for e in range(m):
        a, b = int(eu[e]), int(ev[e])
        tgt = a if disc[a] >= disc[b] else b
        assigned[tgt] += length[e]
    sub = assigned.copy()
    for v in reversed(order):
        p = parent[v]
        if p >= 0:
            sub[p] += sub[v]
    comp_total = np.bincount(comp, weights=assigned, minlength=ncomp) if n else np.zeros(0)
    child_side = np.zeros(m)
    for v in range(n):
        e = parent_edge[v]
        if e >= 0 and is_bridge[e]:
            child_side[e] = sub[v] - length[e]
    return is_bridge, child_side, comp, comp_total


def fragility_map(graph: VesselGraph) -> pd.DataFrame:  # noqa: F821
    import pandas as pd

    c = graph.compact()
    n, eu, ev, length = c["n"], c["eu"], c["ev"], c["length"]
    m = len(length)
    L0 = float(length.sum())
    is_bridge, child_side, comp, comp_total = bridges_and_sides(n, eu, ev, length)
    order = np.argsort(-comp_total)
    top1 = comp_total[order[0]] if len(order) else 0.0
    top2 = comp_total[order[1]] if len(order) > 1 else 0.0
    rows = []
    for i in range(m):
        ci = comp[eu[i]]
        tot = comp_total[ci]
        if is_bridge[i]:
            a = child_side[i]
            b = tot - a - length[i]
            parts = [a, b]
            cutoff = min(a, b)
        else:
            parts = [tot - length[i]]
            cutoff = 0.0
        other_max = top2 if ci == order[0] else top1
        lcc_after = max([other_max] + parts)
        rows.append({
            "edge_id": int(c["edge_ids"][i]),
            "u": int(c["node_ids"][eu[i]]), "v": int(c["node_ids"][ev[i]]),
            "is_bridge": bool(is_bridge[i]),
            "is_self_loop": bool(eu[i] == ev[i]),
            "removed_length": float(length[i]),
            "cutoff_length": float(cutoff),
            "component_length": float(tot),
            "delta_S": (top1 - lcc_after) / L0 if L0 > 0 else float("nan"),
            "width": float(c["width"][i]),
        })
    df = pd.DataFrame(rows)
    df.attrs["units"] = graph.units
    return df
