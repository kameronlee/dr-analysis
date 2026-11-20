"""Serialisable undirected vessel multigraph.

Nodes carry the pixel cluster they were built from; edges carry an ordered
pixel path, physical length and width. Parallel edges and self-loops are
first-class citizens (edges are keyed by id, not by endpoint pair).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

Pixel = tuple[int, int]


@dataclass
class Node:
    id: int
    y: float
    x: float
    pixels: list[Pixel] = field(default_factory=list)
    kind: str = "junction"  # junction | endpoint | isolated | loop_anchor | synthetic


@dataclass
class Edge:
    id: int
    u: int
    v: int
    length: float
    width: float
    path: list[Pixel] = field(default_factory=list)  # interior pixels, ordered u -> v
    n_width_px: int = 1
    source_ids: list[int] = field(default_factory=list)

    @property
    def is_self_loop(self) -> bool:
        return self.u == self.v


class VesselGraph:
    """Undirected multigraph with pixel geometry.

    ``units`` is "um" when ``pixel_size_um`` is known, otherwise "px".
    ``length``/``width`` on edges are in ``units``.
    """

    def __init__(self, shape: tuple[int, int] | None = None, pixel_size_um: float | None = None,
                 meta: dict | None = None):
        self.shape = tuple(shape) if shape is not None else None
        self.pixel_size_um = pixel_size_um
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, Edge] = {}
        self.meta: dict = dict(meta or {})

    # ------------------------------------------------------------------ basics
    @property
    def units(self) -> str:
        return "um" if self.pixel_size_um is not None else "px"

    @property
    def scale(self) -> float:
        """Physical length of one pixel step in ``units``."""
        return float(self.pixel_size_um) if self.pixel_size_um is not None else 1.0

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id {node.id}")
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.id in self.edges:
            raise ValueError(f"duplicate edge id {edge.id}")
        if edge.u not in self.nodes or edge.v not in self.nodes:
            raise ValueError(f"edge {edge.id} references unknown node")
        if not (edge.length > 0 and np.isfinite(edge.length)):
            raise ValueError(f"edge {edge.id} must have positive finite length, got {edge.length}")
        if not edge.source_ids:
            edge.source_ids = [edge.id]
        self.edges[edge.id] = edge

    def copy(self) -> VesselGraph:
        return copy.deepcopy(self)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def total_length(self) -> float:
        return float(sum(e.length for e in self.edges.values()))

    def incidence(self) -> dict[int, list[int]]:
        """node -> list of incident edge ids (a self-loop appears twice)."""
        inc: dict[int, list[int]] = {n: [] for n in self.nodes}
        for eid in sorted(self.edges):
            e = self.edges[eid]
            inc[e.u].append(eid)
            inc[e.v].append(eid)
        return inc

    def degrees(self) -> dict[int, int]:
        return {n: len(v) for n, v in self.incidence().items()}

    def components(self) -> list[tuple[list[int], list[int]]]:
        """Connected components as (node ids, edge ids); isolated nodes included."""
        parent = {n: n for n in self.nodes}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for e in self.edges.values():
            ra, rb = find(e.u), find(e.v)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb
        comp_nodes: dict[int, list[int]] = {}
        for n in sorted(self.nodes):
            comp_nodes.setdefault(find(n), []).append(n)
        comp_edges: dict[int, list[int]] = {r: [] for r in comp_nodes}
        for eid in sorted(self.edges):
            comp_edges[find(self.edges[eid].u)].append(eid)
        return [(comp_nodes[r], comp_edges[r]) for r in sorted(comp_nodes)]

    def stats(self, area_mm2: float | None = None) -> dict:
        deg = self.degrees()
        comps = self.components()
        e, v, c = self.n_edges, self.n_nodes, len(comps)
        seen_pairs: dict[tuple[int, int], int] = {}
        n_self = 0
        for ed in self.edges.values():
            if ed.u == ed.v:
                n_self += 1
            else:
                key = (min(ed.u, ed.v), max(ed.u, ed.v))
                seen_pairs[key] = seen_pairs.get(key, 0) + 1
        n_parallel_extra = int(sum(k - 1 for k in seen_pairs.values() if k > 1))
        out = {
            "n_nodes": v,
            "n_edges": e,
            "n_components": c,
            "total_length": self.total_length(),
            "units": self.units,
            "cyclomatic_number": e - v + c,
            "n_self_loops": n_self,
            "n_parallel_extra_edges": n_parallel_extra,
            "n_junctions": int(sum(1 for d in deg.values() if d >= 3)),
            "n_endpoints": int(sum(1 for d in deg.values() if d == 1)),
            "n_isolated_nodes": int(sum(1 for d in deg.values() if d == 0)),
        }
        if area_mm2 is not None and area_mm2 > 0:
            out["junction_density_per_mm2"] = out["n_junctions"] / area_mm2
            out["end_density_per_mm2"] = out["n_endpoints"] / area_mm2
            out["loop_density_per_mm2"] = out["cyclomatic_number"] / area_mm2
            if self.units == "um":
                out["length_density_mm_per_mm2"] = out["total_length"] / 1000.0 / area_mm2
        return out

    # ------------------------------------------------------------- conversion
    def compact(self) -> dict:
        """Array form used by stress tests. Order is by ascending edge id."""
        node_ids = sorted(self.nodes)
        idx = {n: i for i, n in enumerate(node_ids)}
        edge_ids = sorted(self.edges)
        eu = np.array([idx[self.edges[e].u] for e in edge_ids], dtype=np.int64)
        ev = np.array([idx[self.edges[e].v] for e in edge_ids], dtype=np.int64)
        length = np.array([self.edges[e].length for e in edge_ids], dtype=float)
        width = np.array([self.edges[e].width for e in edge_ids], dtype=float)
        return {"n": len(node_ids), "eu": eu, "ev": ev, "length": length, "width": width,
                "edge_ids": np.array(edge_ids, dtype=np.int64),
                "node_ids": np.array(node_ids, dtype=np.int64)}

    def to_networkx(self):
        import networkx as nx

        G = nx.MultiGraph()
        for n in sorted(self.nodes):
            G.add_node(n, y=self.nodes[n].y, x=self.nodes[n].x)
        for eid in sorted(self.edges):
            e = self.edges[eid]
            G.add_edge(e.u, e.v, key=eid, length=e.length, width=e.width)
        return G

    def to_igraph(self):
        import igraph as ig

        c = self.compact()
        g = ig.Graph(n=c["n"], edges=list(zip(c["eu"].tolist(), c["ev"].tolist())))
        g.es["length"] = c["length"].tolist()
        g.es["edge_id"] = c["edge_ids"].tolist()
        return g

    def subgraph_without(self, remove_edge_ids: Iterable[int]) -> VesselGraph:
        """New graph with the given edges removed (nodes are kept)."""
        rm = set(int(e) for e in remove_edge_ids)
        g = VesselGraph(self.shape, self.pixel_size_um, self.meta)
        for n in sorted(self.nodes):
            g.add_node(copy.deepcopy(self.nodes[n]))
        for eid in sorted(self.edges):
            if eid not in rm:
                g.add_edge(copy.deepcopy(self.edges[eid]))
        return g

    def drop_isolated_nodes(self) -> int:
        deg = self.degrees()
        iso = [n for n, d in deg.items() if d == 0]
        for n in iso:
            del self.nodes[n]
        return len(iso)

    def edge_pixels(self, eid: int, include_nodes: bool = True) -> list[Pixel]:
        e = self.edges[eid]
        px = list(e.path)
        if include_nodes:
            px += list(self.nodes[e.u].pixels)
            if e.v != e.u:
                px += list(self.nodes[e.v].pixels)
        return px

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "format": "retina_fragility.VesselGraph/1",
            "shape": list(self.shape) if self.shape else None,
            "pixel_size_um": self.pixel_size_um,
            "units": self.units,
            "meta": self.meta,
            "nodes": [
                {"id": n.id, "y": n.y, "x": n.x, "kind": n.kind,
                 "pixels": [list(p) for p in n.pixels]}
                for n in (self.nodes[k] for k in sorted(self.nodes))
            ],
            "edges": [
                {"id": e.id, "u": e.u, "v": e.v, "length": e.length, "width": e.width,
                 "n_width_px": e.n_width_px, "source_ids": list(e.source_ids),
                 "path": [list(p) for p in e.path]}
                for e in (self.edges[k] for k in sorted(self.edges))
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> VesselGraph:
        g = cls(tuple(d["shape"]) if d.get("shape") else None, d.get("pixel_size_um"),
                d.get("meta") or {})
        for n in d["nodes"]:
            g.add_node(Node(int(n["id"]), float(n["y"]), float(n["x"]),
                            [tuple(p) for p in n["pixels"]], n.get("kind", "junction")))
        for e in d["edges"]:
            g.add_edge(Edge(int(e["id"]), int(e["u"]), int(e["v"]), float(e["length"]),
                            float(e["width"]), [tuple(p) for p in e["path"]],
                            int(e.get("n_width_px", 1)), [int(s) for s in e.get("source_ids", [])]))
        return g

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, separators=(",", ":"))

    @classmethod
    def load(cls, path: str | Path) -> VesselGraph:
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def renumber(self) -> VesselGraph:
        """Return a copy with node and edge ids renumbered 0..n-1 in current id order."""
        g = VesselGraph(self.shape, self.pixel_size_um, self.meta)
        nmap = {old: new for new, old in enumerate(sorted(self.nodes))}
        for old in sorted(self.nodes):
            n = copy.deepcopy(self.nodes[old])
            n.id = nmap[old]
            g.add_node(n)
        for new, old in enumerate(sorted(self.edges)):
            e = copy.deepcopy(self.edges[old])
            e.id = new
            e.u, e.v = nmap[e.u], nmap[e.v]
            g.add_edge(e)
        return g


def graph_from_edge_list(n_nodes: int, edges: list[tuple[int, int, float]],
                         widths: list[float] | None = None) -> VesselGraph:
    """Abstract graph (no pixel geometry) for synthetic experiments and tests."""
    g = VesselGraph(shape=None, pixel_size_um=None, meta={"synthetic": True})
    for i in range(n_nodes):
        g.add_node(Node(i, 0.0, 0.0, [], "synthetic"))
    for k, (u, v, length) in enumerate(edges):
        w = 1.0 if widths is None else float(widths[k])
        g.add_edge(Edge(k, int(u), int(v), float(length), w, [], 1, [k]))
    return g
