"""Skeleton-tracing artifacts that would bias loop density (review addendum A4).

Synthetic pixel patterns only.
"""
import numpy as np

from retina_fragility.graph.extract import encloses_background, graph_from_mask, trace_graph
from retina_fragility.synthetic import random_network_mask


def corner_pattern():
    m = np.zeros((9, 9), bool)
    m[1:4, 3] = True   # up arm into junction pixel (3, 3)
    m[3, 0:4] = True   # left arm
    m[3, 4] = True     # corner pixel between diagonal junction pixels (3,3) and (4,4)
    m[4, 4:9] = True   # right arm from junction pixel (4, 4)
    m[4:9, 4] = True   # down arm
    return m


def test_corner_pixel_between_diagonal_junction_pixels_is_not_a_loop():
    g = trace_graph(corner_pattern(), 1.0)
    assert g.meta["n_spurious_self_loops_absorbed"] == 1
    assert not any(e.u == e.v for e in g.edges.values())
    assert g.stats()["cyclomatic_number"] == 0 and g.n_edges == 4


def test_genuine_minimal_loops_are_kept():
    diamond = np.zeros((5, 5), bool)
    diamond[1, 2] = diamond[2, 1] = diamond[2, 3] = diamond[3, 2] = True
    g = trace_graph(diamond, 1.0)
    assert g.stats()["cyclomatic_number"] == 1 and g.meta["n_spurious_self_loops_absorbed"] == 0
    ring_with_tail = np.zeros((12, 12), bool)
    ring_with_tail[2, 2:7] = ring_with_tail[6, 2:7] = True
    ring_with_tail[2:7, 2] = ring_with_tail[2:7, 6] = True
    ring_with_tail[6:11, 4] = True
    g2 = trace_graph(ring_with_tail, 1.0)
    assert g2.stats()["cyclomatic_number"] == 1


def test_no_background_free_self_loops_survive_on_fixtures():
    for seed in range(5):
        v, _ = random_network_mask((200, 200), n_seeds=80, seed=seed)
        g, _ = graph_from_mask(v, 10.0)
        for e in g.edges.values():
            if e.u == e.v:
                assert encloses_background(list(e.path) + list(g.nodes[e.u].pixels))
