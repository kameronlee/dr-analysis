"""Independent checks of the scientific invariants in the  review.

Fixtures are synthetic. Passing these tests is not clinical validation.
"""
import itertools

import networkx as nx
import numpy as np
import pytest

from retina_fragility.config import load_config
from retina_fragility.graph.betweenness import edge_betweenness
from retina_fragility.graph.cleanup import clean_graph
from retina_fragility.graph.coverage import CoverageTracker, empty_coverage_fraction
from retina_fragility.graph.extract import trace_graph
from retina_fragility.graph.fragility_map import fragility_map
from retina_fragility.graph.model import Edge, Node, VesselGraph, graph_from_edge_list
from retina_fragility.graph.stress import EmptyGraphError, run_removal, summarize_curve
from retina_fragility.synthetic import grid_graph


def test_straight_skeleton_has_physical_length_and_two_endpoints():
    mask = np.zeros((12, 20), bool)
    mask[5, 3:17] = True
    graph = trace_graph(mask, pixel_size_um=10, vessel_mask=mask)
    assert graph.n_nodes == 2 and graph.n_edges == 1
    assert graph.total_length() == pytest.approx(130)
    assert list(graph.edges.values())[0].width == pytest.approx(20)
    assert graph.stats()["cyclomatic_number"] == 0


def test_isolated_ring_is_preserved_as_one_cycle():
    mask = np.zeros((13, 13), bool)
    mask[3, 3:10] = mask[9, 3:10] = True
    mask[3:10, 3] = mask[3:10, 9] = True
    graph = trace_graph(mask, pixel_size_um=5)
    assert graph.n_edges == 1 and graph.n_nodes == 1
    assert graph.stats()["cyclomatic_number"] == 1
    assert graph.total_length() == pytest.approx(120)


def test_diagonal_staircase_does_not_create_triangle():
    mask = np.zeros((8, 8), bool)
    mask[2, 2] = mask[2, 3] = mask[3, 3] = True
    graph = trace_graph(mask, pixel_size_um=1)
    assert graph.stats()["cyclomatic_number"] == 0
    assert graph.n_edges == 1
    assert graph.total_length() == pytest.approx(2)


def test_multigraph_roundtrip_keeps_parallel_edges_and_self_loops(tmp_path):
    graph = graph_from_edge_list(3, [(0, 1, 1), (0, 1, 1), (1, 2, 2), (1, 1, 3)])
    graph.save(tmp_path / "graph.json")
    restored = VesselGraph.load(tmp_path / "graph.json")
    assert restored.to_dict() == graph.to_dict()
    assert restored.stats()["cyclomatic_number"] == 2
    assert restored.stats()["n_parallel_extra_edges"] == 1
    assert restored.stats()["n_self_loops"] == 1


@pytest.mark.parametrize("backend", ["igraph", "reference"])
def test_simple_graph_betweenness_matches_networkx(backend):
    graph = graph_from_edge_list(5, [(0, 1, 1), (1, 2, 2), (2, 3, 1), (0, 3, 4), (3, 4, 1)])
    c = graph.compact()
    actual = edge_betweenness(c["n"], c["eu"], c["ev"], c["length"], backend=backend)
    simple = nx.Graph()
    simple.add_nodes_from(range(c["n"]))
    for u, v, length in zip(c["eu"], c["ev"], c["length"]):
        simple.add_edge(u, v, length=length)
    expected = nx.edge_betweenness_centrality(simple, weight="length", normalized=False)
    assert actual == pytest.approx([expected.get((u, v), expected.get((v, u))) for u, v in zip(c["eu"], c["ev"])])


@pytest.mark.parametrize("backend", ["igraph", "reference"])
def test_parallel_betweenness_against_independent_edge_path_enumeration(backend):
    graph = graph_from_edge_list(4, [(0, 1, 1), (0, 1, 1), (1, 2, 1), (0, 2, 2), (2, 3, 1), (1, 1, 1)])
    # Enumerate simple edge paths with NetworkX; do not reuse the implementation's Dijkstra.
    network = graph.to_networkx()
    expected = np.zeros(graph.n_edges)
    for u, v in itertools.combinations(network.nodes, 2):
        paths = list(nx.all_simple_edge_paths(network, u, v))
        lengths = [sum(network.edges[a, b, k]["length"] for a, b, k in p) for p in paths]
        if not paths:
            continue
        shortest = [p for p, length in zip(paths, lengths) if length == min(lengths)]
        for path in shortest:
            for _, _, key in path:
                expected[key] += 1 / len(shortest)
    c = graph.compact()
    actual = edge_betweenness(c["n"], c["eu"], c["ev"], c["length"], backend=backend)
    assert actual == pytest.approx(expected)


def test_largest_component_selected_by_length_not_node_count():
    graph = graph_from_edge_list(8, [(0, 1, 100)] + [(i, i + 1, 1) for i in range(2, 7)])
    curve = run_removal(graph, "width")
    assert curve.S[0] == pytest.approx(100 / 105)


def test_unequal_length_index_is_not_clipped():
    graph = graph_from_edge_list(6, [(0, 1, 100)] + [(0, i, 1) for i in range(2, 6)], [1, 5, 4, 3, 2])
    curve = run_removal(graph, "width")
    assert curve.S == pytest.approx(np.array([104, 103, 102, 101, 100, 0]) / 104)
    assert summarize_curve(curve)["FI"] == pytest.approx(1 - 2 * 406 / 520)
    assert summarize_curve(curve)["FI"] < 0


def test_partial_final_batch_and_real_q_interpolation():
    graph = graph_from_edge_list(8, [(i, i + 1, 1) for i in range(7)])
    curve = run_removal(graph, "width", step_frac=0.4)
    assert curve.q_edge == pytest.approx([0, 3 / 7, 6 / 7, 1])
    summary = summarize_curve(curve)
    assert summary["S_q10"] == pytest.approx(np.interp(0.1, curve.q_edge, curve.S))
    assert summary["R_mean"] != pytest.approx(summary["R_right"])


@pytest.mark.parametrize("order", ["random", "adaptive_betweenness", "static_betweenness", "width", "forman"])
def test_removal_is_monotone_reproducible_and_does_not_mutate(order):
    graph = grid_graph(5)
    original = graph.to_dict()
    a = run_removal(graph, order, rng=np.random.default_rng(719))
    b = run_removal(graph, order, rng=np.random.default_rng(719))
    assert np.array_equal(a.S, b.S)
    assert a.removal_batches == b.removal_batches
    assert np.all(np.diff(a.S) <= 1e-12)
    assert a.S[-1] == 0 and a.q_edge[-1] == 1
    assert graph.to_dict() == original
    removed = []
    cycles = graph.stats()["cyclomatic_number"]
    for batch in a.removal_batches:
        removed.extend(batch)
        next_cycles = graph.subgraph_without(removed).stats()["cyclomatic_number"]
        assert next_cycles <= cycles
        cycles = next_cycles


def test_empty_graph_is_not_assigned_a_fake_index():
    with pytest.raises(EmptyGraphError):
        run_removal(graph_from_edge_list(0, []), "width")


def test_bridge_cutoff_excludes_the_removed_vessel():
    graph = graph_from_edge_list(4, [(0, 1, 2), (1, 2, 3), (2, 3, 5)])
    scores = fragility_map(graph).set_index("edge_id")
    assert scores.loc[1, "cutoff_length"] == 2
    assert scores.loc[1, "removed_length"] == 3
    assert scores.loc[1, "delta_S"] == pytest.approx(0.5)
    assert scores.loc[0, "cutoff_length"] == 0


def test_parallel_edges_and_self_loops_are_not_bridges():
    graph = graph_from_edge_list(2, [(0, 1, 1), (0, 1, 2), (0, 0, 3)])
    scores = fragility_map(graph)
    assert not scores.is_bridge.any()
    assert (scores.cutoff_length == 0).all()


def test_coverage_preserves_shared_endpoint_and_empty_graph_is_zero():
    graph = VesselGraph(shape=(5, 5), pixel_size_um=10)
    for i, col in enumerate([1, 2, 3]):
        graph.add_node(Node(i, 2, col, [(2, col)]))
    graph.add_edge(Edge(0, 0, 1, 10, 10))
    graph.add_edge(Edge(1, 1, 2, 10, 10))
    area = np.ones((5, 5), bool)
    coverage = CoverageTracker(graph, area, d_um=0)
    assert coverage.fraction() == pytest.approx(3 / 25)
    assert coverage.unique_area_px().tolist() == [1, 1]
    coverage.remove([0])
    assert coverage.fraction() == pytest.approx(2 / 25)
    assert coverage.covered_mask()[2, 2]
    coverage.remove([1])
    assert coverage.fraction() == 0
    assert empty_coverage_fraction(area) == 0


def test_cleanup_uses_physical_fragment_threshold_and_keeps_input():
    mask = np.zeros((12, 20), bool)
    mask[5, 3:17] = True
    graph = trace_graph(mask, pixel_size_um=10, vessel_mask=mask)
    original = graph.to_dict()
    cfg = load_config()["graph"]
    cfg["min_fragment_length_um"] = 131
    cleaned, audit = clean_graph(graph, cfg)
    assert cleaned.n_edges == 0
    assert graph.to_dict() == original
    assert audit[-1]["n_edges"] == 0
