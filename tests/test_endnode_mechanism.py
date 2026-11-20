"""End-node mechanism helpers (amendment 2026-11-16). Synthetic inputs only: SYNTHETIC - NOT CLINICAL."""
import numpy as np
import pytest

import scripts.endnode_mechanism as em


def test_stage_parts_account_for_every_raw_end_node_in_the_example_log():
    log = [{"stage": s, "n_endpoints": n} for s, n in (
        ("0_raw", 1999), ("1_border", 1898), ("2_faz", 1898), ("3_contract", 1898), ("4_spurs", 604),
        ("5_contract", 604), ("6_fragments", 502), ("7_final", 502))]
    p = em.stage_parts(log)
    assert p == {"end_raw": 1999, "end_border": 101, "end_faz": 0, "end_spur": 1294, "end_fragment": 102,
                 "end_final": 502}
    assert p["end_border"] + p["end_faz"] + p["end_spur"] + p["end_fragment"] + p["end_final"] == p["end_raw"]
    with pytest.raises(em.MechError):
        em.stage_parts(log[:-1])


def test_location_split_uses_degree_and_bands():
    nodes = [{"id": i, "y": y, "x": x} for i, (y, x) in enumerate(
        [(2, 50), (50, 50), (50, 60), (30, 30), (30, 40), (50, 70), (48, 47)])]
    edges = [{"u": 0, "v": 1}, {"u": 1, "v": 2}, {"u": 1, "v": 5},   # node 1 is a junction
             {"u": 3, "v": 4},                                        # a two-node fragment: two ends
             {"u": 6, "v": 1}, {"u": 5, "v": 5}]                      # self-loop: node 5 has degree 3
    faz = np.zeros((100, 100), bool)
    faz[40:46, 40:46] = True
    out = em.location_parts(nodes, edges, (100, 100), faz)
    assert out == {"end_loc_border": 1, "end_loc_perifaz": 1, "end_loc_interior": 3}  # 0 | 6 | 2, 3, 4
    assert em.location_parts(nodes, edges, (100, 100), None)["end_loc_perifaz"] == 0
