"""Per-image pipeline, biomarkers, vessel-territory coverage and physical scaling.

Synthetic mask fixtures only; these checks validate software, not biology.
"""
import json

import numpy as np
import pandas as pd
import pytest

from retina_fragility.biomarkers import box_count_dimension, compute_biomarkers, faz_metrics
from retina_fragility.config import load_config
from retina_fragility.graph.cleanup import clean_graph
from retina_fragility.graph.coverage import (
    CoverageTracker,
    centerline_pixel_sets,
    vessel_territories,
)
from retina_fragility.graph.extract import graph_from_mask
from retina_fragility.graph.model import graph_from_edge_list
from retina_fragility.graph.stress import stress_suite
from retina_fragility.io_masks import faz_disk_mask, make_analysis_mask
from retina_fragility.pipeline import crop_center, process_masks, run_manifest
from retina_fragility.seeds import rng_for
from retina_fragility.synthetic import (
    grid_mask,
    make_fixture_dataset,
    random_network_mask,
    rescale_mask,
)

CFG = load_config(overrides={"stress": {"random_repetitions": 5}, "pipeline": {"overlays": False}})


def network(seed=0, shape=(200, 200)):
    return random_network_mask(shape, n_seeds=60, k=3, radius=1, seed=seed, faz_radius=15)


def test_process_masks_outputs_features_and_is_deterministic(tmp_path):
    v, f = network(1)
    a = process_masks("img", v, f, 10.0, CFG, 40.0, synthetic=True, out_dir=tmp_path / "a")
    b = process_masks("img", v, f, 10.0, CFG, 40.0, synthetic=True, out_dir=tmp_path / "b")
    assert a["status"] == "ok" and a["data_provenance"] == "SYNTHETIC - NOT CLINICAL"
    fa, fb = a["features"], b["features"]
    for k in ("FI_adaptive_betweenness", "FI_random", "G10_adaptive_betweenness", "P0",
              "FIperf_adaptive_betweenness", "loop_density_per_mm2", "vessel_density"):
        assert np.isfinite(fa[k]) and fa[k] == fb[k], k
    for name in ("graph.json", "curves.csv", "features.json", "fragility_map.csv", "cleanup_audit.json",
                 "random_reps.csv"):
        assert (tmp_path / "a" / name).exists()
    feats = json.loads((tmp_path / "a" / "features.json").read_text())
    assert feats["data_provenance"] == "SYNTHETIC - NOT CLINICAL" and "not clipped" in feats["fi_caveats"]
    fmap = pd.read_csv(tmp_path / "a" / "fragility_map.csv")
    assert {"cutoff_length", "removed_length", "unique_coverage_mm2", "removal_q_adaptive_betweenness",
            "initial_betweenness"} <= set(fmap.columns)
    # a different image id gives different random removal orders (seeds are per image)
    c = process_masks("other", v, f, 10.0, CFG, 40.0, synthetic=True)
    assert c["features"]["FI_random"] != fa["FI_random"]
    assert c["features"]["FI_adaptive_betweenness"] == fa["FI_adaptive_betweenness"]


def test_empty_graph_is_excluded_without_fi(tmp_path):
    v = np.zeros((80, 80), bool)
    v[40, 40:43] = True  # 3 px (30 um) fragment < 50 um -> removed
    r = process_masks("tiny", v, None, 10.0, CFG, 40.0, synthetic=True, out_dir=tmp_path)
    assert r["status"] == "excluded" and r["reason"] == "empty_graph"
    assert not any(k.startswith("FI") for k in r["features"])
    assert json.loads((tmp_path / "features.json").read_text())["status"] == "excluded"


def test_missing_d_disables_coverage_explicitly():
    v, f = network(2)
    r = process_masks("x", v, f, 10.0, CFG, None, synthetic=True)
    assert r["features"]["coverage_note"] == "no_frozen_calibration_d"
    assert "FIperf_adaptive_betweenness" not in r["features"]


def test_physical_lengths_and_densities_are_resolution_invariant():
    v = grid_mask(n_lines=4, spacing=24, margin=8, radius=1)
    g1, sk1 = graph_from_mask(v, 10.0)
    v2 = rescale_mask(v, 2)
    g2, sk2 = graph_from_mask(v2, 5.0)
    c1, _ = clean_graph(g1, CFG["graph"])
    c2, _ = clean_graph(g2, CFG["graph"])
    assert c1.total_length() == pytest.approx(c2.total_length(), rel=0.05)
    assert c1.stats()["cyclomatic_number"] == c2.stats()["cyclomatic_number"] == 9
    am1 = make_analysis_mask(v.shape, 10.0, None, {"border_band_um": 50, "exclude_faz": False})
    am2 = make_analysis_mask(v2.shape, 5.0, None, {"border_band_um": 50, "exclude_faz": False})
    b1 = compute_biomarkers(v, None, am1, sk1, g1, c1, 10.0)
    b2 = compute_biomarkers(v2, None, am2, sk2, g2, c2, 5.0)
    for k in ("analysis_area_mm2", "graph_length_density_mm_per_mm2", "loop_density_per_mm2",
              "skeleton_length_density_mm_per_mm2"):
        assert b1[k] == pytest.approx(b2[k], rel=0.06), k
    # the 50-um fragment rule is physical: a 40-um fragment is removed at both resolutions
    frag = np.zeros((60, 60), bool)
    frag[30, 10:14] = True  # 4 px * 10 um = 30-40 um
    for m, px in ((frag, 10.0), (rescale_mask(frag, 2), 5.0)):
        g, _ = graph_from_mask(m, px)
        assert clean_graph(g, CFG["graph"])[0].n_edges == 0


def test_vessel_territory_matches_calibration_geometry_for_thick_vessels():
    v = np.zeros((60, 60), bool)
    v[20:31, 5:55] = True  # 11-px-wide vessel
    g, sk = graph_from_mask(v, 10.0)
    g, _ = clean_graph(g, {**CFG["graph"], "border_band_px": 0})
    am = np.ones_like(v)
    terr = vessel_territories(g, sk, v)
    cent = centerline_pixel_sets(g)
    tot = np.unique(np.concatenate(list(terr.values())))
    assert v.ravel()[tot].all()  # territories only contain vessel pixels
    assert len(tot) > 0.9 * v.sum()
    # tissue 1 px (10 um) outside the vessel wall is covered with d = 10 um by territory
    cov_t = CoverageTracker(g, am, d_um=10.0, edge_pixel_sets=terr).covered_mask()
    cov_c = CoverageTracker(g, am, d_um=10.0, edge_pixel_sets=cent).covered_mask()
    assert cov_t[19, 30] and cov_t[31, 30]
    assert not cov_c[19, 30]  # centreline geometry would leave it uncovered


def test_territory_excludes_pixels_of_pruned_fragments():
    v = np.zeros((60, 60), bool)
    v[10, 5:55] = True
    v[40, 30:33] = True  # 3-px fragment removed by the 50-um rule
    g, sk = graph_from_mask(v, 10.0)
    c, _ = clean_graph(g, {**CFG["graph"], "border_band_px": 0})
    terr = vessel_territories(c, sk, v)
    allpix = np.concatenate(list(terr.values()))
    assert not np.isin(40 * 60 + 31, allpix)


def test_random_trap_len_is_mean_of_repetitions_not_mean_curve():
    g = graph_from_edge_list(6, [(0, 1, 10), (1, 2, 1), (2, 3, 5), (3, 4, 1), (4, 5, 20), (5, 0, 2)])
    out = stress_suite(g, ["random"], 0.2, 7, lambda r: rng_for(1, "t", r))
    reps = pd.DataFrame(out["random_reps"])
    assert out["features"]["FI_trap_len_random"] == pytest.approx(reps.FI_trap_len.mean())
    assert out["features"]["FI_random"] == pytest.approx(reps.FI.mean())
    assert out["features"]["FI_trap_random"] == pytest.approx(reps.FI_trap.mean())


def test_biomarker_basics():
    faz = faz_disk_mask((201, 201), 40)
    m = faz_metrics(faz, 10.0)
    assert m["faz_area_mm2"] == pytest.approx(np.pi * 0.4 ** 2, rel=0.02)
    assert 0.85 < m["faz_circularity"] < 1.15
    assert np.isnan(faz_metrics(np.zeros((10, 10), bool), 10.0)["faz_circularity"])
    line = np.zeros((256, 256), bool)
    line[128, :] = True
    assert box_count_dimension(line)["fractal_dimension"] == pytest.approx(1.0, abs=0.02)
    assert box_count_dimension(np.ones((256, 256), bool))["fractal_dimension"] == pytest.approx(2.0, abs=0.02)


def test_run_manifest_logs_input_errors_reuses_outputs_and_crops(tmp_path):
    f = make_fixture_dataset(tmp_path / "fx", n_normal_per_fov=1, n_dr_per_fov=1, seed=5, n_seeds=40)
    from retina_fragility.manifest import load_manifest

    m = load_manifest(tmp_path / "fx" / "manifest.csv", CFG)
    m.loc[m.index[0], "vessel_mask_path"] = str(tmp_path / "missing.png")
    run = run_manifest(m, CFG, tmp_path, tmp_path / "out", 40.0)
    ex = pd.read_csv(tmp_path / "out" / "exclusions.csv")
    assert run["n_excluded"] >= 1 and "input_error" in set(ex.reason)
    assert run["data_provenance"] == "SYNTHETIC - NOT CLINICAL"
    feats1 = pd.read_csv(tmp_path / "out" / "features.csv")
    run_manifest(m, CFG, tmp_path, tmp_path / "out", 40.0)
    feats2 = pd.read_csv(tmp_path / "out" / "features.csv")
    pd.testing.assert_frame_equal(feats1, feats2)
    with pytest.raises(PermissionError):
        run_manifest(m, CFG, tmp_path, tmp_path / "data" / "raw" / "x", 40.0)
    masks = {"vessel": np.ones((400, 400), bool), "faz": np.zeros((400, 400), bool), "pixel_size_um": 15.0}
    out = crop_center(masks, 3.0)
    assert out["vessel"].shape == (200, 200) and out["crop"]["applied"]
    assert f is not None
