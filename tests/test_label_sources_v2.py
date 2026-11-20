"""Calibrated classical tracers (amendment 2026-11-16). Synthetic images only: SYNTHETIC - NOT CLINICAL."""
import numpy as np

import scripts.label_sources_v2 as ls
from retina_fragility.config import load_config

CFG = load_config()


def test_first_crossing_stops_on_the_rising_branch_of_a_unimodal_curve():
    calls = []

    def f(p):  # rises to 1.6 at p = 0.5, then falls back below 1 (an over-filled mask)
        calls.append(p)
        return 1.6 - 6.4 * (p - 0.5) ** 2

    r = ls.first_crossing(f, [float(x) for x in np.linspace(0, 1, 11)])
    assert r["reached"] and abs(r["value"] - 1.0) <= ls.TOL_CAL and r["param"] < 0.5  # sparse-side crossing
    assert max(calls) <= 0.3 + 1e-9  # never walked into the over-filled side
    low = ls.first_crossing(lambda p: 0.5 * p, [0.0, 0.5, 1.0])
    assert not low["reached"] and low["param"] == 1.0 and low["value"] == 0.5
    first = ls.first_crossing(lambda p: 1.01, [0.2, 0.4])
    assert first["reached"] and first["param"] == 0.2 and len(first["evaluations"]) == 1


def test_tracers_return_masks_that_fill_in_monotonically_on_the_sparse_side():
    rng = np.random.default_rng(0)
    img = rng.normal(0.25, 0.05, (96, 96)).astype(np.float32)
    img[::12, :] += 0.5
    img[:, 40:43] += 0.5
    img = np.clip(img, 0, 1)
    a = [ls.trace(img, "adaptive", o, CFG).mean() for o in (-0.10, -0.02, 0.05)]
    assert a[0] < a[1] < a[2] and ls.trace(img, "adaptive", 0.0, CFG).dtype == bool
    f = [ls.trace(img, "frangi_capillary", s, CFG).mean() for s in (2.0, 1.0, 0.3)]
    assert f[0] <= f[1] <= f[2] and f[2] > 0
    lines = ls.trace(img, "adaptive", -0.10, CFG)
    assert lines[::12, :].mean() > 0.9 and lines[5, 5] == 0  # bright lines kept, background dropped
    assert ls._remove_small(np.eye(30, dtype=bool) & (np.arange(30) < 5)[:, None], 20).sum() == 0
