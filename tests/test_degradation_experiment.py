"""Image-degradation experiment (amendment 2026-11-16). Synthetic data only: SYNTHETIC - NOT CLINICAL."""
import numpy as np
import pandas as pd
import pytest

import scripts.degradation_experiment as de
from retina_fragility.config import load_config

CFG = load_config()


def test_degradations_do_what_they_say():
    rng = np.random.default_rng(0)
    img = rng.uniform(0.2, 0.9, (40, 40)).astype(np.float32)
    assert np.array_equal(de.degrade(img, "original", 0), img)
    assert de.degrade(img, "blur", 1.5).std() < img.std()
    assert de.degrade(img, "dim", 0.4).max() == pytest.approx(0.4 * img.max())
    n = de.degrade(img, "noise", 0.15, np.random.default_rng(1))
    assert n.min() >= 0 and n.max() <= 1 and not np.array_equal(n, img)
    assert np.array_equal(n, de.degrade(img, "noise", 0.15, np.random.default_rng(1)))
    with pytest.raises(ValueError):
        de.degrade(img, "noise", 0.1)
    v = de.versions()
    assert len(v) == 10 and v[0] == ("original", 0, 0.0) and {k for k, _, _ in v} == {"original", "blur", "noise", "dim"}


def _versions(dr_offset, slope_deg, seed=0, n_h=40, n_d=12):
    """Healthy eyes on a line (loose ends = 3 - 0.1 * density); degrading moves along a line of slope ``slope_deg``."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_h + n_d):
        y = int(i >= n_h)
        fov = 3.0 if i % 2 else 6.0
        dens0 = rng.uniform(18, 26) if y == 0 else rng.uniform(12, 18)
        e0 = 3 - 0.1 * dens0 + rng.normal(0, 0.03) + (dr_offset if y else 0.0)
        rows.append({"image_id": f"e{i}", "y": y, "fov_mm": fov, "kind": "original", "level": 0, "density": dens0,
                     "ends_per_mm": e0, "raw_ends_per_mm": 2 * e0})
        for kind in de.LEVELS:
            for lvl in (1, 2, 3):
                d = dens0 * (1 - 0.12 * lvl)
                e = e0 + slope_deg * (d - dens0) + rng.normal(0, 0.03)
                rows.append({"image_id": f"e{i}", "y": y, "fov_mm": fov, "kind": kind, "level": lvl, "density": d,
                             "ends_per_mm": e, "raw_ends_per_mm": 2 * e})
    return pd.DataFrame(rows)


def test_readings_separate_the_three_outcomes():
    ok = de.analyse(_versions(dr_offset=-0.5, slope_deg=-0.1), CFG)  # degradation follows the healthy line; DR below
    assert ok["readings"]["overall"] == "supported" and len(ok["readings"]["T1_types_passing"]) == 3
    assert ok["T3"]["ends_per_mm"]["pooled"]["median"] < -0.3 and ok["readings"]["T2_on_the_curve"]
    same = de.analyse(_versions(dr_offset=0.0, slope_deg=-0.1, seed=1), CFG)  # DR looks like degraded healthy
    assert same["readings"]["overall"] == "not_supported" and not same["readings"]["T3_dr_below_degraded_healthy"]
    flat = de.analyse(_versions(dr_offset=-0.5, slope_deg=+0.05, seed=2), CFG)  # degradation removes loose ends
    assert flat["readings"]["T1_types_passing"] == [] and flat["readings"]["overall"] == "not_supported"


def test_matching_counts_each_eye_once_and_never_the_target_itself():
    pool = pd.DataFrame({"image_id": ["a", "a", "b", "c", "t"], "fov_mm": 3.0, "density": [10.0, 10.2, 9.8, 10.1, 10.0],
                         "v": [1.0, 3.0, 4.0, 6.0, 100.0]})
    tg = pd.DataFrame({"image_id": ["t"], "fov_mm": [3.0], "density": [10.0], "v": [5.0]})
    m = de.matched_difference(tg, pool, "v")
    assert m.n_matched_eyes.iloc[0] == 3 and m.d.iloc[0] == pytest.approx(5.0 - (2.0 + 4.0 + 6.0) / 3)
    few = de.matched_difference(tg, pool[pool.image_id.isin(["a", "t"])], "v")
    assert np.isnan(few.d.iloc[0])
