"""Metric audit (amendment 2026-11-16). Synthetic data only: SYNTHETIC - NOT CLINICAL."""
import numpy as np
import pandas as pd

import scripts.metric_audit as ma


def _frames(effect, noise_pert, seed=0, n=300):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"image_id": [f"e{i}" for i in range(n)], "fov_mm": np.where(np.arange(n) % 2, 3.0, 6.0),
                      "density": rng.uniform(10, 20, n)})
    d["dr"] = (np.arange(n) % 3 == 0).astype(float)
    d["non_dr_disease"] = 0.0
    d["fov_6mm"] = (d.fov_mm == 6.0).astype(float)
    for q in (*ma.qs.PROXIES, *ma.PROXIES_B):
        d[q + "_z"] = rng.normal(size=n)
    raw = 0.3 * d.density + effect * d.dr + rng.normal(0, 1, n)
    d["m_raw"] = raw
    d["m"] = ma.zscore_within_fov(d.assign(m=raw), "m")
    pert = {c: pd.DataFrame({"image_id": d.image_id, "density_p": d.density,
                             "m_p": raw + rng.normal(0, noise_pert, n)}) for c in ma.sp.CONDITIONS}
    allf = pd.concat([d, d.assign(image_id=d.image_id + "x", dr=0.0, non_dr_disease=d.dr)], ignore_index=True)
    return d, allf, pert


def test_real_stable_effect_is_a_robust_candidate_and_density_proxy_is_not():
    d, allf, pert = _frames(effect=1.5, noise_pert=0.1)
    r = ma.audit_metric("m", d, allf, pert, 7)
    c = ma.classify(r, r["adjusted"]["pooled"]["p"])
    assert c["robust_candidate"] and c["rank_stable"] and c["survives_perturbation"]
    assert r["added_discrimination"]["cv_auroc_delta_mean"] > 0.05
    assert abs(r["specificity"]["dr_minus_non_dr"]["coef"]) < 0.3  # same effect given to the other disease
    d, allf, pert = _frames(effect=0.0, noise_pert=0.1, seed=1)
    r = ma.audit_metric("m", d, allf, pert, 7)
    c = ma.classify(r, r["adjusted"]["pooled"]["p"])
    assert not c["density_independent_replicated"] and not c["robust_candidate"]
    assert abs(r["added_discrimination"]["cv_auroc_delta_mean"]) < 0.03


def test_unstable_metric_fails_rank_stability_and_proxies_b_avoid_vessels():
    d, allf, pert = _frames(effect=1.5, noise_pert=3.0, seed=2)
    c = ma.classify(ma.audit_metric("m", d, allf, pert, 7), 0.001)
    assert not c["rank_stable"] and not c["robust_candidate"]
    octa = np.full((60, 60), 50.0)
    faz = np.zeros((60, 60), bool)
    faz[20:40, 20:40] = True
    octa[faz] = 10.0
    large = np.zeros((60, 60), bool)
    large[5, :] = True
    octa[large] = 200.0
    p = ma.proxies_b(octa, faz, large)
    assert p["q_faz_floor"] == 10.0 and p["q_faz_noise"] == 0.0 and p["q_lv_signal"] == 200.0
    assert np.isnan(ma.proxies_b(octa, np.zeros_like(faz), large)["q_faz_floor"])
