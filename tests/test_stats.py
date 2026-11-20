"""Primary LRT, failure detection, grouped CV, effect sizes and multiplicity.

Synthetic statistical fixtures only.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from retina_fragility.config import load_config
from retina_fragility.stats import cv as cvmod
from retina_fragility.stats import primary
from retina_fragility.stats.effects import (
    cliffs_delta,
    group_bootstrap_weights,
    holm,
    jonckheere_terpstra,
    two_group_summary,
    weighted_auc,
)

ST = load_config()["stats"]


def cohort(n=160, seed=0, effect=0.8):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"image_id": [f"i{i}" for i in range(n)],
                      "fov_6mm": rng.integers(0, 2, n).astype(float),
                      "age": rng.normal(60, 10, n), "sex_male": rng.integers(0, 2, n).astype(float),
                      "vessel_density": rng.normal(0.4, 0.05, n), "faz_area_mm2": rng.normal(0.3, 0.1, n),
                      "loop_density_per_mm2": rng.normal(20, 4, n), "fractal_dimension": rng.normal(1.6, 0.05, n),
                      "FI": rng.normal(0.8, 0.05, n)})
    lin = -0.5 + effect * (d.FI - 0.8) / 0.05 - 0.5 * (d.vessel_density - 0.4) / 0.05
    d["y"] = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    d["group_key"] = d.image_id
    return d


BASE = list(ST["baseline_primary"])


def test_lrt_uses_identical_complete_cases_and_matches_manual_fit():
    d = cohort()
    d.loc[:9, "FI"] = np.nan  # missing only in the added variable
    d.loc[10:14, "age"] = np.nan
    r = primary.nested_lrt(d, "y", BASE, "FI", ST, "t", "PRIMARY")
    assert r["status"] == "ok", r["failure_reasons"]
    assert r["n"] == 145 and r["complete_case_audit"]["n_dropped"] == 15
    cc = d.dropna(subset=BASE + ["FI"])
    Xb = sm.add_constant(cc[BASE].to_numpy(float))
    Xf = sm.add_constant(cc[BASE + ["FI"]].to_numpy(float))
    lb = sm.Logit(cc.y.to_numpy(), Xb).fit(disp=0).llf
    lf = sm.Logit(cc.y.to_numpy(), Xf).fit(disp=0).llf
    assert r["lr_statistic"] == pytest.approx(2 * (lf - lb), rel=1e-6)
    assert r["p_value"] == pytest.approx(sps.chi2.sf(2 * (lf - lb), 1), rel=1e-6)
    # a baseline fitted on all rows with FI missing would NOT be nested: guard against it
    lb_all = sm.Logit(d.dropna(subset=BASE).y.to_numpy(),
                      sm.add_constant(d.dropna(subset=BASE)[BASE].to_numpy(float))).fit(disp=0).llf
    assert r["llf_baseline"] != pytest.approx(lb_all)


def test_lrt_invariant_to_standardisation():
    d = cohort(seed=3)
    a = primary.nested_lrt(d, "y", BASE, "FI", dict(ST, standardize_continuous=True), "t", "x")
    b = primary.nested_lrt(d, "y", BASE, "FI", dict(ST, standardize_continuous=False, max_abs_coef_sd=1e9),
                           "t", "x")
    assert a["lr_statistic"] == pytest.approx(b["lr_statistic"], rel=1e-6)
    assert a["odds_ratio_scale"] == "per SD"


def test_rank_deficiency_gives_no_p_value():
    d = cohort()
    d["dup"] = d.vessel_density * 2
    r = primary.nested_lrt(d, "y", BASE + ["dup"], "FI", ST, "t", "PRIMARY")
    assert r["status"] == "not_estimable" and r["p_value"] is None
    assert any(f.startswith("rank_deficient") for f in r["failure_reasons"])
    d["const_fov"] = 1.0
    r2 = primary.nested_lrt(d, "y", ["const_fov", "age"], "FI", ST, "t", "PRIMARY")
    assert r2["p_value"] is None


def test_complete_and_quasi_complete_separation_are_detected():
    d = cohort()
    d["FI"] = d.y + np.random.default_rng(1).normal(0, 0.01, len(d))  # FI separates perfectly
    r = primary.nested_lrt(d, "y", BASE, "FI", ST, "t", "PRIMARY")
    assert r["p_value"] is None and "separation_full" in r["failure_reasons"]
    assert "separation_baseline" not in r["failure_reasons"]
    # quasi-complete: a binary covariate is 1 only among events
    X = np.column_stack([np.ones(8), [0, 0, 0, 0, 1, 1, 0, 0]])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    assert primary.separation_lp(X, y)["separated"] is True
    assert primary.separation_lp(X, np.array([0, 1, 0, 1, 1, 0, 1, 0]))["separated"] is False


def test_insufficient_events_and_nonconvergence_are_failures(monkeypatch):
    d = cohort(n=60)
    d["y"] = 0
    d.loc[:4, "y"] = 1
    r = primary.nested_lrt(d, "y", BASE, "FI", ST, "t", "PRIMARY")
    assert r["p_value"] is None and any("insufficient_events" in f for f in r["failure_reasons"])
    d = cohort()
    real = primary.fit_logit

    def fake(X, y, maxiter=200):
        out = real(X, y, maxiter)
        out["converged"] = False
        return out

    monkeypatch.setattr(primary, "fit_logit", fake)
    r = primary.nested_lrt(d, "y", BASE, "FI", ST, "t", "PRIMARY")
    assert r["p_value"] is None and "nonconvergence_full" in r["failure_reasons"]


def test_ridge_fallback_never_has_p_value():
    f = primary.ridge_fallback(cohort(), "y", BASE + ["FI"], 1.0)
    assert f["role"] == "fallback_no_p_value" and f["p_value"] is None


# --------------------------------------------------------------------- CV
class RecordingModel:
    """Records which rows (by a unique row-id column) each fit sees."""
    seen: list = []

    def __init__(self, C):
        self.inner = cvmod.make_model(C)

    def fit(self, X, y):
        RecordingModel.seen.append(set(X[:, 0].astype(int).tolist()))
        self.inner.fit(X, y)
        self.classes_ = self.inner.classes_
        return self

    def predict_proba(self, X):
        return self.inner.predict_proba(X)


def test_cv_preprocessing_is_fit_on_training_rows_only_and_groups_stay_together():
    d = cohort(n=90)
    d["rowid"] = np.arange(len(d)).astype(float)
    d["group_key"] = [f"s{i // 2}" for i in range(len(d))]  # two images per subject
    d["y"] = d.groupby("group_key").y.transform("max")
    RecordingModel.seen = []
    preds, folds, issues = cvmod.repeated_cv(d, "y", "group_key", {"m": ["rowid", "FI"]}, 5, 2, 1.0, 7,
                                             "t", model_factory=RecordingModel)
    assert not issues
    k = 0
    for (r, f), te in folds.groupby(["repeat", "fold"]):
        test_rows = set(te.row)
        assert not (RecordingModel.seen[k] & test_rows)
        assert RecordingModel.seen[k] | test_rows == set(range(len(d)))
        k += 1
    cvmod.check_group_disjoint(folds, d.group_key.to_numpy())
    bad = folds.copy()
    bad.loc[bad.row == 0, "fold"] = (bad.loc[bad.row == 1, "fold"].to_numpy() + 1) % 5
    with pytest.raises(RuntimeError):
        cvmod.check_group_disjoint(bad, d.group_key.to_numpy())


def test_train_fold_median_imputation_is_not_learned_from_test_rows():
    """Changing one held-out value must not change predictions for the other rows of that
    held-out fold (their model was fit on training rows only). Rows of other folds may change,
    because the altered row is training data for them."""
    d = cohort(n=100, seed=4)
    d.loc[::7, "FI"] = np.nan
    p1, folds, _ = cvmod.repeated_cv(d, "y", "group_key", {"m": BASE + ["FI"]}, 5, 1, 1.0, 3, "t")
    te0 = folds[(folds.repeat == 0) & (folds.fold == 0)].row.to_numpy()
    d2 = d.copy()
    victim = te0[0]
    d2.loc[victim, "FI"] = 1e6  # extreme value in one held-out row
    p2, _, _ = cvmod.repeated_cv(d2, "y", "group_key", {"m": BASE + ["FI"]}, 5, 1, 1.0, 3, "t")
    a = p1.set_index("row").prob
    b = p2.set_index("row").prob
    same_fold = np.setdiff1d(te0, [victim])
    assert np.allclose(a.loc[same_fold], b.loc[same_fold])
    other_folds = np.setdiff1d(np.arange(len(d)), te0)
    assert not np.allclose(a.loc[other_folds], b.loc[other_folds])  # the check has power


def test_aligned_oof_predictions_and_paired_delta():
    d = cohort(n=120, seed=5)
    preds, folds, _ = cvmod.repeated_cv(d, "y", "group_key", {"base": BASE, "full": BASE + ["FI"]}, 5, 3,
                                        1.0, 11, "t")
    per = preds.groupby(["repeat", "model"]).row.agg(["count", "nunique"])
    assert (per["count"] == len(d)).all() and (per["nunique"] == len(d)).all()
    delta = cvmod.paired_delta_auc(preds, "base", "full", 300, 1, "t")
    manual = []
    for r, g in preds.groupby("repeat"):
        w = g.pivot(index="row", columns="model", values="prob").join(g.drop_duplicates("row").set_index("row").y)
        manual.append(roc_auc_score(w.y, w.full) - roc_auc_score(w.y, w.base))
    assert delta["delta_auroc_mean_over_repeats"] == pytest.approx(np.mean(manual))
    assert delta["ci"][0] <= delta["delta_auroc_mean_over_repeats"] <= delta["ci"][1]
    assert "not a full refit" in delta["note"]


# ------------------------------------------------------------ effect sizes
def test_weighted_auc_matches_sklearn_and_row_replication():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50)
    s = np.round(rng.normal(size=50), 1)  # ties
    assert weighted_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    w = rng.integers(0, 3, 50).astype(float)
    rep = np.repeat(np.arange(50), w.astype(int))
    assert weighted_auc(y, s, w) == pytest.approx(roc_auc_score(y[rep], s[rep]))


def test_cliffs_delta_orientation_is_fixed():
    assert cliffs_delta(np.array([3, 4]), np.array([1, 2])) == 1.0
    assert cliffs_delta(np.array([1, 2]), np.array([3, 4])) == -1.0
    s = two_group_summary(np.r_[np.zeros(20), np.ones(20)], np.r_[np.ones(20), np.zeros(20)],
                          np.arange(40), 100, np.random.default_rng(0))
    assert s["auroc"] == 0.0 and s["cliffs_delta"] == -1.0  # lower in DR stays < 0.5, not flipped


def test_group_bootstrap_moves_groups_together_and_respects_strata():
    groups = np.array(["a", "a", "b", "c", "c", "d"])
    strata = np.array([1, 1, 1, 0, 0, 0])
    W = group_bootstrap_weights(groups, strata, 200, np.random.default_rng(0))
    assert np.all(W[:, 0] == W[:, 1]) and np.all(W[:, 3] == W[:, 4])
    # each stratum keeps its number of groups (2 groups each)
    assert np.allclose(W[:, [0, 2]].sum(1) - W[:, 0], 2 - W[:, 0])
    assert np.all((W[:, 0] + W[:, 2]) == 2) and np.all((W[:, 3] + W[:, 5]) == 2)


def test_holm_and_jonckheere():
    # Fixed planned family (resume_review.md D7): the unavailable member "d" keeps
    # m = 4 and counts as p = 1; it is never dropped to shrink the correction.
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj.pop("d") is None and adj == pytest.approx({"a": 0.04, "b": 0.09, "c": 0.09})
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    x = np.array([1, 2, 2, 3, 3, 4, 5, 5, 6])
    g = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    r = jonckheere_terpstra(x, g, 2000, np.random.default_rng(0))
    # pairs: (0,1): 9 wins; (0,2): 9; (1,2): 9 -> J = 27 (no ties across groups here)
    assert r["J"] == 27 and r["p_increasing"] < 0.01
    tie = jonckheere_terpstra(np.array([1, 1, 1, 1]), np.array([0, 0, 1, 1]), 200, np.random.default_rng(0))
    assert tie["J"] == 2.0 and tie["p_increasing"] == 1.0
