"""Effect sizes, resampling and multiplicity helpers.

Orientation is fixed in advance and never flipped by outcome: AUROC is the
probability that a DR (positive) image has a HIGHER value than a normal image
(ties count 1/2). Cliff's delta = 2 AUROC - 1, oriented DR - normal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def weighted_auc(y: np.ndarray, score: np.ndarray, w: np.ndarray | None = None) -> float:
    """AUROC P(score_pos > score_neg) + 0.5 P(tie), with optional case weights."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, float)
    w = np.ones(len(s)) if w is None else np.asarray(w, float)
    wp, wn = w * y, w * ~y
    Wp, Wn = wp.sum(), wn.sum()
    if Wp <= 0 or Wn <= 0:
        return float("nan")
    u, inv = np.unique(s, return_inverse=True)
    pos = np.bincount(inv, weights=wp, minlength=len(u))
    neg = np.bincount(inv, weights=wn, minlength=len(u))
    neg_below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    return float(np.sum(pos * (neg_below + 0.5 * neg)) / (Wp * Wn))


def cliffs_delta(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    y = np.r_[np.ones(len(x_pos)), np.zeros(len(x_neg))]
    return 2.0 * weighted_auc(y, np.r_[x_pos, x_neg]) - 1.0


def group_bootstrap_weights(groups: np.ndarray, strata: np.ndarray | None, n_boot: int,
                            rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_rows) case weights from resampling groups (within strata if given).

    Each group's rows all receive the multiplicity of the group, so repeated
    images of one subject move together.
    """
    groups = np.asarray(groups).astype(str)
    ug, ginv = np.unique(groups, return_inverse=True)
    if strata is None:
        gstrat = np.zeros(len(ug), int)
    else:
        strata = np.asarray(strata).astype(str)
        gs = {}
        for g, s in zip(ginv, strata):
            gs.setdefault(g, set()).add(s)
        # A group spanning strata (e.g. two eyes with different labels) is kept in
        # one pseudo-stratum so it is never split.
        labels = ["|".join(sorted(gs[i])) for i in range(len(ug))]
        _, gstrat = np.unique(labels, return_inverse=True)
    W = np.zeros((n_boot, len(groups)))
    for st in np.unique(gstrat):
        members = np.flatnonzero(gstrat == st)
        draws = rng.integers(0, len(members), size=(n_boot, len(members)))
        counts = np.zeros((n_boot, len(ug)))
        for b in range(n_boot):
            np.add.at(counts[b], members[draws[b]], 1.0)
        W += counts[:, ginv] * (gstrat[ginv] == st)[None, :]
    return W


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> list[float]:
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))]


def two_group_summary(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int,
                      rng: np.random.Generator, alpha: float = 0.05) -> dict:
    """MWU (two-sided), AUROC and Cliff's delta with group-bootstrap CIs (stratified by class)."""
    x = np.asarray(x, float)
    y = np.asarray(y).astype(int)
    ok = np.isfinite(x)
    x, y, groups = x[ok], y[ok], np.asarray(groups)[ok]
    out = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
           "n_missing_excluded": int((~ok).sum())}
    if out["n_pos"] == 0 or out["n_neg"] == 0:
        out["status"] = "one_group_empty"
        return out
    mw = sps.mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")
    auc = weighted_auc(y, x)
    W = group_bootstrap_weights(groups, y, n_boot, rng)
    boots = np.array([weighted_auc(y, x, w) for w in W])
    out.update(status="ok", mwu_U=float(mw.statistic), mwu_p=float(mw.pvalue),
               median_pos=float(np.median(x[y == 1])), median_neg=float(np.median(x[y == 0])),
               auroc=auc, auroc_ci=percentile_ci(boots, alpha),
               cliffs_delta=2 * auc - 1, cliffs_delta_ci=percentile_ci(2 * boots - 1, alpha),
               orientation="higher value = DR (fixed, never flipped)")
    return out


def holm(pvals: dict[str, float | None]) -> dict[str, float | None]:
    """Holm step-down adjusted p-values over a FIXED family (every key is a planned member).

    Missing-test policy (pre-outcome amendment 2026-09-30, resume_review.md D7):
    an unavailable member (None/NaN) keeps its place in the family size m. It is
    treated as p = 1, so it can never be rejected, and its adjusted value is
    reported as None. This is Holm's procedure on the complete planned family
    with the missing hypothesis retained. It controls the familywise error
    rate. Dropping the member would shrink m after the fact, which is not
    conservative.
    """
    filled = {k: (float(v) if v is not None and np.isfinite(v) else 1.0) for k, v in pvals.items()}
    m = len(filled)
    items = sorted(filled.items(), key=lambda kv: kv[1])
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return {k: (adj[k] if v is not None and np.isfinite(v) else None) for k, v in pvals.items()}


def jonckheere_terpstra(x: np.ndarray, grade: np.ndarray, n_perm: int,
                        rng: np.random.Generator) -> dict:
    """JT statistic for increasing trend with ties (1/2), permutation p (one-sided, increasing).

    Permutation reshuffles grade labels over images (exchangeable under H0). The
    p-value uses (1 + #{J_perm >= J}) / (1 + n_perm).
    """
    x = np.asarray(x, float)
    g = np.asarray(grade)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    levels = np.unique(g)

    def stat(labels):
        J = 0.0
        for i, a in enumerate(levels):
            xa = x[labels == a]
            for b in levels[i + 1:]:
                xb = x[labels == b]
                d = xb[:, None] - xa[None, :]
                J += float((d > 0).sum() + 0.5 * (d == 0).sum())
        return J

    J = stat(g)
    perm = np.array([stat(rng.permutation(g)) for _ in range(int(n_perm))])
    return {"J": J, "n": int(len(x)), "levels": levels.tolist(), "n_permutations": int(n_perm),
            "p_increasing": float((1 + np.sum(perm >= J - 1e-12)) / (1 + len(perm))),
            "p_two_sided": float((1 + np.sum(np.abs(perm - perm.mean()) >= abs(J - perm.mean()) - 1e-12))
                                 / (1 + len(perm))),
            "J_perm_mean": float(perm.mean()), "J_perm_sd": float(perm.std(ddof=1))}


def spearman(x, y) -> dict:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"rho": None, "p": None, "n": int(ok.sum())}
    if np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {"rho": None, "p": None, "n": int(ok.sum()), "reason": "constant_input_undefined"}
    r = sps.spearmanr(x[ok], y[ok])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(ok.sum())}


def correlation_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
