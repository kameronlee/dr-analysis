"""Primary test (Experiment B, H3): nested unpenalised logistic likelihood-ratio test.

Contract (preregistration + scientific_review.md 7.1):
* baseline and full (= baseline + FI) models are fit by unpenalised maximum
  likelihood on the IDENTICAL complete-case sample;
* LR = 2 (llf_full - llf_base), df = 1, referred to chi-square;
* rank deficiency, non-convergence, (quasi-)complete separation, too few
  events, implausible coefficients or a negative LR give ``status != "ok"``
  and ``p_value = None``. Ridge fits never supply a chi-square p-value; a
  ridge fit may be reported as ``role = "fallback_no_p_value"``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import linprog


def complete_cases(df: pd.DataFrame, cols: list[str], outcome: str) -> tuple[pd.DataFrame, dict]:
    """Rows with every column in ``cols`` + outcome finite; per-column missing counts."""
    need = list(dict.fromkeys([outcome] + list(cols)))
    missing_cols = [c for c in need if c not in df.columns]
    if missing_cols:
        raise KeyError(f"columns absent from analysis table: {missing_cols}")
    sub = df[need].apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(sub.to_numpy(float)).all(axis=1)
    audit = {"n_input": int(len(df)), "n_complete": int(ok.sum()),
             "n_dropped": int((~ok).sum()),
             "missing_by_column": {c: int((~np.isfinite(sub[c].to_numpy(float))).sum()) for c in need}}
    return df.loc[ok].copy(), audit


def is_binary(x: np.ndarray) -> bool:
    u = np.unique(x[np.isfinite(x)])
    return len(u) <= 2 and set(u.tolist()) <= {0.0, 1.0}


def design(sub: pd.DataFrame, cols: list[str], standardize: bool) -> tuple[np.ndarray, list[str], dict]:
    """Design with intercept. Continuous columns z-scored on this sample if requested."""
    X = [np.ones(len(sub))]
    scales = {}
    for c in cols:
        x = sub[c].to_numpy(float)
        if standardize and not is_binary(x):
            sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            mu = float(np.mean(x))
            scales[c] = {"mean": mu, "sd": sd}
            x = (x - mu) / sd if sd > 0 else x - mu
        X.append(x)
    return np.column_stack(X), ["const"] + list(cols), scales


def separation_lp(X: np.ndarray, y: np.ndarray, tol: float = 1e-7) -> dict:
    """Konis (2007) LP check for complete or quasi-complete separation.

    With s_i = 2 y_i - 1, the data are (quasi-)separated iff there exists
    beta != 0 with s_i x_i beta >= 0 for all i and strict inequality for some i.
    We maximise sum_i s_i x_i beta subject to s_i x_i beta >= 0, |beta_j| <= 1.
    """
    s = 2.0 * np.asarray(y, float) - 1.0
    A = s[:, None] * X
    c = -A.sum(axis=0)
    res = linprog(c, A_ub=-A, b_ub=np.zeros(len(A)), bounds=[(-1, 1)] * X.shape[1],
                  method="highs")
    if res.status != 0:
        return {"separated": None, "lp_status": int(res.status), "lp_message": res.message}
    value = float(-res.fun)
    scale = float(np.abs(A).sum()) or 1.0
    return {"separated": bool(value > tol * scale), "lp_objective": value}


def fit_logit(X: np.ndarray, y: np.ndarray, maxiter: int = 200) -> dict:
    import statsmodels.api as sm

    out: dict = {"converged": False, "error": None}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = sm.Logit(y, X)
            res = model.fit(method="newton", maxiter=maxiter, disp=0)
        except Exception as exc:  # PerfectSeparationError, LinAlgError, ...
            out["error"] = f"{type(exc).__name__}: {exc}"
            return out
    out["warnings"] = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    out["converged"] = bool(res.mle_retvals.get("converged", False))
    out["llf"] = float(res.llf)
    out["params"] = np.asarray(res.params, float)
    try:
        out["bse"] = np.asarray(res.bse, float)
    except Exception:  # singular Hessian
        out["bse"] = np.full(X.shape[1], np.nan)
    out["fitted"] = np.asarray(res.predict(X), float)
    if any("PerfectSeparation" in w for w in out["warnings"]):
        out["error"] = "perfect separation warning from statsmodels"
    return out


def nested_lrt(df: pd.DataFrame, outcome: str, baseline: list[str], added: str, cfg_stats: dict,
               label: str, role: str) -> dict:
    """Nested LRT on identical complete cases. Never returns a p-value on failure."""
    sub, audit = complete_cases(df, baseline + [added], outcome)
    y = sub[outcome].to_numpy(float)
    res: dict = {"label": label, "role": role, "outcome": outcome, "baseline": list(baseline),
                 "added": added, "complete_case_audit": audit, "n": int(len(sub)),
                 "n_events": int((y == 1).sum()), "n_nonevents": int((y == 0).sum()),
                 "status": "ok", "failure_reasons": [], "p_value": None, "lr_statistic": None,
                 "df": 1, "estimator": "statsmodels.Logit unpenalised MLE (Newton)"}
    if not set(np.unique(y).tolist()) <= {0.0, 1.0}:
        raise ValueError("outcome must be coded 0/1")
    fails = res["failure_reasons"]
    min_ev = int(cfg_stats["min_events_per_class"])
    if min(res["n_events"], res["n_nonevents"]) < min_ev:
        fails.append(f"insufficient_events(<{min_ev} per class)")
    std = bool(cfg_stats["standardize_continuous"])
    Xb, nb, _ = design(sub, baseline, std)
    Xf, nf, scales = design(sub, baseline + [added], std)
    res["design_columns_full"] = nf
    for name, X in (("baseline", Xb), ("full", Xf)):
        rank = int(np.linalg.matrix_rank(X))
        res[f"rank_{name}"] = rank
        if rank < X.shape[1]:
            fails.append(f"rank_deficient_{name}(rank {rank} < {X.shape[1]} columns)")
    if len(sub) and len(np.unique(y)) == 2 and not any(f.startswith("rank") for f in fails):
        for name, X in (("baseline", Xb), ("full", Xf)):
            sep = separation_lp(X, y)
            res[f"separation_check_{name}"] = sep
            if sep["separated"]:
                fails.append(f"separation_{name}")
            elif sep["separated"] is None:
                fails.append(f"separation_check_failed_{name}")
    if fails:
        res["status"] = "not_estimable"
        return res
    fb, ff = fit_logit(Xb, y), fit_logit(Xf, y)
    for name, f in (("baseline", fb), ("full", ff)):
        if f["error"]:
            fails.append(f"fit_error_{name}: {f['error']}")
            continue
        if not f["converged"]:
            fails.append(f"nonconvergence_{name}")
        tol = float(cfg_stats["separation_prob_tol"])
        if np.any(f["fitted"] < tol) or np.any(f["fitted"] > 1 - tol):
            fails.append(f"fitted_probabilities_at_boundary_{name}")
        if np.any(np.abs(f["params"][1:]) > float(cfg_stats["max_abs_coef_sd"])):
            fails.append(f"implausible_coefficient_{name}(>|{cfg_stats['max_abs_coef_sd']}|)")
    if fails:
        res["status"] = "not_estimable"
        return res
    lr = 2.0 * (ff["llf"] - fb["llf"])
    tol = 1e-8 * (1.0 + abs(ff["llf"]))
    if lr < -tol:
        res.update(status="not_estimable", lr_statistic=float(lr))
        fails.append("negative_likelihood_ratio")
        return res
    lr = max(lr, 0.0)
    k = nf.index(added)
    beta, se = float(ff["params"][k]), float(ff["bse"][k])
    z = sps.norm.ppf(1 - float(cfg_stats["alpha"]) / 2)
    per = "per SD" if added in scales else "per unit"
    res.update(
        lr_statistic=float(lr), p_value=float(sps.chi2.sf(lr, 1)),
        llf_baseline=fb["llf"], llf_full=ff["llf"],
        coef_added=beta, se_added=se,
        odds_ratio_added=float(np.exp(beta)),
        odds_ratio_ci=[float(np.exp(beta - z * se)), float(np.exp(beta + z * se))],
        odds_ratio_scale=per, added_scale=scales.get(added),
        coefficients_full=dict(zip(nf, map(float, ff["params"]))),
        coefficients_baseline=dict(zip(nb, map(float, fb["params"]))),
        fit_warnings=sorted(set(fb.get("warnings", []) + ff.get("warnings", []))),
    )
    return res


def ridge_fallback(df: pd.DataFrame, outcome: str, cols: list[str], C: float) -> dict:
    """Penalised fit reported ONLY as a labelled fallback with no p-value."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sub, audit = complete_cases(df, cols, outcome)
    y = sub[outcome].to_numpy(int)
    if len(np.unique(y)) < 2:
        return {"role": "fallback_no_p_value", "status": "single_class", "p_value": None}
    X = StandardScaler().fit_transform(sub[cols].to_numpy(float))
    m = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    return {"role": "fallback_no_p_value", "status": "fitted", "p_value": None,
            "penalty": "l2", "C": C, "n": int(len(sub)),
            "coefficients_standardized": dict(zip(cols, map(float, m.coef_[0]))),
            "note": "Penalised likelihood; NOT referred to a chi-square distribution."}


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X, names, _ = design(df, cols, True)
    rows = []
    for i, n in enumerate(names):
        if n == "const":
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                v = float(variance_inflation_factor(X, i))
            except Exception:
                v = float("nan")
        rows.append({"variable": n, "vif": v})
    return pd.DataFrame(rows)
