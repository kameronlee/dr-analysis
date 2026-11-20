"""M3/M4 primary-result table, M5 cross-validated prediction, M6 collinearity, X1 ablations.

Every estimate is copied from Experiment B's ``results.json`` and its CSVs, or
from the ablation runs; the status string is reported verbatim. A
``not_estimable`` model is shown with its failure reasons and no p-value, odds
ratio or interval. Nothing is refitted. ROC curves are drawn from the saved
out-of-fold predictions; their per-repeat AUROC must equal the recorded value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats.effects import weighted_auc
from .core import ReportRefusal, fmt, fmt_ci, fmt_p, is_missing

ROLE_TAG = {"primary": "PRIM", "sensitivity_uncorrected": "SENS"}


def model_row(name: str, tag: str, r: dict | None) -> dict:
    if r is None:
        return {"model": name, "tag": tag, "status": "missing_input", "n": "", "events": "",
                "non_events": "", "LR (df 1)": "", "p": "", "OR per SD of FI": "", "95% CI": "",
                "failure_reasons": "result absent"}
    ok = r.get("status") == "ok"
    return {"model": name, "tag": tag, "status": r.get("status"), "n": fmt(r.get("n")),
            "events": fmt(r.get("n_events")), "non_events": fmt(r.get("n_nonevents")),
            "LR (df 1)": fmt(r.get("lr_statistic")) if ok else "not estimable",
            "p": fmt_p(r.get("p_value")) if ok else "no p-value (not estimable)",
            "OR per SD of FI": fmt(r.get("odds_ratio_added")) if ok else "not estimable",
            "95% CI": fmt_ci(r.get("odds_ratio_ci")) if ok else "not estimable",
            "failure_reasons": "; ".join(r.get("failure_reasons") or []) or "none"}


def primary_tables(rb: dict) -> dict:
    prim = rb.get("primary")
    if prim is None:
        raise ReportRefusal(["Experiment B results have no primary block"])
    if prim.get("p_value") is not None and prim.get("status") != "ok":
        raise ReportRefusal(["primary result has a p-value although its status is not ok"])
    if prim.get("p_value_role") not in (None, "primary"):
        raise ReportRefusal([f"primary p_value_role is {prim.get('p_value_role')!r}, not 'primary'"])
    rows = [model_row("H3 primary: baseline + FI vs baseline (Section 7B)", "PRIM", prim)]
    for name, r in sorted((rb.get("sensitivities") or {}).items()):
        if r.get("p_value_role") != "sensitivity_uncorrected":
            raise ReportRefusal([f"sensitivity {name} lacks the sensitivity_uncorrected role"])
        rows.append(model_row(f"Sensitivity: {name}", "SENS", r))
    coef = []
    if prim.get("status") == "ok":
        for k in prim.get("coefficients_full", {}):
            coef.append({"term": k, "full_model_coef": fmt(prim["coefficients_full"][k], 4),
                         "baseline_model_coef": fmt(prim.get("coefficients_baseline", {}).get(k), 4)})
    fb = rb.get("primary_ridge_fallback")
    fallback = None
    if fb:
        fallback = pd.DataFrame([{"term": k, "standardised_ridge_coef": fmt(v, 4),
                                  "role": "fallback_no_p_value (penalised; never referred to chi-square)"}
                                 for k, v in (fb.get("coefficients_standardized") or {}).items()])
    meta = {"status_verbatim": prim.get("status"), "failure_reasons": prim.get("failure_reasons"),
            "baseline": prim.get("baseline"), "added": prim.get("added"),
            "llf_baseline": prim.get("llf_baseline"), "llf_full": prim.get("llf_full"),
            "odds_ratio_scale": prim.get("odds_ratio_scale"), "estimator": prim.get("estimator"),
            "complete_case_audit": prim.get("complete_case_audit"),
            "fallback_status": (fb or {}).get("status"), "discrepancy_note": rb.get("discrepancy_note")}
    return {"models": pd.DataFrame(rows), "coefficients": pd.DataFrame(coef), "fallback": fallback,
            "meta": meta}


def prediction_table(rb: dict) -> tuple[pd.DataFrame, dict]:
    cv = rb.get("prediction_cv") or {}
    if cv.get("status") != "ok":
        return pd.DataFrame([{"status": cv.get("status", "missing_input"),
                              "reason": cv.get("reason", "prediction_cv block absent")}]), cv
    d = cv["paired_delta"]
    rows = [{"model": m, "mean cross-validated AUROC": fmt(v)} for m, v in sorted(cv["mean_auroc"].items())]
    rows.append({"model": f"Δ ({d['model_b']} − {d['model_a']}), mean over repeats",
                 "mean cross-validated AUROC": f"{fmt(d['delta_auroc_mean_over_repeats'])} "
                                               f"(95% CI {fmt_ci(d['ci'])}; conditional, see note)"})
    return pd.DataFrame(rows), cv


def roc_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    distinct = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (1 + distinct) - tps
    return np.r_[0, fps / max(fps[-1], 1)], np.r_[0, tps / max(tps[-1], 1)]


def check_oof(preds: pd.DataFrame, auc_by_repeat: pd.DataFrame, cv: dict) -> list[dict]:
    """Recomputed per-repeat AUROC from saved OOF predictions must equal the recorded values."""
    checks = []
    for (r, m), g in preds.groupby(["repeat", "model"]):
        rec = auc_by_repeat[(auc_by_repeat.repeat == r) & (auc_by_repeat.model == m)].auroc
        val = weighted_auc(g.y.to_numpy(), g.prob.to_numpy())
        ok = len(rec) == 1 and np.isclose(rec.iloc[0], val, rtol=0, atol=1e-12)
        checks.append({"check": f"AUROC repeat {r} {m}", "ok": bool(ok)})
    means = auc_by_repeat.groupby("model").auroc.mean().to_dict()
    for m, v in (cv.get("mean_auroc") or {}).items():
        checks.append({"check": f"mean AUROC {m} equals results.json",
                       "ok": bool(m in means and np.isclose(means[m], v, rtol=0, atol=1e-12))})
    bad = [c["check"] for c in checks if not c["ok"]]
    if bad:
        raise ReportRefusal([f"saved out-of-fold predictions do not reproduce the recorded AUROC: {bad[:5]}"])
    return checks


def roc_figure(preds: pd.DataFrame, cv: dict, title_note: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    colours = {"baseline": "#4c6a92", "baseline_plus_FI": "#b04b27"}
    for m, gm in preds.groupby("model"):
        for _, g in gm.groupby("repeat"):
            fpr, tpr = roc_points(g.y.to_numpy(), g.prob.to_numpy())
            ax.plot(fpr, tpr, color=colours.get(m, "#555"), alpha=0.35, lw=0.8)
        ax.plot([], [], color=colours.get(m, "#555"),
                label=f"{m} (mean AUROC {fmt((cv.get('mean_auroc') or {}).get(m))})")
    ax.plot([0, 1], [0, 1], ls=":", color="#888", lw=0.8)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.0),
           title="Cross-validated out-of-fold ROC, one curve per repeat")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.text(0.5, 0.01, title_note, ha="center", fontsize=7, wrap=True)
    fig.subplots_adjust(bottom=0.16, top=0.9)
    return fig


# ------------------------------------------------------------------ ablations (X1)
PRIMARY_SETTING_PATHS = {"step_frac": ("stress", "step_frac"),
                         "spur_length_factor": ("graph", "spur_length_factor"),
                         "min_fragment_length_um": ("graph", "min_fragment_length_um"),
                         "crop_center_mm": ("pipeline", "crop_center_mm"),
                         "coverage_geometry": ("stress", "coverage_geometry"),
                         "d_percentile": ("calibration", "percentile")}


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def ablation_rows(summary: dict, variant_results: dict[str, dict | None], cfg: dict) -> pd.DataFrame:
    """Every configured (parameter, value), its status and the primary-model estimate."""
    configured = cfg.get("ablations", {})
    runs = {(r["parameter"], str(r["value"])): r for r in summary.get("runs", [])}
    rows = []
    for param, values in configured.items():
        for value in values:
            r = runs.get((param, str(value)))
            path = PRIMARY_SETTING_PATHS.get(param)
            primary_value = cfg[path[0]].get(path[1]) if path else None
            dup = bool(path) and _same(value, primary_value)
            row = {"parameter": param, "value": "none" if value is None else value,
                   "equals primary setting": ("not applicable" if not path else
                                              "yes (duplicates the primary configuration)" if dup else "no")}
            if r is None:
                row.update(status="missing_input", detail="no run recorded in ablation_summary.json")
            elif r.get("status") == "not_applicable":
                row.update(status="not_applicable", detail=r.get("reason", ""))
            elif r.get("status") == "failed":
                row.update(status="failed", detail=r.get("error", ""))
            else:
                vr = variant_results.get(f"{param}={value}")
                pr = (vr or {}).get("primary") or {}
                est = pr.get("status") == "ok"
                row.update(status=f"ok; primary {r.get('primary_status')}",
                           detail="; ".join(pr.get("failure_reasons") or []) or "none",
                           n=fmt(r.get("primary_n")), LR=fmt(r.get("primary_lr")) if est else "not estimable",
                           p=fmt_p(r.get("primary_p")) if est else "no p-value",
                           OR=fmt(r.get("primary_or")) if est else "not estimable",
                           CI=fmt_ci(pr.get("odds_ratio_ci")) if est else "not estimable",
                           pooled_AUROC_FI=fmt(r.get("a_pooled_auroc_fi")),
                           pooled_AUROC_FIperf=fmt(r.get("a_pooled_auroc_fiperf")),
                           p_value_role=r.get("p_value_role"))
                if vr is None:
                    row["detail"] += "; variant exp_b/results.json missing (CI unavailable)"
                elif not _same(pr.get("odds_ratio_added"), r.get("primary_or")):
                    raise ReportRefusal([f"ablation {param}={value}: summary OR differs from its "
                                         "exp_b/results.json"])
            rows.append(row)
    unknown = sorted(set(runs) - {(p, str(v)) for p, vs in configured.items() for v in vs})
    if unknown:
        raise ReportRefusal([f"ablation runs not in the configured ablation set: {unknown[:5]}"])
    return pd.DataFrame(rows)


def forest_figure(entries: list[dict], note: str):
    """entries: label, tag, status, or, lo, hi, reasons. Non-estimable rows are shown as text."""
    import matplotlib.pyplot as plt

    n = len(entries)
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * n + 1.8))
    ys = np.arange(n)[::-1]
    finite = [e for e in entries if not any(is_missing(e.get(k)) for k in ("or", "lo", "hi"))]
    lo = min([e["lo"] for e in finite] + [0.5])
    hi = max([e["hi"] for e in finite] + [2.0])
    for y, e in zip(ys, entries):
        colour = {"PRIM": "#b04b27", "SENS": "#4c6a92"}.get(e["tag"], "#666")
        if e in finite:
            ax.plot([e["lo"], e["hi"]], [y, y], color=colour, lw=1.4)
            ax.plot([e["or"]], [y], marker="s" if e["tag"] == "PRIM" else "o", color=colour, ms=6)
        else:
            ax.text(np.sqrt(lo * hi), y, f"{e['status']}: no estimate", ha="center", va="center",
                    fontsize=7, color="#444", style="italic")
    ax.axvline(1.0, color="#888", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(lo / 1.3, hi * 1.3)
    ax.set_ylim(-0.7, n - 0.3)
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    ticks = [t for t in (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10, 20, 50, 100) if lo / 1.3 <= t <= hi * 1.3]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(ys, [f"[{e['tag']}] {e['label']}" for e in entries], fontsize=7)
    ax.set_xlabel("Odds ratio for DR per SD of FI (Wald 95% CI), full nested model")
    ax.set_title("Primary H3 model, prespecified sensitivities and processing ablations", fontsize=9)
    fig.text(0.01, 0.005, note, fontsize=6.5, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.42, right=0.97, bottom=0.2, top=0.9)
    return fig


def forest_entries(rb: dict, abl: pd.DataFrame | None, variant_results: dict[str, dict | None]) -> list[dict]:
    def entry(lab, tag, r):
        ok = (r or {}).get("status") == "ok"
        ci = (r or {}).get("odds_ratio_ci") or [None, None]
        return {"label": lab, "tag": tag, "status": (r or {}).get("status", "missing_input"),
                "or": r.get("odds_ratio_added") if ok else None, "lo": ci[0] if ok else None,
                "hi": ci[1] if ok else None}

    out = [entry("H3 primary", "PRIM", rb.get("primary"))]
    for name, r in sorted((rb.get("sensitivities") or {}).items()):
        out.append(entry(f"sensitivity: {name}", "SENS", r))
    if abl is not None:
        for row in abl.itertuples():
            key = f"{row.parameter}={row.value if row.value != 'none' else None}"
            vr = variant_results.get(key)
            if str(row.status).startswith("ok"):
                out.append(entry(f"ablation {row.parameter}={row.value}", "SENS", (vr or {}).get("primary")))
            else:
                out.append({"label": f"ablation {row.parameter}={row.value}", "tag": "SENS",
                            "status": row.status, "or": None, "lo": None, "hi": None})
    return out
