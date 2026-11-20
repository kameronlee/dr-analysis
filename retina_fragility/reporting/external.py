"""E1/E2 external DRAC tables and grade figure, E3 QC outcome, F1 exploratory embeddings.

Experiment E writes ``results.json`` only after its human-QC gate passes;
otherwise it writes ``readiness.json``. A readiness record is reported as
``not_run`` with its reasons, never as a result. All E p-values keep the
``exploratory_uncorrected`` role; grouping is image level because DRAC has no
patient identifiers. QC fail reasons are reported as aggregate counts only.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .core import ReportRefusal, fmt, fmt_ci, fmt_p

GRADE_LABEL = {0: "0 (no DR)", 1: "1 (NPDR)", 2: "2 (PDR)"}


def qc_table(qc: dict) -> pd.DataFrame:
    reasons = Counter((qc.get("failed_reasons") or {}).values())
    return pd.DataFrame([{
        "QC rows": qc.get("n_rows"), "audited (valid human decisions)": qc.get("n_audited"),
        "failed": qc.get("n_failed"), "fail fraction": fmt(qc.get("fail_fraction")),
        "gate ready": qc.get("ready"), "binding verified": qc.get("binding_verified"),
        "fail reasons (count)": "; ".join(f"{k}: {v}" for k, v in sorted(reasons.items())) or "none",
        "gate messages": "; ".join(qc.get("reasons") or []) or "none"}])


def flow_table(results: dict, seg_manifest: pd.DataFrame | None, run: dict | None) -> tuple[pd.DataFrame, list[dict]]:
    checks = []
    n_train = int((seg_manifest.split.str.lower() == "train").sum()) if seg_manifest is not None else None
    n_qc = results.get("analysis_table_audit", {}).get("n_excluded_qc_fail")
    n_an = results.get("analysis_table_audit", {}).get("n_analysed")
    rows = [{"stage": "segmented training images (grade labels public)", "n": n_train},
            {"stage": "processed by the graph pipeline", "n": (run or {}).get("n_rows")},
            {"stage": "pipeline exclusions", "n": (run or {}).get("n_excluded"),
             "detail": "; ".join(f"{k}: {v}" for k, v in ((run or {}).get("exclusion_reasons") or {}).items())},
            {"stage": "images with features", "n": (run or {}).get("n_ok")},
            {"stage": "excluded: failed human QC audit", "n": n_qc},
            {"stage": "analysed", "n": n_an}]
    if run is not None and n_train is not None:
        checks.append({"check": "pipeline processed every segmented training image", "ok": run.get("n_rows") == n_train})
    if run is not None and n_qc is not None and n_an is not None:
        checks.append({"check": "features − QC-failed = analysed", "ok": run.get("n_ok") - n_qc == n_an})
    bad = [c["check"] for c in checks if not c["ok"]]
    if bad:
        raise ReportRefusal([f"Experiment E count reconciliation failed: {bad}"])
    return pd.DataFrame(rows), checks


def trend_table(results: dict) -> pd.DataFrame:
    rows = []
    for t in results.get("trend", []):
        sp, jt = t.get("spearman") or {}, t.get("jt") or {}
        med = t.get("median_by_grade") or {}
        rows.append({"feature": t["feature"], "n": sp.get("n"),
                     "Spearman ρ (p)": (f"{fmt(sp.get('rho'))} ({fmt_p(sp.get('p'))})" if sp.get("rho") is not None
                                        else f"undefined ({sp.get('reason', 'n < 3')})"),
                     "JT J; one-sided p (increasing)": f"{fmt(jt.get('J'), 1)}; {fmt_p(jt.get('p_increasing'))}",
                     "permutations": jt.get("n_permutations"),
                     **{f"median grade {g}": fmt(_by_grade(med, g)) for g in (0, 1, 2)},
                     "role": "exploratory_uncorrected"})
    return pd.DataFrame(rows)


def _by_grade(medians: dict, grade: int):
    """JSON turns float grade keys into strings such as '0.0'."""
    for k, v in medians.items():
        try:
            if float(k) == grade:
                return v
        except (TypeError, ValueError):
            continue
    return None


def cv_table(results: dict) -> pd.DataFrame:
    rows = []
    for outcome, c in (results.get("cv") or {}).items():
        if c.get("status") != "ok":
            rows.append({"outcome": outcome, "status": c.get("status"), "detail": c.get("reason", "")})
            continue
        d = c["paired_delta"]
        rows.append({"outcome": outcome, "status": "ok",
                     "baseline AUROC": fmt(c["mean_auroc"].get("baseline")),
                     "baseline + FI AUROC": fmt(c["mean_auroc"].get("baseline_plus_FI")),
                     "Δ AUROC (95% CI, conditional)": f"{fmt(d['delta_auroc_mean_over_repeats'])} "
                                                      f"({fmt_ci(d['ci'])})",
                     "fit issues": len(c.get("fit_issues") or [])})
    return pd.DataFrame(rows)


def grade_figure(features: pd.DataFrame, seg_manifest: pd.DataFrame, failed_ids, fi_col: str,
                 segmenter: str, note: str):
    import matplotlib.pyplot as plt

    m = seg_manifest[seg_manifest.split.str.lower() == "train"][["image_id", "grade_num"]]
    t = m.merge(features[["image_id", fi_col]], on="image_id", how="inner")
    t = t[~t.image_id.isin(set(failed_ids or []))]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for i, g in enumerate((0, 1, 2), start=1):
        v = pd.to_numeric(t[t.grade_num == g][fi_col], errors="coerce").to_numpy(float)
        ok = v[np.isfinite(v)]
        if len(ok):
            rng = np.random.default_rng(i)
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(ok)), ok, s=8, alpha=0.45, color="#4c6a92",
                       edgecolors="none")
            ax.boxplot([ok], positions=[i], widths=0.5, showfliers=False, manage_ticks=False,
                       medianprops={"color": "#111"})
        ax.text(i, 1.0, f"n={len(ok)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7)
    ax.set_xticks([1, 2, 3], [GRADE_LABEL[g] for g in (0, 1, 2)])
    ax.set(ylabel=f"{fi_col} ({segmenter}-derived graph)", xlabel="DRAC grade (training labels)")
    ax.set_title(f"Experiment E ({segmenter}): FI by grade, image-level", fontsize=9, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.01, note, ha="center", fontsize=6.5, wrap=True)
    fig.subplots_adjust(bottom=0.22, top=0.86)
    return fig


def exp_f_table(rf: dict) -> pd.DataFrame:
    rows = []
    a, b = rf.get("model_a_embeddings") or {}, rf.get("model_b_embeddings_plus_fragility") or {}
    for k, name in (("qwk", "quadratic weighted κ"), ("auc_any_dr", "AUROC any DR"), ("auc_pdr", "AUROC PDR")):
        d = (rf.get("delta_b_minus_a") or {}).get(k, {})
        rows.append({"metric": name, "A: embeddings": fmt(a.get(k)), "B: + fragility": fmt(b.get(k)),
                     "B − A (95% CI, paired, conditional)": f"{fmt(d.get('point'))} ({fmt_ci(d.get('ci'))})",
                     "role": "exploratory (H5); no p-value"})
    return pd.DataFrame(rows)
