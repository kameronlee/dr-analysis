"""M7 (Experiment A), M8 curves index, M9 (Experiment D) with X4 Forman ties, M10 (Experiment C), M11 Holm.

Values come from the experiments' saved outputs. Holm-adjusted p-values are
read from ``secondary_holm.json``; the report checks that they follow from the
recorded raw p-values and fixed family sizes, and that every p-value role in
the comparison tables agrees with the prespecified families. No p-value role
is assigned or changed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import multiplicity
from ..stats.effects import holm
from .cohort import _fov, _group
from .core import ReportRefusal, fmt, fmt_ci, fmt_p, is_missing, stable_hash


def holm_lookup(holm_json: dict | None) -> dict[str, float | None]:
    out = {}
    for fam in (holm_json or {}).get("families", {}).values():
        out.update(fam.get("holm_adjusted_p", {}))
    return out


def member_key(cfg: dict) -> dict[tuple[str, str, str], str]:
    """(experiment, scope, feature) -> member label for MWU members (A/D)."""
    return {(k[0], k[1], k[2]): v[1] for k, v in multiplicity.family_members(cfg).items() if k[3] == "mwu_p"}


def comparison_table(rows: list[dict], experiment: str, cfg: dict, holm_adj: dict | None,
                     labels: dict[str, str] | None = None, holm_note: str | None = None) -> pd.DataFrame:
    """``holm_adj`` must come from a validated summary; pass None with ``holm_note`` otherwise."""
    members = member_key(cfg)
    out = []
    for r in rows:
        key = (experiment, r.get("scope"), r.get("feature"))
        expected_role = multiplicity.role(cfg, experiment, r.get("scope"), r.get("feature"), "mwu_p")
        if r.get("status") == "ok" and r.get("mwu_p_role") != expected_role:
            raise ReportRefusal([f"Experiment {experiment} {key[1:]}: recorded mwu_p_role "
                                 f"{r.get('mwu_p_role')!r} differs from the prespecified role {expected_role!r}"])
        member = members.get(key)
        feat = r.get("feature")
        name = f"{feat} [{labels[feat]}]" if labels and feat in labels else feat
        if r.get("status") != "ok":
            out.append({"scope": r.get("scope"), "feature": name, "status": r.get("status"),
                        "n DR / NORMAL": f"{r.get('n_pos', '')} / {r.get('n_neg', '')}",
                        "role": expected_role})
            continue
        out.append({
            "scope": r["scope"], "feature": name, "status": "ok",
            "n DR / NORMAL": f"{r['n_pos']} / {r['n_neg']}",
            "missing excluded": r.get("n_missing_excluded"),
            "median DR / NORMAL": f"{fmt(r.get('median_pos'))} / {fmt(r.get('median_neg'))}",
            "AUROC (95% CI)": f"{fmt(r.get('auroc'))} ({fmt_ci(r.get('auroc_ci'))})",
            "Cliff's δ (95% CI)": f"{fmt(r.get('cliffs_delta'))} ({fmt_ci(r.get('cliffs_delta_ci'))})",
            "MWU p": fmt_p(r.get("mwu_p")), "role": r.get("mwu_p_role"),
            "Holm-adjusted p": _holm_cell(member, holm_adj, holm_note),
            "adjusted LRT (covariates)": (f"{r.get('adjusted_status')}; p {fmt_p(r.get('adjusted_lr_p'))}; "
                                          f"OR {fmt(r.get('adjusted_or'))} ({fmt_ci(r.get('adjusted_or_ci'))}) "
                                          f"{r.get('adjusted_or_scale') or ''}; "
                                          f"{'+'.join(r.get('adjusted_covariates') or [])}"
                                          + (f"; reasons: {'; '.join(r.get('adjusted_failure_reasons'))}"
                                             if r.get("adjusted_failure_reasons") else "")),
            "adjusted p role": r.get("adjusted_lr_p_role")})
    return pd.DataFrame(out)


def _holm_cell(member: str | None, holm_adj: dict | None, holm_note: str | None) -> str:
    if not member:
        return "not in a family"
    if holm_adj is None:
        return holm_note or "withheld: Holm summary not validated"
    v = holm_adj.get(member)
    return fmt_p(v) if v is not None else "unavailable (counted as p = 1 in m)"


def source_p_values(results: dict[str, dict | None], cfg: dict) -> dict[str, tuple[str, float | None]]:
    """Each planned member's p-value as saved by its experiment: member -> (state, p)."""
    out = {}
    for (exp, scope, feat, test), (_, member) in multiplicity.family_members(cfg).items():
        r = results.get(exp)
        if r is None:
            out[member] = ("experiment_missing", None)
        elif exp in ("A", "D"):
            row = next((c for c in r.get("comparisons", [])
                        if c.get("scope") == scope and c.get("feature") == feat), None)
            out[member] = ("row_missing", None) if row is None else ("saved", row.get(test))
        else:
            out[member] = ("saved", (r.get("summary") or {}).get(test))
    return out


def holm_table(holm_json: dict, results: dict[str, dict | None], cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Families with raw/adjusted p.

    Checks that every raw member p equals the p-value saved by its experiment
    (absent experiments must appear as missing, i.e. counted as p = 1), that it
    equals the register, and that adjusted values follow from raw values and m.
    """
    checks = []
    rows = []
    source = source_p_values(results, cfg)
    planned = cfg["stats"]["holm_families"]
    if set(holm_json.get("families", {})) != set(planned):
        raise ReportRefusal(["secondary_holm.json families differ from the configured Holm families"])
    if holm_json.get("primary_excluded") is not True:
        raise ReportRefusal(["secondary_holm.json does not state that the primary test is excluded"])
    register = {r["member"]: r for r in holm_json.get("p_value_register", []) if r.get("member")}
    for fam, d in holm_json["families"].items():
        if list(d.get("planned_members", [])) != list(planned[fam]) or d.get("family_size_m") != len(planned[fam]):
            raise ReportRefusal([f"Holm family {fam}: members or m differ from the configuration"])
        recomputed = holm(d["raw_p"])
        for m in planned[fam]:
            raw, adj = d["raw_p"].get(m), d["holm_adjusted_p"].get(m)
            ok = (adj is None and recomputed[m] is None) or (
                adj is not None and recomputed[m] is not None and np.isclose(adj, recomputed[m], rtol=0, atol=1e-15))
            reg = register.get(m)
            ok_reg = reg is None and raw is None or (reg is not None and (
                (reg.get("p") is None and raw is None) or
                (reg.get("p") is not None and raw is not None and np.isclose(reg["p"], raw, rtol=0, atol=0))))
            state, src = source.get(m, ("not_a_member", None))
            ok_src = (raw is None and src is None) or (
                raw is not None and src is not None and float(raw) == float(src))
            checks.append({"check": f"{fam}/{m} raw p equals the saved analysis result ({state})",
                           "ok": bool(ok_src)})
            checks.append({"check": f"{fam}/{m} Holm value follows from raw p and m", "ok": bool(ok)})
            checks.append({"check": f"{fam}/{m} raw p equals the register", "ok": bool(ok_reg)})
            rows.append({"family": fam, "m": len(planned[fam]), "member": m, "raw p": fmt_p(raw),
                         "Holm-adjusted p": fmt_p(adj) if adj is not None else "unavailable",
                         "note": d.get("notes", {}).get(m, "")})
    bad = [c["check"] for c in checks if not c["ok"]]
    if bad:
        raise ReportRefusal([f"Holm summary inconsistent: {bad[:5]}"])
    return pd.DataFrame(rows), checks


def distribution_figure(table: pd.DataFrame, columns: list[tuple[str, str]], title: str, note: str):
    """Box + jittered points by group × FOV for each (column, label); missing values counted, not drawn."""
    import matplotlib.pyplot as plt

    t = table.assign(_g=_group(table.analysis_group), _f=_fov(table.fov_mm))
    cells = [(g, f) for f in sorted(t._f.unique()) for g in ("NORMAL", "DR")]
    fig, axes = plt.subplots(1, len(columns), figsize=(4.2 * len(columns), 4.6), squeeze=False)
    for ax, (col, lab) in zip(axes[0], columns):
        for i, (g, f) in enumerate(cells, start=1):
            sub = t[(t._g == g) & (t._f == f)]
            v = pd.to_numeric(sub.get(col, pd.Series(dtype=float)), errors="coerce").to_numpy(float)
            ok = v[np.isfinite(v)]
            if len(ok):
                # identifier-derived jitter is stable; identifiers are never displayed
                key = sub.image_id.astype(str).map(lambda s: int(stable_hash("jitter", s)[:6], 16)
                                                   / 0xFFFFFF).to_numpy()[np.isfinite(v)]
                ax.scatter(i + (key - 0.5) * 0.3, ok, s=9, alpha=0.5,
                           color="#b04b27" if g == "DR" else "#4c6a92", edgecolors="none")
                ax.boxplot([ok], positions=[i], widths=0.5, showfliers=False, manage_ticks=False,
                           medianprops={"color": "#111"})
            ax.text(i, 1.0, f"n={len(ok)}" + (f"\n({int((~np.isfinite(v)).sum())} missing)"
                                              if (~np.isfinite(v)).any() else ""),
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(1, len(cells) + 1), [f"{g}\n{f}" for g, f in cells], fontsize=7)
        ax.set_ylabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=10, y=0.95)
    fig.text(0.5, 0.01, note, ha="center", fontsize=6.5, wrap=True)
    fig.subplots_adjust(bottom=0.2, top=0.82, wspace=0.35)
    return fig


def forman_tie_table(table: pd.DataFrame) -> pd.DataFrame:
    cols = [("forman_initial_tied_edge_fraction", "Initial tied-edge fraction"),
            ("forman_first_batch_tiebreak_fraction", "First-batch fraction decided by edge-ID tie-break"),
            ("forman_initial_n_distinct_scores", "Distinct initial Forman-type scores")]
    t = table.assign(_g=_group(table.analysis_group), _f=_fov(table.fov_mm))
    cells = [("all", "all")] + [(g, f) for f in sorted(t._f.unique()) for g in ("NORMAL", "DR")]
    rows = []
    for col, lab in cols:
        r = {"diagnostic": lab}
        for g, f in cells:
            sub = t if g == "all" else t[(t._g == g) & (t._f == f)]
            v = pd.to_numeric(sub.get(col, pd.Series(dtype=float)), errors="coerce").to_numpy(float)
            ok = v[np.isfinite(v)]
            r[f"{g} {f}" if g != "all" else "all images"] = (
                f"{fmt(np.median(ok))} [{fmt(np.quantile(ok, .25))}, {fmt(np.quantile(ok, .75))}] (n={len(ok)})"
                if len(ok) else f"unavailable (n=0; {len(v)} missing)")
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Experiment C
def exp_c_tables(rc: dict, per_dr: pd.DataFrame, audit: pd.DataFrame, holm_adj: dict | None,
                 mode_is_clinical: bool, holm_note: str | None = None) -> dict:
    s = rc.get("summary") or {}
    checks, flags = [], []
    counts = per_dr.status.value_counts().to_dict() if len(per_dr) else {}
    checks.append({"check": "status counts equal results.json",
                   "ok": {k: int(v) for k, v in counts.items()} == {k: int(v) for k, v in
                                                                    (s.get("status_counts") or {}).items()}})
    ok_rows = per_dr[per_dr.status.eq("ok")] if len(per_dr) else per_dr
    checks.append({"check": "n_z_defined equals rows with status ok", "ok": int(len(ok_rows)) == s.get("n_z_defined")})
    checks.append({"check": "n_dr equals per-DR rows", "ok": int(len(per_dr)) == s.get("n_dr")})
    if len(ok_rows):
        z = (ok_rows.FI_DR - ok_rows.null_mean) / ok_rows.null_sd
        checks.append({"check": "z = (FI_DR − null mean) / null SD for every defined row",
                       "ok": bool(np.allclose(z, ok_rows.z, rtol=0, atol=1e-12))})
        if s.get("median_z") is not None:
            checks.append({"check": "median z equals the median of defined z",
                           "ok": bool(np.isclose(np.median(ok_rows.z), s["median_z"], rtol=0, atol=1e-12))})
    if len(audit) and len(per_dr):
        n_tot = audit.groupby("dr_image_id").size()
        n_oot = audit.assign(o=~audit.within_tolerance.astype(bool)).groupby("dr_image_id").o.sum()
        pend = per_dr[per_dr.n_reps_total.notna()] if "n_reps_total" in per_dr else per_dr.iloc[:0]
        checks.append({"check": "per-DR repetition counts equal the matching audit",
                       "ok": bool(all(int(n_tot.get(r.image_id, 0)) == int(r.n_reps_total)
                                      and int(n_oot.get(r.image_id, 0)) == int(r.n_reps_out_of_tolerance)
                                      for r in pend.itertuples()))})
    bad = [c["check"] for c in checks if not c["ok"]]
    if bad:
        raise ReportRefusal([f"Experiment C outputs inconsistent: {bad}"])
    ex = rc.get("execution") or {}
    if ex.get("mode") != "sharded":
        flags.append(f"execution mode is {ex.get('mode')!r}, not the planned resumable 'sharded' run; "
                     "no plan hash binds the inputs" + (" (required evidence for the clinical result)"
                                                         if mode_is_clinical else ""))
    if rc.get("deviations"):
        flags.append(f"deviations recorded: {rc['deviations']}")
    member = "C:wilcoxon_z_gt_0"
    row = {"DR images": s.get("n_dr"), "z defined": s.get("n_z_defined"),
           "status counts": "; ".join(f"{k}: {v}" for k, v in sorted((s.get("status_counts") or {}).items())),
           "Wilcoxon z > 0 (one-sided) p": fmt_p(s.get("wilcoxon_p_greater")),
           "role": s.get("wilcoxon_p_role"),
           "Holm-adjusted p (H4, m = 1)": _holm_cell(member, holm_adj, holm_note),
           "median z (95% CI)": f"{fmt(s.get('median_z'))} ({fmt_ci(s.get('median_z_ci'))})",
           "fraction z > 0": fmt(s.get("fraction_z_positive")),
           "repetitions (all DR)": int(audit.shape[0]) if len(audit) else 0,
           "out of tolerance": int((~audit.within_tolerance.astype(bool)).sum()) if len(audit) else 0,
           "empty after thinning": int(audit.status.eq("empty_after_thinning").sum()) if "status" in audit else 0,
           "median |relative residual|": fmt(audit.residual_rel.abs().median()) if len(audit) else "unavailable",
           "execution": f"{ex.get('mode')} (plan {str(ex.get('plan_sha256'))[:12]})",
           "deviations": "; ".join(rc.get("deviations") or []) or "none"}
    if s.get("wilcoxon_p_greater") is None:
        row["Wilcoxon z > 0 (one-sided) p"] = f"no p-value ({s.get('reason', 'not computed')})"
    return {"summary": pd.DataFrame([row]), "checks": checks, "flags": flags,
            "settings": rc.get("settings"), "caveats": rc.get("caveats")}


def exp_c_figure(per_dr: pd.DataFrame, audit: pd.DataFrame, tol: float, note: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    ok = per_dr[per_dr.status.eq("ok")] if len(per_dr) else per_dr
    ax = axes[0]
    if len(ok):
        ax.hist(ok.z, bins=min(30, max(5, len(ok) // 2)), color="#4c6a92", alpha=0.8)
        ax.axvline(0, color="#888", ls=":")
    undefined = per_dr.status.ne("ok").sum() if len(per_dr) else 0
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel="z = (FI_DR − null mean) / null SD", ylabel="DR images",
           title=f"Per-DR z (defined: {len(ok)}; undefined: {int(undefined)})")
    ax = axes[1]
    if len(audit):
        ax.hist(audit.residual_rel, bins=40, color="#777", alpha=0.85)
        for v in (-tol, tol):
            ax.axvline(v, color="#b04b27", ls="--", lw=0.9)
        inside = audit.within_tolerance.astype(bool).mean()
        ax.set_title(f"Density-matching residuals ({100 * inside:.1f}% within ±{tol:g})")
    else:
        ax.set_title("Density-matching residuals (none recorded)")
    ax.set(xlabel="(achieved − target) / target", ylabel="repetitions")
    ax = axes[2]
    if len(ok):
        ax.errorbar(ok.null_mean, ok.FI_DR, xerr=ok.null_sd, fmt="o", ms=3, color="#b04b27",
                    ecolor="#bbb", elinewidth=0.8)
        lim = [np.nanmin([ok.null_mean.min(), ok.FI_DR.min()]), np.nanmax([ok.null_mean.max(), ok.FI_DR.max()])]
        ax.plot(lim, lim, ls=":", color="#888")
    ax.set(xlabel="null mean FI (± SD)\n(density-matched thinned NORMAL donors)", ylabel="FI of the DR image",
           title="DR FI against its uniform-thinning null")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.01, note, ha="center", fontsize=6.5, wrap=True)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.27, top=0.85, wspace=0.32)
    return fig


def missing_z_reasons(per_dr: pd.DataFrame) -> pd.DataFrame:
    if not len(per_dr):
        return pd.DataFrame(columns=["status", "n"])
    return per_dr.status.value_counts().rename_axis("status").reset_index(name="n")


def finite_or_none(x):
    return None if is_missing(x) else float(x)
