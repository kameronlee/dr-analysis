"""Loose ends compared only where DR and healthy eyes share image contrast and vessel density.

Plan: ``docs/ENDNODE_CONTRAST_OVERLAP_AMENDMENT_2026-11-17.md`` (exploratory, post hoc). Builds the
item-12 table unchanged (``scripts/endnode_quality.py: build_table``), keeps the eyes whose
``q_cnr`` and vessel density lie in the range both groups share within their scan size, and fits
the item-12 set-A quality ladder there. Also, as fixed in the plan: a healthy-only prediction for the
DR eyes in that range (guards "supports"), the DR eyes kept against those dropped, each scan size,
the contrast-only range, and a nearest-neighbour matched comparison.

Usage::

    python scripts/endnode_contrast_overlap.py run --performed-by "<exact model>"
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retina_fragility.config import load_config  # noqa: E402
from retina_fragility.provenance import environment, now_utc, read_json, sha256_file, write_json  # noqa: E402
from scripts import endnode_quality as eq  # noqa: E402
from scripts import step0_quality_specificity as qs  # noqa: E402

PLAN = "docs/ENDNODE_CONTRAST_OVERLAP_AMENDMENT_2026-11-17.md"
RECORD = "reports/endnode_contrast_overlap_amendment_2026-11-17.json"
OUT = "outputs/exploratory_20261116_step0/endnode_contrast_overlap"
ITEM12 = "outputs/exploratory_20261116_step0/endnode_quality/results.json"
# Every file this run depends on must be hashed in the lock record, or the run refuses (pre-locking review, issue 5).
REQUIRED = (
    "scripts/endnode_contrast_overlap.py",
    "scripts/endnode_contrast_overlap_design.py",
    "scripts/endnode_quality.py",
    "scripts/step0_quality_specificity.py",
    "scripts/step0_null_controls.py",
    "retina_fragility/experiments/common.py",
    "retina_fragility/config.py",
    "retina_fragility/provenance.py",
    ITEM12,
    "config.yaml",
    "data/manifests/octa500_manifest.csv",
    "outputs/calibration/calibration.json",
    "outputs/exploratory_20261009/pipeline/features.csv",
    "outputs/exploratory_20261116_step0/endnode_mechanism/end_node_parts.csv",
    "outputs/exploratory_20261116_step0/endnode_null/analysis/per_target.csv",
    "outputs/exploratory_20261116_step0/quality_specificity/quality_proxies.csv",
    "outputs/exploratory_20261116_step0/metric_audit/quality_proxies_b.csv",
)

DECISION_OUTCOMES = ("interior", "null_z")
OUTCOMES = ("interior", "null_z", "raw")
FOVS = (3.0, 6.0)
Q = [p + "_z" for p in qs.PROXIES]  # set A: q_signal, q_cnr, q_sharp, q_stripe; z within FOV over the 295 eyes
BASE = ["dr", "density", "fov_6mm"]
SPECS = {"M0_density_fov": BASE,
         "M1_plus_quality": BASE + Q,
         "M3_plus_age_sex_quality": BASE + ["age", "sex_male"] + Q}
# Design counts fixed in the plan before any outcome was looked at: (DR, healthy) per scan size.
EXPECTED_JOINT = {3.0: (15, 57), 6.0: (32, 38)}
EXPECTED_CONTRAST = {3.0: (16, 79), 6.0: (32, 41)}
K, CALIPER, N_BOOT, SEED = 3, 0.5, 2000, 20261117
TOL = 1e-9
MATCH_LIMITS = ("interval holds matches fixed, so it ignores matching uncertainty and the reuse of healthy eyes "
                "(too narrow); matching uses density and q_cnr only, not the other proxies, so it is not "
                "comparable to M1; for null_z it matches on density, while the models use density_r")


def overlap(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Shared range of two samples: [larger of the minima, smaller of the maxima]."""
    return float(max(a.min(), b.min())), float(min(a.max(), b.max()))


def add_support(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Flag eyes inside the shared q_cnr range (in_contrast) and inside both shared ranges (in_joint), per FOV."""
    t = t.copy()
    t["in_contrast"] = False
    t["in_joint"] = False
    bounds = {}
    for fov in FOVS:
        m = t.fov_mm == fov
        h, d = t[m & (t.dr == 0)], t[m & (t.dr == 1)]
        clo, chi = overlap(h.q_cnr, d.q_cnr)
        dlo, dhi = overlap(h.density, d.density)
        inc = m & t.q_cnr.between(clo, chi)
        inj = inc & t.density.between(dlo, dhi)
        t.loc[inc, "in_contrast"] = True
        t.loc[inj, "in_joint"] = True
        bounds[f"fov_{int(fov)}mm"] = {
            "q_cnr_shared": [clo, chi], "density_shared": [dlo, dhi],
            "contrast": {"dr": int(t[inc].dr.sum()), "healthy": int((t[inc].dr == 0).sum())},
            "joint": {"dr": int(t[inj].dr.sum()), "healthy": int((t[inj].dr == 0).sum())}}
    return t, bounds


def _model_frame(s: pd.DataFrame, oc: str) -> pd.DataFrame:
    d = s.copy()
    if oc == "null_z":
        d["density"] = d.density_r  # the null was matched on re-cleaned density, as in item 12
    return d[np.isfinite(d[oc].astype(float))]


def fit_ladder(s: pd.DataFrame, oc: str) -> dict:
    """M0, M1, M3 of the item-12 set-A ladder (OLS, HC3) on the eyes in ``s`` with a finite outcome."""
    d = _model_frame(s, oc)
    out = {"n": int(len(d)), "n_dr": int(d.dr.sum())}
    for name, terms in SPECS.items():
        r = qs.ols(d, oc, terms, ["dr", "q_cnr_z"])
        r.pop("_fit", None)
        out[name] = r
    return out


def dr_coef(m: dict):
    return (m.get("coef") or {}).get("dr") if m.get("status") == "ok" else None


def reading(lad: dict, full_m1: float) -> str:
    """holds: M1 and M3 intervals below zero and M1 at or below half the full-sample M1;
    shrinks: M1 above half the full-sample M1 (smaller in size, or positive); otherwise inconclusive."""
    c1, c3 = dr_coef(lad["M1_plus_quality"]), dr_coef(lad["M3_plus_age_sex_quality"])
    if c1 is None or c3 is None:
        return "not_estimable"
    threshold = 0.5 * full_m1  # full_m1 < 0 (stop-check)
    if c1["ci"][1] < 0 and c3["ci"][1] < 0 and c1["coef"] <= threshold:
        return "holds"
    if c1["coef"] > threshold:
        return "shrinks"
    return "inconclusive"


def decide(readings: dict, guard_below_zero: dict, selection_flag: dict) -> dict:
    """Plan decision rule, with the healthy-only guard on "supports" and the selection wording."""
    v = [readings[oc] for oc in DECISION_OUTCOMES]
    if "not_estimable" in v:
        return {"outcome": "not_estimable"}
    if all(x == "holds" for x in v):
        failed = [oc for oc in DECISION_OUTCOMES if not guard_below_zero.get(oc)]
        if not failed:
            return {"outcome": "supports"}
        return {"outcome": "inconclusive",
                "why": "both hold, but the healthy-only prediction does not place DR eyes below healthy eyes for "
                       + ", ".join(failed)}
    n_shrinks = sum(x == "shrinks" for x in v)
    outcome = "refutes" if n_shrinks == 2 else "weakens" if n_shrinks == 1 else "inconclusive"
    res = {"outcome": outcome}
    if outcome in ("refutes", "weakens"):
        shrunk = [oc for oc in DECISION_OUTCOMES if readings[oc] == "shrinks"]
        res["wording"] = ("not separable from image contrast or from milder disease among the retained eyes"
                          if any(selection_flag.get(oc) for oc in shrunk) else "not separable from image contrast")
    return res


def healthy_prediction(s: pd.DataFrame, oc: str) -> dict:
    """Fit M1's terms without the DR term on the healthy eyes of ``s``; mean residual of the DR eyes of ``s``.

    Percentile bootstrap: healthy and DR eyes resampled separately, the healthy fit redone each time.
    """
    d = _model_frame(s, oc)
    terms = ["density", "fov_6mm", *Q]
    h, r = d[d.dr == 0], d[d.dr == 1]
    keep = [c for c in terms if h[c].astype(float).nunique() > 1]

    def mean_resid(hh: pd.DataFrame, rr: pd.DataFrame) -> float:
        a = np.column_stack([np.ones(len(hh)), hh[keep].to_numpy(float)])
        beta = np.linalg.lstsq(a, hh[oc].to_numpy(float), rcond=None)[0]
        b = np.column_stack([np.ones(len(rr)), rr[keep].to_numpy(float)])
        return float((rr[oc].to_numpy(float) - b @ beta).mean())

    if len(h) <= len(keep) + 2 or len(r) < 2:
        return {"status": "not_estimable", "n_healthy": int(len(h)), "n_dr": int(len(r))}
    rng = np.random.default_rng(SEED)
    boot = np.array([mean_resid(h.iloc[rng.integers(0, len(h), len(h))], r.iloc[rng.integers(0, len(r), len(r))])
                     for _ in range(N_BOOT)])
    est = mean_resid(h, r)
    return {"status": "ok", "n_healthy": int(len(h)), "n_dr": int(len(r)), "terms": keep,
            "mean_dr_residual": est, "ci_percentile": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "below_zero": bool(est < 0)}


def kept_vs_dropped(t: pd.DataFrame) -> dict:
    """DR eyes inside the joint range against DR eyes outside it, per scan size (descriptive)."""
    out = {}
    for fov in FOVS:
        d = t[(t.fov_mm == fov) & (t.dr == 1)]
        out[f"fov_{int(fov)}mm"] = {
            name: {"n": int(len(g)), "mean_density": float(g.density.mean()) if len(g) else None,
                   "mean_q_cnr": float(g.q_cnr.mean()) if len(g) else None,
                   "mean_age": float(g.age.mean()) if len(g) else None}
            for name, g in (("kept", d[d.in_joint]), ("dropped", d[~d.in_joint]))}
    return out


def matched(t: pd.DataFrame, oc: str) -> dict:
    """Each DR eye against its <= K nearest healthy eyes of the same scan size within CALIPER.

    Distance: Euclidean in (density_z, q_cnr_z), both z-scored within scan size over the 295 eyes;
    matching with replacement; DR eyes with no healthy eye within CALIPER are dropped. Estimate: mean
    of (DR outcome - mean outcome of its matches); percentile bootstrap over matched DR eyes with the
    matches held fixed (see MATCH_LIMITS).
    """
    d = t[np.isfinite(t[oc].astype(float))]
    diffs, dists, uses, per_fov = [], [], Counter(), {}
    for fov in FOVS:
        s = d[d.fov_mm == fov]
        h, r = s[s.dr == 0], s[s.dr == 1]
        hx = h[["density_z", "q_cnr_z"]].to_numpy(float)
        kept = 0
        for _, row in r.iterrows():
            dist = np.sqrt(((hx - row[["density_z", "q_cnr_z"]].to_numpy(float)) ** 2).sum(axis=1))
            idx = np.argsort(dist, kind="stable")[:K]
            idx = idx[dist[idx] <= CALIPER]
            if len(idx) == 0:
                continue
            kept += 1
            diffs.append(float(row[oc]) - float(h.iloc[idx][oc].astype(float).mean()))
            dists.append(float(dist[idx].mean()))
            uses.update(h.iloc[idx].image_id)
        per_fov[f"fov_{int(fov)}mm"] = {"dr_matched": kept, "dr_dropped": int(len(r)) - kept, "dr_total": int(len(r))}
    out = {"n_dr_matched": len(diffs), "per_fov": per_fov, "distinct_healthy_used": len(uses),
           "max_reuse_of_one_healthy_eye": max(uses.values()) if uses else 0,
           "k": K, "caliper_sd": CALIPER, "n_boot": N_BOOT, "seed": SEED, "limits": MATCH_LIMITS}
    if len(diffs) < 2:
        return {**out, "status": "not_estimable"}
    x = np.asarray(diffs)
    rng = np.random.default_rng(SEED)
    boot = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)])
    return {**out, "status": "ok", "estimate": float(x.mean()),
            "ci_percentile": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "mean_match_distance_sd": float(np.mean(dists))}


def item12_coef(item12: dict, oc: str, model: str) -> dict:
    return item12["outcomes"][oc]["pooled"]["ladder_set_a"][model]["coef"]["dr"]


def checks(t: pd.DataFrame, item12: dict) -> list[str]:
    """Stop-checks from the plan; nothing is written if any fails."""
    bad = []
    if len(t) != 295 or int(t.dr.sum()) != 64:
        bad.append(f"cohort is {len(t)} eyes with {int(t.dr.sum())} DR, expected 295 with 64")
    n_null = int(np.isfinite(t.null_z.astype(float)).sum())
    if n_null != 275:
        bad.append(f"{n_null} usable null targets, expected 275")
    for oc in DECISION_OUTCOMES:
        for model in ("M0_density_fov", "M1_plus_quality"):
            saved = item12_coef(item12, oc, model)["coef"]
            if not saved < 0:
                bad.append(f"{oc}: item-12 full-sample {model} is not negative ({saved})")
            got = dr_coef(fit_ladder(t, oc)[model])
            if got is None or abs(got["coef"] - saved) > TOL:
                bad.append(f"{oc}: full-sample {model} {None if got is None else got['coef']!r} does not reproduce "
                           f"item 12 ({saved!r}) within {TOL}")
    tt, bounds = add_support(t)
    for fov in FOVS:
        b = bounds[f"fov_{int(fov)}mm"]
        if (b["joint"]["dr"], b["joint"]["healthy"]) != EXPECTED_JOINT[fov]:
            bad.append(f"{fov:.0f} mm joint range has {b['joint']}, plan says {EXPECTED_JOINT[fov]}")
        if (b["contrast"]["dr"], b["contrast"]["healthy"]) != EXPECTED_CONTRAST[fov]:
            bad.append(f"{fov:.0f} mm contrast range has {b['contrast']}, plan says {EXPECTED_CONTRAST[fov]}")
        j = tt[tt.in_joint & (tt.fov_mm == fov) & np.isfinite(tt.null_z.astype(float))]
        if (int(j.dr.sum()), int((j.dr == 0).sum())) != EXPECTED_JOINT[fov]:
            bad.append(f"{fov:.0f} mm joint range: usable null targets differ from the plan")
    return bad


def analyse(t: pd.DataFrame, item12: dict) -> tuple[dict, pd.DataFrame]:
    t, bounds = add_support(t)
    t["density_z"] = eq.zscore_within_fov(t, "density")
    full = {oc: {m: item12_coef(item12, oc, m) for m in ("M0_density_fov", "M1_plus_quality")} for oc in OUTCOMES}
    res: dict = {"support": bounds, "dr_kept_vs_dropped": kept_vs_dropped(t),
                 "full_sample_item12": {oc: {m: {k: c[k] for k in ("coef", "ci", "se_hc3")} for m, c in d.items()}
                                        for oc, d in full.items()},
                 "scopes": {}}
    scopes = {"joint_pooled": t[t.in_joint],
              "joint_fov_3mm": t[t.in_joint & (t.fov_mm == 3.0)],
              "joint_fov_6mm": t[t.in_joint & (t.fov_mm == 6.0)],
              "contrast_pooled": t[t.in_contrast]}
    for name, s in scopes.items():
        res["scopes"][name] = {}
        for oc in OUTCOMES:
            lad = fit_ladder(s, oc)
            c1 = dr_coef(lad["M1_plus_quality"])
            res["scopes"][name][oc] = {
                "ladder": lad,
                "reading": reading(lad, full[oc]["M1_plus_quality"]["coef"]) if oc in DECISION_OUTCOMES else "descriptive",
                "share_removed_M1_pct": eq.share_removed(lad, "M0_density_fov", "M1_plus_quality"),
                "share_removed_M3_pct": eq.share_removed(lad, "M0_density_fov", "M3_plus_age_sex_quality"),
                "se_ratio_M1_vs_full_sample_M1": None if c1 is None
                else float(c1["se_hc3"] / full[oc]["M1_plus_quality"]["se_hc3"])}
    joint = t[t.in_joint]
    res["healthy_only_prediction_joint"] = {oc: healthy_prediction(joint, oc) for oc in OUTCOMES}
    res["matched"] = {oc: matched(t, oc) for oc in OUTCOMES}
    thr = {oc: 0.5 * full[oc]["M1_plus_quality"]["coef"] for oc in DECISION_OUTCOMES}
    sel = {}
    for oc in DECISION_OUTCOMES:
        m0 = dr_coef(res["scopes"]["joint_pooled"][oc]["ladder"]["M0_density_fov"])
        sel[oc] = None if m0 is None else bool(m0["coef"] > 0.5 * full[oc]["M0_density_fov"]["coef"])
    readings = {oc: res["scopes"]["joint_pooled"][oc]["reading"] for oc in DECISION_OUTCOMES}
    guard = {oc: res["healthy_only_prediction_joint"][oc].get("below_zero") for oc in DECISION_OUTCOMES}
    res["decision"] = {"readings_joint_pooled_set_a": readings, **decide(readings, guard, sel),
                       "healthy_only_guard_below_zero": guard,
                       "selection_flag_joint_m0_under_half_full_m0": sel,
                       "shrink_thresholds_half_full_m1": thr,
                       "matched_above_shrink_threshold": {
                           oc: (res["matched"][oc]["estimate"] > thr[oc]) if res["matched"][oc]["status"] == "ok" else None
                           for oc in DECISION_OUTCOMES}}
    return res, t


def _fmt(c) -> str:
    return "not estimable" if c is None else f"{c['coef']:+.3f} ({c['ci'][0]:+.3f} to {c['ci'][1]:+.3f})"


def text(res: dict) -> str:
    lines = [f"Contrast-overlap check of the loose-end lead. Plan: {PLAN}. Exploratory, post hoc.", ""]
    for k, b in res["support"].items():
        lines.append(f"{k}: shared q_cnr {b['q_cnr_shared'][0]:.4f} to {b['q_cnr_shared'][1]:.4f}; shared density "
                     f"{b['density_shared'][0]:.3f} to {b['density_shared'][1]:.3f}; joint range DR {b['joint']['dr']} / "
                     f"healthy {b['joint']['healthy']}; contrast-only DR {b['contrast']['dr']} / healthy {b['contrast']['healthy']}")
    for k, g in res["dr_kept_vs_dropped"].items():
        lines.append(f"{k} DR kept vs dropped: " + "; ".join(
            f"{n} n={v['n']} density {v['mean_density'] or float('nan'):.2f} q_cnr {v['mean_q_cnr'] or float('nan'):.3f} "
            f"age {v['mean_age'] or float('nan'):.1f}" for n, v in g.items()))
    lines.append("")
    for oc, d in res["full_sample_item12"].items():
        lines.append(f"full sample (item 12) {oc:9s} M0 {_fmt(d['M0_density_fov'])}  M1 {_fmt(d['M1_plus_quality'])}")
    for scope, ocs in res["scopes"].items():
        for oc, e in ocs.items():
            lad = e["ladder"]
            lines.append(f"{scope:16s} {oc:9s} n={lad['n']:3d} (DR {lad['n_dr']:2d})  "
                         f"M0 {_fmt(dr_coef(lad['M0_density_fov']))}  M1 {_fmt(dr_coef(lad['M1_plus_quality']))}  "
                         f"M3 {_fmt(dr_coef(lad['M3_plus_age_sex_quality']))}  [{e['reading']}]")
    lines.append("")
    for oc, h in res["healthy_only_prediction_joint"].items():
        lines.append(f"healthy-only prediction, joint range, {oc}: " + (
            "not estimable" if h["status"] != "ok" else
            f"mean DR residual {h['mean_dr_residual']:+.3f} ({h['ci_percentile'][0]:+.3f} to {h['ci_percentile'][1]:+.3f}); "
            f"{h['n_dr']} DR eyes against a fit on {h['n_healthy']} healthy eyes"))
    for oc, m in res["matched"].items():
        if m["status"] != "ok":
            lines.append(f"matched {oc}: not estimable ({m['n_dr_matched']} DR eyes matched)")
            continue
        lines.append(f"matched {oc}: DR minus matched healthy {m['estimate']:+.3f} "
                     f"({m['ci_percentile'][0]:+.3f} to {m['ci_percentile'][1]:+.3f}); {m['n_dr_matched']} DR eyes "
                     f"matched {m['per_fov']}; {m['distinct_healthy_used']} healthy eyes used, one used up to "
                     f"{m['max_reuse_of_one_healthy_eye']} times; mean distance {m['mean_match_distance_sd']:.2f} SD")
    lines.append(f"matching limits: {MATCH_LIMITS}")
    dec = res["decision"]
    lines += ["", f"DECISION: {dec['outcome']}" + (f" ({dec['wording']})" if dec.get("wording") else "")
              + (f"; {dec['why']}" if dec.get("why") else ""),
              f"  readings (joint range, pooled, set A): {dec['readings_joint_pooled_set_a']}",
              f"  healthy-only guard below zero: {dec['healthy_only_guard_below_zero']}",
              f"  joint-range M0 under half the full-sample M0: {dec['selection_flag_joint_m0_under_half_full_m0']}",
              f"  matched estimate above the shrink threshold: {dec['matched_above_shrink_threshold']}"]
    return "\n".join(lines) + "\n"


def lock_problems(rec: dict) -> list[str]:
    locked = {**rec.get("files_sha256", {}), **rec.get("script_sha256_at_adoption", {})}
    missing = [f for f in REQUIRED if f not in locked]
    if missing:
        return ["the lock record does not cover: " + ", ".join(missing)]
    return ["locked file changed: " + f for f, h in locked.items() if sha256_file(ROOT / f) != h]


def main() -> int:
    os.chdir(ROOT)
    if len(sys.argv) != 4 or sys.argv[1] != "run" or sys.argv[2] != "--performed-by":
        sys.exit(__doc__)
    performed_by = sys.argv[3]
    if not (ROOT / RECORD).exists():
        sys.exit("REFUSED: the plan is not locked (no ledger record)")
    bad = lock_problems(read_json(ROOT / RECORD))
    if bad:
        sys.exit("REFUSED: " + "; ".join(bad))
    out = ROOT / OUT
    if out.exists() and any(out.iterdir()):
        sys.exit("REFUSED: the output directory already has files")
    item12 = read_json(ROOT / ITEM12)
    t = eq.build_table(load_config(ROOT / "config.yaml"))
    bad = checks(t, item12)
    if bad:
        sys.exit("REFUSED (checks): " + "; ".join(bad))
    res, t2 = analyse(t, item12)
    res.update(analysis_status=eq.s0.STATUS_LABEL, plan=PLAN, record=RECORD,
               checks="passed (item-12 values read from " + ITEM12 + f", tolerance {TOL})",
               provenance={"created_utc": now_utc(), "performed_by": performed_by,
                           "script_sha256": sha256_file(Path(__file__)), "environment": environment()})
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "results.json", res)
    t2[["image_id", "y", "fov_mm", "q_cnr", "density", "in_contrast", "in_joint"]].to_csv(out / "support.csv", index=False)
    (out / "summary.txt").write_text(text(res))
    print(text(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
