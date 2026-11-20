"""Image-quality ladder for the loose-end results (backlog item 12).

Plan and decision rule: ``docs/ENDNODE_QUALITY_AMENDMENT_2026-11-17.md`` (locked by
``reports/endnode_quality_amendment_2026-11-17.json``). Reuses the exact ladder
that was applied to the fragility index. Reads saved outputs only; writes
``outputs/exploratory_20261116_step0/endnode_quality/``.

Usage::

    python scripts/endnode_quality.py run --performed-by ""
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retina_fragility.config import load_config  # noqa: E402
from retina_fragility.provenance import environment, now_utc, read_json, sha256_file, write_json  # noqa: E402
from scripts import step0_null_controls as s0  # noqa: E402
from scripts import step0_quality_specificity as qs  # noqa: E402

PLAN = "docs/ENDNODE_QUALITY_AMENDMENT_2026-11-17.md"
RECORD = "reports/endnode_quality_amendment_2026-11-17.json"
BASE = "outputs/exploratory_20261116_step0"
OUT = f"{BASE}/endnode_quality"
PARTS = f"{BASE}/endnode_mechanism/end_node_parts.csv"
NULL = f"{BASE}/endnode_null/analysis/per_target.csv"
PROXY_A = f"{BASE}/quality_specificity/quality_proxies.csv"
PROXY_B = f"{BASE}/metric_audit/quality_proxies_b.csv"
SET_A = tuple(qs.PROXIES)                       # q_signal, q_cnr, q_sharp, q_stripe
SET_B = ("q_faz_floor", "q_faz_noise", "q_lv_signal")
SCOPES = (("pooled", None), ("fov_3mm", 3.0), ("fov_6mm", 6.0))
DECISION_OUTCOMES = ("interior", "null_z")


def zscore_within_fov(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("fov_mm")[col]
    return (df[col] - g.transform("mean")) / g.transform("std")


def build_table(cfg: dict) -> pd.DataFrame:
    """One row per analysis eye with the three outcomes, both densities and all proxies."""
    table, _ = s0.load_table(cfg)
    t = table[["image_id", "y", "fov_mm", "age", "sex_male", "density", "end_density_per_mm2_pre"]].copy()
    parts = pd.read_csv(ROOT / PARTS, dtype={"image_id": str})[["image_id", "end_loc_interior"]]
    nul = pd.read_csv(ROOT / NULL, dtype={"image_id": str})
    nul = nul[nul.group.isin(["DR", "NORMAL"])][["image_id", "density_r", "z_interior_uniform", "usable"]]
    pa = pd.read_csv(ROOT / PROXY_A, dtype={"image_id": str})[["image_id", *SET_A]]
    pb = pd.read_csv(ROOT / PROXY_B, dtype={"image_id": str})[["image_id", *SET_B]]
    t = t.merge(parts, on="image_id", how="left").merge(nul, on="image_id", how="left") \
         .merge(pa, on="image_id", how="left").merge(pb, on="image_id", how="left")
    if len(t) != len(table):
        raise RuntimeError("row count changed while joining inputs")
    t["dr"] = t.y.astype(float)
    t["fov_6mm"] = (t.fov_mm == 6.0).astype(float)
    t["interior"] = zscore_within_fov(t, "end_loc_interior")
    t["raw"] = zscore_within_fov(t, "end_density_per_mm2_pre")
    t["null_z"] = t.z_interior_uniform.where(t.usable.astype(str).str.lower() == "true")
    for p in (*SET_A, *SET_B):
        t[p + "_z"] = zscore_within_fov(t, p)
    return t


def direction_aware_reading(lad: dict, m0: str, m1: str, m3: str) -> str:
    """Survives only if M1 and M3 exclude zero on the same side as M0 (plan, decision rule)."""
    def coef(m):
        return lad.get(m, {}).get("coef", {}).get("dr")
    c0, c1, c3 = coef(m0), coef(m1), coef(m3)
    if c0 is None or c1 is None or c3 is None:
        return "not_estimable"
    side = np.sign(c0["coef"])

    def same_side(c):
        return bool((c["ci"][1] < 0 and side < 0) or (c["ci"][0] > 0 and side > 0))
    return "survives_quality_adjustment" if same_side(c1) and same_side(c3) else "attenuated"


def ladder_b(df: pd.DataFrame, outcome: str) -> dict:
    """Secondary ladder: set B in place of set A (same structure and reading)."""
    q = [p + "_z" for p in SET_B]
    base = ["dr", "density", "fov_6mm"]
    out = {}
    for name, terms in (("M1b_plus_quality_b", base + q), ("M3b_plus_age_sex_quality_b", base + ["age", "sex_male"] + q)):
        r = qs.ols(df, outcome, terms, ["dr", *q])
        r.pop("_fit", None)
        out[name] = r

    return out


def share_removed(lad: dict, frm: str, to: str):
    """100 x (1 - coefficient / M0 coefficient): -1.22 to -0.51 is 58% removed."""
    a = lad[frm].get("coef", {}).get("dr", {}).get("coef")
    b = lad[to].get("coef", {}).get("dr", {}).get("coef")
    return None if a in (None, 0) or b is None else 100.0 * (1.0 - b / a)


def se_ratio(lad: dict, frm: str, to: str):
    a = lad[frm].get("coef", {}).get("dr", {}).get("se_hc3")
    b = lad[to].get("coef", {}).get("dr", {}).get("se_hc3")
    return None if not a or b is None else float(b / a)


RAW_M0 = -1.2214  # metric_audit/results.json: end_density_per_mm2_pre.adjusted.pooled.coef
RAW_M1 = (-0.5135, -0.7970, -0.2301)  # ... end_density_per_mm2_pre.quality_A (coef, ci)


def sanity_checks(t: pd.DataFrame) -> list[str]:
    """Structural checks fixed in the plan; any failure stops the run before results are written."""
    bad = []
    if len(t) != 295 or int(t.dr.sum()) != 64:
        bad.append(f"expected 295 eyes with 64 DR, got {len(t)} with {int(t.dr.sum())}")
    for c in ("end_loc_interior", *SET_A, *SET_B):
        if t[c].isna().any():
            bad.append(f"missing values in {c}: {int(t[c].isna().sum())}")
    if int(t.null_z.notna().sum()) != 275:
        bad.append(f"expected 275 usable null_z targets, got {int(t.null_z.notna().sum())}")
    la = qs.ladder(t[np.isfinite(t.raw)], "raw")
    m0 = la["M0_density_fov"]["coef"]["dr"]["coef"]
    m1 = la["M1_plus_quality"]["coef"]["dr"]
    if abs(m0 - RAW_M0) > 1e-4 or abs(m1["coef"] - RAW_M1[0]) > 1e-4 or \
            abs(m1["ci"][0] - RAW_M1[1]) > 1e-4 or abs(m1["ci"][1] - RAW_M1[2]) > 1e-4:
        bad.append(f"raw outcome does not reproduce the audit: M0 {m0:.4f}, M1 {m1['coef']:.4f} {m1['ci']}")
    return bad


def analyse(t: pd.DataFrame) -> dict:
    res: dict = {"outcomes": {}}
    for oc in ("interior", "raw", "null_z"):
        d = t.copy()
        if oc == "null_z":
            d["density"] = d.density_r  # the null was matched on re-cleaned density
        res["outcomes"][oc] = {}
        for scope, fov in SCOPES:
            s = d if fov is None else d[d.fov_mm == fov]
            s = s[np.isfinite(s[oc].astype(float))]
            la = qs.ladder(s, oc)
            la["reading_ladder_function_direction_blind"] = la.pop("reading")
            la["reading"] = direction_aware_reading(la, "M0_density_fov", "M1_plus_quality", "M3_plus_age_sex_quality")
            lb = ladder_b(s, oc)
            lb["M0_density_fov"] = la["M0_density_fov"]
            lb["reading"] = direction_aware_reading(lb, "M0_density_fov", "M1b_plus_quality_b", "M3b_plus_age_sex_quality_b")
            lb.pop("M0_density_fov")
            res["outcomes"][oc][scope] = {
                "n": int(len(s)), "n_dr": int(s.dr.sum()), "ladder_set_a": la, "ladder_set_b": lb,
                "share_removed_M1_pct": share_removed(la, "M0_density_fov", "M1_plus_quality"),
                "share_removed_M3_pct": share_removed(la, "M0_density_fov", "M3_plus_age_sex_quality"),
                "se_ratio_M1_over_M0": se_ratio(la, "M0_density_fov", "M1_plus_quality")}
    r = {oc: res["outcomes"][oc]["pooled"]["ladder_set_a"]["reading"] for oc in DECISION_OUTCOMES}
    surv = [v == "survives_quality_adjustment" for v in r.values()]
    res["decision"] = {"readings_pooled_set_a": r,
                       "outcome": ("supports" if all(surv) else "refutes" if not any(surv) else "weakens")
                       if "not_estimable" not in r.values() else "not_estimable"}
    return res


def _c(m: dict) -> str:
    c = m.get("coef", {}).get("dr")
    return "not estimable" if c is None else f"{c['coef']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] (n={m['n']})"


def text(res: dict) -> str:
    L = ["IMAGE-QUALITY LADDER FOR THE LOOSE-END RESULTS (exploratory, post hoc)", "",
         "DR coefficient (SD of the outcome; null_z in z units): M0 | M1 (+ set A) | M3 (+ age, sex, set A) | reading"]
    for oc, scopes in res["outcomes"].items():
        for scope, b in scopes.items():
            la = b["ladder_set_a"]
            L.append(f"  {oc:9s} {scope:8s} {_c(la['M0_density_fov'])} | {_c(la['M1_plus_quality'])} | "
                     f"{_c(la['M3_plus_age_sex_quality'])} | {la['reading']}")
    L += ["", "Set B (FAZ floor, FAZ noise, large-vessel signal) in place of set A: M1b | M3b | reading"]
    for oc, scopes in res["outcomes"].items():
        for scope, b in scopes.items():
            lb = b["ladder_set_b"]
            L.append(f"  {oc:9s} {scope:8s} {_c(lb['M1b_plus_quality_b'])} | {_c(lb['M3b_plus_age_sex_quality_b'])} | "
                     f"{lb['reading']}")
    L += ["", "Share of the M0 effect removed by set A (M1, M3) and SE ratio M1/M0, pooled:"]
    for oc, scopes in res["outcomes"].items():
        b = scopes["pooled"]
        f = lambda v: "n/a" if v is None else f"{v:.0f}%"  # noqa: E731
        L.append(f"  {oc:9s} M1 {f(b['share_removed_M1_pct'])}, M3 {f(b['share_removed_M3_pct'])}, "
                 f"SE ratio {b['se_ratio_M1_over_M0']:.2f}")
        q = b["ladder_set_a"]["M1_plus_quality"].get("coef", {}).get("q_cnr_z")
        if q:
            L.append(f"            q_cnr in M1: {q['coef']:+.3f} [{q['ci'][0]:+.3f}, {q['ci'][1]:+.3f}]")
    L += ["", "DECISION (plan rule, pooled set A, interior and null_z): " + json.dumps(res["decision"]),
          "Attenuated means: not separable from image contrast with these proxies."]
    return "\n".join(L) + "\n"


def main() -> int:
    os.chdir(ROOT)
    if len(sys.argv) != 4 or sys.argv[1] != "run" or sys.argv[2] != "--performed-by":
        sys.exit('usage: endnode_quality.py run --performed-by "<exact model>"')
    performed_by = sys.argv[3]
    if not (ROOT / RECORD).exists():
        sys.exit("REFUSED: the plan is not locked (no ledger record)")
    if (ROOT / OUT).exists() and any((ROOT / OUT).iterdir()):
        sys.exit("REFUSED: the output directory already has files")
    rec = read_json(ROOT / RECORD)
    bad = [f for f, h in rec["files_sha256"].items() if sha256_file(ROOT / f) != h]
    if bad:
        sys.exit("REFUSED: locked inputs changed: " + ", ".join(bad))
    cfg = load_config(ROOT / "config.yaml")
    t = build_table(cfg)
    bad = sanity_checks(t)
    if bad:
        sys.exit("REFUSED (sanity checks): " + "; ".join(bad))
    res = analyse(t)
    res.update(analysis_status=s0.STATUS_LABEL, plan=PLAN, record=RECORD,
               provenance={"created_utc": now_utc(), "performed_by": performed_by,
                           "script_sha256": sha256_file(Path(__file__)), "environment": environment()})
    (ROOT / OUT).mkdir(parents=True, exist_ok=True)
    write_json(ROOT / OUT / "results.json", res)
    t[["image_id", "y", "fov_mm", "density", "density_r", "interior", "raw", "null_z"]].to_csv(
        ROOT / OUT / "per_eye.csv", index=False)
    (ROOT / OUT / "summary.txt").write_text(text(res))
    print(text(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
