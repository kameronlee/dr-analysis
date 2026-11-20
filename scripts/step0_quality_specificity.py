"""Image-quality and disease-specificity checks for the Step 0 matched-density finding.

Post-outcome exploratory. Authority and fixed readings:
``docs/STEP0_QUALITY_SPECIFICITY_AMENDMENT_2026-11-16.md``. Reads saved outputs
only; writes under ``outputs/exploratory_20261116_step0/``.

Subcommands::

    record    dated machine record (timestamp + input hashes); once only
    ids       print the comma-separated image IDs of the set-aside eyes
    proxies   four image-quality proxies for every eye in the source catalogue
    analyse   quality ladder (A) and disease-specificity contrasts (B)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retina_fragility.config import load_config  # noqa: E402
from retina_fragility.provenance import environment, now_utc, read_json, sha256_file, write_json  # noqa: E402
from retina_fragility.seeds import rng_for  # noqa: E402
from scripts import step0_null_controls as s0  # noqa: E402

AMENDMENT = "docs/STEP0_QUALITY_SPECIFICITY_AMENDMENT_2026-11-16.md"
RECORD = "reports/step0_quality_specificity_amendment_2026-11-16.json"
CATALOG = "data/manifests/octa500_source_catalog.csv"
OUT = "outputs/exploratory_20261116_step0/quality_specificity"
PIPE_OTHER = "outputs/exploratory_20261116_step0/pipeline_other"
STEP0_WIDE = "outputs/exploratory_20261116_step0/analysis/per_target_wide.csv"
PROXIES = ("q_signal", "q_cnr", "q_sharp", "q_stripe")
OTHER_DISEASES = ("AMD", "CNV", "CSC", "RVO", "OTHERS")


class CheckError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- proxies
def quality_proxies(oct_img: np.ndarray, octa_img: np.ndarray, vessel: np.ndarray) -> dict:
    """Four crude image-quality proxies (see the amendment). Inputs are 2-D arrays."""
    oct_img, octa = np.asarray(oct_img, float), np.asarray(octa_img, float)
    vessel = np.asarray(vessel, bool)
    bg = octa[~vessel]
    sd_bg = float(bg.std()) if bg.size else np.nan
    cnr = (float(octa[vessel].mean()) - float(bg.mean())) / sd_bg if vessel.any() and sd_bg > 0 else np.nan
    rng_ = float(octa.max() - octa.min())
    u = (octa - octa.min()) / rng_ if rng_ > 0 else np.zeros_like(octa)
    lap = u[1:-1, 1:-1] * 4 - u[:-2, 1:-1] - u[2:, 1:-1] - u[1:-1, :-2] - u[1:-1, 2:]
    mean = float(octa.mean())
    stripe = max(float(np.abs(np.diff(octa.mean(axis=1))).mean()),
                 float(np.abs(np.diff(octa.mean(axis=0))).mean())) / mean if mean > 0 else np.nan
    return {"q_signal": float(oct_img.mean()), "q_cnr": cnr, "q_sharp": float(lap.var()), "q_stripe": stripe}


def _plane(path: str) -> np.ndarray:
    from skimage import io

    a = io.imread(ROOT / path)
    return a if a.ndim == 2 else a[..., 0]


def compute_proxies() -> pd.DataFrame:
    cat = pd.read_csv(ROOT / CATALOG, dtype=str)
    rows = []
    for r in cat.itertuples():
        vessel = (_plane(r.capillary_path) > 0) | (_plane(r.large_vessel_path) > 0)
        rows.append({"image_id": r.image_id, "disease": r.disease, "fov_mm": float(r.fov_mm),
                     **quality_proxies(_plane(r.oct_ilm_opl_path), _plane(r.octa_ilm_opl_path), vessel)})
    return pd.DataFrame(rows)


def standardise_within_fov(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        g = out.groupby("fov_mm")[c]
        out[c + "_z"] = (out[c] - g.transform("mean")) / g.transform("std")
    return out


# ------------------------------------------------------------------------------ models
def ols(df: pd.DataFrame, outcome: str, terms: list[str], report: list[str]) -> dict:
    """OLS with HC3 standard errors on complete cases; coefficient, interval and p for ``report``."""
    import statsmodels.api as sm

    d = df[[outcome, *terms]].astype(float).dropna()
    X = d[terms].loc[:, d[terms].nunique() > 1]
    dropped = [t for t in terms if t not in X.columns]
    X = sm.add_constant(X, has_constant="add")
    if len(d) <= X.shape[1] + 1 or np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        return {"status": "not_estimable", "n": int(len(d)), "dropped_constant_terms": dropped}
    fit = sm.OLS(d[outcome], X).fit(cov_type="HC3")
    ci = fit.conf_int()
    out = {"status": "ok", "n": int(len(d)), "terms": list(X.columns), "dropped_constant_terms": dropped,
           "coef": {}}
    for t in report:
        if t in X.columns:
            out["coef"][t] = {"coef": float(fit.params[t]), "se_hc3": float(fit.bse[t]),
                              "ci": [float(ci.loc[t, 0]), float(ci.loc[t, 1])], "p": float(fit.pvalues[t])}
    out["_fit"] = fit
    return out


def ladder(df: pd.DataFrame, outcome: str) -> dict:
    q = [p + "_z" for p in PROXIES]
    base = ["dr", "density", "fov_6mm"]
    specs = {"M0_density_fov": base, "M1_plus_quality": base + q, "M2_plus_age_sex": base + ["age", "sex_male"],
             "M3_plus_age_sex_quality": base + ["age", "sex_male"] + q}
    out = {}
    for name, terms in specs.items():
        r = ols(df, outcome, terms, ["dr", *q])
        r.pop("_fit", None)
        out[name] = r

    def excl(m):
        c = out[m].get("coef", {}).get("dr")
        return None if c is None else bool(c["ci"][0] > 0 or c["ci"][1] < 0)

    e1, e3 = excl("M1_plus_quality"), excl("M3_plus_age_sex_quality")
    out["reading"] = ("not_estimable" if e1 is None or e3 is None
                      else "survives_quality_adjustment" if (e1 and e3) else "attenuated")
    return out


def group_difference(df: pd.DataFrame, col: str) -> dict:
    out = {}
    for scope, sub in (("fov_3mm", df[df.fov_mm == 3.0]), ("fov_6mm", df[df.fov_mm == 6.0])):
        a, b = sub.loc[sub.dr == 1, col].dropna(), sub.loc[sub.dr == 0, col].dropna()
        if len(a) < 2 or len(b) < 2:
            out[scope] = {"status": "not_estimable"}
            continue
        u = sps.mannwhitneyu(a, b, alternative="two-sided")
        out[scope] = {"n_dr": int(len(a)), "n_healthy": int(len(b)), "median_dr": float(a.median()),
                      "median_healthy": float(b.median()), "auroc_dr_higher": float(u.statistic / (len(a) * len(b))),
                      "mwu_p": float(u.pvalue)}
    return out


def spearman_by_fov(df: pd.DataFrame, x: str, cols) -> dict:
    out = {}
    for scope, sub in (("fov_3mm", df[df.fov_mm == 3.0]), ("fov_6mm", df[df.fov_mm == 6.0])):
        out[scope] = {}
        for c in cols:
            d = sub[[x, c]].dropna()
            r = sps.spearmanr(d[x], d[c]) if len(d) >= 3 else None
            out[scope][c] = None if r is None else {"n": int(len(d)), "rho": float(r.statistic), "p": float(r.pvalue)}
    return out


def specificity(healthy: pd.DataFrame, cases: pd.DataFrame, base_seed: int) -> dict:
    """Matched-density FI contrast for DR, each other disease, and all non-DR disease together."""
    out = {"groups": {}, "descriptive": {}}
    groups = {"DR": cases[cases.disease == "DR"], **{g: cases[cases.disease == g] for g in OTHER_DISEASES},
              "ALL_NON_DR": cases[cases.disease != "DR"]}
    h = healthy.assign(y=0)
    for name, sub in groups.items():
        frame = pd.concat([h, sub.assign(y=1)], ignore_index=True)
        out["groups"][name] = s0.contrast_by_scope(frame, "FI", base_seed, f"specificity|{name}")
        out["descriptive"][name] = {
            f"fov_{int(f)}mm": {"n": int(len(s)), "median_FI": float(s.FI.median()),
                                "median_density": float(s.density.median())}
            for f, s in sub.groupby("fov_mm")}
    out["descriptive"]["HEALTHY"] = {f"fov_{int(f)}mm": {"n": int(len(s)), "median_FI": float(s.FI.median()),
                                                       "median_density": float(s.density.median())}
                                     for f, s in healthy.groupby("fov_mm")}
    allf = pd.concat([healthy, cases], ignore_index=True)
    allf["dr"] = (allf.disease == "DR").astype(float)
    allf["non_dr_disease"] = ((allf.disease != "DR") & (allf.disease != "NORMAL")).astype(float)
    q = [p + "_z" for p in PROXIES]
    for label, extra in (("density_fov", []), ("plus_age_sex_quality", ["age", "sex_male", *q])):
        r = ols(allf, "FI", ["dr", "non_dr_disease", "density", "fov_6mm", *extra], ["dr", "non_dr_disease"])
        fit = r.pop("_fit", None)
        if fit is not None:
            t = fit.t_test("dr - non_dr_disease = 0")
            ci = np.asarray(t.conf_int()).ravel()
            r["dr_minus_non_dr"] = {"coef": float(np.asarray(t.effect).ravel()[0]), "ci": [float(ci[0]), float(ci[1])],
                                    "p": float(np.asarray(t.pvalue).ravel()[0])}
        out[f"ols_{label}"] = r
    dr_c, non_c = out["groups"]["DR"]["pooled"], out["groups"]["ALL_NON_DR"]["pooled"]
    diff = out["ols_density_fov"].get("dr_minus_non_dr")

    def excl(ci):
        return ci is not None and (ci[0] > 0 or ci[1] < 0)

    same_sign = (dr_c.get("median") is not None and non_c.get("median") is not None
                 and np.sign(dr_c["median"]) == np.sign(non_c["median"]) and dr_c["median"] != 0)
    if excl(non_c.get("ci")) and same_sign:
        reading = "not_specific_to_dr"
    elif excl(dr_c.get("ci")) and not excl(non_c.get("ci")) and diff is not None and excl(diff["ci"]):
        reading = "specific_to_dr"
    else:
        reading = "undetermined"
    out["reading"] = reading
    return out


# -------------------------------------------------------------------------------- data
def input_files(cfg: dict) -> dict[str, str]:
    files = [AMENDMENT, s0.AMENDMENT, s0.RECORD, "config.yaml", cfg["data"]["manifest"], CATALOG,
             cfg["calibration"]["frozen_file"], f"{s0.SRC_RUN}/pipeline/features.csv",
             f"{s0.OUT_ROOT}/saved/results.json"]
    return {f: sha256_file(ROOT / f) for f in files}


def check_record(cfg: dict) -> dict:
    if not (ROOT / RECORD).exists():
        raise CheckError(f"no dated record at {RECORD}; run 'record' first")
    rec = read_json(ROOT / RECORD)
    changed = [f for f, h in rec["files_sha256"].items() if input_files(cfg).get(f) != h]
    if changed:
        raise CheckError("inputs changed since the dated record: " + ", ".join(changed))
    return rec


def set_aside_ids(cfg: dict) -> list[str]:
    m = pd.read_csv(ROOT / cfg["data"]["manifest"], dtype=str, keep_default_na=False)
    sel = m[m.disease.isin(OTHER_DISEASES) & (m.calibration.str.lower() != "true")]
    return sorted(sel.image_id)


def other_table(cfg: dict) -> pd.DataFrame:
    """Features of the set-aside eyes joined with manifest covariates."""
    m = pd.read_csv(ROOT / cfg["data"]["manifest"], dtype=str, keep_default_na=False)
    f = pd.read_csv(ROOT / PIPE_OTHER / "features.csv", dtype={"image_id": str})
    want = set(set_aside_ids(cfg))
    if set(f.image_id) - want:
        raise CheckError("pipeline_other holds eyes outside the set-aside list")
    t = m[m.image_id.isin(want)][["image_id", "disease", "fov_mm", "age", "sex"]].merge(f, on="image_id", how="inner",
                                                                                      suffixes=("", "_feat"))
    t["fov_mm"] = t.fov_mm.astype(float)
    t["age"] = pd.to_numeric(t.age, errors="coerce")
    t["sex_male"] = t.sex.str.upper().map({"M": 1.0, "F": 0.0})
    t["density"], t["FI"] = t[s0.DENSITY_COL].astype(float), t[s0.FI_COL].astype(float)
    return t[["image_id", "disease", "fov_mm", "age", "sex_male", "density", "FI"]]


def analyse(cfg: dict) -> tuple[dict, pd.DataFrame]:
    base_seed = cfg["project"]["base_seed"]
    prox = pd.read_csv(ROOT / OUT / "quality_proxies.csv", dtype={"image_id": str})
    table, _ = s0.load_table(cfg)
    a = table[["image_id", "y", "fov_mm", "age", "sex_male", "density", "FI"]].copy()
    a["disease"] = np.where(a.y == 1, "DR", "NORMAL")
    frames = [a.drop(columns="y")]
    other_missing = None
    if (ROOT / PIPE_OTHER / "features.csv").exists():
        o = other_table(cfg)
        other_missing = sorted(set(set_aside_ids(cfg)) - set(o.image_id))
        frames.append(o)
    eyes = pd.concat(frames, ignore_index=True).merge(prox[["image_id", *PROXIES]], on="image_id", how="left")
    eyes = standardise_within_fov(eyes, PROXIES)
    eyes["dr"] = (eyes.disease == "DR").astype(float)
    eyes["fov_6mm"] = (eyes.fov_mm == 6.0).astype(float)
    cohort = eyes[eyes.disease.isin(["DR", "NORMAL"])].copy()
    res: dict = {"analysis_status": s0.STATUS_LABEL, "human_qc_satisfied": False,
                 "A_quality": {
                     "n": int(len(cohort)),
                     "group_difference": {p: group_difference(cohort, p) for p in PROXIES},
                     "spearman_FI": spearman_by_fov(cohort, "FI", PROXIES),
                     "spearman_density": spearman_by_fov(cohort, "density", PROXIES),
                     "ladder_FI": ladder(cohort, "FI")}}
    if (ROOT / STEP0_WIDE).exists():
        w = pd.read_csv(ROOT / STEP0_WIDE, dtype={"image_id": str})
        w = w[w.n_eligible_donors >= s0.MIN_DONORS][["image_id", "z_uniform", "patchiness_rank"]]
        c2 = cohort.merge(w, on="image_id", how="inner")
        res["A_quality"]["ladder_z_uniform"] = ladder(c2, "z_uniform")
        res["A_quality"]["ladder_patchiness_rank"] = ladder(c2, "patchiness_rank")
    else:
        res["A_quality"]["ladder_z_uniform"] = res["A_quality"]["ladder_patchiness_rank"] = {
            "status": "pending_step0_simulation"}
    if len(frames) == 2:
        healthy = eyes[eyes.disease == "NORMAL"]
        res["B_specificity"] = specificity(healthy, eyes[eyes.disease != "NORMAL"], base_seed)
        res["B_specificity"]["set_aside_without_features"] = other_missing
    else:
        res["B_specificity"] = {"status": "pending_pipeline_other"}
    return res, eyes


def text(res: dict) -> str:
    A = res["A_quality"]
    L = ["IMAGE-QUALITY AND SPECIFICITY CHECKS (post-outcome exploratory; human QC unmet)", "",
         "A. Does the matched-density finding survive image-quality adjustment?",
         "   DR coefficient for FI (negative = DR less fragile than healthy at the same density):"]
    for m, r in A["ladder_FI"].items():
        if m == "reading":
            continue
        c = r.get("coef", {}).get("dr")
        L.append(f"   {m:26s} " + ("not estimable" if c is None else
                 f"{c['coef']:+.4f}  [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]  p = {c['p']:.4f}  n = {r['n']}"))
    L.append(f"   READING: {A['ladder_FI']['reading']}")
    L.append("   Quality proxies, DR versus healthy (AUROC > 0.5 = higher in DR):")
    for p in PROXIES:
        g = A["group_difference"][p]
        L.append(f"   {p:9s} " + "  ".join(
            f"{s}: AUROC {g[s]['auroc_dr_higher']:.2f}, p = {g[s]['mwu_p']:.4f}" for s in ("fov_3mm", "fov_6mm")
            if "auroc_dr_higher" in g[s]))
    for k in ("ladder_z_uniform", "ladder_patchiness_rank"):
        L.append(f"   {k}: {A[k].get('reading', A[k].get('status'))}")
    B = res["B_specificity"]
    L += ["", "B. Is the shift specific to diabetic eyes?"]
    if "groups" not in B:
        L.append(f"   {B.get('status')}")
    else:
        L.append("   FI minus matched healthy FI (same FOV, density within 5%), pooled:")
        for g, c in B["groups"].items():
            p = c["pooled"]
            ci = p.get("ci")
            L.append(f"   {g:11s} median {s0._fmt(p.get('median'), 4)} "
                     f"[{s0._fmt(ci[0], 4) if ci else 'n/a'}, {s0._fmt(ci[1], 4) if ci else 'n/a'}]  "
                     f"matched {p['n_dr_matched']}/{p['n_dr_total']}  p = {s0._fmt(p.get('wilcoxon_p_two_sided'), 4)}")
        d = B["ols_density_fov"].get("dr_minus_non_dr")
        if d:
            L.append(f"   DR minus non-DR coefficient: {d['coef']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}], p = {d['p']:.4f}")
        L.append(f"   READING: {B['reading']}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("record", "ids", "proxies", "analyse"):
        sub.add_parser(c)
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    cfg = load_config(ROOT / "config.yaml")
    if args.cmd == "record":
        if (ROOT / RECORD).exists():
            raise CheckError(f"{RECORD} already exists; it is written once")
        if (ROOT / OUT).exists() or (ROOT / PIPE_OTHER).exists():
            raise CheckError("outputs already exist; the record must precede them")
        write_json(ROOT / RECORD, {
            "adopted_utc": now_utc(), "decided_by": "",
            "instructed_by": "user, in session, 2026-11-16", "is_human_review": False,
            "human_approval_claimed": False, "analysis_status": s0.STATUS_LABEL,
            "outcomes_available_at_adoption": True, "step0a_saved_results_available_at_adoption": True,
            "step0_simulation_results_available_at_adoption": (ROOT / STEP0_WIDE).exists(),
            "quality_or_set_aside_results_available_at_adoption": False,
            "script_sha256_at_adoption": sha256_file(Path(__file__)), "files_sha256": input_files(cfg)})
        print(json.dumps(read_json(ROOT / RECORD), indent=1)[:900])
        return 0
    if args.cmd == "ids":
        print(",".join(set_aside_ids(cfg)))
        return 0
    check_record(cfg)
    (ROOT / OUT).mkdir(parents=True, exist_ok=True)
    if args.cmd == "proxies":
        df = compute_proxies()
        df.to_csv(ROOT / OUT / "quality_proxies.csv", index=False)
        print(df.groupby(["fov_mm"])[list(PROXIES)].median().to_string())
    elif args.cmd == "analyse":
        res, eyes = analyse(cfg)
        res["provenance"] = {"created_utc": now_utc(), "performed_by": "",
                             "amendment": AMENDMENT, "record": RECORD, "script_sha256": sha256_file(Path(__file__)),
                             "environment": environment(), "inputs_sha256": input_files(cfg)}
        write_json(ROOT / OUT / "results.json", res)
        eyes.to_csv(ROOT / OUT / "per_eye.csv", index=False)
        (ROOT / OUT / "summary.txt").write_text(text(res))
        print(text(res))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CheckError, s0.Step0Error) as exc:
        sys.exit(f"REFUSED: {exc}")
