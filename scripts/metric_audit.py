"""Audit of OCTA graph metrics: information beyond density, and stability under label change.

Post-outcome exploratory. Authority and fixed classification:
``docs/METRIC_AUDIT_AMENDMENT_2026-11-16.md``. Reads saved feature tables and
released images only; rebuilds no graph; writes under
``outputs/exploratory_20261116_step0/metric_audit/``.

Subcommands::

    record   dated machine record; once only, before any audit number
    run      proxies (second set), every analysis, tables and summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retina_fragility.config import load_config  # noqa: E402
from retina_fragility.provenance import environment, now_utc, read_json, sha256_file, write_json  # noqa: E402
from retina_fragility.seeds import derive_seed, rng_for  # noqa: E402
from retina_fragility.stats.effects import holm  # noqa: E402
from scripts import step0_null_controls as s0  # noqa: E402
from scripts import step0_quality_specificity as qs  # noqa: E402
from scripts import step0_segmentation_perturbation as sp  # noqa: E402

AMENDMENT = "docs/METRIC_AUDIT_AMENDMENT_2026-11-16.md"
RECORD = "reports/metric_audit_amendment_2026-11-16.json"
OUT = "outputs/exploratory_20261116_step0/metric_audit"
METRICS = ("FI_adaptive_betweenness", "FI_random", "FIperf_adaptive_betweenness", "loop_density_per_mm2",
           "junction_density_per_mm2_post", "end_density_per_mm2_post", "junction_density_per_mm2_pre",
           "end_density_per_mm2_pre", "fractal_dimension", "vessel_density", "faz_area_mm2")
PROXIES_B = ("q_faz_floor", "q_faz_noise", "q_lv_signal")
MILD = ("dropout1", "boundary10", "dilate1")
FOVS = ((3.0, "fov_3mm"), (6.0, "fov_6mm"))


class AuditError(RuntimeError):
    pass


# -------------------------------------------------------------------------- proxies B
def proxies_b(octa: np.ndarray, faz: np.ndarray, large: np.ndarray) -> dict:
    """Quality proxies that avoid the capillary bed (see the amendment)."""
    octa = np.asarray(octa, float)
    core = ndi.binary_erosion(np.asarray(faz, bool), iterations=2)
    large = np.asarray(large, bool)
    ok = int(core.sum()) >= 20
    return {"q_faz_floor": float(octa[core].mean()) if ok else np.nan,
            "q_faz_noise": float(octa[core].std()) if ok else np.nan,
            "q_lv_signal": float(octa[large].mean()) if large.any() else np.nan,
            "n_faz_core_px": int(core.sum())}


def compute_proxies_b() -> pd.DataFrame:
    cat = pd.read_csv(ROOT / qs.CATALOG, dtype=str)
    rows = []
    for r in cat.itertuples():
        rows.append({"image_id": r.image_id,
                     **proxies_b(qs._plane(r.octa_ilm_opl_path), qs._plane(r.faz_path) > 0,
                                 qs._plane(r.large_vessel_path) > 0)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- analyses
def zscore_within_fov(df: pd.DataFrame, col: str, ref: pd.DataFrame | None = None) -> pd.Series:
    """Standardise ``col`` within FOV using the mean and SD of ``ref`` (default: ``df``)."""
    ref = df if ref is None else ref
    mu, sd = ref.groupby("fov_mm")[col].mean(), ref.groupby("fov_mm")[col].std()
    return (df[col] - df.fov_mm.map(mu)) / df.fov_mm.map(sd)


def coef(df: pd.DataFrame, y: str, terms: list[str], which: str = "dr") -> dict:
    r = qs.ols(df, y, terms, [which])
    r.pop("_fit", None)
    c = r.get("coef", {}).get(which)
    return {"status": r["status"], "n": r["n"], **(c or {})}


def excludes_zero(c: dict) -> bool:
    return "ci" in c and (c["ci"][0] > 0 or c["ci"][1] < 0)


def added_discrimination(df: pd.DataFrame, y: str, base_seed: int) -> dict:
    """Nested logistic LRT and repeated-CV AUROC change for adding metric ``y`` to density + FOV."""
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    d = df[["dr", "density", "fov_6mm", y]].astype(float).dropna()
    Xb = sm.add_constant(d[["density", "fov_6mm"]])
    Xf = sm.add_constant(d[["density", "fov_6mm", y]])
    out: dict = {"n": int(len(d))}
    try:
        fb, ff = sm.Logit(d.dr, Xb).fit(disp=0), sm.Logit(d.dr, Xf).fit(disp=0)
        lr = 2 * (ff.llf - fb.llf)
        out.update(lrt_statistic=float(lr), lrt_p=float(sps.chi2.sf(lr, 1)), logit_coef_per_sd=float(ff.params[y]))
    except Exception as exc:  # separation or non-convergence is reported, not hidden
        out.update(lrt_statistic=None, lrt_p=None, lrt_failure=f"{type(exc).__name__}: {exc}"[:200])
    Xb_, Xf_, yv = d[["density", "fov_6mm"]].to_numpy(), d[["density", "fov_6mm", y]].to_numpy(), d.dr.to_numpy()
    mu, sd = Xf_.mean(0), Xf_.std(0)
    sd[sd == 0] = 1.0
    Xf_, Xb_ = (Xf_ - mu) / sd, (Xb_ - mu[:2]) / sd[:2]
    deltas, base, full = [], [], []
    for rep in range(10):
        skf = StratifiedKFold(5, shuffle=True, random_state=derive_seed(base_seed, "audit_cv", rep) % (2 ** 31))
        pb, pf = np.zeros(len(yv)), np.zeros(len(yv))
        for tr, te in skf.split(Xb_, yv):
            pb[te] = LogisticRegression(C=1e6, max_iter=2000).fit(Xb_[tr], yv[tr]).predict_proba(Xb_[te])[:, 1]
            pf[te] = LogisticRegression(C=1e6, max_iter=2000).fit(Xf_[tr], yv[tr]).predict_proba(Xf_[te])[:, 1]
        a, b = roc_auc_score(yv, pb), roc_auc_score(yv, pf)
        base.append(a), full.append(b), deltas.append(b - a)
    out.update(cv_auroc_base=float(np.mean(base)), cv_auroc_full=float(np.mean(full)),
               cv_auroc_delta_mean=float(np.mean(deltas)),
               cv_auroc_delta_range=[float(np.min(deltas)), float(np.max(deltas))])
    return out


def audit_metric(m: str, cohort: pd.DataFrame, allf: pd.DataFrame, pert: dict, base_seed: int) -> dict:
    """Every analysis of the amendment for one metric. ``cohort`` holds z-scored ``m`` as ``m``."""
    qa = [p + "_z" for p in qs.PROXIES]
    qb = [p + "_z" for p in PROXIES_B]
    res: dict = {"adjusted": {"pooled": coef(cohort, m, ["dr", "density", "fov_6mm"])}}
    for fov, name in FOVS:
        res["adjusted"][name] = coef(cohort[cohort.fov_mm == fov], m, ["dr", "density"])
    c2 = cohort.assign(density_sq=cohort.density ** 2)
    res["adjusted"]["pooled_quadratic"] = coef(c2, m, ["dr", "density", "density_sq", "fov_6mm"])
    res["matched_contrast"] = s0.contrast_by_scope(cohort.assign(y=cohort.dr.astype(int)), m, base_seed, f"audit|{m}")
    res["added_discrimination"] = added_discrimination(cohort, m, base_seed)
    res["quality_A"] = coef(cohort, m, ["dr", "density", "fov_6mm", *qa])
    res["quality_B"] = coef(cohort, m, ["dr", "density", "fov_6mm", *qb])
    r = qs.ols(allf, m, ["dr", "non_dr_disease", "density", "fov_6mm"], ["dr", "non_dr_disease"])
    fit = r.pop("_fit", None)
    spec = {"status": r["status"], "n": r["n"], **r.get("coef", {})}
    if fit is not None:
        t = fit.t_test("dr - non_dr_disease = 0")
        ci = np.asarray(t.conf_int()).ravel()
        spec["dr_minus_non_dr"] = {"coef": float(np.asarray(t.effect).ravel()[0]), "ci": [float(ci[0]), float(ci[1])],
                                   "p": float(np.asarray(t.pvalue).ravel()[0])}
    res["specificity"] = spec
    res["perturbation"] = {}
    for cond, pf in pert.items():
        j = cohort[["image_id", "fov_mm", "dr", "fov_6mm", m + "_raw"]].merge(pf, on="image_id", how="inner")
        j[m] = zscore_within_fov(j, m + "_p")  # SD units of the perturbed metric
        blk = {"adjusted_pooled": coef(j.assign(density=j.density_p), m, ["dr", "density", "fov_6mm"])}
        for fov, name in FOVS:
            s = j[j.fov_mm == fov]
            blk[f"spearman_{name}"] = float(sps.spearmanr(s[m + "_raw"], s[m + "_p"]).statistic)
        res["perturbation"][cond] = blk
    return res


def classify(res: dict, holm_p: float | None) -> dict:
    a = res["adjusted"]
    sign = np.sign(a["pooled"].get("coef", np.nan))
    both = all(excludes_zero(a[n]) and np.sign(a[n]["coef"]) == sign for _, n in FOVS)
    replicated = bool(holm_p is not None and holm_p < 0.05 and both)
    stable = bool(all(min(res["perturbation"][c][f"spearman_{n}"] for _, n in FOVS) >= 0.8 for c in MILD))
    survives_p = bool(all(excludes_zero(b["adjusted_pooled"]) and np.sign(b["adjusted_pooled"]["coef"]) == sign
                          for b in res["perturbation"].values()))
    qa, qb = excludes_zero(res["quality_A"]), excludes_zero(res["quality_B"])
    return {"holm_p_pooled": holm_p, "density_independent_replicated": replicated,
            "significant_3mm": bool(excludes_zero(a["fov_3mm"])), "significant_6mm": bool(excludes_zero(a["fov_6mm"])),
            "rank_stable": stable, "survives_perturbation": survives_p,
            "survives_quality_A": bool(qa), "survives_quality_B": bool(qb),
            "robust_candidate": bool(replicated and stable and survives_p and qa and qb)}


# -------------------------------------------------------------------------------- data
def load_frames(cfg: dict, prox_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    table, _ = s0.load_table(cfg)
    keep = ["image_id", "y", "fov_mm", "age", "sex_male", "density", *METRICS]
    a = table[keep].copy()
    a["disease"] = np.where(a.y == 1, "DR", "NORMAL")
    m = pd.read_csv(ROOT / cfg["data"]["manifest"], dtype=str, keep_default_na=False)
    f = pd.read_csv(ROOT / qs.PIPE_OTHER / "features.csv", dtype={"image_id": str})
    o = m[m.image_id.isin(f.image_id)][["image_id", "disease", "fov_mm", "age", "sex"]].merge(f, on="image_id")
    o["fov_mm"] = o.fov_mm.astype(float)
    o["age"] = pd.to_numeric(o.age, errors="coerce")
    o["sex_male"] = o.sex.str.upper().map({"M": 1.0, "F": 0.0})
    o["density"] = o[s0.DENSITY_COL].astype(float)
    eyes = pd.concat([a.drop(columns="y"), o[["image_id", "disease", "fov_mm", "age", "sex_male", "density", *METRICS]]],
                     ignore_index=True)
    pa = pd.read_csv(ROOT / qs.OUT / "quality_proxies.csv", dtype={"image_id": str})
    eyes = eyes.merge(pa[["image_id", *qs.PROXIES]], on="image_id", how="left").merge(
        prox_b[["image_id", *PROXIES_B]], on="image_id", how="left")
    eyes["dr"] = (eyes.disease == "DR").astype(float)
    eyes["non_dr_disease"] = ((eyes.disease != "DR") & (eyes.disease != "NORMAL")).astype(float)
    eyes["fov_6mm"] = (eyes.fov_mm == 6.0).astype(float)
    is_cohort = eyes.disease.isin(["DR", "NORMAL"])
    ref = eyes[is_cohort]
    for q in (*qs.PROXIES, *PROXIES_B):
        eyes[q + "_z"] = zscore_within_fov(eyes, q, ref)
    for mm in METRICS:
        eyes[mm + "_raw"] = eyes[mm]
        eyes[mm] = zscore_within_fov(eyes, mm, ref)
    pert = {}
    for cond in sp.CONDITIONS:
        pf = pd.read_csv(ROOT / sp.OUT / cond / "pipeline" / "features.csv", dtype={"image_id": str})
        pert[cond] = pf.rename(columns={s0.DENSITY_COL: "density_p", **{mm: mm + "_p" for mm in METRICS}})[
            ["image_id", "density_p", *[mm + "_p" for mm in METRICS]]]
    return eyes[is_cohort].copy(), eyes, pert


def input_files(cfg: dict) -> dict[str, str]:
    files = [AMENDMENT, "config.yaml", cfg["data"]["manifest"], qs.CATALOG,
             f"{s0.SRC_RUN}/pipeline/features.csv", f"{qs.PIPE_OTHER}/features.csv",
             f"{qs.OUT}/quality_proxies.csv", f"{qs.OUT}/results.json", f"{sp.OUT}/results.json",
             *[f"{sp.OUT}/{c}/pipeline/features.csv" for c in sp.CONDITIONS]]
    return {f: sha256_file(ROOT / f) for f in files}


def run(cfg: dict) -> tuple[dict, pd.DataFrame]:
    base_seed = cfg["project"]["base_seed"]
    out = ROOT / OUT
    out.mkdir(parents=True, exist_ok=True)
    pb = compute_proxies_b()
    pb.to_csv(out / "quality_proxies_b.csv", index=False)
    cohort, allf, pert = load_frames(cfg, pb)
    res: dict = {"analysis_status": s0.STATUS_LABEL, "human_qc": "20-graph review 2026-11-16: 3 mm 10/10 pass; 6 mm 4/10 fail",
                 "n_cohort": int(len(cohort)), "n_all": int(len(allf)),
                 "proxies_b_group_difference": {p: qs.group_difference(cohort, p) for p in PROXIES_B},
                 "metrics": {}}
    for m in METRICS:
        pm = {c: pf[["image_id", "density_p", m + "_p"]] for c, pf in pert.items()}
        res["metrics"][m] = audit_metric(m, cohort, allf, pm, base_seed)
    adj = holm({m: res["metrics"][m]["adjusted"]["pooled"].get("p") for m in METRICS})
    rows = []
    for m in METRICS:
        r = res["metrics"][m]
        r["classification"] = classify(r, adj[m])
        a, d = r["adjusted"], r["added_discrimination"]
        rows.append({"metric": m, "coef_pooled_sd": a["pooled"].get("coef"), "ci_lo": a["pooled"].get("ci", [None, None])[0],
                     "ci_hi": a["pooled"].get("ci", [None, None])[1], "holm_p": adj[m],
                     "coef_3mm": a["fov_3mm"].get("coef"), "coef_6mm": a["fov_6mm"].get("coef"),
                     "cv_auroc_delta": d["cv_auroc_delta_mean"], "lrt_p": d.get("lrt_p"),
                     "coef_quality_A": r["quality_A"].get("coef"), "coef_quality_B": r["quality_B"].get("coef"),
                     "dr_minus_non_dr": r["specificity"].get("dr_minus_non_dr", {}).get("coef"),
                     **{f"spearman_{c}_3mm": r["perturbation"][c]["spearman_fov_3mm"] for c in sp.CONDITIONS},
                     **{f"spearman_{c}_6mm": r["perturbation"][c]["spearman_fov_6mm"] for c in sp.CONDITIONS},
                     **r["classification"]})
    return res, pd.DataFrame(rows)


def text(res: dict, tab: pd.DataFrame) -> str:
    L = ["AUDIT OF OCTA GRAPH METRICS (post-outcome exploratory)", res["human_qc"], "",
         "DR minus healthy at equal vessel length density, in SD of the metric (pooled; 3 mm; 6 mm),",
         "change in cross-validated AUROC when the metric is added to density, and worst rank agreement",
         "under the three mild label perturbations:", ""]
    for r in tab.itertuples():
        worst = min(getattr(r, f"spearman_{c}_{f}") for c in MILD for f in ("3mm", "6mm"))
        hp = "n/a" if r.holm_p is None or pd.isna(r.holm_p) else f"{r.holm_p:.3g}"
        L.append(f"  {r.metric:32s} {r.coef_pooled_sd:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}] Holm p {hp}; "
                 f"3mm {r.coef_3mm:+.2f}, 6mm {r.coef_6mm:+.2f}; dAUROC {r.cv_auroc_delta:+.3f}; worst rho {worst:.2f}")
    L += ["", "Classification (amendment rules):"]
    for r in tab.itertuples():
        tags = [k for k in ("density_independent_replicated", "rank_stable", "survives_perturbation",
                            "survives_quality_A", "survives_quality_B") if getattr(r, k)]
        L.append(f"  {r.metric:32s} {'ROBUST CANDIDATE' if r.robust_candidate else ''} {', '.join(tags) or 'none'}")
    L += ["", "Second proxy set, DR versus healthy AUROC (0.5 = unrelated to disease):"]
    for p, g in res["proxies_b_group_difference"].items():
        L.append(f"  {p:12s} " + "  ".join(f"{s}: {g[s]['auroc_dr_higher']:.2f}" for s in ("fov_3mm", "fov_6mm")
                                             if "auroc_dr_higher" in g[s]))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record")
    sub.add_parser("run")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    cfg = load_config(ROOT / "config.yaml")
    if args.cmd == "record":
        if (ROOT / RECORD).exists() or (ROOT / OUT).exists():
            raise AuditError("record or outputs already exist; the record is written once, first")
        write_json(ROOT / RECORD, {
            "adopted_utc": now_utc(), "decided_by": "",
            "instructed_by": "user, in session, 2026-11-16", "is_human_review": False,
            "analysis_status": s0.STATUS_LABEL, "audit_results_available_at_adoption": False,
            "script_sha256_at_adoption": sha256_file(Path(__file__)), "files_sha256": input_files(cfg)})
        print(json.dumps(read_json(ROOT / RECORD), indent=1)[:500])
        return 0
    if not (ROOT / RECORD).exists():
        raise AuditError("no dated record; run 'record' first")
    rec = read_json(ROOT / RECORD)
    bad = [f for f, h in rec["files_sha256"].items() if input_files(cfg).get(f) != h]
    if bad:
        raise AuditError("inputs changed since the dated record: " + ", ".join(bad))
    res, tab = run(cfg)
    res["provenance"] = {"created_utc": now_utc(), "performed_by": "",
                         "amendment": AMENDMENT, "record": RECORD, "script_sha256": sha256_file(Path(__file__)),
                         "environment": environment(), "inputs_sha256": input_files(cfg)}
    write_json(ROOT / OUT / "results.json", res)
    tab.to_csv(ROOT / OUT / "audit_table.csv", index=False)
    (ROOT / OUT / "summary.txt").write_text(text(res, tab))
    print(text(res, tab))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AuditError, s0.Step0Error) as exc:
        sys.exit(f"REFUSED: {exc}")
