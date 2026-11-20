"""What the end-node difference is made of, and whether it survives another vessel tracing.

Post-outcome exploratory. Authority and fixed readings:
``docs/ENDNODE_MECHANISM_AMENDMENT_2026-11-16.md``. Reads saved pipeline
outputs; the only new pipeline runs are on alternative label sources, through
the unmodified CLI.

Subcommands::

    record      dated machine record; once only
    ids         CSV (image_id) of the analysis eyes, for the Frangi segmenter
    manifests   derived manifests for the Frangi and held-out U-Net label sources
    m123        stage decomposition, location split, added value beyond the baseline
    m4          label-source agreement and the audit model on each source
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
from retina_fragility.io_masks import read_mask  # noqa: E402
from retina_fragility.provenance import environment, now_utc, read_json, sha256_file, write_json  # noqa: E402
from retina_fragility.seeds import derive_seed  # noqa: E402
from scripts import metric_audit as ma  # noqa: E402
from scripts import step0_null_controls as s0  # noqa: E402
from scripts import step0_quality_specificity as qs  # noqa: E402

AMENDMENT = "docs/ENDNODE_MECHANISM_AMENDMENT_2026-11-16.md"
RECORD = "reports/endnode_mechanism_amendment_2026-11-16.json"
OUT = "outputs/exploratory_20261116_step0/endnode_mechanism"
SOURCES = "outputs/exploratory_20261116_step0/label_sources"
UNET_MASKS = "outputs/unet/octa500_v1/heldout_octa500/masks"
UNET_META = "outputs/unet/octa500_v1/model/meta.json"
STAGE_PARTS = ("end_raw", "end_border", "end_faz", "end_spur", "end_fragment", "end_final")
LOCATION_PARTS = ("end_loc_border", "end_loc_perifaz", "end_loc_interior")
BAND_PX = 5
BASELINE = ["fov_6mm", "age", "sex_male", "vessel_density", "faz_area_mm2", "loop_density_per_mm2"]
M3_METRICS = ("end_density_per_mm2_pre", "end_density_per_mm2_post", "end_loc_interior", "FI_random",
              "FI_adaptive_betweenness")


class MechError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- M1 / M2
def stage_parts(audit: list[dict]) -> dict:
    """End-node counts by cleanup stage from one eye's cleanup log."""
    e = {s["stage"]: int(s["n_endpoints"]) for s in audit}
    need = ("0_raw", "1_border", "2_faz", "3_contract", "4_spurs", "5_contract", "6_fragments", "7_final")
    missing = [k for k in need if k not in e]
    if missing:
        raise MechError(f"cleanup log lacks stages {missing}")
    return {"end_raw": e["0_raw"], "end_border": e["0_raw"] - e["1_border"], "end_faz": e["1_border"] - e["2_faz"],
            "end_spur": e["3_contract"] - e["4_spurs"], "end_fragment": e["5_contract"] - e["6_fragments"],
            "end_final": e["7_final"]}


def location_parts(nodes: list[dict], edges: list[dict], shape, faz: np.ndarray | None, band: int = BAND_PX) -> dict:
    """Degree-one nodes of a cleaned graph by location: border band, peri-FAZ band, interior."""
    deg: dict[int, int] = {}
    for ed in edges:
        deg[ed["u"]] = deg.get(ed["u"], 0) + 1
        deg[ed["v"]] = deg.get(ed["v"], 0) + 1
    H, W = shape
    dist = ndi.distance_transform_edt(~np.asarray(faz, bool)) if faz is not None and np.any(faz) else None
    out = dict.fromkeys(LOCATION_PARTS, 0)
    for nd in nodes:
        if deg.get(nd["id"], 0) != 1:
            continue
        y, x = float(nd["y"]), float(nd["x"])
        if min(y, x, H - 1 - y, W - 1 - x) <= band:
            out["end_loc_border"] += 1
        elif dist is not None and dist[int(round(min(max(y, 0), H - 1))), int(round(min(max(x, 0), W - 1)))] <= band:
            out["end_loc_perifaz"] += 1
        else:
            out["end_loc_interior"] += 1
    return out


def collect_parts(cfg: dict) -> pd.DataFrame:
    table, _ = s0.load_table(cfg)
    man = pd.read_csv(ROOT / cfg["data"]["manifest"], dtype=str, keep_default_na=False).set_index("image_id")
    proot = ROOT / s0.SRC_RUN / "pipeline"
    rows = []
    for r in table.itertuples():
        d = s0.graph_path(proot, r.image_id).parent
        area = float(r.analysis_area_mm2)
        sp_ = stage_parts(read_json(d / "cleanup_audit.json"))
        g = json.loads((d / "graph.json").read_text())
        faz = read_mask(ROOT / man.loc[r.image_id, "faz_mask_path"], cfg["data"]["label_maps"]["faz"])
        lp = location_parts(g["nodes"], g["edges"], g["shape"], faz)
        rows.append({"image_id": r.image_id, **{k: v / area for k, v in sp_.items()},
                     **{k: v / area for k, v in lp.items()}, **{f"n_{k}": v for k, v in sp_.items()},
                     "location_minus_logged_final": int(sum(lp.values()) - sp_["end_final"])})
    return pd.DataFrame(rows)


def added_value(df: pd.DataFrame, metric: str, base_seed: int) -> dict:
    """Nested LRT and repeated-CV AUROC change for adding ``metric`` to the preregistered baseline."""
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    d = df[["dr", *BASELINE, metric]].astype(float).dropna()
    out: dict = {"n": int(len(d)), "n_dr": int(d.dr.sum())}
    Z = d[[*BASELINE, metric]].copy()
    for c in Z.columns:
        if c not in ("fov_6mm", "sex_male"):
            Z[c] = (Z[c] - Z[c].mean()) / Z[c].std()
    try:
        fb = sm.Logit(d.dr, sm.add_constant(Z[BASELINE])).fit(disp=0, maxiter=200)
        ff = sm.Logit(d.dr, sm.add_constant(Z)).fit(disp=0, maxiter=200)
        if not (fb.mle_retvals["converged"] and ff.mle_retvals["converged"]):
            raise RuntimeError("not converged")
        lr = 2 * (ff.llf - fb.llf)
        ci = ff.conf_int().loc[metric]
        out.update(lrt_statistic=float(lr), lrt_p=float(sps.chi2.sf(lr, 1)),
                   odds_ratio_per_sd=float(np.exp(ff.params[metric])),
                   odds_ratio_ci=[float(np.exp(ci[0])), float(np.exp(ci[1]))])
    except Exception as exc:  # separation or non-convergence is a result, not hidden
        out.update(lrt_p=None, lrt_failure=f"{type(exc).__name__}: {exc}"[:200])
    Xb, Xf, y = Z[BASELINE].to_numpy(), Z.to_numpy(), d.dr.to_numpy()
    base, full, delta = [], [], []
    for rep in range(10):
        skf = StratifiedKFold(5, shuffle=True, random_state=derive_seed(base_seed, "mech_cv", rep) % (2 ** 31))
        pb, pf = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in skf.split(Xb, y):
            pb[te] = LogisticRegression(C=1.0, max_iter=5000).fit(Xb[tr], y[tr]).predict_proba(Xb[te])[:, 1]
            pf[te] = LogisticRegression(C=1.0, max_iter=5000).fit(Xf[tr], y[tr]).predict_proba(Xf[te])[:, 1]
        a, b = roc_auc_score(y, pb), roc_auc_score(y, pf)
        base.append(a), full.append(b), delta.append(b - a)
    out.update(cv_auroc_base=float(np.mean(base)), cv_auroc_full=float(np.mean(full)),
               cv_auroc_delta_mean=float(np.mean(delta)),
               cv_auroc_delta_range=[float(np.min(delta)), float(np.max(delta))],
               cv_note="L2 logistic, C = 1, as in the primary analysis's prediction check")
    return out


def adjusted_block(df: pd.DataFrame, m: str) -> dict:
    out = {"pooled": ma.coef(df, m, ["dr", "density", "fov_6mm"])}
    for fov, name in ma.FOVS:
        out[name] = ma.coef(df[df.fov_mm == fov], m, ["dr", "density"])
    return out


def run_m123(cfg: dict) -> dict:
    base_seed = cfg["project"]["base_seed"]
    out = ROOT / OUT
    out.mkdir(parents=True, exist_ok=True)
    parts = collect_parts(cfg)
    parts.to_csv(out / "end_node_parts.csv", index=False)
    table, _ = s0.load_table(cfg)
    df = table[["image_id", "y", "fov_mm", "age", "sex_male", "density", "vessel_density", "faz_area_mm2",
                "loop_density_per_mm2", "end_density_per_mm2_pre", "end_density_per_mm2_post", "FI_random",
                "FI_adaptive_betweenness"]].merge(parts, on="image_id")
    df["dr"] = df.y.astype(float)
    df["fov_6mm"] = (df.fov_mm == 6.0).astype(float)
    chk = float((df.end_raw - df.end_density_per_mm2_pre).abs().max())
    res: dict = {"analysis_status": s0.STATUS_LABEL, "n": int(len(df)),
                 "consistency_end_raw_vs_feature_max_abs_diff": chk,
                 "consistency_location_vs_logged_final_max_abs_diff": int(df.location_minus_logged_final.abs().max()),
                 "M1": {}, "M2": {}, "M3": {}}
    cnt = df.assign(group=np.where(df.dr == 1, "dr", "healthy"))
    shares = {}
    for (fov, grp), s in cnt.groupby(["fov_mm", "group"]):
        tot = s.n_end_raw.sum()
        shares[f"fov_{int(fov)}mm|{grp}"] = {
            "median_raw_per_mm2": float(s.end_raw.median()),
            **{k: float(s[f"n_{k}"].sum() / tot) for k in STAGE_PARTS if k != "end_raw"}}
    res["M1"]["share_of_raw_end_nodes"] = shares
    z = df.copy()
    for m in (*STAGE_PARTS, *LOCATION_PARTS):
        z[m] = ma.zscore_within_fov(df, m)
    for m in STAGE_PARTS:
        res["M1"][m] = adjusted_block(z, m)
    for m in LOCATION_PARTS:
        res["M2"][m] = adjusted_block(z, m)
    for m in M3_METRICS:
        res["M3"][m] = added_value(df, m, base_seed)

    def ex(c):
        return ma.excludes_zero(c)

    interior = res["M2"]["end_loc_interior"]
    res["readings"] = {
        "texture_not_terminations": bool((ex(res["M1"]["end_spur"]["pooled"]) or ex(res["M1"]["end_fragment"]["pooled"]))
                                         and not ex(interior["pooled"])),
        "dangling_ends": bool(all(ex(interior[n]) for _, n in ma.FOVS)
                              and len({np.sign(interior[n]["coef"]) for _, n in ma.FOVS}) == 1)}
    return res


# -------------------------------------------------------------------------------- M4
def analysis_ids(cfg: dict) -> list[str]:
    table, _ = s0.load_table(cfg)
    return sorted(table.image_id.astype(str))


def make_manifests(cfg: dict) -> dict:
    raw0 = pd.read_csv(ROOT / cfg["data"]["manifest"], dtype=str, keep_default_na=False)
    ids = set(analysis_ids(cfg))
    trained = set(read_json(ROOT / UNET_META)["train_image_ids"])
    info = {}
    for name, mask_dir, keep in (("frangi", f"{SOURCES}/frangi/seg/masks", ids),
                                 ("unet_heldout", UNET_MASKS, ids - trained)):
        raw = raw0.copy()
        used = []
        for i, r in raw.iterrows():
            if r.image_id not in keep:
                continue
            mp = ROOT / mask_dir / f"{r.image_id}.png"
            if not mp.exists():
                continue
            raw.at[i, "vessel_mask_path"] = str(mp.relative_to(ROOT))
            raw.at[i, "large_vessel_mask_path"] = ""
            for c in ("vessel_mask_sha256", "large_vessel_mask_sha256"):
                if c in raw.columns:
                    raw.at[i, c] = sha256_file(mp) if c.startswith("vessel") else ""
            used.append(r.image_id)
        d = ROOT / SOURCES / name
        d.mkdir(parents=True, exist_ok=True)
        raw.to_csv(d / "manifest.csv", index=False)
        (d / "ids.txt").write_text(",".join(sorted(used)))
        info[name] = {"n": len(used), "n_requested": len(keep), "mask_dir": mask_dir,
                      "trained_on_these_eyes": False if name == "unet_heldout" else None}
        write_json(d / "manifest_info.json", {**info[name], "created_utc": now_utc(),
                                              "note": "alternative vessel mask only; reference FAZ kept; large-vessel label not added"})
    return info


def run_m4(cfg: dict) -> dict:
    table, _ = s0.load_table(cfg)
    ref = table[["image_id", "y", "fov_mm", "density", *ma.METRICS]].copy()
    ref["dr"] = ref.y.astype(float)
    ref["fov_6mm"] = (ref.fov_mm == 6.0).astype(float)
    metrics = [m for m in ma.METRICS if m != "faz_area_mm2"]
    res: dict = {"analysis_status": s0.STATUS_LABEL, "sources": {}}
    for name in ("frangi", "unet_heldout"):
        fp = ROOT / SOURCES / name / "pipeline" / "features.csv"
        if not fp.exists():
            res["sources"][name] = {"status": "pending"}
            continue
        f = pd.read_csv(fp, dtype={"image_id": str})
        alt = f.rename(columns={s0.DENSITY_COL: "density_alt", **{m: m + "_alt" for m in metrics}})[
            ["image_id", "density_alt", *[m + "_alt" for m in metrics]]]
        j = ref.merge(alt, on="image_id", how="inner")
        n_req = int(read_json(ROOT / SOURCES / name / "manifest_info.json")["n"])
        blk: dict = {"status": "ok", "n": int(len(j)), "n_dr": int(j.dr.sum()), "n_missing": n_req - int(len(j)),
                     "density_ratio_median": {n: float((j.density_alt / j.density)[j.fov_mm == f_].median())
                                              for f_, n in ma.FOVS},
                     "agreement": {}, "audit_model": {}}
        for m in ["density", *metrics]:
            blk["agreement"][m] = {n: float(sps.spearmanr(j.loc[j.fov_mm == f_, m], j.loc[j.fov_mm == f_, m + "_alt"]).statistic)
                                   for f_, n in ma.FOVS}
        k = j.assign(density=j.density_alt)
        for m in metrics:
            k[m] = ma.zscore_within_fov(j, m + "_alt")
            blk["audit_model"][m] = adjusted_block(k, m)
        res["sources"][name] = blk
    fr = res["sources"].get("frangi", {})
    if fr.get("status") == "ok":
        c = fr["audit_model"]["end_density_per_mm2_pre"]["pooled"]
        res["readings"] = {"end_node_pre_frangi": c,
                           "holds_with_independent_tracing": bool(ma.excludes_zero(c) and c["coef"] < 0),
                           "label_specific": bool(not ma.excludes_zero(c) or c["coef"] > 0)}
    return res


# -------------------------------------------------------------------------------- text
def _c(c: dict) -> str:
    return "n/a" if "coef" not in c else f"{c['coef']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]"


def text_m123(res: dict) -> str:
    L = ["WHAT THE END-NODE DIFFERENCE IS MADE OF (post-outcome exploratory)", "",
         "M1. Share of raw end nodes by what removes them (border, FAZ, spur, fragment) or keeps them (final):"]
    for k, v in res["M1"]["share_of_raw_end_nodes"].items():
        L.append(f"  {k:18s} raw {v['median_raw_per_mm2']:.0f}/mm2: border {v['end_border']:.0%}, faz {v['end_faz']:.0%}, "
                 f"spur {v['end_spur']:.0%}, fragment {v['end_fragment']:.0%}, final {v['end_final']:.0%}")
    L += ["", "DR minus healthy at equal density (SD): pooled; 3 mm; 6 mm"]
    for sec, parts in (("M1", STAGE_PARTS), ("M2", LOCATION_PARTS)):
        for m in parts:
            b = res[sec][m]
            L.append(f"  {sec} {m:18s} {_c(b['pooled'])}; {_c(b['fov_3mm'])}; {_c(b['fov_6mm'])}")
    L += ["", "M3. Added to the preregistered baseline (FOV, age, sex, vessel density, FAZ area, loop density):"]
    for m, v in res["M3"].items():
        lp = "not estimable" if v.get("lrt_p") is None else f"LRT p = {v['lrt_p']:.3g}, OR/SD {v['odds_ratio_per_sd']:.2f}"
        L.append(f"  {m:28s} {lp}; CV AUROC {v['cv_auroc_base']:.3f} -> {v['cv_auroc_full']:.3f} "
                 f"({v['cv_auroc_delta_mean']:+.3f})")
    L += ["", "READINGS: " + json.dumps(res["readings"])]
    return "\n".join(L) + "\n"


def text_m4(res: dict) -> str:
    L = ["INDEPENDENT LABEL SOURCES (post-outcome exploratory; graphs not reviewed by a human)", ""]
    for name, b in res["sources"].items():
        if b.get("status") != "ok":
            L.append(f"{name}: {b.get('status')}")
            continue
        L.append(f"{name}: n = {b['n']} ({b['n_dr']} DR); density relative to reference labels "
                 f"x{b['density_ratio_median']['fov_3mm']:.2f} (3 mm), x{b['density_ratio_median']['fov_6mm']:.2f} (6 mm)")
        L.append("  metric: rank agreement with reference labels (3 mm, 6 mm) | DR minus healthy at equal density (SD): pooled; 3 mm; 6 mm")
        for m in ["density", *b["audit_model"].keys()]:
            a = b["agreement"][m]
            mod = b["audit_model"].get(m)
            tail = "" if mod is None else f" | {_c(mod['pooled'])}; {_c(mod['fov_3mm'])}; {_c(mod['fov_6mm'])}"
            L.append(f"  {m:32s} {a['fov_3mm']:.2f}, {a['fov_6mm']:.2f}{tail}")
        L.append("")
    if "readings" in res:
        r = res["readings"]
        L.append(f"READING: holds with an independent tracing = {r['holds_with_independent_tracing']}; "
                 f"label-specific = {r['label_specific']}")
    return "\n".join(L) + "\n"


def input_files(cfg: dict) -> dict[str, str]:
    files = [AMENDMENT, "config.yaml", cfg["data"]["manifest"], f"{s0.SRC_RUN}/pipeline/features.csv",
             f"{ma.OUT}/results.json", UNET_META]
    return {f: sha256_file(ROOT / f) for f in files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("record", "ids", "manifests", "m123", "m4"):
        sub.add_parser(c)
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    cfg = load_config(ROOT / "config.yaml")
    if args.cmd == "record":
        if (ROOT / RECORD).exists() or (ROOT / OUT).exists() or (ROOT / SOURCES).exists():
            raise MechError("record or outputs already exist; the record is written once, first")
        write_json(ROOT / RECORD, {
            "adopted_utc": now_utc(), "decided_by": "",
            "instructed_by": "user, in session, 2026-11-16", "is_human_review": False,
            "analysis_status": s0.STATUS_LABEL, "results_available_at_adoption": False,
            "one_cleanup_log_opened_for_format": "OCTA500_10001",
            "script_sha256_at_adoption": sha256_file(Path(__file__)), "files_sha256": input_files(cfg)})
        print(json.dumps(read_json(ROOT / RECORD), indent=1)[:400])
        return 0
    rec = read_json(ROOT / RECORD) if (ROOT / RECORD).exists() else None
    if rec is None:
        raise MechError("no dated record; run 'record' first")
    bad = [f for f, h in rec["files_sha256"].items() if input_files(cfg).get(f) != h]
    if bad:
        raise MechError("inputs changed since the dated record: " + ", ".join(bad))
    prov = {"created_utc": now_utc(), "performed_by": "", "amendment": AMENDMENT,
            "record": RECORD, "script_sha256": sha256_file(Path(__file__)), "environment": environment()}
    if args.cmd == "ids":
        print("image_id")
        print("\n".join(analysis_ids(cfg)))
    elif args.cmd == "manifests":
        print(json.dumps(make_manifests(cfg), indent=1))
    elif args.cmd == "m123":
        res = run_m123(cfg)
        res["provenance"] = prov
        write_json(ROOT / OUT / "m123_results.json", res)
        (ROOT / OUT / "m123_summary.txt").write_text(text_m123(res))
        print(text_m123(res))
    elif args.cmd == "m4":
        res = run_m4(cfg)
        res["provenance"] = prov
        (ROOT / OUT).mkdir(parents=True, exist_ok=True)
        write_json(ROOT / OUT / "m4_results.json", res)
        (ROOT / OUT / "m4_summary.txt").write_text(text_m4(res))
        print(text_m4(res))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (MechError, s0.Step0Error) as exc:
        sys.exit(f"REFUSED: {exc}")
