"""Image-degradation experiment: does degrading a healthy scan move it along the healthy curve?

Post-outcome exploratory. Authority and fixed readings:
``docs/DEGRADATION_EXPERIMENT_AMENDMENT_2026-11-16.md``. Uses only eyes the
U-Net never trained on. Modifies nothing in the package, configuration or
saved results.

Subcommands::

    record    dated machine record; once only
    run       trace every version of every held-out eye; write versions.csv
    analyse   T1-T3 and the overall reading
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
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retina_fragility.config import load_config  # noqa: E402
from retina_fragility.io_masks import read_intensity, read_mask  # noqa: E402
from retina_fragility.provenance import environment, now_utc, read_json, sha256_file, write_json  # noqa: E402
from retina_fragility.seeds import rng_for  # noqa: E402
from scripts import endnode_mechanism as em  # noqa: E402
from scripts import endnode_null as en  # noqa: E402
from scripts import step0_null_controls as s0  # noqa: E402

AMENDMENT = "docs/DEGRADATION_EXPERIMENT_AMENDMENT_2026-11-16.md"
RECORD = "reports/degradation_experiment_amendment_2026-11-16.json"
OUT = "outputs/exploratory_20261116_step0/degradation"
MODEL_DIR = "outputs/unet/octa500_v1/model"
LEVELS = {"blur": (0.5, 1.0, 1.5), "noise": (0.05, 0.10, 0.15), "dim": (0.8, 0.6, 0.4)}
CALIPER = 0.05
MIN_MATCHED_EYES = 3


class DegradeError(RuntimeError):
    pass


def degrade(img: np.ndarray, kind: str, value: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """One degradation of an image with intensities in [0, 1]."""
    a = np.asarray(img, np.float32)
    if kind == "original":
        return a
    if kind == "blur":
        return gaussian_filter(a, float(value)).astype(np.float32)
    if kind == "noise":
        if rng is None:
            raise ValueError("noise needs an rng")
        return np.clip(a + rng.normal(0.0, float(value), a.shape), 0.0, 1.0).astype(np.float32)
    if kind == "dim":
        return (a * float(value)).astype(np.float32)
    raise ValueError(f"unknown degradation {kind}")


def versions() -> list[tuple[str, int, float]]:
    out = [("original", 0, 0.0)]
    for kind, vals in LEVELS.items():
        out += [(kind, i + 1, float(v)) for i, v in enumerate(vals)]
    return out


def measure(mask: np.ndarray, faz: np.ndarray, px_um: float, cfg: dict) -> dict:
    """Density and end-node measures of one traced mask (pipeline graph builder, frozen config)."""
    from retina_fragility.pipeline import build_graph

    b = build_graph(mask, faz, px_um, cfg)
    area = float(b["analysis"].sum()) * (px_um / 1000.0) ** 2
    clean = b["clean"]
    length_mm = clean.total_length() / 1000.0
    ends = en.end_counts(clean, en.faz_distance(faz))
    raw_ends = int(b["audit"][0]["n_endpoints"])
    return {"area_mm2": area, "density": length_mm / area, "n_edges": int(clean.n_edges),
            "end_final_per_mm2": ends["total"] / area, "end_interior_per_mm2": ends["interior"] / area,
            "end_raw_per_mm2": raw_ends / area,
            "ends_per_mm": ends["interior"] / length_mm if length_mm > 0 else np.nan,
            "raw_ends_per_mm": raw_ends / length_mm if length_mm > 0 else np.nan,
            "vessel_fraction": float(np.asarray(mask, bool).mean())}


def heldout_rows(cfg: dict) -> pd.DataFrame:
    table, _ = s0.load_table(cfg)
    trained = set(read_json(ROOT / MODEL_DIR / "meta.json")["train_image_ids"])
    man = pd.read_csv(ROOT / cfg["data"]["manifest"], dtype=str, keep_default_na=False).set_index("image_id")
    t = table[~table.image_id.isin(trained)][["image_id", "y", "fov_mm", "group_key"]].copy()
    t["image_path"] = t.image_id.map(man.image_path)
    t["faz_mask_path"] = t.image_id.map(man.faz_mask_path)
    t["pixel_size_um"] = t.image_id.map(man.pixel_size_um).astype(float)
    return t.reset_index(drop=True)


def run(cfg: dict) -> pd.DataFrame:
    from retina_fragility.segmentation import unet as unet_mod

    net, meta = unet_mod.load_model(ROOT / MODEL_DIR)
    ucfg = cfg["segmentation"]["unet"]
    rows = heldout_rows(cfg)
    unet_mod.check_leakage(meta, rows.image_id, rows.group_key.astype(str))
    base_seed = cfg["project"]["base_seed"]
    out = []
    for r in rows.itertuples():
        img, _ = read_intensity(ROOT / r.image_path)
        faz = read_mask(ROOT / r.faz_mask_path, cfg["data"]["label_maps"]["faz"])
        for kind, level, value in versions():
            x = degrade(img, kind, value, rng_for(base_seed, "degrade", r.image_id, kind, level))
            mask = unet_mod.segment(unet_mod.predict_prob(net, meta, x), ucfg)
            out.append({"image_id": r.image_id, "y": int(r.y), "fov_mm": float(r.fov_mm), "kind": kind,
                        "level": level, "value": value, **measure(mask, faz, r.pixel_size_um, cfg)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- analyse
def matched_difference(targets: pd.DataFrame, pool: pd.DataFrame, col: str) -> pd.DataFrame:
    """Target value minus the mean of other eyes' pool versions of the same FOV within the density caliper.

    Pool values are averaged within eye first, so each matched eye counts once.
    """
    rows = []
    for t in targets.itertuples():
        p = pool[(pool.fov_mm == t.fov_mm) & (pool.image_id != t.image_id)
                 & ((pool.density / t.density - 1.0).abs() <= CALIPER) & np.isfinite(pool[col])]
        per_eye = p.groupby("image_id")[col].mean()
        rows.append({"image_id": t.image_id, "fov_mm": t.fov_mm, "density": t.density, "value": getattr(t, col),
                     "n_matched_eyes": int(len(per_eye)),
                     "d": float(getattr(t, col) - per_eye.mean()) if len(per_eye) >= MIN_MATCHED_EYES else np.nan})
    return pd.DataFrame(rows)


def summarise_d(m: pd.DataFrame, rng: np.random.Generator) -> dict:
    d = m.d.to_numpy(float)
    d = d[np.isfinite(d)]
    out = {"n_targets": int(len(m)), "n_with_matches": int(d.size), **{k: v for k, v in s0.median_ci(d, rng).items() if k != "n"}}
    if d.size >= 1 and np.any(d != 0):
        w = sps.wilcoxon(d, alternative="two-sided")
        out.update(wilcoxon_p_two_sided=float(w.pvalue))
    else:
        out.update(wilcoxon_p_two_sided=None)
    return out


def analyse(v: pd.DataFrame, cfg: dict, reference_check: dict | None = None) -> dict:
    base_seed = cfg["project"]["base_seed"]
    orig = v[v.kind == "original"]
    healthy0, dr0 = orig[orig.y == 0], orig[orig.y == 1]
    deg_h = v[(v.kind != "original") & (v.y == 0)]
    res: dict = {"analysis_status": s0.STATUS_LABEL, "n_eyes": {"healthy": int(len(healthy0)), "dr": int(len(dr0))},
                 "reference_check": reference_check, "T1": {}, "T2": {}, "T3": {}}
    sd_h = {f: float(healthy0[healthy0.fov_mm == f].ends_per_mm.std()) for f in sorted(healthy0.fov_mm.unique())}
    res["healthy_sd_of_loose_ends_per_mm"] = {f"fov_{int(f)}mm": s for f, s in sd_h.items()}
    passes = {}
    for kind, vals in LEVELS.items():
        top = v[(v.kind == kind) & (v.level == len(vals)) & (v.y == 0)].set_index("image_id")
        j = healthy0.set_index("image_id").join(top, rsuffix="_deg", how="inner")
        ratio = j.density_deg / j.density
        diff = j.ends_per_mm_deg - j.ends_per_mm
        rdiff = j.raw_ends_per_mm_deg - j.raw_ends_per_mm
        pw = lambda x, alt: float(sps.wilcoxon(x, alternative=alt).pvalue) if np.any(x != 0) else None  # noqa: E731
        res["T1"][kind] = {"strongest_level_value": float(vals[-1]), "n": int(len(j)),
                           "median_density_ratio": float(ratio.median()),
                           "p_density_falls": pw((ratio - 1).to_numpy(), "less"),
                           "median_change_loose_ends_per_mm": float(diff.median()),
                           "p_loose_ends_rise": pw(diff.to_numpy(), "greater"),
                           "median_change_raw_ends_per_mm": float(rdiff.median()),
                           "by_level_median_density_ratio": {
                               int(l): float((v[(v.kind == kind) & (v.level == l) & (v.y == 0)].set_index("image_id").density
                                              / healthy0.set_index("image_id").density).median()) for l in (1, 2, 3)}}
        t = res["T1"][kind]
        passes[kind] = bool(t["median_density_ratio"] < 1 and t["median_change_loose_ends_per_mm"] > 0
                            and (t["p_density_falls"] or 1) < 0.05 and (t["p_loose_ends_rise"] or 1) < 0.05)
        res["T1"][kind]["passes"] = passes[kind]
    for col in ("ends_per_mm", "raw_ends_per_mm"):
        m2 = matched_difference(healthy0, deg_h, col)
        m3 = matched_difference(dr0, deg_h, col)
        res["T2"][col] = {"pooled": summarise_d(m2, rng_for(base_seed, "degrade_boot", "T2", col)),
                          **{f"fov_{int(f)}mm": summarise_d(m2[m2.fov_mm == f], rng_for(base_seed, "degrade_boot", "T2", col, f))
                             for f in sorted(m2.fov_mm.unique())}}
        res["T3"][col] = {"pooled": summarise_d(m3, rng_for(base_seed, "degrade_boot", "T3", col)),
                          **{f"fov_{int(f)}mm": summarise_d(m3[m3.fov_mm == f], rng_for(base_seed, "degrade_boot", "T3", col, f))
                             for f in sorted(m3.fov_mm.unique())}}
        if col == "ends_per_mm":
            m2["sd"] = m2.fov_mm.map(sd_h)
            res["T2"]["median_in_healthy_sd"] = float((m2.d / m2.sd).median())
    t2, t3 = res["T2"]["ends_per_mm"]["pooled"], res["T3"]["ends_per_mm"]["pooled"]
    ci = t2.get("ci")
    on_curve = bool((ci is not None and ci[0] <= 0 <= ci[1]) or abs(res["T2"]["median_in_healthy_sd"]) <= 0.25)
    below = bool(t3.get("median") is not None and t3["median"] < 0 and (t3.get("wilcoxon_p_two_sided") or 1) < 0.05)
    n_pass = sum(passes.values())
    res["readings"] = {"T1_types_passing": [k for k, p in passes.items() if p], "T2_on_the_curve": on_curve,
                       "T3_dr_below_degraded_healthy": below,
                       "overall": ("supported" if n_pass >= 2 and on_curve and below
                                   else "not_supported" if n_pass <= 1 or not below else "mixed")}
    return res


def text(res: dict) -> str:
    def s(c):
        if c.get("median") is None:
            return f"n/a ({c['n_with_matches']} of {c['n_targets']} matched)"
        return (f"{c['median']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}], {c['n_with_matches']} of {c['n_targets']} matched, "
                f"p = {s0._fmt(c.get('wilcoxon_p_two_sided'), 4)}")

    L = ["IMAGE-DEGRADATION EXPERIMENT (post-outcome exploratory)", "",
         f"Held-out eyes: {res['n_eyes']}; reference check: {json.dumps(res['reference_check'])}", "",
         "T1. Strongest level of each degradation, healthy eyes, against the same eye's original:"]
    for k, t in res["T1"].items():
        L.append(f"   {k:6s} density x{t['median_density_ratio']:.3f} (p {s0._fmt(t['p_density_falls'], 4)}); "
                 f"loose ends per mm {t['median_change_loose_ends_per_mm']:+.3f} (p {s0._fmt(t['p_loose_ends_rise'], 4)}); "
                 f"passes: {t['passes']}")
    L += ["", "T2. Healthy original minus other healthy eyes degraded to the same density (loose ends per mm):",
          "   pooled " + s(res["T2"]["ends_per_mm"]["pooled"]),
          f"   median difference in healthy SD units: {res['T2']['median_in_healthy_sd']:+.2f}", "",
          "T3. DR original minus healthy eyes degraded to the same density (loose ends per mm):"]
    for k, c in res["T3"]["ends_per_mm"].items():
        L.append(f"   {k:8s} " + s(c))
    L += ["   raw-skeleton ends, pooled: " + s(res["T3"]["raw_ends_per_mm"]["pooled"]), "",
          "READINGS: " + json.dumps(res["readings"])]
    return "\n".join(L) + "\n"


def reference_check(v: pd.DataFrame) -> dict:
    """Originals must reproduce the held-out U-Net pipeline features for the same eyes."""
    fp = ROOT / em.SOURCES / "unet_heldout" / "pipeline" / "features.csv"
    if not fp.exists():
        return {"status": "reference features absent"}
    f = pd.read_csv(fp, dtype={"image_id": str})
    j = v[v.kind == "original"].merge(f[["image_id", s0.DENSITY_COL, "end_density_per_mm2_post",
                                         "end_density_per_mm2_pre"]], on="image_id")
    return {"n": int(len(j)),
            "max_abs_diff_density": float((j.density - j[s0.DENSITY_COL]).abs().max()),
            "max_abs_diff_end_final": float((j.end_final_per_mm2 - j.end_density_per_mm2_post).abs().max()),
            "max_abs_diff_end_raw": float((j.end_raw_per_mm2 - j.end_density_per_mm2_pre).abs().max())}


def input_files(cfg: dict) -> dict[str, str]:
    files = [AMENDMENT, "config.yaml", cfg["data"]["manifest"], f"{MODEL_DIR}/model.pt", f"{MODEL_DIR}/meta.json",
             f"{s0.SRC_RUN}/pipeline/features.csv"]
    return {f: sha256_file(ROOT / f) for f in files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("record", "run", "analyse"):
        sub.add_parser(c)
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    cfg = load_config(ROOT / "config.yaml")
    if args.cmd == "record":
        if (ROOT / RECORD).exists() or (ROOT / OUT).exists():
            raise DegradeError("record or outputs already exist; the record is written once, first")
        write_json(ROOT / RECORD, {
            "adopted_utc": now_utc(), "decided_by": "",
            "instructed_by": "user, in session, 2026-11-16", "is_human_review": False,
            "analysis_status": s0.STATUS_LABEL, "degraded_images_traced_at_adoption": False,
            "script_sha256_at_adoption": sha256_file(Path(__file__)), "files_sha256": input_files(cfg)})
        print(json.dumps(read_json(ROOT / RECORD), indent=1)[:400])
        return 0
    if not (ROOT / RECORD).exists():
        raise DegradeError("no dated record; run 'record' first")
    rec = read_json(ROOT / RECORD)
    bad = [f for f, h in rec["files_sha256"].items() if input_files(cfg).get(f) != h]
    if bad:
        raise DegradeError("inputs changed since the dated record: " + ", ".join(bad))
    out = ROOT / OUT
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "run":
        v = run(cfg)
        v.to_csv(out / "versions.csv", index=False)
        print(v.groupby(["kind", "level"]).density.median().round(2).to_string())
    elif args.cmd == "analyse":
        v = pd.read_csv(out / "versions.csv", dtype={"image_id": str})
        res = analyse(v, cfg, reference_check(v))
        res["provenance"] = {"created_utc": now_utc(), "performed_by": "",
                             "amendment": AMENDMENT, "record": RECORD, "script_sha256": sha256_file(Path(__file__)),
                             "environment": environment()}
        write_json(out / "results.json", res)
        (out / "summary.txt").write_text(text(res))
        print(text(res))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (DegradeError, s0.Step0Error) as exc:
        sys.exit(f"REFUSED: {exc}")
