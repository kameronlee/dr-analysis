"""Density-calibrated classical tracers as independent label sources.

Post-outcome exploratory. Authority and fixed readings:
``docs/LABEL_SOURCES_V2_AMENDMENT_2026-11-16.md``. Calibration uses calibration
eyes only. Nothing in the package, configuration or saved results is modified.

Subcommands::

    record      dated machine record; once only
    calibrate   set each tracer's one parameter per FOV on calibration eyes; freeze
    segment     trace all analysis eyes with the frozen parameters; derived manifests
    analyse     agreement with reference labels and the audit model per tracer
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
from retina_fragility.io_masks import read_intensity, read_mask, write_mask_png  # noqa: E402
from retina_fragility.provenance import environment, now_utc, read_json, sha256_file, write_json  # noqa: E402
from scripts import endnode_mechanism as em  # noqa: E402
from scripts import metric_audit as ma  # noqa: E402
from scripts import step0_null_controls as s0  # noqa: E402

AMENDMENT = "docs/LABEL_SOURCES_V2_AMENDMENT_2026-11-16.md"
RECORD = "reports/label_sources_v2_amendment_2026-11-16.json"
OUT = "outputs/exploratory_20261116_step0/label_sources_v2"
PARAMS = f"{OUT}/calibrated_parameters.json"
TRACERS = ("frangi_capillary", "adaptive")
FRANGI_SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0]
ADAPTIVE_BLOCK = 31
MIN_OBJECT_PX = 20
# search grids, ordered from the sparse end (few vessel pixels) to the dense end
GRID = {"frangi_capillary": [float(x) for x in np.geomspace(3.0, 0.05, 12)],
        "adaptive": [float(x) for x in np.linspace(-0.15, 0.15, 13)]}
TARGET, TOL_CAL, MAX_STEPS = 1.0, 0.02, 18
END_METRICS = ("end_density_per_mm2_pre", "end_density_per_mm2_post")


class SourceError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- tracers
def _remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    from scipy import ndimage as ndi

    if min_px <= 0:
        return mask
    lab, n = ndi.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def trace(image: np.ndarray, tracer: str, param: float, cfg: dict) -> np.ndarray:
    """Vessel mask of one OCTA image for ``tracer`` at its single parameter."""
    img = np.asarray(image, np.float32)
    if tracer == "frangi_capillary":
        from retina_fragility.segmentation import frangi as frangi_mod

        fcfg = {**cfg["segmentation"]["frangi"], "sigmas": list(FRANGI_SIGMAS), "min_object_px": MIN_OBJECT_PX,
                "threshold": "otsu", "threshold_scale": float(param)}
        mask, _ = frangi_mod.segment(img, fcfg, float(param))
        return np.asarray(mask, bool)
    if tracer == "adaptive":
        from skimage.filters import threshold_local

        thr = threshold_local(img, ADAPTIVE_BLOCK, method="gaussian", offset=float(param))
        return _remove_small(img > thr, MIN_OBJECT_PX)
    raise ValueError(f"unknown tracer {tracer}")


def graph_length(mask: np.ndarray, faz: np.ndarray, px_um: float, cfg: dict) -> float:
    from retina_fragility.pipeline import build_graph

    return float(build_graph(mask, faz, px_um, cfg)["clean"].total_length())


def first_crossing(f, grid, target: float = TARGET, tol: float = TOL_CAL, max_steps: int = MAX_STEPS) -> dict:
    """First parameter, walking ``grid`` from its sparse end, where ``f`` reaches ``target``.

    Vessel length rises as a mask fills in and falls again when it over-fills, so
    the search stops at the first grid point at or above the target and bisects
    only inside that last interval. If the target is never reached, the grid
    point with the largest value is returned and ``reached`` is False.
    """
    evals = []
    prev = None
    for g in grid:
        v = f(g)
        evals.append((float(g), float(v)))
        if v >= target:
            if prev is None or abs(v - target) <= tol:
                return {"param": float(g), "value": float(v), "reached": bool(abs(v - target) <= tol),
                        "evaluations": evals}
            a, fa, b, fb = prev[0], prev[1], g, v  # f(a) < target <= f(b)
            best = (a, fa) if abs(fa - target) < abs(fb - target) else (b, fb)
            for _ in range(max_steps):
                if abs(best[1] - target) <= tol:
                    break
                mid = 0.5 * (a + b)
                fm = f(mid)
                evals.append((float(mid), float(fm)))
                if abs(fm - target) < abs(best[1] - target):
                    best = (mid, fm)
                if fm < target:
                    a, fa = mid, fm
                else:
                    b, fb = mid, fm
            return {"param": float(best[0]), "value": float(best[1]),
                    "reached": bool(abs(best[1] - target) <= tol), "evaluations": evals}
        prev = (g, v)
    g, v = max(evals, key=lambda e: e[1])
    return {"param": float(g), "value": float(v), "reached": False, "evaluations": evals}


# --------------------------------------------------------------------------- calibrate
def calibration_rows(cfg: dict) -> pd.DataFrame:
    cal = read_json(ROOT / cfg["calibration"]["frozen_file"])["calibration_ids"]
    m = pd.read_csv(ROOT / cfg["data"]["manifest"], dtype=str, keep_default_na=False)
    rows = m[m.image_id.isin(cal)].copy()
    if len(rows) != len(cal):
        raise SourceError("calibration eyes missing from the manifest")
    return rows


def calibrate(cfg: dict) -> dict:
    from retina_fragility.pipeline import load_row_masks

    rows = calibration_rows(cfg)
    eyes = []
    for r in rows.to_dict("records"):
        r["pixel_size_um"] = float(r["pixel_size_um"])
        masks = load_row_masks(r, cfg, ROOT, {"mode": "reference"})
        if masks["intensity"] is None:
            raise SourceError(f"{r['image_id']}: no intensity image")
        ref = graph_length(masks["vessel"], masks["faz"], masks["pixel_size_um"], cfg)
        eyes.append({"fov": float(r["fov_mm"]), "img": masks["intensity"], "faz": masks["faz"],
                     "px": masks["pixel_size_um"], "ref": ref})
    out: dict = {"created_utc": now_utc(), "calibration_eyes_only": True, "n_eyes": len(eyes),
                 "target_ratio": TARGET, "tolerance": TOL_CAL, "tracers": {}}
    for tracer in TRACERS:
        out["tracers"][tracer] = {}
        for fov in sorted({e["fov"] for e in eyes}):
            sub = [e for e in eyes if e["fov"] == fov]

            def f(p, sub=sub, tracer=tracer):
                return float(np.median([graph_length(trace(e["img"], tracer, p, cfg), e["faz"], e["px"], cfg) / e["ref"]
                                        for e in sub]))

            r = first_crossing(f, GRID[tracer])
            out["tracers"][tracer][f"fov_{int(fov)}mm"] = {**r, "n_eyes": len(sub)}
    return out


# ----------------------------------------------------------------------------- segment
def segment_all(cfg: dict) -> dict:
    params = read_json(ROOT / PARAMS)
    raw0 = pd.read_csv(ROOT / cfg["data"]["manifest"], dtype=str, keep_default_na=False)
    ids = set(em.analysis_ids(cfg))
    info = {}
    for tracer in TRACERS:
        raw = raw0.copy()
        used, frac = [], []
        for i, r in raw.iterrows():
            if r.image_id not in ids:
                continue
            p = params["tracers"][tracer][f"fov_{int(float(r.fov_mm))}mm"]["param"]
            img, _ = read_intensity(ROOT / r.image_path)
            mask = trace(img, tracer, p, cfg)
            mp = ROOT / OUT / tracer / "masks" / f"{r.image_id}.png"
            write_mask_png(mp, mask)
            raw.at[i, "vessel_mask_path"] = str(mp.relative_to(ROOT))
            raw.at[i, "large_vessel_mask_path"] = ""
            if "vessel_mask_sha256" in raw.columns:
                raw.at[i, "vessel_mask_sha256"] = sha256_file(mp)
            if "large_vessel_mask_sha256" in raw.columns:
                raw.at[i, "large_vessel_mask_sha256"] = ""
            used.append(r.image_id)
            frac.append(float(mask.mean()))
        d = ROOT / OUT / tracer
        raw.to_csv(d / "manifest.csv", index=False)
        (d / "ids.txt").write_text(",".join(sorted(used)))
        info[tracer] = {"n": len(used), "median_vessel_fraction": float(np.median(frac))}
        write_json(d / "manifest_info.json", {**info[tracer], "created_utc": now_utc(),
                                              "parameters_sha256": sha256_file(ROOT / PARAMS)})
    return info


# ----------------------------------------------------------------------------- analyse
def analyse(cfg: dict) -> dict:
    table, _ = s0.load_table(cfg)
    ref = table[["image_id", "y", "fov_mm", "density", *ma.METRICS]].copy()
    ref["dr"] = ref.y.astype(float)
    ref["fov_6mm"] = (ref.fov_mm == 6.0).astype(float)
    metrics = [m for m in ma.METRICS if m != "faz_area_mm2"]
    res: dict = {"analysis_status": s0.STATUS_LABEL, "calibrated_parameters": read_json(ROOT / PARAMS)["tracers"],
                 "tracers": {}, "readings": {}}
    for tracer in TRACERS:
        fp = ROOT / OUT / tracer / "pipeline" / "features.csv"
        if not fp.exists():
            res["tracers"][tracer] = {"status": "pending"}
            continue
        f = pd.read_csv(fp, dtype={"image_id": str})
        alt = f.rename(columns={s0.DENSITY_COL: "density_alt", **{m: m + "_alt" for m in metrics}})[
            ["image_id", "density_alt", *[m + "_alt" for m in metrics]]]
        j = ref.merge(alt, on="image_id", how="inner")
        blk: dict = {"status": "ok", "n": int(len(j)), "n_dr": int(j.dr.sum()),
                     "n_missing": int(len(ref) - len(j)),
                     "density_ratio_median": {n: float((j.density_alt / j.density)[j.fov_mm == f_].median())
                                              for f_, n in ma.FOVS},
                     "agreement": {}, "audit_model": {}}
        for m in ["density", *metrics]:
            blk["agreement"][m] = {n: float(sps.spearmanr(j.loc[j.fov_mm == f_, m], j.loc[j.fov_mm == f_, m + "_alt"]).statistic)
                                   for f_, n in ma.FOVS}
        k = j.assign(density=j.density_alt)
        for m in metrics:
            k[m] = ma.zscore_within_fov(j, m + "_alt")
            blk["audit_model"][m] = em.adjusted_block(k, m)
        res["tracers"][tracer] = blk
        c = blk["audit_model"]["end_density_per_mm2_pre"]["pooled"]
        res["readings"][tracer] = ("holds" if ma.excludes_zero(c) and c["coef"] < 0 else
                                   "reverses" if ma.excludes_zero(c) and c["coef"] > 0 else "not_detected")
    r = [res["readings"].get(t) for t in TRACERS]
    if all(x is not None for x in r):
        res["readings"]["overall"] = ("robust_to_the_tracer" if all(x == "holds" for x in r) else
                                      "specific_to_the_reference_labelling_style" if all(x != "holds" for x in r)
                                      else "mixed")
    return res


def text(res: dict) -> str:
    L = ["DENSITY-CALIBRATED CLASSICAL TRACERS (post-outcome exploratory; graphs not reviewed by a human)", ""]
    for t, per in res["calibrated_parameters"].items():
        L.append(f"calibration {t}: " + "; ".join(
            f"{k}: parameter {v['param']:.4g}, length ratio {v['value']:.3f}, reached {v['reached']}" for k, v in per.items()))
    L.append("")
    for name, b in res["tracers"].items():
        if b.get("status") != "ok":
            L.append(f"{name}: {b.get('status')}")
            continue
        L.append(f"{name}: n = {b['n']} ({b['n_dr']} DR); density relative to reference labels "
                 f"x{b['density_ratio_median']['fov_3mm']:.2f} (3 mm), x{b['density_ratio_median']['fov_6mm']:.2f} (6 mm)")
        L.append("  metric: rank agreement with reference labels (3 mm, 6 mm) | DR minus healthy at equal density (SD): pooled; 3 mm; 6 mm")
        for m in ["density", *b["audit_model"].keys()]:
            a = b["agreement"][m]
            mod = b["audit_model"].get(m)
            tail = "" if mod is None else f" | {em._c(mod['pooled'])}; {em._c(mod['fov_3mm'])}; {em._c(mod['fov_6mm'])}"
            L.append(f"  {m:32s} {a['fov_3mm']:.2f}, {a['fov_6mm']:.2f}{tail}")
        L.append("")
    L.append("READINGS (end-node density, raw skeleton): " + json.dumps(res["readings"]))
    return "\n".join(L) + "\n"


def input_files(cfg: dict) -> dict[str, str]:
    files = [AMENDMENT, "config.yaml", cfg["data"]["manifest"], cfg["calibration"]["frozen_file"],
             f"{s0.SRC_RUN}/pipeline/features.csv", f"{em.OUT}/m4_results.json"]
    return {f: sha256_file(ROOT / f) for f in files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("record", "calibrate", "segment", "analyse"):
        sub.add_parser(c)
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    cfg = load_config(ROOT / "config.yaml")
    if args.cmd == "record":
        if (ROOT / RECORD).exists() or (ROOT / OUT).exists():
            raise SourceError("record or outputs already exist; the record is written once, first")
        write_json(ROOT / RECORD, {
            "adopted_utc": now_utc(), "decided_by": "",
            "instructed_by": "user, in session, 2026-11-16", "is_human_review": False,
            "analysis_status": s0.STATUS_LABEL, "first_frangi_result_known_at_adoption": True,
            "calibration_or_segmentation_done_at_adoption": False,
            "script_sha256_at_adoption": sha256_file(Path(__file__)), "files_sha256": input_files(cfg)})
        print(json.dumps(read_json(ROOT / RECORD), indent=1)[:400])
        return 0
    if not (ROOT / RECORD).exists():
        raise SourceError("no dated record; run 'record' first")
    rec = read_json(ROOT / RECORD)
    bad = [f for f, h in rec["files_sha256"].items() if input_files(cfg).get(f) != h]
    if bad:
        raise SourceError("inputs changed since the dated record: " + ", ".join(bad))
    (ROOT / OUT).mkdir(parents=True, exist_ok=True)
    if args.cmd == "calibrate":
        if (ROOT / PARAMS).exists():
            raise SourceError("parameters are already frozen")
        out = calibrate(cfg)
        out["script_sha256"] = sha256_file(Path(__file__))
        write_json(ROOT / PARAMS, out)
        print(json.dumps({t: {k: {"param": v["param"], "ratio": v["value"], "reached": v["reached"]}
                              for k, v in per.items()} for t, per in out["tracers"].items()}, indent=1))
    elif args.cmd == "segment":
        print(json.dumps(segment_all(cfg), indent=1))
    elif args.cmd == "analyse":
        res = analyse(cfg)
        res["provenance"] = {"created_utc": now_utc(), "performed_by": "",
                             "amendment": AMENDMENT, "record": RECORD, "script_sha256": sha256_file(Path(__file__)),
                             "environment": environment()}
        write_json(ROOT / OUT / "results.json", res)
        (ROOT / OUT / "summary.txt").write_text(text(res))
        print(text(res))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SourceError, em.MechError, s0.Step0Error) as exc:
        sys.exit(f"REFUSED: {exc}")
