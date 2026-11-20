"""Segmentation-perturbation robustness of FI and of the Step 0 matched-density contrast.

Post-outcome exploratory. Authority and fixed readings:
``docs/STEP0_PERTURBATION_AMENDMENT_2026-11-16.md``. The released capillary
label of every analysis eye is perturbed in a fixed, seeded way, the unmodified
pipeline command is run on the perturbed labels, and FI is compared with the
original run. Nothing in the package, configuration or saved results changes.

Subcommands::

    record            dated machine record; once only
    make  --cond C    perturbed masks + derived manifest for one condition
    analyse           stability, non-differential and differential-error contrasts
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
from retina_fragility.io_masks import read_mask, write_mask_png  # noqa: E402
from retina_fragility.provenance import environment, now_utc, read_json, sha256_file, write_json  # noqa: E402
from retina_fragility.seeds import rng_for  # noqa: E402
from scripts import step0_null_controls as s0  # noqa: E402
from scripts.step0_quality_specificity import ols  # noqa: E402

AMENDMENT = "docs/STEP0_PERTURBATION_AMENDMENT_2026-11-16.md"
RECORD = "reports/step0_perturbation_amendment_2026-11-16.json"
OUT = "outputs/exploratory_20261116_step0/perturbation"
CONDITIONS = ("dropout1", "dropout3", "boundary10", "dilate1")
CROSS = ndi.generate_binary_structure(2, 1)


class PerturbError(RuntimeError):
    pass


def perturb(mask: np.ndarray, cond: str, rng: np.random.Generator) -> np.ndarray:
    """Seeded perturbation of a boolean capillary label (see the amendment)."""
    m = np.asarray(mask, bool)
    if cond in ("dropout1", "dropout3"):
        p = {"dropout1": 0.01, "dropout3": 0.03}[cond]
        return m & ~(rng.random(m.shape) < p)
    if cond == "boundary10":
        inner = m & ~ndi.binary_erosion(m, CROSS, border_value=1)
        outer = ~m & ndi.binary_dilation(m, CROSS)
        flip = (inner | outer) & (rng.random(m.shape) < 0.10)
        return m ^ flip
    if cond == "dilate1":
        return ndi.binary_dilation(m, CROSS)
    raise ValueError(f"unknown condition {cond}")


def input_files(cfg: dict) -> dict[str, str]:
    files = [AMENDMENT, "config.yaml", cfg["data"]["manifest"], cfg["calibration"]["frozen_file"],
             f"{s0.SRC_RUN}/pipeline/features.csv", f"{s0.OUT_ROOT}/saved/results.json"]
    return {f: sha256_file(ROOT / f) for f in files}


def check_record(cfg: dict) -> None:
    if not (ROOT / RECORD).exists():
        raise PerturbError(f"no dated record at {RECORD}; run 'record' first")
    rec = read_json(ROOT / RECORD)
    changed = [f for f, h in rec["files_sha256"].items() if input_files(cfg).get(f) != h]
    if changed:
        raise PerturbError("inputs changed since the dated record: " + ", ".join(changed))


def make(cfg: dict, cond: str) -> dict:
    from retina_fragility.calibration import load_calibration
    from retina_fragility.manifest import analysis_rows, load_manifest

    base_seed = cfg["project"]["base_seed"]
    man_path = ROOT / cfg["data"]["manifest"]
    raw = pd.read_csv(man_path, dtype=str, keep_default_na=False)
    cal = load_calibration(ROOT / cfg["calibration"]["frozen_file"])
    rows, _ = analysis_rows(load_manifest(man_path, cfg), cal["calibration_ids"],
                            dataset=cfg["data"]["primary_dataset"])
    ids = set(rows.image_id.astype(str))
    out = ROOT / OUT / cond
    frac = []
    for i, r in raw.iterrows():
        if r.image_id not in ids:
            continue
        m = read_mask(ROOT / r.vessel_mask_path, cfg["data"]["label_maps"]["vessel"])
        p = perturb(m, cond, rng_for(base_seed, "step0_perturb", cond, r.image_id))
        dst = out / "masks" / f"{r.image_id}.png"
        write_mask_png(dst, p)
        raw.at[i, "vessel_mask_path"] = str(dst.relative_to(ROOT))
        if "vessel_mask_sha256" in raw.columns:
            raw.at[i, "vessel_mask_sha256"] = sha256_file(dst)
        frac.append(float((p != m).mean()))
    raw.to_csv(out / "manifest.csv", index=False)
    info = {"condition": cond, "n_perturbed": len(frac), "median_fraction_pixels_changed": float(np.median(frac)),
            "created_utc": now_utc(), "note": "capillary label only; the pipeline re-adds the large-vessel label"}
    write_json(out / "make.json", info)
    return info


def _contrast_block(frame: pd.DataFrame, base_seed: int, label: str) -> dict:
    f = frame.assign(dr=frame.y.astype(float), fov_6mm=(frame.fov_mm == 6.0).astype(float))
    r = ols(f, "FI", ["dr", "density", "fov_6mm"], ["dr"])
    r.pop("_fit", None)
    return {"matched_contrast": s0.contrast_by_scope(frame, "FI", base_seed, label)["pooled"], "ols_M0": r}


def analyse(cfg: dict) -> dict:
    base_seed = cfg["project"]["base_seed"]
    table, _ = s0.load_table(cfg)
    orig = table[["image_id", "y", "fov_mm", "density", "FI"]].copy()
    res: dict = {"analysis_status": s0.STATUS_LABEL, "human_qc_satisfied": False,
                 "original": _contrast_block(orig, base_seed, "perturb|orig"), "conditions": {}}
    stable, robust = [], []
    for cond in CONDITIONS:
        fp = ROOT / OUT / cond / "pipeline" / "features.csv"
        if not fp.exists():
            res["conditions"][cond] = {"status": "pending"}
            continue
        f = pd.read_csv(fp, dtype={"image_id": str})
        f = f.rename(columns={s0.DENSITY_COL: "density_p", s0.FI_COL: "FI_p"})[["image_id", "density_p", "FI_p"]]
        j = orig.merge(f, on="image_id", how="left")
        pert = j.assign(density=j.density_p, FI=j.FI_p)[["image_id", "y", "fov_mm", "density", "FI"]]
        block: dict = {"status": "ok", "n_missing_after_perturbation": int(j.FI_p.isna().sum()),
                       "make": read_json(ROOT / OUT / cond / "make.json"), "stability": {}}
        for scope, sub in (("fov_3mm", j[j.fov_mm == 3.0]), ("fov_6mm", j[j.fov_mm == 6.0])):
            d = sub.dropna(subset=["FI_p"])
            block["stability"][scope] = {
                "n": int(len(d)), "spearman_FI": float(sps.spearmanr(d.FI, d.FI_p).statistic),
                "median_FI_shift": float((d.FI_p - d.FI).median()),
                "median_density_ratio": float((d.density_p / d.density).median()),
                "median_FI_shift_dr": float((d.FI_p - d.FI)[d.y == 1].median()),
                "median_FI_shift_healthy": float((d.FI_p - d.FI)[d.y == 0].median())}
        block["non_differential"] = _contrast_block(pert, base_seed, f"perturb|{cond}|both")
        if cond.startswith("dropout"):
            dr_only = pd.concat([pert[pert.y == 1], orig[orig.y == 0]], ignore_index=True)
            h_only = pd.concat([orig[orig.y == 1], pert[pert.y == 0]], ignore_index=True)
            block["differential_dr_scans_degraded"] = _contrast_block(dr_only, base_seed, f"perturb|{cond}|dr")
            block["differential_healthy_scans_degraded"] = _contrast_block(h_only, base_seed, f"perturb|{cond}|h")
        res["conditions"][cond] = block
        if cond != "dropout3":
            stable.append(min(v["spearman_FI"] for v in block["stability"].values()) >= 0.8)
        c = block["non_differential"]["ols_M0"].get("coef", {}).get("dr")
        robust.append(c is not None and c["ci"][1] < 0)
    done = [c for c in CONDITIONS if res["conditions"][c].get("status") == "ok"]
    res["readings"] = {
        "complete": len(done) == len(CONDITIONS),
        "fi_ranking_stable_to_mild_perturbation": bool(stable and all(stable)) if len(done) == len(CONDITIONS) else None,
        "matched_contrast_robust_to_non_differential_perturbation":
            bool(robust and all(robust)) if len(done) == len(CONDITIONS) else None}
    return res


def text(res: dict) -> str:
    def c(b):
        o = b["ols_M0"].get("coef", {}).get("dr")
        m = b["matched_contrast"]
        return ("n/a" if o is None else f"OLS {o['coef']:+.4f} [{o['ci'][0]:+.4f}, {o['ci'][1]:+.4f}]") + \
            f"; matched median {s0._fmt(m.get('median'), 4)}, p = {s0._fmt(m.get('wilcoxon_p_two_sided'), 4)}"

    L = ["SEGMENTATION-PERTURBATION ROBUSTNESS (post-outcome exploratory; human QC unmet)", "",
         "DR minus healthy FI at matched density (negative = DR less fragile):", f"  original labels:      {c(res['original'])}"]
    for cond, b in res["conditions"].items():
        if b.get("status") != "ok":
            L.append(f"  {cond}: {b.get('status')}")
            continue
        s3, s6 = b["stability"]["fov_3mm"], b["stability"]["fov_6mm"]
        L += [f"  {cond}: FI rank agreement with original, Spearman {s3['spearman_FI']:.3f} (3 mm), "
              f"{s6['spearman_FI']:.3f} (6 mm); density x{s3['median_density_ratio']:.3f} / x{s6['median_density_ratio']:.3f}; "
              f"FI shift {s3['median_FI_shift']:+.4f} / {s6['median_FI_shift']:+.4f}",
              f"    all scans perturbed:   {c(b['non_differential'])}"]
        if "differential_dr_scans_degraded" in b:
            L += [f"    only DR scans:         {c(b['differential_dr_scans_degraded'])}",
                  f"    only healthy scans:    {c(b['differential_healthy_scans_degraded'])}"]
    L += ["", "READINGS: " + json.dumps(res["readings"])]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record")
    p = sub.add_parser("make")
    p.add_argument("--cond", required=True, choices=CONDITIONS)
    sub.add_parser("analyse")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    cfg = load_config(ROOT / "config.yaml")
    if args.cmd == "record":
        if (ROOT / RECORD).exists() or (ROOT / OUT).exists():
            raise PerturbError("record or outputs already exist; the record is written once, first")
        write_json(ROOT / RECORD, {
            "adopted_utc": now_utc(), "decided_by": "",
            "instructed_by": "user, in session, 2026-11-16", "is_human_review": False,
            "human_approval_claimed": False, "analysis_status": s0.STATUS_LABEL,
            "outcomes_available_at_adoption": True, "perturbation_results_available_at_adoption": False,
            "script_sha256_at_adoption": sha256_file(Path(__file__)), "files_sha256": input_files(cfg)})
        print(json.dumps(read_json(ROOT / RECORD), indent=1)[:600])
        return 0
    check_record(cfg)
    if args.cmd == "make":
        print(json.dumps(make(cfg, args.cond)))
    else:
        res = analyse(cfg)
        res["provenance"] = {"created_utc": now_utc(), "performed_by": "",
                             "amendment": AMENDMENT, "record": RECORD, "script_sha256": sha256_file(Path(__file__)),
                             "environment": environment(), "inputs_sha256": input_files(cfg)}
        write_json(ROOT / OUT / "results.json", res)
        (ROOT / OUT / "summary.txt").write_text(text(res))
        print(text(res))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (PerturbError, s0.Step0Error) as exc:
        sys.exit(f"REFUSED: {exc}")
