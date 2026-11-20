"""Experiment A (secondary, H1): FI and biomarkers, DR vs normal, per FOV and pooled.

For each configured feature: two-sided Mann-Whitney U, Cliff's delta (DR -
normal) and AUROC with the fixed orientation "higher = DR", both with
group-bootstrap CIs stratified by class; plus an age/sex(/FOV when pooled)
adjusted logistic model (nested LRT on complete cases; failures reported, no
p-value). Mean S(q) curves on a common q grid with group-bootstrap bands.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..seeds import rng_for
from ..stats import multiplicity
from ..stats.effects import group_bootstrap_weights, two_group_summary
from ..stats.primary import nested_lrt
from .common import header, write_results


def scopes(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"pooled": table}
    for fov in sorted(table.fov_mm.dropna().unique()):
        out[f"fov_{fov:g}mm"] = table[table.fov_mm == fov]
    return out


def compare_features(table: pd.DataFrame, features: list[str], cfg: dict, purpose: str) -> list[dict]:
    st = cfg["stats"]
    rows = []
    for scope, sub in scopes(table).items():
        adj = ["age", "sex_male"] + (["fov_6mm"] if scope == "pooled" and sub.fov_mm.nunique() > 1 else [])
        for feat in features:
            if feat not in sub.columns:
                rows.append({"scope": scope, "feature": feat, "status": "feature_absent"})
                continue
            rng = rng_for(cfg["project"]["base_seed"], purpose, scope, feat)
            s = two_group_summary(sub[feat].to_numpy(float), sub.y.to_numpy(), sub.group_key.to_numpy(),
                                  int(st["n_boot"]), rng, float(st["alpha"]))
            lrt = nested_lrt(sub, "y", adj, feat, st, f"{purpose}:{scope}:{feat}:adjusted",
                             "secondary_adjusted")
            s.update(scope=scope, feature=feat, adjusted_covariates=adj,
                     adjusted_status=lrt["status"], adjusted_failure_reasons=lrt["failure_reasons"],
                     adjusted_n=lrt["n"], adjusted_lr_p=lrt["p_value"],
                     adjusted_or=lrt.get("odds_ratio_added"), adjusted_or_ci=lrt.get("odds_ratio_ci"),
                     adjusted_or_scale=lrt.get("odds_ratio_scale"))
            rows.append(s)
    return multiplicity.annotate_comparisons(rows, purpose.split("_")[-1].upper(), cfg)


def load_curve(pipeline_root: Path, image_id: str, order: str, grid: np.ndarray) -> np.ndarray | None:
    from ..pipeline import safe_name

    p = pipeline_root / "images" / safe_name(image_id) / "curves.csv"
    if not p.exists():
        return None
    c = pd.read_csv(p)
    c = c[c.order == order]
    if c.empty:
        return None
    return np.interp(grid, c.q_edge.to_numpy(), c.S.to_numpy())


def group_curves(table: pd.DataFrame, pipeline_root: Path, order: str, grid: np.ndarray,
                 n_boot: int, rng: np.random.Generator) -> tuple[dict, list[str]]:
    curves, keep, missing = [], [], []
    for iid in table.image_id:
        c = load_curve(pipeline_root, iid, order, grid)
        if c is None:
            missing.append(iid)
        else:
            curves.append(c)
            keep.append(iid)
    if not curves:
        return {}, missing
    C = np.vstack(curves)
    sub = table.set_index("image_id").loc[keep]
    y, g = sub.y.to_numpy(), sub.group_key.astype(str).to_numpy()
    out = {}
    for label, cls in (("normal", 0), ("DR", 1)):
        m = y == cls
        if not m.any():
            continue
        W = group_bootstrap_weights(g[m], None, n_boot, rng)
        boots = (W @ C[m]) / W.sum(axis=1, keepdims=True)
        out[label] = {"mean": C[m].mean(axis=0), "lo": np.quantile(boots, 0.025, axis=0),
                      "hi": np.quantile(boots, 0.975, axis=0), "n": int(m.sum())}
    return out, missing


def run(table: pd.DataFrame, info: dict, cfg: dict, pipeline_root: str | Path, out_dir: str | Path,
        calibration: dict | None) -> dict:
    from .. import viz

    pipeline_root, out_dir = Path(pipeline_root), Path(out_dir)
    feats = list(cfg["exp_a"]["features"])
    rows = compare_features(table, feats, cfg, "exp_a")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "exp_a_comparisons.csv", index=False)
    grid = np.linspace(0, 1, int(cfg["exp_a"]["curve_grid"]))
    curve_meta = {}
    synthetic = info.get("data_provenance", "").startswith("SYNTHETIC")
    for order in (cfg["stress"]["primary_order"], "random"):
        for scope, sub in scopes(table).items():
            rng = rng_for(cfg["project"]["base_seed"], "exp_a_curves", order, scope)
            gc, missing = group_curves(sub, pipeline_root, order, grid, int(cfg["stats"]["n_boot"]), rng)
            if not gc:
                continue
            frame = pd.DataFrame({"q": grid})
            for label, d in gc.items():
                for k in ("mean", "lo", "hi"):
                    frame[f"{label}_{k}"] = d[k]
            frame.to_csv(out_dir / f"exp_a_curves_{order}_{scope}.csv", index=False)
            viz.plot_group_curves(grid, gc, out_dir / f"exp_a_curves_{order}_{scope}.png",
                                  f"Exp A mean S(q), {order}, {scope} (95% group-bootstrap band)",
                                  synthetic)
            curve_meta[f"{order}:{scope}"] = {"missing_curves": missing}
    from ..pipeline import COVERAGE_METHOD_NOTES

    res = header(cfg, "A", "secondary (H1); p-value roles per row", info, calibration)
    geom = cfg["stress"]["coverage_geometry"]
    res.update(comparisons=rows, curves=curve_meta,
               p_value_roles=("Each row carries mwu_p_role and adjusted_lr_p_role. Only the pooled "
                              "Mann-Whitney p of prespecified family members (holm:<family>) is "
                              "Holm-corrected, in secondary_holm.json. Every other p-value here "
                              "(per-FOV scopes, other features, adjusted LRTs) is "
                              "exploratory_uncorrected."),
               method_notes={"FIperf/P0 coverage_geometry": geom,
                             "coverage_method_note": COVERAGE_METHOD_NOTES[geom]})
    write_results(out_dir, res)
    return res
