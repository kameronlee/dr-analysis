"""Experiment D (secondary, H2): early targeted-vs-random damage, by order.

G10_<order> = mean random S(0.10) - S_<order>(0.10), interpolated at actual q.
qc_<order> = first recorded q with S < 0.5 (qc_random: mean random curve).
Group comparisons as in Experiment A (per FOV and pooled).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import header, write_results
from .exp_a import compare_features


def feature_names(cfg: dict) -> list[str]:
    q = int(round(cfg["stress"]["q_eval"] * 100))
    feats = []
    for order in cfg["exp_d"]["orders"]:
        feats += [f"G{q}_{order}", f"qc_{order}"]
    return feats + ["qc_random"]


def run(table: pd.DataFrame, info: dict, cfg: dict, out_dir: str | Path, calibration: dict | None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = compare_features(table, feature_names(cfg), cfg, "exp_d")
    pd.DataFrame(rows).to_csv(out_dir / "exp_d_comparisons.csv", index=False)
    from ..graph.stress import ORDER_LABELS

    res = header(cfg, "D", "secondary (H2); p-value roles per row", info, calibration)
    res.update(comparisons=rows,
               order_labels={o: ORDER_LABELS[o] for o in cfg["exp_d"]["orders"]},
               p_value_roles=("Only the pooled Mann-Whitney p of prespecified family members "
                              "(holm:<family>) is Holm-corrected, in secondary_holm.json. All other "
                              "orders (including the Forman-type heuristic), scopes and adjusted LRTs "
                              "are exploratory_uncorrected."),
               definitions={"G10": "mean random S(0.10) - targeted S(0.10), linear interpolation at "
                                   "actual edge-count q", "qc": "first recorded q with S < 0.5 "
                                   "(0 when S0 < 0.5); random uses the mean curve"})
    write_results(out_dir, res)
    return res
