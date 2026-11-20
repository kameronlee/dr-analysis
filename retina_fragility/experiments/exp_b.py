"""Experiment B: PRIMARY nested logistic LRT (H3) plus labelled sensitivities and CV.

* Primary: baseline = stats.baseline_primary (Section 7B: FOV, age, sex,
  vessel density, FAZ area, loop density); full = baseline + stats.fi_column.
  Unpenalised MLE on identical complete cases. Failure -> no p-value, plus a
  ridge fit labelled ``fallback_no_p_value``.
* Sensitivities (never replace the primary): fractal dimension substituted
  for / added to loop density (the H3 prose vs Section 7B discrepancy).
* Prediction: repeated stratified grouped 5-fold x 10, train-fold-only
  imputation/scaling/ridge, aligned OOF predictions, paired ΔAUROC with a
  conditional group bootstrap CI. Fold assignments and predictions saved.
* Correlation matrix and VIFs on the primary complete-case sample.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..stats import cv as cvmod
from ..stats.effects import correlation_table
from ..stats.primary import complete_cases, nested_lrt, ridge_fallback, vif_table
from .common import header, write_results


def run(table: pd.DataFrame, info: dict, cfg: dict, out_dir: str | Path,
        calibration: dict | None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    st = cfg["stats"]
    fi = st["fi_column"]
    base = list(st["baseline_primary"])
    primary = nested_lrt(table, "y", base, fi, st, "H3 primary nested LRT", "PRIMARY")
    fallback = None
    if primary["status"] != "ok":
        fallback = ridge_fallback(table, "y", base + [fi], float(st["cv"]["C"]))
    sens = {name: nested_lrt(table, "y", list(cols), fi, st, f"sensitivity:{name}", "sensitivity")
            for name, cols in st["sensitivity_sets"].items()}

    cc, _ = complete_cases(table, base + [fi], "y")
    if len(cc) > len(base) + 2:
        vif_table(cc, base + [fi]).to_csv(out_dir / "exp_b_vif_primary_sample.csv", index=False)
        correlation_table(cc, base + [fi]).to_csv(out_dir / "exp_b_spearman_primary_sample.csv")

    cvres: dict = {}
    cvc = st["cv"]
    if table.y.nunique() == 2 and min((table.y == 1).sum(), (table.y == 0).sum()) >= cvc["n_splits"]:
        preds, folds, issues = cvmod.repeated_cv(
            table.reset_index(drop=True), "y", "group_key",
            {"baseline": base, "baseline_plus_FI": base + [fi]},
            int(cvc["n_splits"]), int(cvc["n_repeats"]), float(cvc["C"]),
            cfg["project"]["base_seed"], "exp_b_cv")
        preds.to_csv(out_dir / "exp_b_oof_predictions.csv", index=False)
        folds.assign(image_id=table.reset_index(drop=True).image_id.to_numpy()[folds.row]).to_csv(
            out_dir / "exp_b_fold_assignments.csv", index=False)
        aucs = cvmod.auc_by_repeat(preds)
        aucs.to_csv(out_dir / "exp_b_auroc_by_repeat.csv", index=False)
        delta = cvmod.paired_delta_auc(preds, "baseline", "baseline_plus_FI", int(st["n_boot"]),
                                       cfg["project"]["base_seed"], "exp_b_cv", float(st["alpha"]))
        cvres = {"status": "ok", "fit_issues": issues,
                 "mean_auroc": aucs.groupby("model").auroc.mean().to_dict(),
                 "paired_delta": delta,
                 "preprocessing": "median imputation + standardisation + L2 logistic fit on "
                                  "training folds only (sklearn Pipeline)",
                 "missing_value_policy": "train-fold imputation for prediction; complete cases "
                                         "for the primary LRT"}
    else:
        cvres = {"status": "not_run", "reason": "fewer positives/negatives than folds"}

    res = header(cfg, "B", "PRIMARY (H3) + labelled sensitivities", info, calibration)
    from ..stats.multiplicity import PRIMARY, SENSITIVITY

    primary["p_value_role"] = PRIMARY
    for v in sens.values():
        v["p_value_role"] = SENSITIVITY
    res.update(primary=primary, primary_ridge_fallback=fallback, sensitivities=sens,
               p_value_roles={"primary": ("single prespecified H3 test; stands alone, never in a "
                                          "Holm family"),
                              "sensitivities": ("sensitivity_uncorrected: labelled sensitivities, no "
                                                "multiplicity correction, never replace the primary")},
               prediction_cv=cvres,
               discrepancy_note=("Section 7B baseline (loop density) is primary; the H3 prose "
                                 "names fractal dimension. Fractal substitution/addition are "
                                 "sensitivities recorded before data inspection."))
    write_results(out_dir, res)
    return res
