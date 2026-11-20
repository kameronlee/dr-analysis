"""Repeated stratified *grouped* cross-validation with aligned out-of-fold predictions.

* Folds come from ``StratifiedGroupKFold`` (groups = subject key; image key
  when subject IDs are unavailable) with a distinct shuffle seed per repeat.
  Every model in a comparison uses the SAME folds in a repeat (paired).
* Median imputation, standardisation and L2 logistic regression are fit on the
  training fold only (sklearn Pipeline), then applied to the held-out fold.
* The paired ΔAUROC CI resamples groups of the aligned OOF predictions and
  recomputes the per-repeat AUROC difference averaged over repeats. It is
  conditional on the fitted fold models; it is NOT a full refit bootstrap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..seeds import derive_seed, rng_for
from .effects import group_bootstrap_weights, percentile_ci, weighted_auc

CONDITIONAL_NOTE = ("Bootstrap of aligned out-of-fold predictions, resampling groups; "
                    "conditional on the fitted fold models, not a full refit bootstrap.")


def make_model(C: float, multiclass: bool = False):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fold_assignments(y: np.ndarray, groups: np.ndarray, n_splits: int, n_repeats: int,
                     base_seed: int, purpose: str) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold

    rows = []
    for r in range(n_repeats):
        seed = derive_seed(base_seed, purpose, "cv_repeat", r) % (2**32 - 1)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        for k, (_, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            for i in te:
                rows.append({"repeat": r, "fold": k, "row": int(i)})
    fa = pd.DataFrame(rows)
    # every row exactly once per repeat
    chk = fa.groupby("repeat").row.nunique()
    if not (chk == len(y)).all():
        raise RuntimeError("fold assignment does not cover every row once per repeat")
    return fa


def check_group_disjoint(folds: pd.DataFrame, groups: np.ndarray) -> None:
    g = np.asarray(groups).astype(str)
    f = folds.assign(group=g[folds.row.to_numpy()])
    per = f.groupby(["repeat", "group"]).fold.nunique()
    if (per > 1).any():
        raise RuntimeError("a group appears in more than one test fold within a repeat")


def repeated_cv(df: pd.DataFrame, outcome: str, group_col: str, feature_sets: dict[str, list[str]],
                n_splits: int, n_repeats: int, C: float, base_seed: int,
                purpose: str, model_factory=None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Returns (oof predictions long table, fold assignments, per-fit issues).

    ``model_factory(C)`` defaults to ``make_model``; tests inject recording models.
    """
    model_factory = model_factory or make_model
    y = df[outcome].to_numpy(int)
    groups = df[group_col].astype(str).to_numpy()
    folds = fold_assignments(y, groups, n_splits, n_repeats, base_seed, purpose)
    check_group_disjoint(folds, groups)
    preds, issues = [], []
    ids = df["image_id"].astype(str).to_numpy()
    for (r, k), te_df in folds.groupby(["repeat", "fold"]):
        te = te_df.row.to_numpy()
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(np.unique(y[tr])) < 2:
            issues.append({"repeat": int(r), "fold": int(k), "issue": "single_class_training_fold"})
            continue
        for name, cols in feature_sets.items():
            X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            model = model_factory(C)
            model.fit(X[tr], y[tr])  # imputer/scaler learned on the training fold only
            p = model.predict_proba(X[te])[:, list(model.classes_).index(1)]
            for i, pi in zip(te, p):
                preds.append({"repeat": int(r), "fold": int(k), "row": int(i), "image_id": ids[i],
                              "group": groups[i], "y": int(y[i]), "model": name, "prob": float(pi)})
    return pd.DataFrame(preds), folds, issues


def auc_by_repeat(preds: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (r, m), g in preds.groupby(["repeat", "model"]):
        out.append({"repeat": r, "model": m, "auroc": weighted_auc(g.y.to_numpy(), g.prob.to_numpy())})
    return pd.DataFrame(out)


def paired_delta_auc(preds: pd.DataFrame, model_a: str, model_b: str, n_boot: int,
                     base_seed: int, purpose: str, alpha: float = 0.05) -> dict:
    """Mean over repeats of AUROC(model_b) - AUROC(model_a) with a group bootstrap CI."""
    wide = preds.pivot_table(index=["repeat", "row"], columns="model", values="prob").reset_index()
    meta = preds.drop_duplicates(["repeat", "row"]).set_index(["repeat", "row"])[["y", "group"]]
    wide = wide.join(meta, on=["repeat", "row"])
    wide = wide.dropna(subset=[model_a, model_b])
    rows = np.sort(wide.row.unique())
    pos = {r: i for i, r in enumerate(rows)}
    groups_by_row = wide.drop_duplicates("row").set_index("row").group.loc[rows].to_numpy()
    y_by_row = wide.drop_duplicates("row").set_index("row").y.loc[rows].to_numpy()
    rng = rng_for(base_seed, purpose, "paired_delta_auc_bootstrap")
    W = group_bootstrap_weights(groups_by_row, y_by_row, n_boot, rng)
    reps = sorted(wide.repeat.unique())
    point, boot = [], np.zeros((n_boot, len(reps)))
    for j, r in enumerate(reps):
        g = wide[wide.repeat == r]
        idx = g.row.map(pos).to_numpy()
        yy = g.y.to_numpy()
        pa, pb = g[model_a].to_numpy(), g[model_b].to_numpy()
        point.append(weighted_auc(yy, pb) - weighted_auc(yy, pa))
        for b in range(n_boot):
            w = W[b, idx]
            boot[b, j] = weighted_auc(yy, pb, w) - weighted_auc(yy, pa, w)
    bmean = np.nanmean(boot, axis=1)
    return {"model_a": model_a, "model_b": model_b, "delta_auroc_mean_over_repeats": float(np.mean(point)),
            "delta_auroc_per_repeat": [float(v) for v in point],
            "ci": percentile_ci(bmean, alpha), "n_boot": int(n_boot), "note": CONDITIONAL_NOTE}
