"""Experiment F (exploratory, H5): frozen embeddings vs embeddings + fragility on DRAC.

Five grouped folds (image-level groups for DRAC), multinomial logistic
regression (L2, train-fold imputation/scaling) for model A = embeddings and
model B = embeddings + fragility features. Predicted probabilities are aligned
to the fixed class order [0, 1, 2] even if a training fold lacks a class.
Metrics: quadratic weighted kappa (argmax), any-DR AUROC (P1 + P2) and PDR
AUROC (P2), each with paired group-bootstrap CIs of B - A. Exploratory only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ..seeds import derive_seed, rng_for
from ..stats.cv import make_model
from ..stats.effects import group_bootstrap_weights, percentile_ci, weighted_auc
from .common import header, write_results

CLASSES = [0, 1, 2]


def aligned_proba(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(model.classes_):
        out[:, CLASSES.index(int(c))] = p[:, j]
    return out


def metrics(y: np.ndarray, P: np.ndarray, w: np.ndarray | None = None) -> dict:
    pred = np.asarray(CLASSES)[P.argmax(axis=1)]
    try:
        qwk = float(cohen_kappa_score(y, pred, weights="quadratic", sample_weight=w))
    except Exception:
        qwk = float("nan")
    return {"qwk": qwk, "auc_any_dr": weighted_auc(y >= 1, P[:, 1] + P[:, 2], w),
            "auc_pdr": weighted_auc(y == 2, P[:, 2], w)}


def run(cfg: dict, table: pd.DataFrame, emb_ids: np.ndarray, X_emb: np.ndarray, emb_meta: dict,
        out_dir, info: dict) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = cfg["exp_f"]
    if emb_meta.get("random_init_test_only") and not info.get("data_provenance", "").startswith("SYNTHETIC"):
        raise RuntimeError("random-initialised embeddings may only be used on synthetic fixtures")
    idx = {str(i): k for k, i in enumerate(emb_ids)}
    keep = table.image_id.astype(str).isin(idx)
    missing = table.loc[~keep, "image_id"].tolist()
    t = table[keep].reset_index(drop=True)
    E = X_emb[[idx[i] for i in t.image_id.astype(str)]]
    y = t.grade_num.to_numpy(int)
    groups = t.group_key.astype(str).to_numpy()
    frag = [c for c in f["fragility_features"] if c in t]
    F = t[frag].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    seed = derive_seed(cfg["project"]["base_seed"], "exp_f_folds") % (2**32 - 1)
    cv = StratifiedGroupKFold(n_splits=int(f["n_splits"]), shuffle=True, random_state=int(seed))
    PA, PB = np.zeros((len(t), 3)), np.zeros((len(t), 3))
    fold_of = np.full(len(t), -1)
    for k, (tr, te) in enumerate(cv.split(E, y, groups)):
        fold_of[te] = k
        for P, X in ((PA, E), (PB, np.hstack([E, F]))):
            m = make_model(float(f["C"]))
            m.fit(X[tr], y[tr])
            P[te] = aligned_proba(m, X[te])
    pd.DataFrame({"image_id": t.image_id, "fold": fold_of, "y": y,
                  **{f"A_p{c}": PA[:, c] for c in CLASSES}, **{f"B_p{c}": PB[:, c] for c in CLASSES}}
                 ).to_csv(out_dir / "exp_f_oof.csv", index=False)
    ma, mb = metrics(y, PA), metrics(y, PB)
    W = group_bootstrap_weights(groups, y, int(cfg["stats"]["n_boot"]), rng_for(cfg["project"]["base_seed"], "exp_f_boot"))
    deltas = {k: [] for k in ma}
    for w in W:
        a, b = metrics(y, PA, w), metrics(y, PB, w)
        for k in ma:
            deltas[k].append(b[k] - a[k])
    res = header(cfg, "F", "exploratory (H5)", info, None)
    res.update(embedding=emb_meta.get("model"), embedding_weights=emb_meta.get("weights"),
               fragility_features=frag, n=int(len(t)), missing_embeddings=missing,
               model_a_embeddings=ma, model_b_embeddings_plus_fragility=mb,
               delta_b_minus_a={k: {"point": mb[k] - ma[k], "ci": percentile_ci(np.array(v))}
                                for k, v in deltas.items()},
               note="Paired bootstrap of out-of-fold predictions (groups resampled); conditional on "
                    "fitted fold models. Exploratory; no leaderboard submission.")
    write_results(out_dir, res)
    return res
