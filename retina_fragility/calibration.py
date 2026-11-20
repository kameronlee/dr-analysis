"""Deterministic, frozen calibration of the perfusion distance d.

* Selection (outcome-blind): healthy rows of the calibration dataset, not in a
  test split, with valid inputs; ``n_per_fov`` per FOV in ascending order of
  ``sha256(base_seed|CALIBRATION|image_id)``. Too few eligible rows raise an
  error; a smaller calibration set is never used silently.
* d: weighted percentile (default 95th) of the distance, in µm, from every
  calibration *tissue* pixel (analysis mask minus vessel mask) to the nearest
  vessel pixel. With ``weighting: equal_image`` every image has total weight 1.
* The record (IDs, their hash, d, per-FOV diagnostics, input hashes, config
  hash) is frozen to JSON. Recomputing with different IDs or d is refused
  unless ``force=True``; a forced change keeps the previous record in
  ``history``. Calibration IDs are excluded from every analysis.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .config import config_hash
from .io_masks import load_reference_masks, make_analysis_mask
from .manifest import resolve_input
from .provenance import now_utc, read_json, sha256_file, sha256_strings, write_json


class CalibrationError(ValueError):
    pass


def selection_key(base_seed: int, image_id: str) -> str:
    return hashlib.sha256(f"{int(base_seed)}|CALIBRATION|{image_id}".encode()).hexdigest()


def calibration_config_subset(cfg: dict) -> dict:
    return {"calibration": {k: v for k, v in cfg["calibration"].items() if k != "frozen_file"},
            "analysis_mask": cfg["analysis_mask"], "label_maps": cfg["data"]["label_maps"],
            "base_seed": cfg["project"]["base_seed"]}


def select_calibration_ids(frame: pd.DataFrame, cfg: dict, dataset: str,
                           valid_ids: set[str] | None = None) -> tuple[list[str], pd.DataFrame]:
    ccfg = cfg["calibration"]
    base_seed = cfg["project"]["base_seed"]
    rows = []
    for r in frame.itertuples():
        reasons = []
        if r.dataset != dataset:
            reasons.append("different_dataset")
        if r.analysis_group != ccfg["eligible_group"]:
            reasons.append("not_eligible_group")
        if str(r.split).lower() == "test":
            reasons.append("test_split")
        if valid_ids is not None and r.image_id not in valid_ids:
            reasons.append("invalid_inputs")
        fov_key = f"{float(r.fov_mm):g}" if pd.notna(r.fov_mm) else ""
        if fov_key not in ccfg["n_per_fov"]:
            reasons.append("fov_not_calibrated")
        rows.append({"image_id": r.image_id, "fov_key": fov_key, "eligible": not reasons,
                     "reasons": ";".join(reasons), "selection_key": selection_key(base_seed, r.image_id)})
    audit = pd.DataFrame(rows)
    selected: list[str] = []
    audit["selected"] = False
    for fov_key, n in ccfg["n_per_fov"].items():
        pool = audit[audit.eligible & audit.fov_key.eq(str(fov_key))].sort_values("selection_key")
        if len(pool) < int(n):
            raise CalibrationError(f"FOV {fov_key}: {len(pool)} eligible calibration images, "
                                   f"{n} required")
        pick = pool.image_id.head(int(n)).tolist()
        selected += pick
        audit.loc[audit.image_id.isin(pick), "selected"] = True
    return sorted(selected), audit


def tissue_distances_um(vessel: np.ndarray, faz: np.ndarray | None, pixel_size_um: float,
                        cfg: dict) -> np.ndarray:
    vessel = np.asarray(vessel, bool)
    if not vessel.any():
        raise CalibrationError("calibration vessel mask is empty")
    analysis = make_analysis_mask(vessel.shape, pixel_size_um, faz, cfg["analysis_mask"])
    tissue = analysis & ~vessel
    dist = ndi.distance_transform_edt(~vessel) * float(pixel_size_um)
    return dist[tissue]


def weighted_percentile(samples: list[np.ndarray], p: float, weighting: str = "equal_image") -> float:
    """Inverted-CDF weighted percentile; each sample array is one image."""
    parts = [np.asarray(s, float) for s in samples if len(s)]
    if not parts:
        raise CalibrationError("no tissue pixels in calibration images")
    vals = np.concatenate(parts)
    if weighting == "equal_image":
        w = np.concatenate([np.full(len(s), 1.0 / len(s)) for s in parts])
    elif weighting == "pixel":
        w = np.ones_like(vals)
    else:
        raise CalibrationError(f"unknown calibration weighting {weighting}")
    order = np.argsort(vals, kind="stable")
    vals, w = vals[order], w[order]
    cum = np.cumsum(w) / w.sum()
    k = int(np.searchsorted(cum, p / 100.0 - 1e-12, side="left"))
    return float(vals[min(k, len(vals) - 1)])


def compute_calibration(frame: pd.DataFrame, ids: list[str], cfg: dict, root: str | Path,
                        dataset: str, percentiles_extra=()) -> dict:
    ccfg = cfg["calibration"]
    sub = frame.set_index("image_id").loc[ids]
    per_image, samples, by_fov = [], [], {}
    input_hashes = {}
    for image_id, r in sub.iterrows():
        vpath = resolve_input(r["vessel_mask_path"], root)
        fpath = resolve_input(r["faz_mask_path"], root)
        lpath = resolve_input(r["large_vessel_mask_path"], root) if r["large_vessel_mask_path"] else None
        vessel, faz = load_reference_masks(vpath, fpath, cfg["data"]["label_maps"], lpath)
        d = tissue_distances_um(vessel, faz, float(r["pixel_size_um"]), cfg)
        samples.append(d)
        key = f"{float(r['fov_mm']):g}"
        by_fov.setdefault(key, []).append(d)
        input_hashes[image_id] = {"vessel": sha256_file(vpath), "faz": sha256_file(fpath),
                                  **({"large_vessel": sha256_file(lpath)} if lpath else {})}
        per_image.append({"image_id": image_id, "fov_key": key, "n_tissue_px": int(d.size),
                          "median_um": float(np.median(d)) if d.size else None,
                          "p95_um": float(np.percentile(d, 95)) if d.size else None})
    p = float(ccfg["percentile"])
    d_um = weighted_percentile(samples, p, ccfg["weighting"])
    record = {
        "format": "retina_fragility.calibration/1",
        "dataset": dataset,
        "calibration_ids": sorted(ids),
        "calibration_ids_sha256": sha256_strings(ids),
        "percentile": p,
        "weighting": ccfg["weighting"],
        "d_um": d_um,
        "d_um_per_fov_diagnostic": {k: weighted_percentile(v, p, ccfg["weighting"])
                                    for k, v in sorted(by_fov.items())},
        "d_um_other_percentiles": {f"{q:g}": weighted_percentile(samples, float(q), ccfg["weighting"])
                                   for q in percentiles_extra},
        "per_image": per_image,
        "input_sha256": input_hashes,
        "calibration_config_sha256": config_hash(calibration_config_subset(cfg)),
        "created_utc": now_utc(),
        "data_provenance": ("SYNTHETIC - NOT CLINICAL" if str(dataset).upper().startswith("SYNTHETIC")
                            else "real"),
        "history": [],
    }
    return record


def compare_records(old: dict, new: dict) -> dict:
    """Field-by-field identity check of two calibration records.

    Covers the dataset, IDs, d, the other stored percentiles, the calibration
    config hash and the per-image input file hashes. ``changed_inputs`` lists
    the image IDs whose stored input hashes differ.
    """
    def close(a, b):
        return a is not None and b is not None and bool(np.isclose(a, b, rtol=1e-9, atol=0))

    old_in, new_in = old.get("input_sha256") or {}, new.get("input_sha256") or {}
    changed = sorted(i for i in set(old_in) | set(new_in) if old_in.get(i) != new_in.get(i))
    old_p, new_p = old.get("d_um_other_percentiles") or {}, new.get("d_um_other_percentiles") or {}
    checks = {
        "dataset": old.get("dataset") == new.get("dataset"),
        "ids": old.get("calibration_ids") == new.get("calibration_ids"),
        "d": close(old.get("d_um"), new.get("d_um")),
        "other_percentiles": set(old_p) == set(new_p) and all(close(old_p[k], new_p[k]) for k in old_p),
        "config": old.get("calibration_config_sha256") == new.get("calibration_config_sha256"),
        "inputs": not changed,
    }
    return {"identical": all(checks.values()), "checks": checks, "changed_inputs": changed}


def freeze_calibration(record: dict, path: str | Path, force: bool = False) -> dict:
    """Write the record unless a different frozen record exists (then require force).

    "Different" includes changed input files with unchanged IDs, d and config. A
    re-labelled or re-exported mask can leave the percentile unchanged, but it
    must not silently keep stale hashes in the provenance record. A forced
    replacement stores the previous record, including its input hashes, in
    ``history`` together with the comparison.
    """
    path = Path(path)
    if path.exists():
        old = read_json(path)
        cmp = compare_records(old, record)
        if cmp["identical"]:
            return old
        if not force:
            failed = [k for k, ok in cmp["checks"].items() if not ok]
            detail = (f"; input hashes changed for {cmp['changed_inputs'][:10]}"
                      if cmp["changed_inputs"] else "")
            raise CalibrationError(
                f"frozen calibration at {path} differs in {failed}{detail}; refusing to overwrite "
                "without --force")
        hist = list(old.get("history", []))
        old_copy = {k: v for k, v in old.items() if k != "history"}
        hist.append({"replaced_utc": now_utc(), "forced": True, "comparison": cmp,
                     "previous": old_copy})
        record = dict(record, history=hist)
    write_json(path, record)
    return record


def verify_calibration_inputs(record: dict, frame: pd.DataFrame, root: str | Path) -> list[str]:
    """Re-hash the calibration input files named in ``frame``; return a list of problems.

    Used before calibrated processing, so a calibration record whose masks have
    since changed on disk cannot be used silently.
    """
    problems = []
    idx = frame.set_index("image_id") if "image_id" in frame.columns else frame
    for iid, stored in (record.get("input_sha256") or {}).items():
        if iid not in idx.index:
            problems.append(f"{iid}: calibration image absent from the manifest")
            continue
        r = idx.loc[iid]
        cols = {"vessel": "vessel_mask_path", "faz": "faz_mask_path",
                "large_vessel": "large_vessel_mask_path"}
        for key, col in cols.items():
            if key not in stored:
                continue
            try:
                h = sha256_file(resolve_input(r[col], root))
            except (OSError, ValueError) as exc:
                problems.append(f"{iid}: {key} unreadable ({exc})")
                continue
            if h != stored[key]:
                problems.append(f"{iid}: {key} file hash differs from the frozen calibration record")
    return problems


def load_calibration(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise CalibrationError(f"no frozen calibration at {path}; run the calibrate command first")
    rec = read_json(path)
    if rec.get("format") != "retina_fragility.calibration/1":
        raise CalibrationError(f"unrecognised calibration file {path}")
    if sha256_strings(rec["calibration_ids"]) != rec["calibration_ids_sha256"]:
        raise CalibrationError("calibration ID hash mismatch (file edited?)")
    return rec


def d_for_percentile(record: dict, percentile: float) -> float:
    """d at a different percentile (ablation). Only percentiles computed at freeze time."""
    if np.isclose(percentile, record["percentile"]):
        return float(record["d_um"])
    key = f"{float(percentile):g}"
    if key not in record.get("d_um_other_percentiles", {}):
        raise CalibrationError(f"percentile {percentile} was not computed at calibration time")
    return float(record["d_um_other_percentiles"][key])
