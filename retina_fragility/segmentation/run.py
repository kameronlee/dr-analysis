"""Manifest-level segmentation: writes masks + a derived manifest with provenance.

The derived manifest is the input to the per-image pipeline for external data
(Experiment E). It keeps every original column, sets ``vessel_mask_path`` to
the produced mask, and records the segmenter, its parameters and hashes.

Label identities (``DERIVED_MANIFEST_FORMAT``): every ``<label>_path`` /
``<label>_sha256`` pair in a derived manifest names the same file.
``vessel_mask_*`` is the generated mask; ``large_vessel_mask_*`` and
``faz_mask_*`` are cleared together. The source row's label identities are kept
only under ``source_manifest_<label>_{path,sha256}`` and are never output
identities (segmentation_manifest_identity_findings_2026-09-30.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..io_masks import read_intensity, read_mask, write_mask_png
from ..manifest import resolve_input
from ..pipeline import check_output_names, safe_name
from ..provenance import code_fingerprint, now_utc, sha256_file, sha256_strings, write_json
from ..qc import segmentation_identity
from ..seeds import derive_seed
from . import frangi as frangi_mod
from . import unet as unet_mod


def _rows(frame: pd.DataFrame, splits: tuple[str, ...] | None,
          image_ids=None, seg_splits: tuple[str, ...] | None = None,
          exclude_ids=()) -> pd.DataFrame:
    """Rows by label-availability ``split`` and, optionally, by segmentation ``seg_split``.

    OCTA-500 has ``split = public`` for every row, so held-out U-Net prediction
    selects ``seg_splits = (val, test)``. ``exclude_ids`` (frozen calibration
    eyes) are removed before anything is segmented.
    """
    rows = frame if splits is None else frame[frame.split.str.lower().isin([s.lower() for s in splits])]
    if seg_splits is not None:
        rows = rows[rows.seg_split.astype(str).str.lower().isin([s.lower() for s in seg_splits])]
    exclude = set(map(str, exclude_ids))
    if exclude:
        rows = rows[~rows.image_id.isin(exclude)]
    if image_ids is not None:
        want = [str(i) for i in image_ids]
        missing = sorted(set(want) - set(rows.image_id))
        if missing:
            raise ValueError(f"requested images are not in the selected splits: {missing[:10]}")
        rows = rows[rows.image_id.isin(want)]
    check_output_names(rows.image_id)
    return rows


DERIVED_MANIFEST_FORMAT = "retina_fragility.derived_manifest/2"
LABEL_FIELDS = ("vessel_mask", "large_vessel_mask", "faz_mask")
SOURCE_PREFIX = "source_manifest_"


def check_derived_label_identities(derived: pd.DataFrame, records: pd.DataFrame | None = None) -> None:
    """Raise unless every label path/hash pair corresponds (metadata only; no file is read).

    A cleared path has a cleared hash, a generated path has a 64-hex hash and,
    given the segmentation ``records`` (indexed by image_id), the vessel pair
    is exactly the logged mask path and hash.
    """
    bad = []
    for f in LABEL_FIELDS:
        p, h = derived[f"{f}_path"].astype(str), derived[f"{f}_sha256"].astype(str)
        if (p.eq("") != h.eq("")).any():
            bad.append(f"{f}: path/hash emptiness differs")
        if not h[h.ne("")].str.fullmatch(r"[0-9a-f]{64}").all():
            bad.append(f"{f}: malformed sha256")
    if records is not None:
        idx = derived.index if derived.index.name == "image_id" else derived.image_id
        rec = records.loc[list(idx)]
        if (derived.vessel_mask_path.astype(str).to_numpy() != rec.mask_path.astype(str).to_numpy()).any() or \
                (derived.vessel_mask_sha256.astype(str).to_numpy() != rec.mask_sha256.astype(str).to_numpy()).any():
            bad.append("vessel_mask path/hash differ from the segmentation log")
    if bad:
        raise ValueError(f"derived manifest label identities are inconsistent: {bad}")


def verify_derived_mask_files(manifest: pd.DataFrame, root=None) -> int:
    """Re-hash every generated label file named in a derived manifest; return how many were checked."""
    check_derived_label_identities(manifest)
    n, bad = 0, []
    for f in LABEL_FIELDS:
        for iid, p, h in zip(manifest.image_id, manifest[f"{f}_path"], manifest[f"{f}_sha256"]):
            if not str(p):
                continue
            n += 1
            if sha256_file(resolve_input(p, root) if root is not None else Path(p)) != h:
                bad.append(f"{iid}:{f}")
    if bad:
        raise ValueError(f"{len(bad)} derived label files do not match their recorded sha256 (e.g. {bad[:5]})")
    return n


def _write_derived(frame: pd.DataFrame, records: list[dict], out_dir: Path, segmenter: str,
                   params: dict, model_sha256: str | None = None, extra: dict | None = None) -> Path:
    """Derived manifest + segmentation_run.json, both carrying the segmentation identity.

    ``segmentation_config_sha256`` identifies the method (segmenter, effective
    parameters, weights, segmenter source fingerprint), independent of which
    images were segmented, so a QC sample and a full run of the same method match.
    Label path/hash pairs follow the module docstring; ``records`` need
    ``image_id``, ``mask_path``, ``mask_sha256`` and ``status``.
    """
    ident = segmentation_identity(segmenter, params, model_sha256)
    rec = pd.DataFrame(records).set_index("image_id")
    derived = frame.loc[rec.index].copy()
    # source label identities, explicitly named; appended after all other columns below
    source = {f"{SOURCE_PREFIX}{f}_{s}": derived[f"{f}_{s}"].copy()
              for f in LABEL_FIELDS for s in ("path", "sha256") if f"{f}_{s}" in derived}
    derived["vessel_mask_path"] = rec.mask_path
    derived["vessel_mask_sha256"] = rec.mask_sha256
    for f in ("large_vessel_mask", "faz_mask"):
        derived[f"{f}_path"] = ""
        derived[f"{f}_sha256"] = ""
    check_derived_label_identities(derived, rec)
    derived["segmenter"] = segmenter
    derived["segmentation_status"] = rec.status
    derived["segmentation_config_sha256"] = ident["segmentation_config_sha256"]
    for c, v in source.items():
        derived[c] = v
    keep = [c for c in derived.columns if c not in {"analysis_group", "subject_id_assumed", "group_key",
                                                     "grade_num", "sex_male"}]
    out = out_dir / "manifest_segmented.csv"
    derived[keep].to_csv(out, index=False)
    write_json(out_dir / "segmentation_run.json", {
        "segmenter": segmenter, "params": params, "created_utc": now_utc(), "n": len(records),
        "derived_manifest_format": DERIVED_MANIFEST_FORMAT,
        "derived_manifest_sha256": sha256_file(out),
        "status_counts": rec.status.value_counts().to_dict(),
        "segmentation_config_sha256": ident["segmentation_config_sha256"],
        "segmentation_identity": ident["identity"], "segmenter_code_files": ident["code_files"],
        "image_ids_sha256": sha256_strings(rec.index), "fi_computed": False, **(extra or {})})
    return out


def frangi_manifest(frame: pd.DataFrame, cfg: dict, root, out_dir, splits=("train",),
                    threshold_scale: float | None = None, image_ids=None, seg_splits=None,
                    exclude_ids=()) -> Path:
    out_dir = Path(out_dir)
    fcfg = cfg["segmentation"]["frangi"]
    records = []
    for r in _rows(frame, splits, image_ids, seg_splits, exclude_ids).itertuples():
        path = resolve_input(r.image_path, root)
        img, info = read_intensity(path)
        mask, sinfo = frangi_mod.segment(img, fcfg, threshold_scale)
        mp = out_dir / "masks" / f"{safe_name(r.image_id)}.png"
        write_mask_png(mp, mask)
        records.append({"image_id": r.image_id, "mask_path": str(mp), "status": "ok",
                        "image_sha256": sha256_file(path), "mask_sha256": sha256_file(mp),
                        **{f"seg_{k}": v for k, v in sinfo.items() if not isinstance(v, (list, dict))},
                        **{f"seg_px_sigma_{s}": n for s, n in sinfo.get("mask_px_by_sigma", {}).items()}})
    pd.DataFrame(records).to_csv(out_dir / "segmentation_log.csv", index=False)
    used = fcfg["threshold_scale"] if threshold_scale is None else threshold_scale
    return _write_derived(frame, records, out_dir, "frangi",
                          {**fcfg, "threshold_scale": used, "threshold_scale_used": used})


def unet_training_rows(frame: pd.DataFrame, calibration_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = set(map(str, calibration_ids))
    audit = []
    for r in frame.itertuples():
        reasons = []
        if str(r.seg_split).lower() != "train":
            reasons.append("not_seg_train_split")
        if r.image_id in cal or bool(r.calibration):
            reasons.append("calibration")
        if not r.vessel_mask_path:
            reasons.append("no_reference_label")
        audit.append({"image_id": r.image_id, "included": not reasons, "reasons": ";".join(reasons)})
    a = pd.DataFrame(audit)
    return frame[frame.image_id.isin(a[a.included].image_id)], a


def _reference_vessel(r, cfg, root) -> np.ndarray:
    maps = cfg["data"]["label_maps"]
    v = read_mask(resolve_input(r.vessel_mask_path, root), maps["vessel"])
    if r.large_vessel_mask_path:
        v |= read_mask(resolve_input(r.large_vessel_mask_path, root), maps["large_vessel"])
    return v


UNET_TRAINING_CODE = ("retina_fragility.segmentation.run", "retina_fragility.segmentation.unet")


def _label_paths(r, root) -> list[Path]:
    paths = [resolve_input(r.vessel_mask_path, root)]
    if r.large_vessel_mask_path:
        paths.append(resolve_input(r.large_vessel_mask_path, root))
    return paths


def _json_norm(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def check_model_identity(out_dir, expected: dict) -> dict:
    """Return the existing model's metadata if it was trained exactly as ``expected``; else raise.

    Compared: the U-Net config, seed, training rows and groups, per-image input
    and label hashes, calibration-ID hash, effective patches per epoch,
    training-code fingerprint and the CLI provenance (manifest, calibration file,
    full config hash, overrides). The weights must still match their recorded hash.
    """
    out_dir = Path(out_dir)
    meta_p, model_p = out_dir / "meta.json", out_dir / "model.pt"
    if not (meta_p.exists() and model_p.exists()):
        raise FileExistsError(f"{out_dir} holds an incomplete model (model.pt/meta.json); remove it or use "
                              "a new output directory")
    meta = json.loads(meta_p.read_text())
    diffs = [] if sha256_file(model_p) == meta.get("model_sha256") else ["model_sha256 (weights changed)"]
    have = dict(meta, training_code_sha256=(meta.get("training_code") or {}).get("sha256"))
    diffs += [k for k, v in expected.items() if _json_norm(have.get(k)) != _json_norm(v)]
    if diffs:
        raise FileExistsError(f"{out_dir} holds a U-Net trained differently ({', '.join(diffs)}); refusing to "
                              "reuse it. Use a new output directory (e.g. a new UNET_TAG).")
    return dict(meta, reused_existing_model=True)


def unet_train_manifest(frame: pd.DataFrame, cfg: dict, root, calibration_ids, out_dir,
                        patches_per_epoch: int | None = None, provenance: dict | None = None,
                        reuse_identical: bool = False) -> dict:
    """Train on ``seg_split == train``, non-calibration, labelled rows.

    ``meta.json`` records the image and label hashes, the selection audit hash,
    the training-code fingerprint and ``provenance`` (e.g. manifest and
    calibration identities supplied by the CLI). An existing model is refused,
    unless ``reuse_identical`` and its recorded identity equals this request
    (``check_model_identity``); then it is returned unchanged.
    """
    out_dir = Path(out_dir)
    rows, audit = unet_training_rows(frame, calibration_ids)
    if rows.empty:
        raise ValueError("no eligible U-Net training rows (seg_split=train, non-calibration, labelled)")
    ucfg = cfg["segmentation"]["unet"]
    seed = derive_seed(cfg["project"]["base_seed"], ucfg["seed_purpose"])
    paths = [str(resolve_input(r.image_path, root)) for r in rows.itertuples()]
    label_sha = {r.image_id: [sha256_file(q) for q in _label_paths(r, root)] for r in rows.itertuples()}
    code = code_fingerprint(UNET_TRAINING_CODE)
    expected = {"config": ucfg, "seed": int(seed), "train_image_ids": rows.image_id.tolist(),
                "train_group_keys": rows.group_key.astype(str).tolist(),
                "train_input_sha256": {p: sha256_file(p) for p in paths}, "train_label_sha256": label_sha,
                "calibration_ids_sha256": sha256_strings(calibration_ids),
                "patches_per_epoch": patches_per_epoch or max(int(ucfg["batch_size"]), len(rows)),
                "training_code_sha256": code["sha256"], **(provenance or {})}
    if (out_dir / "model.pt").exists() or (out_dir / "meta.json").exists():
        if reuse_identical:
            return check_model_identity(out_dir, expected)
        raise FileExistsError(f"{out_dir} already holds a trained U-Net; choose a new output directory")
    images = [read_intensity(p)[0] for p in paths]
    masks = [_reference_vessel(r, cfg, root) for r in rows.itertuples()]
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "training_selection_audit.csv", index=False)
    extra = {"train_label_sha256": label_sha,
             "training_selection_audit_sha256": sha256_file(out_dir / "training_selection_audit.csv"),
             "calibration_ids_sha256": expected["calibration_ids_sha256"],
             "n_calibration_ids_excluded": int(audit.reasons.str.contains("calibration").sum()),
             "training_code": code,
             "seed_rule": f"derive_seed(base_seed, {ucfg['seed_purpose']!r})",
             "vessel_label_rule": "vessel_mask_path OR large_vessel_mask_path (label_maps decoding)",
             "input_rule": "image_path via read_intensity: nominal dtype range to [0, 1], no per-image stretch",
             **(provenance or {})}
    meta = unet_mod.train(images, masks, rows.image_id.tolist(), rows.group_key.astype(str).tolist(),
                          ucfg, seed, out_dir, paths, patches_per_epoch, extra_meta=extra)
    check_model_identity(out_dir, expected)  # what was saved is exactly what was requested
    return meta


def unet_predict_manifest(frame: pd.DataFrame, cfg: dict, root, model_dir, out_dir, splits=("train",),
                          threshold_scale: float | None = None, image_ids=None, seg_splits=None,
                          exclude_ids=()) -> Path:
    net, meta = unet_mod.load_model(model_dir)
    rows = _rows(frame, splits, image_ids, seg_splits, exclude_ids)
    if rows.empty:
        raise ValueError("no rows selected for U-Net prediction (check --splits / --seg-splits)")
    unet_mod.check_leakage(meta, rows.image_id, rows.group_key.astype(str))
    ucfg = cfg["segmentation"]["unet"]
    out_dir = Path(out_dir)
    records = []
    for r in rows.itertuples():
        path = resolve_input(r.image_path, root)
        img, _ = read_intensity(path)
        prob = unet_mod.predict_prob(net, meta, img)
        mask = unet_mod.segment(prob, ucfg, threshold_scale)
        mp = out_dir / "masks" / f"{safe_name(r.image_id)}.png"
        write_mask_png(mp, mask)
        records.append({"image_id": r.image_id, "mask_path": str(mp), "status": "ok",
                        "image_sha256": sha256_file(path), "mask_sha256": sha256_file(mp),
                        "vessel_fraction": float(mask.mean())})
    pd.DataFrame(records).to_csv(out_dir / "segmentation_log.csv", index=False)
    used = ucfg["threshold_scale"] if threshold_scale is None else threshold_scale
    return _write_derived(frame, records, out_dir, "unet",
                          {"model_sha256": meta["model_sha256"], "threshold": ucfg["threshold"],
                           "threshold_scale_used": used},
                          model_sha256=meta["model_sha256"],
                          extra={"model_dir": str(model_dir), "model_train_n": meta["n_train"],
                                 "selection": {"splits": list(splits) if splits else None,
                                               "seg_splits": list(seg_splits) if seg_splits else None,
                                               "n_excluded_ids": len(set(map(str, exclude_ids)))}})


def evaluate_against_reference(frame: pd.DataFrame, cfg: dict, root, derived_manifest,
                               calibration_ids, model_meta: dict | None = None) -> pd.DataFrame:
    """Dice and clDice vs reference labels on non-training, non-calibration rows.

    For a U-Net run the model metadata is required (read from the run's
    ``segmentation_run.json`` when not supplied) and must match the weights that
    produced the masks; its training images and subjects are never scored.
    A derived manifest with label hash columns must name exactly the masks that
    are scored (``verify_derived_mask_files``); pre-fix manifests that copied the
    source label hashes are refused.
    """
    derived_manifest = Path(derived_manifest)
    seg = pd.read_csv(derived_manifest, dtype=str, keep_default_na=False)
    if any(f"{f}_sha256" in seg for f in LABEL_FIELDS):
        try:
            verify_derived_mask_files(seg)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{derived_manifest}: label path/hash identities do not correspond ({exc}); a "
                             "derived manifest written before the identity fix needs a repaired copy") from exc
    seg = seg.set_index("image_id")
    run_path = derived_manifest.parent / "segmentation_run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    if run.get("segmenter") == "unet":
        if model_meta is None:
            model_meta = json.loads((Path(run["model_dir"]) / "meta.json").read_text())
        if model_meta["model_sha256"] != run["params"]["model_sha256"]:
            raise ValueError("model metadata does not match the weights that produced these masks")
    cal = set(map(str, calibration_ids))
    rows = []
    for r in frame.itertuples():
        if r.image_id not in seg.index or r.image_id in cal or bool(getattr(r, "calibration", False)) \
                or not r.vessel_mask_path:
            continue
        if model_meta is not None and (r.image_id in model_meta["train_image_ids"]
                                       or str(r.group_key) in model_meta["train_group_keys"]):
            continue
        ref = _reference_vessel(r, cfg, root)
        pred = read_mask(seg.loc[r.image_id, "vessel_mask_path"], {"mode": "binary"})
        if pred.shape != ref.shape:
            raise ValueError(f"{r.image_id}: prediction shape {pred.shape} != reference {ref.shape}")
        rows.append({"image_id": r.image_id, "seg_split": r.seg_split, "fov_mm": r.fov_mm,
                     "dice": unet_mod.dice(pred, ref), "cl_dice": unet_mod.cl_dice(pred, ref),
                     "pred_vessel_fraction": float(pred.mean()), "ref_vessel_fraction": float(ref.mean())})
    return pd.DataFrame(rows, columns=["image_id", "seg_split", "fov_mm", "dice", "cl_dice",
                                       "pred_vessel_fraction", "ref_vessel_fraction"])


def summarize_evaluation(df: pd.DataFrame) -> dict:
    """Descriptive summaries by seg_split and FOV (no model or threshold is chosen from them).

    Means and medians are over finite values only, so every block states its
    denominators (``n_dice_finite``, ``n_cl_dice_finite``) and the empty-mask
    counts. A complete miss (empty prediction, non-empty reference) scores 0 and
    stays in every average; only images with both masks empty are undefined.
    """
    def block(g):
        pe, re_ = g.pred_vessel_fraction.astype(float).eq(0), g.ref_vessel_fraction.astype(float).eq(0)
        out = {"n": len(g), "n_dice_finite": int(np.isfinite(g.dice.to_numpy(float)).sum()),
               "n_cl_dice_finite": int(np.isfinite(g.cl_dice.to_numpy(float)).sum()),
               "n_pred_empty_ref_nonempty": int((pe & ~re_).sum()),
               "n_ref_empty_pred_nonempty": int((re_ & ~pe).sum()), "n_both_empty": int((pe & re_).sum())}
        for m in ("dice", "cl_dice"):
            v = g[m].to_numpy(float)
            v = v[np.isfinite(v)]
            out[f"{m}_median"] = float(np.median(v)) if v.size else None
            out[f"{m}_mean"] = float(np.mean(v)) if v.size else None
        return out

    out = {"overall": block(df),
           "convention": ("dice/cl_dice: one empty mask = 0 (complete miss, kept in averages); both empty = "
                          "undefined, excluded from the finite denominators")}
    for keys in (["seg_split"], ["seg_split", "fov_mm"]):
        for k, g in df.groupby(keys):
            out[" / ".join(f"{a}={b}" for a, b in zip(keys, k if isinstance(k, tuple) else (k,)))] = block(g)
    return out
