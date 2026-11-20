"""Human segmentation-QC gate for external data (Experiment E/F), bound to exact masks.

A QC decision is valid only for the mask a human actually looked at. The gate
therefore works on a *hash-bound review sheet* for one segmentation run
(``make_qc_sheet``), not on image IDs alone. Each row carries

    image_id, segmenter, segmentation_config_sha256, graph_review_sha256,
    image_sha256, mask_sha256, reviewer, review_date, decision, fail_reason

``segmentation_config_sha256`` identifies the segmentation method: segmenter,
effective parameters, model weights and a fingerprint of the segmenter source.
``graph_review_sha256`` identifies how the reviewed graph overlays were built:
graph cleanup config, analysis-mask settings, FAZ handling and a fingerprint
of the graph-extraction source. ``mask_sha256`` is the reviewed mask file.

``qc_status`` reports ready only if all of the following hold:

* every required column is present (a legacy ID-only queue gets an actionable
  message naming the ``qc-sheet`` command);
* image IDs are unique (duplicates never count as independent reviews);
* every row belongs to the analysed training cohort and to the requested
  segmenter;
* each decision is ``pass`` or ``fail`` with a named reviewer and an ISO date
  that is not in the future, and every ``fail`` has a reason;
* SYNTHETIC reviewers are refused for real data;
* the sheet's segmentation and graph identities equal those of the analysed
  run, and every reviewed mask/image hash equals the file that is analysed;
* at least ``min_audited`` distinct images are audited, and the failure
  fraction is at most ``max_fail_fraction`` (prespecified,
  resume_review.md B4).

Without binding information (``binding=None``) the gate is never ready. It
then only reports schema and completeness problems.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd

from .config import config_hash
from .provenance import code_fingerprint, now_utc, read_json, sha256_file, write_json

QC_SHEET_FORMAT = "retina_fragility.qc_sheet/1"
REQUIRED_COLUMNS = ("image_id", "segmenter", "segmentation_config_sha256", "graph_review_sha256",
                    "image_sha256", "mask_sha256", "reviewer", "review_date", "decision",
                    "fail_reason")
SHEET_COLUMNS = REQUIRED_COLUMNS + ("mask_path", "review_directory", "notes")
PASS = {"pass"}
FAIL = {"fail"}
SEGMENTER_CODE = {"frangi": ("retina_fragility.segmentation.frangi", "retina_fragility.io_masks"),
                  "unet": ("retina_fragility.segmentation.unet", "retina_fragility.io_masks")}
GRAPH_CODE = ("retina_fragility.graph.extract", "retina_fragility.graph.cleanup")


class QCError(ValueError):
    pass


def segmentation_identity(segmenter: str, params: dict, model_sha256: str | None = None) -> dict:
    """Identity of a segmentation method (not of an image list)."""
    if segmenter not in SEGMENTER_CODE:
        raise QCError(f"unknown segmenter {segmenter}")
    code = code_fingerprint(SEGMENTER_CODE[segmenter])
    ident = {"segmenter": segmenter, "params": params, "model_sha256": model_sha256,
             "code_sha256": code["sha256"]}
    return {"segmentation_config_sha256": config_hash(ident), "identity": ident,
            "code_files": code["files"]}


def graph_review_identity(cfg: dict, faz_spec: dict) -> dict:
    code = code_fingerprint(GRAPH_CODE)
    ident = {"graph": cfg["graph"], "analysis_mask": cfg["analysis_mask"], "faz_spec": dict(faz_spec),
             "code_sha256": code["sha256"]}
    return {"graph_review_sha256": config_hash(ident), "identity": ident}


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def qc_status(qc_csv: str | Path, segmenter: str, min_audited: int, max_fail_fraction: float,
              allow_synthetic_reviewer: bool = False, binding: dict | None = None) -> dict:
    """Readiness of human QC for ``segmenter``. See the module docstring for the rules.

    ``binding`` (from ``experiments.exp_e.qc_binding``) holds ``cohort_ids``,
    ``mask_sha256`` (image -> analysed mask hash), ``image_sha256`` (optional),
    ``segmentation_config_sha256`` and ``graph_review_sha256`` of the analysed run.
    Reviewer names starting with SYNTHETIC mark automated fixture QC; they count
    only when ``allow_synthetic_reviewer`` (synthetic fixture data), never for real data.
    """
    p = Path(qc_csv)
    base = {"ready": False, "n_rows": 0, "n_audited": 0, "n_failed": 0, "fail_fraction": float("nan"),
            "failed_image_ids": [], "failed_reasons": {}, "binding_verified": False,
            "qc_file": str(p), "required_columns": list(REQUIRED_COLUMNS)}
    incomplete = f"human QC incomplete for {segmenter}: 0/{int(min_audited)} audited rows"
    if not p.exists():
        return base | {"reasons": [f"QC file missing: {p}", incomplete]}
    q = pd.read_csv(p, dtype=str, keep_default_na=False)
    base["n_rows"] = int(len(q))
    missing = [c for c in REQUIRED_COLUMNS if c not in q.columns]
    if missing:
        legacy = f"{segmenter}_pass" in q.columns
        msg = (f"QC file {p} is missing required columns {missing}"
               + (" (legacy ID-only queue: decisions would not be bound to masks)" if legacy else "")
               + ". Create a hash-bound review sheet for the segmentation run being analysed with "
                 "`retina-fragility qc-sheet --segmented <segmentation dir> --selection <queue csv> "
                 "--out <sheet csv>` (or `qc-export`), record decisions there, and pass it via --qc.")
        return base | {"reasons": [msg, incomplete]}
    for c in REQUIRED_COLUMNS:
        q[c] = _norm(q[c])
    reasons: list[str] = []
    ids = q.image_id
    if ids.eq("").any():
        reasons.append(f"{int(ids.eq('').sum())} QC rows have no image_id")
    dup = sorted(ids[ids.duplicated(keep=False) & ids.ne("")].unique())
    if dup:
        reasons.append(f"duplicate image IDs in QC sheet (never counted as independent reviews): {dup[:10]}")
    wrong_seg = q.segmenter.ne(segmenter)
    if wrong_seg.any():
        reasons.append(f"{int(wrong_seg.sum())} QC rows are for another segmenter "
                       f"({sorted(set(q.segmenter[wrong_seg]))}), not {segmenter}")
    dec = q.decision.str.lower()
    decided = dec.ne("")
    bad_dec = decided & ~dec.isin(PASS | FAIL)
    if bad_dec.any():
        reasons.append(f"unrecognised QC decisions (use pass or fail): {sorted(set(dec[bad_dec]))}")
    no_attr = decided & (q.reviewer.eq("") | q.review_date.eq(""))
    if no_attr.any():
        reasons.append(f"decisions without reviewer or review_date: {ids[no_attr].tolist()[:10]}")
    bad_date = []
    today = _dt.datetime.now(_dt.timezone.utc).date()
    for i, d in zip(ids, q.review_date):
        if not d:
            continue
        try:
            if _dt.date.fromisoformat(d) > today:
                bad_date.append(i)
        except ValueError:
            bad_date.append(i)
    if bad_date:
        reasons.append(f"review_date must be an ISO date (YYYY-MM-DD) not in the future: {bad_date[:10]}")
    fail_no_reason = dec.isin(FAIL) & q.fail_reason.eq("")
    if fail_no_reason.any():
        reasons.append(f"failed decisions without fail_reason: {ids[fail_no_reason].tolist()[:10]}")
    synthetic_rev = q.reviewer.str.upper().str.startswith("SYNTHETIC")
    if synthetic_rev.any() and not allow_synthetic_reviewer:
        reasons.append(f"{int(synthetic_rev.sum())} QC rows have a SYNTHETIC (non-human) reviewer; "
                       "they do not count as human QC for real data")
    valid = (decided & ~bad_dec & ~no_attr & ~ids.isin(bad_date) & ~fail_no_reason & ~wrong_seg
             & ~ids.isin(dup) & ids.ne(""))
    if not allow_synthetic_reviewer:
        valid &= ~synthetic_rev

    bind_reasons: list[str] = []
    if binding is None:
        bind_reasons.append("QC binding not verified: no analysed segmentation run/pipeline supplied "
                            "(cohort, mask hashes and run identities are checked only by the E/F gate)")
    else:
        cohort = set(map(str, binding.get("cohort_ids") or ()))
        outside = sorted(set(ids[ids.ne("")]) - cohort)
        if outside:
            bind_reasons.append(f"QC rows outside the analysed training cohort: {outside[:10]}")
        for col, key in (("segmentation_config_sha256", "segmentation_config_sha256"),
                         ("graph_review_sha256", "graph_review_sha256")):
            want = binding.get(key)
            vals = set(q[col])
            if not want:
                bind_reasons.append(f"analysed run has no {key}; cannot bind QC decisions")
            elif vals != {want}:
                bind_reasons.append(f"QC sheet {col} {sorted(vals)[:3]} does not match the analysed "
                                    f"run ({want[:12]}…); the reviewed masks/graphs are not the "
                                    "analysed ones")
        masks = binding.get("mask_sha256") or {}
        mism, absent = [], []
        for i, h in zip(ids, q.mask_sha256):
            if i not in masks:
                absent.append(i)
            elif masks[i] != h:
                mism.append(i)
        if absent:
            bind_reasons.append(f"reviewed images have no analysed mask: {absent[:10]}")
        if mism:
            bind_reasons.append(f"reviewed mask hash differs from the analysed mask: {mism[:10]}")
        imgs = binding.get("image_sha256") or {}
        img_mism = [i for i, h in zip(ids, q.image_sha256) if i in imgs and imgs[i] and imgs[i] != h]
        if img_mism:
            bind_reasons.append(f"reviewed source image hash differs from the manifest: {img_mism[:10]}")
        valid &= ~ids.isin(outside + mism + absent + img_mism)
    reasons += bind_reasons
    n_aud = int(ids[valid].nunique())
    failed_mask = valid & dec.isin(FAIL)
    n_fail = int(ids[failed_mask].nunique())
    if n_aud < int(min_audited):
        reasons.append(f"human QC incomplete for {segmenter}: {n_aud}/{int(min_audited)} audited rows")
    frac = n_fail / n_aud if n_aud else float("nan")
    if n_aud and frac > float(max_fail_fraction):
        reasons.append(f"{segmenter} failed QC: {n_fail}/{n_aud} audited images failed "
                       f"(> {max_fail_fraction:.0%} prespecified maximum)")
    failed = q.loc[failed_mask, ["image_id", "fail_reason"]]
    return base | {"ready": not reasons, "reasons": reasons, "n_audited": n_aud, "n_failed": n_fail,
                   "fail_fraction": frac, "failed_image_ids": failed.image_id.tolist(),
                   "failed_reasons": dict(zip(failed.image_id, failed.fail_reason)),
                   "binding_verified": binding is not None and not bind_reasons}


# ------------------------------------------------------------------ review sheets
def load_segmentation_run(segmented_dir: str | Path) -> tuple[dict, pd.DataFrame]:
    d = Path(segmented_dir)
    run_p, man_p = d / "segmentation_run.json", d / "manifest_segmented.csv"
    if not run_p.exists() or not man_p.exists():
        raise QCError(f"{d} is not a segmentation run directory (needs segmentation_run.json and "
                      "manifest_segmented.csv)")
    run = read_json(run_p)
    if not run.get("segmentation_config_sha256"):
        raise QCError(
            f"segmentation run {d} predates segmentation identities (no segmentation_config_sha256), "
            "so reviewed masks cannot be bound to a method. Re-run `segment` with the same settings "
            "into a new directory (Frangi is deterministic; mask hashes can be compared with the old "
            "inventory) and create the review sheet from that run.")
    man = pd.read_csv(man_p, dtype=str, keep_default_na=False)
    return run, man


def _has_decisions(path: Path) -> bool:
    q = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = [c for c in ("reviewer", "review_date", "decision", "fail_reason") if c in q.columns]
    return bool(cols) and bool((q[cols].apply(lambda s: s.str.strip()).ne("")).any().any())


def make_qc_sheet(segmented_dir: str | Path, selection_csv: str | Path, cfg: dict, root: str | Path,
                  faz_spec: dict, out_csv: str | Path, review_dirs: dict[str, str] | None = None,
                  overwrite_blank: bool = False) -> pd.DataFrame:
    """Write a hash-bound review sheet with blank decisions for the selected images.

    Never fills reviewer, date or decision. Refuses to overwrite a sheet that
    already holds any human entry.
    """
    from .manifest import resolve_input

    out_csv = Path(out_csv)
    if out_csv.exists() and (_has_decisions(out_csv) or not overwrite_blank):
        raise QCError(f"{out_csv} exists" + (" and contains review entries; it is never overwritten"
                                              if _has_decisions(out_csv) else "; pass --overwrite-blank"))
    run, man = load_segmentation_run(segmented_dir)
    sel = pd.read_csv(selection_csv, dtype=str, keep_default_na=False)
    if "image_id" not in sel.columns:
        raise QCError(f"selection {selection_csv} has no image_id column")
    sel["image_id"] = _norm(sel.image_id)
    dup = sorted(sel.image_id[sel.image_id.duplicated()].unique())
    if dup:
        raise QCError(f"selection has duplicate image IDs: {dup[:10]}")
    man = man.set_index("image_id")
    absent = sorted(set(sel.image_id) - set(man.index))
    if absent:
        raise QCError(f"selected images are not in segmentation run {segmented_dir}: {absent[:10]}")
    graph_id = graph_review_identity(cfg, faz_spec)
    rows = []
    for r in sel.itertuples():
        m = man.loc[r.image_id]
        if m.get("split", "").lower() == "test":
            raise QCError(f"{r.image_id} is a hidden-label test image; not part of the training cohort")
        img_h = sha256_file(resolve_input(m["image_path"], root))
        if getattr(r, "image_sha256", "") and r.image_sha256 != img_h:
            raise QCError(f"selected source image changed since selection: {r.image_id}")
        if m.get("image_sha256", "") and m["image_sha256"] != img_h:
            raise QCError(f"source image hash differs from the manifest: {r.image_id}")
        mask_p = resolve_input(m["vessel_mask_path"], root)
        rows.append({"image_id": r.image_id, "segmenter": run["segmenter"],
                     "segmentation_config_sha256": run["segmentation_config_sha256"],
                     "graph_review_sha256": graph_id["graph_review_sha256"],
                     "image_sha256": img_h, "mask_sha256": sha256_file(mask_p),
                     "reviewer": "", "review_date": "", "decision": "", "fail_reason": "",
                     "mask_path": str(m["vessel_mask_path"]),
                     "review_directory": (review_dirs or {}).get(r.image_id, ""), "notes": ""})
    sheet = pd.DataFrame(rows, columns=list(SHEET_COLUMNS))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out_csv, index=False)
    write_json(out_csv.with_suffix(".json"), {
        "format": QC_SHEET_FORMAT, "created_utc": now_utc(), "segmented_dir": str(segmented_dir),
        "segmenter": run["segmenter"], "segmentation_config_sha256": run["segmentation_config_sha256"],
        "segmentation_identity": run.get("segmentation_identity"),
        "graph_review_sha256": graph_id["graph_review_sha256"], "graph_review_identity": graph_id["identity"],
        "selection_csv": str(selection_csv), "selection_sha256": sha256_file(selection_csv),
        "n_rows": len(sheet), "human_qc_completed": False,
        "instructions": ("Fill reviewer, review_date (YYYY-MM-DD), decision (pass|fail) and, for fail, "
                         "fail_reason, after viewing the native-resolution images of exactly this mask. "
                         "Do not edit the hash columns. Any re-segmentation needs a new sheet."),
    })
    return sheet
