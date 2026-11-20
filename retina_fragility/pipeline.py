"""Per-image processing: masks -> graph -> cleanup -> stress tests -> features.

Every image gets its own output directory containing

    graph.json            cleaned multigraph (pixel paths, µm lengths/widths)
    cleanup_audit.json    per-stage graph statistics (pre- and post-clean)
    curves.csv            S(q), P(q), q_edge, q_len per order (random = mean curve)
    random_reps.csv       per-repetition random summaries
    fragility_map.csv     per-edge bridge cut-off, ΔS, unique coverage, removal q
    features.json         biomarkers + stress features (NaN never replaced)
    provenance.json       cache identity (config subset, image geometry, input hashes,
                          code fingerprint, numerical library versions), seeds, versions
    overlay_*.png         edge values over the OCTA intensity image; ``*_on_mask.png``
                          when no intensity image is available (mask-only diagnostic)
    qc_graph_on_intensity.png  native-resolution graph over the intensity image
    curves.png

Images whose cleaned graph is empty, or whose analysis area is empty, are
*excluded* with a logged reason. They never receive a fake FI.

Integration safeguards (resume_review.md D1-D5):

* Output directory names are collision-safe (``safe_name``). Duplicate IDs and
  name collisions are refused before anything is written.
* Cached outputs are reused only when the full cache identity matches. The
  identity covers the pipeline config subset, the per-image geometry
  (pixel size, shape, crop), input file hashes (masks and intensity image),
  a fingerprint of the package source the pipeline imports, and numerical
  library versions.
* Each image is processed into a private staging directory, which then
  atomically replaces the previous output directory. An input error, a
  processing error or an empty graph therefore never leaves curves, overlays
  or graphs from an earlier successful run next to the new exclusion.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .biomarkers import compute_biomarkers
from .config import config_hash
from .graph.betweenness import edge_betweenness
from .graph.cleanup import clean_graph
from .graph.coverage import CoverageTracker, centerline_pixel_sets, vessel_territories
from .graph.extract import skeletonize_mask, trace_graph
from .graph.fragility_map import fragility_map
from .graph.stress import FI_CAVEATS, ORDER_LABELS, batch_size, forman_curvature, stress_suite
from .io_masks import (
    MaskError,
    faz_auto_mask,
    faz_disk_mask,
    make_analysis_mask,
    read_intensity,
    read_mask,
)
from .manifest import SYNTHETIC_PREFIX, resolve_input
from .provenance import (
    SYNTHETIC_BANNER,
    assert_outside_inputs,
    code_fingerprint,
    environment,
    now_utc,
    numerical_library_versions,
    read_json,
    sha256_file,
    write_json,
)
from .seeds import derive_seed, rng_for

PIPELINE_VERSION = "1"
# Entry points whose static import closure defines the per-image code fingerprint.
PIPELINE_ENTRY_MODULES = ("retina_fragility.pipeline", "retina_fragility.viz")

COVERAGE_METHOD_NOTES = {
    "vessel_territory": (
        "SECONDARY-METHOD DEVIATION from the guide's pixel-path coverage recipe: each edge "
        "covers tissue within d of its vessel territory (vessel-mask pixels nearest to its "
        "skeleton pixels), to match the calibration distance measured to vessel pixels "
        "(resume_review.md B1). The guide-compatible pixel-path calculation is "
        "stress.coverage_geometry: centerline (also run by the coverage_geometry ablation). "
        "Coverage features (P0, FIperf) are secondary; the primary FI does not use d."),
    "centerline": (
        "Guide-compatible pixel-path coverage: each edge covers tissue within d of its "
        "interior path pixels plus attachment node clusters."),
}

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class PipelineInputError(ValueError):
    pass


def safe_name(image_id: str) -> str:
    """Deterministic, collision-resistant directory/file name for an image ID.

    IDs that are already portable path components (start alphanumeric, then only
    ``[A-Za-z0-9._-]``, at most 128 characters) are used unchanged. Any other ID,
    including ``.``/``..``, IDs containing ``/`` and empty-looking IDs, is
    sanitised and suffixed with 16 hex digits of its SHA-256, so ``a/b`` and
    ``a_b`` map to different names. ``check_output_names`` still verifies
    uniqueness for every run before anything is written.
    """
    s = str(image_id)
    if not s.strip():
        raise PipelineInputError("image_id must be non-empty")
    if _SAFE_ID.fullmatch(s):
        return s
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-")[:64] or "id"
    return f"{stem}__{hashlib.sha256(s.encode()).hexdigest()[:16]}"


def check_output_names(image_ids) -> dict[str, str]:
    """Map ID -> output name; raise on duplicate IDs or on colliding names.

    Collisions are also checked case-insensitively, so outputs stay distinct on
    case-insensitive file systems.
    """
    ids = [str(i) for i in image_ids]
    dup = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dup:
        raise PipelineInputError(f"duplicate image IDs in one run: {dup[:10]}")
    names = {i: safe_name(i) for i in ids}
    seen: dict[str, str] = {}
    clashes = []
    for i, n in names.items():
        key = n.casefold()
        if key in seen:
            clashes.append((seen[key], i, n))
        else:
            seen[key] = i
    if clashes:
        raise PipelineInputError(f"image IDs map to the same output name: {clashes[:10]}")
    return names


@functools.cache
def pipeline_code_fingerprint() -> dict:
    return code_fingerprint(PIPELINE_ENTRY_MODULES)


def pipeline_config_subset(cfg: dict, d_um: float | None, faz_spec: dict) -> dict:
    return {"version": PIPELINE_VERSION, "graph": cfg["graph"], "stress": cfg["stress"],
            "analysis_mask": cfg["analysis_mask"], "label_maps": cfg["data"]["label_maps"],
            "base_seed": cfg["project"]["base_seed"], "d_um": d_um, "faz_spec": faz_spec,
            "crop_center_mm": cfg["pipeline"]["crop_center_mm"],
            "overlays": cfg["pipeline"]["overlays"]}


def cache_identity(cfg: dict, d_um: float | None, faz_spec: dict, geometry: dict,
                   input_sha256: dict) -> dict:
    """Everything that determines an image's saved outputs (compared before reuse)."""
    return {"pipeline_config_sha256": config_hash(pipeline_config_subset(cfg, d_um, faz_spec)),
            "geometry": geometry, "input_sha256": input_sha256,
            "code_sha256": pipeline_code_fingerprint()["sha256"],
            "numerical_libraries": numerical_library_versions()}


def crop_center(masks: dict, crop_mm: float | None) -> dict:
    """Central square crop of side ``crop_mm`` (no-op when the image is not larger)."""
    if crop_mm is None:
        return masks
    px = masks["pixel_size_um"]
    H, W = masks["vessel"].shape
    side = int(round(float(crop_mm) * 1000.0 / px))
    if side >= min(H, W):
        masks["crop"] = {"crop_mm": crop_mm, "applied": False}
        return masks
    y0, x0 = (H - side) // 2, (W - side) // 2
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    masks["vessel"] = masks["vessel"][sl]
    if masks["faz"] is not None:
        masks["faz"] = masks["faz"][sl]
    if masks.get("intensity") is not None:
        masks["intensity"] = masks["intensity"][sl]
    masks["crop"] = {"crop_mm": crop_mm, "applied": True, "origin_px": [y0, x0], "side_px": side}
    return masks


# ----------------------------------------------------------------------------- inputs
def load_row_masks(row: dict, cfg: dict, root: str | Path, faz_spec: dict) -> dict:
    """Decode the vessel (union of capillary + large vessel) and FAZ masks of a row.

    The OCTA intensity image (``image_path``) is read when present, for overlays
    only. It never enters any feature. An unreadable or mis-shaped intensity
    image is recorded in ``intensity_note``; overlays then fall back to the
    labelled mask-only diagnostic.
    """
    maps = cfg["data"]["label_maps"]
    hashes = {}
    vpath = resolve_input(row["vessel_mask_path"], root)
    vessel = read_mask(vpath, maps["vessel"])
    hashes["vessel"] = sha256_file(vpath)
    large = str(row.get("large_vessel_mask_path") or "")
    if large:
        lpath = resolve_input(large, root)
        lv = read_mask(lpath, maps["large_vessel"])
        if lv.shape != vessel.shape:
            raise MaskError("Large-vessel and capillary mask dimensions differ")
        vessel = vessel | lv
        hashes["large_vessel"] = sha256_file(lpath)
    mode = faz_spec.get("mode", "reference")
    faz_info = {"mode": mode}
    if mode == "reference":
        fpath = resolve_input(row["faz_mask_path"], root)
        faz = read_mask(fpath, maps["faz"])
        if faz.shape != vessel.shape:
            raise MaskError("Vessel and FAZ mask dimensions differ")
        hashes["faz"] = sha256_file(fpath)
    elif mode == "disk":
        faz = faz_disk_mask(vessel.shape, float(faz_spec["radius_px"]),
                            faz_spec.get("center", "image_center"))
        faz_info.update(radius_px=float(faz_spec["radius_px"]))
    elif mode == "auto":
        faz, info = faz_auto_mask(vessel)
        faz_info.update(info)
    elif mode == "none":
        faz = None
    else:
        raise PipelineInputError(f"unknown FAZ mode {mode}")
    px = row.get("pixel_size_um")
    if px is None or (isinstance(px, float) and not np.isfinite(px)) or px == "":
        raise PipelineInputError("pixel_size_um is required: the per-image pipeline works in "
                                 "physical units only (pixel-unit mode is not implemented)")
    intensity, note = None, "no image_path in manifest row"
    ipath_text = str(row.get("image_path") or "").strip()
    if ipath_text:
        try:
            ipath = resolve_input(ipath_text, root)
            img, _ = read_intensity(ipath)
            if img.shape != vessel.shape:
                note = f"intensity shape {img.shape} differs from mask shape {vessel.shape}"
            else:
                intensity, note = img, "ok"
                hashes["intensity"] = sha256_file(ipath)
        except (OSError, ValueError) as exc:
            note = f"intensity image unreadable: {exc}"
    return {"vessel": vessel, "faz": faz, "pixel_size_um": float(px), "input_sha256": hashes,
            "faz_info": faz_info, "intensity": intensity, "intensity_note": note}


# --------------------------------------------------------------------------- core
def build_graph(vessel: np.ndarray, faz: np.ndarray | None, pixel_size_um: float, cfg: dict,
                meta: dict | None = None) -> dict:
    """Analysis mask, skeleton, raw and cleaned graph (shared by the pipeline and QC export)."""
    vessel = np.asarray(vessel, bool)
    gcfg = cfg["graph"]
    am_settings = dict(cfg["analysis_mask"])
    if faz is None:
        am_settings["exclude_faz"] = False
    analysis = make_analysis_mask(vessel.shape, pixel_size_um, faz, am_settings)
    skel = skeletonize_mask(vessel)
    raw = trace_graph(skel, pixel_size_um, vessel_mask=vessel,
                      force_2x2=gcfg["force_2x2_blocks_as_junctions"], meta=meta)
    area_mm2 = float(analysis.sum()) * (pixel_size_um / 1000.0) ** 2
    clean, audit = clean_graph(raw, gcfg, faz=faz, area_mm2=area_mm2)
    return {"analysis": analysis, "skeleton": skel, "raw": raw, "clean": clean, "audit": audit}


def forman_tie_diagnostics(clean, step_frac: float) -> dict:
    """How much of the first Forman-type batch is decided by the edge-id tie-break."""
    c = clean.compact()
    m = len(c["eu"])
    F = forman_curvature(c["n"], c["eu"], c["ev"], np.ones(m, bool))
    vals, counts = np.unique(F, return_counts=True)
    b = batch_size(m, step_frac)
    score = np.sort(-F)[::-1]  # removal score, most negative curvature first
    v = score[b - 1]
    n_above = int((score > v).sum())
    n_tied = int((score == v).sum())
    tie_broken = (b - n_above) if n_tied > (b - n_above) else 0
    return {"forman_initial_n_distinct_scores": int(len(vals)),
            "forman_initial_tied_edge_fraction": float(counts[counts > 1].sum() / m),
            "forman_first_batch_tiebreak_fraction": float(tie_broken / b)}


def process_masks(image_id: str, vessel: np.ndarray, faz: np.ndarray | None, pixel_size_um: float,
                  cfg: dict, d_um: float | None, synthetic: bool, out_dir: str | Path | None = None,
                  meta: dict | None = None, intensity: np.ndarray | None = None) -> dict:
    t0 = time.monotonic()
    vessel = np.asarray(vessel, bool)
    scfg = cfg["stress"]
    base_seed = cfg["project"]["base_seed"]
    result: dict = {"image_id": image_id, "status": "ok", "reason": None,
                    "data_provenance": SYNTHETIC_BANNER if synthetic else "real"}
    try:
        g = build_graph(vessel, faz, pixel_size_um, cfg, meta={"image_id": image_id, **(meta or {})})
    except MaskError as exc:
        result.update(status="excluded", reason="empty_analysis_area", details=str(exc))
        return result
    analysis, skel, raw, clean, audit = g["analysis"], g["skeleton"], g["raw"], g["clean"], g["audit"]
    bio = compute_biomarkers(vessel, faz, analysis, skel, raw, clean, pixel_size_um)
    feats: dict = {"image_id": image_id, **bio}
    result["cleanup_audit"] = audit
    if clean.n_edges == 0 or clean.total_length() <= 0:
        result.update(status="excluded", reason="empty_graph",
                      details="cleaned graph has no edges; FI undefined (not imputed)")
        result["features"] = feats
        if out_dir is not None:
            _save(out_dir, result, clean, None, None, vessel, None, cfg, synthetic, intensity=intensity)
        return result

    tracker = None
    coverage_note = None
    if scfg["compute_coverage"]:
        if d_um is None:
            coverage_note = "no_frozen_calibration_d"
        else:
            if scfg["coverage_geometry"] == "vessel_territory":
                sets = vessel_territories(clean, skel, vessel)
            else:
                sets = centerline_pixel_sets(clean)
            tracker = CoverageTracker(clean, analysis, d_um=d_um, edge_pixel_sets=sets)
            feats["P0"] = tracker.fraction()
    else:
        coverage_note = "coverage_disabled"
    feats["coverage_d_um"] = d_um if tracker is not None else np.nan
    feats["coverage_note"] = coverage_note
    feats["coverage_geometry"] = scfg["coverage_geometry"] if tracker is not None else None

    def rng_factory(rep: int):
        return rng_for(base_seed, "random_removal", image_id, rep)

    suite = stress_suite(clean, scfg["orders"], scfg["step_frac"], scfg["random_repetitions"],
                         rng_factory, scfg["betweenness_backend"], tracker,
                         scfg["coverage_orders"] if tracker is not None else (),
                         scfg["q_eval"], scfg["s_threshold"], scfg["tie_rel_quantum"])
    feats.update(suite["features"])
    feats["m0_edges"] = clean.n_edges
    feats["L0_um"] = clean.total_length()
    feats["batch_size_edges"] = batch_size(clean.n_edges, scfg["step_frac"])
    if "forman" in scfg["orders"]:
        feats.update(forman_tie_diagnostics(clean, scfg["step_frac"]))

    fmap = fragility_map(clean)
    c = clean.compact()
    bt = edge_betweenness(c["n"], c["eu"], c["ev"], c["length"], None, scfg["betweenness_backend"])
    fmap["initial_betweenness"] = bt
    if tracker is not None:
        tracker.reset()
        fmap["unique_coverage_mm2"] = tracker.unique_area_px() * tracker.pixel_area_mm2
    for order, cv in suite["curves"].items():
        if cv.removal_batches:
            qmap = {}
            for i, batch in enumerate(cv.removal_batches):
                for eid in batch:
                    qmap[eid] = float(cv.q_edge[i + 1])
            fmap[f"removal_q_{order}"] = fmap.edge_id.map(qmap)
    result["features"] = feats
    result["elapsed_seconds"] = time.monotonic() - t0
    if out_dir is not None:
        _save(out_dir, result, clean, suite, fmap, vessel, analysis, cfg, synthetic, raw=raw,
              intensity=intensity)
    return result


def _save(out_dir, result, clean, suite, fmap, vessel, analysis, cfg, synthetic, raw=None,
          intensity=None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean.save(out / "graph.json")
    if raw is not None and cfg["pipeline"]["save_raw_graph"]:
        raw.save(out / "graph_raw.json")
    write_json(out / "cleanup_audit.json", result.get("cleanup_audit", []))
    feats = dict(result.get("features", {}))
    feats.update(status=result["status"], reason=result["reason"],
                 data_provenance=result["data_provenance"], fi_caveats=FI_CAVEATS)
    write_json(out / "features.json", feats)
    if synthetic:
        (out / "SYNTHETIC_NOT_CLINICAL.txt").write_text(
            "Outputs in this directory derive from SYNTHETIC fixtures; not clinical findings.\n")
    if suite is not None:
        frames = [cv.to_frame() for cv in suite["curves"].values()]
        pd.concat(frames, ignore_index=True).to_csv(out / "curves.csv", index=False)
        pd.DataFrame(suite["random_reps"]).to_csv(out / "random_reps.csv", index=False)
        fmap.to_csv(out / "fragility_map.csv", index=False)
    if not cfg["pipeline"]["overlays"] or vessel is None:
        return
    from . import viz

    if intensity is not None:
        # native-resolution graph over the scan (empty graph -> scan + mask only)
        files = viz.qc_composites(intensity, vessel, clean if clean.n_edges else None, out / "_qc",
                                  synthetic)
        shutil.move(files["graph_on_intensity"], out / "qc_graph_on_intensity.png")
        shutil.rmtree(out / "_qc")
    if suite is None:
        return
    px = clean.pixel_size_um
    iid = result["image_id"]
    suffix = "" if intensity is not None else "_on_mask"
    viz.plot_curves(suite["curves"], out / "curves.png", f"{iid}: removal curves", synthetic)
    cut = dict(zip(fmap.edge_id, fmap.cutoff_length))
    viz.overlay(vessel, clean, cut, out / f"overlay_bridge_cutoff{suffix}.png",
                f"{iid}: bridge cut-off length", "cut-off vessel length (µm)", px, synthetic,
                cmap="magma", analysis=analysis, intensity=intensity)
    if "unique_coverage_mm2" in fmap:
        uc = dict(zip(fmap.edge_id, fmap.unique_coverage_mm2 * 1e3))
        viz.overlay(vessel, clean, uc, out / f"overlay_unique_coverage{suffix}.png",
                    f"{iid}: unique coverage", "unique coverage (10⁻³ mm²)", px, synthetic,
                    cmap="viridis", analysis=analysis, intensity=intensity)
    col = f"removal_q_{cfg['stress']['primary_order']}"
    if col in fmap:
        rq = dict(zip(fmap.edge_id, fmap[col]))
        viz.overlay(vessel, clean, rq, out / f"overlay_primary_removal_q{suffix}.png",
                    f"{iid}: removal fraction q ({cfg['stress']['primary_order']})",
                    "q at removal (edge fraction)", px, synthetic, cmap="plasma_r",
                    analysis=analysis, intensity=intensity)


# ----------------------------------------------------------------------- manifest run
def _staging_dir(images_dir: Path, name: str) -> Path:
    return images_dir / f".staging-{name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _publish(staging: Path, out_dir: Path) -> bool:
    """Atomically replace ``out_dir`` by ``staging``. Returns True if an old directory existed."""
    staging.mkdir(parents=True, exist_ok=True)
    old = None
    if out_dir.exists():
        old = out_dir.with_name(f".replaced-{out_dir.name}-{uuid.uuid4().hex[:8]}")
        out_dir.rename(old)
    staging.rename(out_dir)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    return old is not None


def _record_failure(images_dir: Path, name: str, rec: dict) -> dict:
    """Replace any previous outputs of this image by a provenance-only failure record."""
    out_dir = images_dir / name
    staging = _staging_dir(images_dir, name)
    staging.mkdir(parents=True, exist_ok=True)
    prov = {"image_id": rec["image_id"], "created_utc": now_utc(),
            "pipeline_version": PIPELINE_VERSION, "status": rec["status"], "stage": rec["stage"],
            "reason": rec["reason"], "details": rec.get("details"),
            "previous_outputs_removed": out_dir.exists(),
            "note": "No graph, curve, feature or overlay from any earlier run is kept for this image.",
            "environment": environment()}
    write_json(staging / "provenance.json", prov)
    _publish(staging, out_dir)
    return rec


def _job(args) -> dict:
    row, cfg, root, out_root, d_um, faz_spec = args
    image_id = row["image_id"]
    synthetic = str(row["dataset"]).upper().startswith(SYNTHETIC_PREFIX)
    name = safe_name(image_id)
    images_dir = Path(out_root) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_dir = images_dir / name
    try:
        masks = crop_center(load_row_masks(row, cfg, root, faz_spec),
                            cfg["pipeline"]["crop_center_mm"])
    except (OSError, ValueError) as exc:
        return _record_failure(images_dir, name, {"image_id": image_id, "status": "excluded",
                                                  "stage": "inputs", "reason": "input_error",
                                                  "details": str(exc)})
    geometry = {"pixel_size_um": masks["pixel_size_um"], "shape": list(masks["vessel"].shape),
                "crop": masks.get("crop"), "dataset": str(row["dataset"]),
                "intensity_background": masks["intensity"] is not None}
    identity = cache_identity(cfg, d_um, faz_spec, geometry, masks["input_sha256"])
    key = config_hash(identity)
    prov_path = out_dir / "provenance.json"
    if cfg["pipeline"]["skip_existing"] and prov_path.exists():
        prov = read_json(prov_path)
        if prov.get("cache_key_sha256") == key and (out_dir / "features.json").exists():
            feats = read_json(out_dir / "features.json")
            meta = {k: feats.pop(k, None) for k in ("status", "reason", "data_provenance", "fi_caveats")}
            feats = {k: (np.nan if v is None else v) for k, v in feats.items()}
            return {"image_id": image_id, "status": meta["status"] or "ok",
                    "stage": "reused", "reason": meta["reason"], "features": feats,
                    "details": "reused existing output with an identical cache identity"}
    staging = _staging_dir(images_dir, name)
    try:
        res = process_masks(image_id, masks["vessel"], masks["faz"], masks["pixel_size_um"],
                            cfg, d_um, synthetic, staging,
                            meta={"dataset": row["dataset"], "faz": masks["faz_info"],
                                  "crop": masks.get("crop")},
                            intensity=masks["intensity"])
    except Exception as exc:  # recorded, never silently dropped
        shutil.rmtree(staging, ignore_errors=True)
        return _record_failure(images_dir, name, {
            "image_id": image_id, "status": "excluded", "stage": "processing",
            "reason": "processing_error", "details": f"{exc!r}\n{traceback.format_exc()}"})
    geom = cfg["stress"]["coverage_geometry"]
    prov = {"image_id": image_id, "output_dir_name": name, "created_utc": now_utc(),
            "pipeline_version": PIPELINE_VERSION, "cache_key_sha256": key, "cache_identity": identity,
            "pipeline_config_sha256": identity["pipeline_config_sha256"],
            "full_config_sha256": config_hash(cfg),
            "input_sha256": masks["input_sha256"], "faz": masks["faz_info"],
            "pixel_size_um": masks["pixel_size_um"], "d_um": d_um,
            "coverage_geometry": geom, "coverage_method_note": COVERAGE_METHOD_NOTES[geom],
            "intensity_background": {"available": masks["intensity"] is not None,
                                     "note": masks["intensity_note"]},
            "random_seed_rule": "derive_seed(base_seed, 'random_removal', image_id, rep)",
            "random_seed_rep0": derive_seed(cfg["project"]["base_seed"], "random_removal",
                                            image_id, 0),
            "data_provenance": res["data_provenance"], "status": res["status"],
            "reason": res.get("reason"), "elapsed_seconds": res.get("elapsed_seconds"),
            "environment": environment()}
    staging.mkdir(parents=True, exist_ok=True)
    write_json(staging / "provenance.json", prov)
    _publish(staging, out_dir)
    return {"image_id": image_id, "status": res["status"], "stage": "processing",
            "reason": res.get("reason"), "details": res.get("details"),
            "features": res.get("features")}


def run_manifest(frame: pd.DataFrame, cfg: dict, root: str | Path, out_root: str | Path,
                 d_um: float | None, faz_spec: dict | None = None,
                 n_workers: int | None = None, calibration_record: dict | None = None) -> dict:
    """Process every row of ``frame``; write features.csv, exclusions.csv and run.json."""
    out_root = Path(out_root)
    assert_outside_inputs(out_root, [Path(root) / r for r in cfg["data"]["input_roots"]])
    names = check_output_names(frame["image_id"].astype(str))  # before anything is written
    out_root.mkdir(parents=True, exist_ok=True)
    faz_spec = faz_spec or {"mode": "reference"}
    rows = frame.to_dict("records")
    jobs = [(r, cfg, str(root), str(out_root), d_um, faz_spec) for r in rows]
    n_workers = int(n_workers or cfg["project"]["n_workers"])
    t0 = time.monotonic()
    if n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_job, jobs, chunksize=1))
    else:
        results = [_job(j) for j in jobs]
    feats, excl = [], []
    for r in results:
        if r["status"] == "ok" and r.get("features"):
            feats.append(r["features"])
        else:
            excl.append({"image_id": r["image_id"], "stage": r.get("stage"),
                         "reason": r.get("reason"), "details": r.get("details")})
    synthetic = bool(frame["dataset"].astype(str).str.upper().str.startswith(SYNTHETIC_PREFIX).any())
    fdf = pd.DataFrame(feats)
    if not fdf.empty:
        fdf = fdf.sort_values("image_id")
    fdf.to_csv(out_root / "features.csv", index=False)
    pd.DataFrame(excl, columns=["image_id", "stage", "reason", "details"]).to_csv(
        out_root / "exclusions.csv", index=False)
    current = set(names.values())
    images_dir = out_root / "images"
    other_dirs = sorted(p.name for p in images_dir.iterdir()
                        if p.is_dir() and not p.name.startswith(".") and p.name not in current) \
        if images_dir.exists() else []
    geom = cfg["stress"]["coverage_geometry"]
    run = {"created_utc": now_utc(), "n_rows": len(rows), "n_ok": len(feats),
           "n_excluded": len(excl), "exclusion_reasons": pd.Series(
               [e["reason"] for e in excl], dtype=object).value_counts().to_dict(),
           "stage_counts": pd.Series([r.get("stage") for r in results],
                                     dtype=object).value_counts().to_dict(),
           "pipeline_config_sha256": config_hash(pipeline_config_subset(cfg, d_um, faz_spec)),
           "full_config_sha256": config_hash(cfg), "d_um": d_um, "faz_spec": faz_spec,
           "calibration_ids_sha256": (calibration_record or {}).get("calibration_ids_sha256"),
           "coverage_geometry": geom, "coverage_method_note": COVERAGE_METHOD_NOTES[geom],
           "stress_order_labels": {o: ORDER_LABELS[o] for o in cfg["stress"]["orders"]},
           "code_fingerprint": pipeline_code_fingerprint(),
           "numerical_libraries": numerical_library_versions(),
           "image_dirs_not_in_this_run": other_dirs,
           "data_provenance": SYNTHETIC_BANNER if synthetic else "real",
           "elapsed_seconds": time.monotonic() - t0, "n_workers": n_workers,
           "environment": environment()}
    write_json(out_root / "run.json", run)
    (out_root / "config_used.json").write_text(json.dumps(cfg, indent=1, default=str))
    return run
