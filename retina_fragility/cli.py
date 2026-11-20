"""retina-fragility command-line interface.

Run from the project root. Every command reads one config file (default
config.yaml). Outputs from synthetic fixtures are labelled SYNTHETIC - NOT
CLINICAL. Commands that need unavailable data or dependencies exit with a
precise readiness report instead of substituting anything.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import runpy
import sys
from pathlib import Path

ROOT = Path.cwd()


def _cfg(args, overrides: dict | None = None) -> dict:
    from .config import load_config

    path = Path(args.config)
    return load_config(path if path.exists() else None, overrides)


def _manifest(args, cfg):
    from .manifest import load_manifest

    return load_manifest(args.manifest, cfg)


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ------------------------------------------------------------------ commands
def cmd_synthetic(args):
    sys.argv = ["scripts.run_synthetic"] + (["--quick"] if args.quick else [])
    runpy.run_module("scripts.run_synthetic", run_name="__main__")


def cmd_inventory_drac(args):
    sys.argv = ["scripts.build_drac_manifest"]
    runpy.run_module("scripts.build_drac_manifest", run_name="__main__")


def cmd_make_fixtures(args):
    from .synthetic import make_fixture_dataset

    f = make_fixture_dataset(args.out, args.n_normal, args.n_dr, args.seed, n_seeds=args.n_seeds,
                             dropout_patches=args.dropout_patches)
    _print({"written": str(Path(args.out) / "manifest.csv"), "n": len(f),
            "data_provenance": "SYNTHETIC - NOT CLINICAL"})


def cmd_validate(args):
    from .manifest import validate_manifest
    from .provenance import write_json

    cfg = _cfg(args)
    rep = validate_manifest(_manifest(args, cfg), cfg, ROOT, reference_masks=not args.no_reference_masks)
    if args.out:
        write_json(args.out, rep)
    _print({k: rep[k] for k in ("n_images", "n_valid_inputs", "all_inputs_valid", "note")})
    if not rep["all_inputs_valid"]:
        sys.exit(2)


def cmd_calibrate(args):
    from .calibration import compute_calibration, freeze_calibration, select_calibration_ids
    from .manifest import validate_manifest

    cfg = _cfg(args)
    m = _manifest(args, cfg)
    dataset = args.dataset or cfg["data"]["primary_dataset"]
    rep = validate_manifest(m[m.dataset == dataset], cfg, ROOT)
    valid = {r["image_id"] for r in rep["rows"] if r["valid_inputs"]}
    ids, audit = select_calibration_ids(m, cfg, dataset, valid)
    rec = compute_calibration(m, ids, cfg, ROOT, dataset, cfg["ablations"]["d_percentile"])
    out = Path(args.out or cfg["calibration"]["frozen_file"])
    rec = freeze_calibration(rec, out, force=args.force)
    audit.to_csv(out.with_name("calibration_selection_audit.csv"), index=False)
    _print({"frozen": str(out), "n_ids": len(rec["calibration_ids"]), "d_um": rec["d_um"],
            "ids_sha256": rec["calibration_ids_sha256"], "data_provenance": rec["data_provenance"]})


def _rows_for_processing(args, cfg, m, cal):
    from .manifest import analysis_rows

    dataset = args.dataset or cfg["data"]["primary_dataset"]
    if args.rows == "analysis":
        rows, _ = analysis_rows(m, cal["calibration_ids"] if cal else [], dataset=dataset,
                                require_calibration_present=cal is not None)
    else:  # all non-test rows of the dataset (external data with graded, unmapped labels)
        rows = m[(m.dataset == dataset) & (m.split.str.lower() != "test")]
        if cal:
            rows = rows[~rows.image_id.isin(cal["calibration_ids"])]
    if args.image_ids:
        rows = rows[rows.image_id.isin(args.image_ids.split(","))]
    return rows


def cmd_process(args):
    from .calibration import load_calibration, verify_calibration_inputs
    from .experiments.exp_e import external_faz_spec
    from .pipeline import run_manifest

    over = {"stress": {"compute_coverage": False}} if args.no_coverage else None
    cfg = _cfg(args, over)
    m = _manifest(args, cfg)
    cal = load_calibration(args.calibration) if args.calibration else None
    if cal is None and args.rows == "analysis":
        sys.exit("process --rows analysis requires --calibration (calibration IDs must be excluded)")
    if cal is not None and args.rows == "analysis":
        problems = verify_calibration_inputs(cal, m, ROOT)
        if problems:
            sys.exit("frozen calibration inputs no longer match the files on disk: " + "; ".join(problems[:10]))
    rows = _rows_for_processing(args, cfg, m, cal)
    faz = external_faz_spec(cfg) if args.faz_mode == "disk" else {"mode": args.faz_mode}
    d_um = cal["d_um"] if (cal and not args.no_coverage) else None
    run = run_manifest(rows, cfg, ROOT, args.out, d_um, faz, args.workers, cal)
    _print({k: run[k] for k in ("n_rows", "n_ok", "n_excluded", "exclusion_reasons",
                                "data_provenance", "elapsed_seconds")})


def cmd_analyze(args):
    from .calibration import load_calibration
    from .experiments import exp_a, exp_b, exp_c, exp_d, secondary_summary
    from .experiments.common import load_analysis_table

    cfg = _cfg(args)
    m = _manifest(args, cfg)
    dataset = args.dataset or cfg["data"]["primary_dataset"]
    cal = load_calibration(args.calibration)
    table, info = load_analysis_table(args.pipeline, m, cal, dataset)
    out = Path(args.out)
    exps = [e.strip().upper() for e in args.experiments.split(",")]
    done = {}
    if "B" in exps:
        r = exp_b.run(table, info, cfg, out / "exp_b", cal)
        done["B"] = {"primary_status": r["primary"]["status"], "p_value": r["primary"]["p_value"],
                     "failure_reasons": r["primary"]["failure_reasons"]}
    if "A" in exps:
        exp_a.run(table, info, cfg, args.pipeline, out / "exp_a", cal)
        done["A"] = "written"
    if "C" in exps:
        r = exp_c.run(table, info, cfg, args.pipeline, out / "exp_c", cal, args.workers)
        done["C"] = r["summary"]
    if "D" in exps:
        exp_d.run(table, info, cfg, out / "exp_d", cal)
        done["D"] = "written"
    s = secondary_summary.run(cfg, {k: out / f"exp_{k.lower()}" for k in ("A", "C", "D")},
                              out / "secondary_holm.json")
    _print({"data_provenance": info["data_provenance"], "grouping_level": info["grouping_level"],
            "n_analysed": len(table), "results": done, "holm": s["families"]})


def cmd_exp_c_plan(args):
    from .calibration import load_calibration, verify_calibration_inputs
    from .experiments import exp_c_shard
    from .experiments.common import load_analysis_table

    cfg = _cfg(args)
    m = _manifest(args, cfg)
    cal = load_calibration(args.calibration)
    problems = verify_calibration_inputs(cal, m, ROOT)
    if problems:
        sys.exit("frozen calibration inputs no longer match: " + "; ".join(problems[:10]))
    table, info = load_analysis_table(args.pipeline, m, cal, args.dataset or cfg["data"]["primary_dataset"])
    plan = exp_c_shard.make_plan(table, info, cfg, args.pipeline, cal, args.n_shards,
                                 {"manifest": args.manifest, "calibration_file": args.calibration})
    path = exp_c_shard.write_plan(plan, args.run_dir)
    _print({"plan": str(path), "plan_sha256": plan["plan_sha256"], **plan["summary"],
            "data_provenance": info["data_provenance"]})


def cmd_exp_c_shard(args):
    from .experiments import exp_c_shard

    _print(exp_c_shard.run_shard(args.run_dir, args.shard, _cfg(args), workers=args.workers or 1,
                                 max_new_units=args.max_new_units))


def cmd_exp_c_status(args):
    from .experiments import exp_c_shard

    _print(exp_c_shard.shard_status(args.run_dir))


def cmd_exp_c_reduce(args):
    from .experiments import exp_c_shard

    res = exp_c_shard.reduce(args.run_dir, args.out, _cfg(args))
    _print({"out": args.out, "data_provenance": res["data_provenance"], "summary": res["summary"],
            "plan_sha256": res["execution"]["plan_sha256"],
            "note": "re-run `analyze --experiments B,A,D --out <parent>` (or its Holm step) so "
                    "secondary_holm.json includes this H4 result"})


def cmd_segment(args):
    from .segmentation import run as segrun

    cfg = _cfg(args)
    m = _manifest(args, cfg)
    splits = tuple(args.splits.split(","))
    seg_splits = tuple(args.seg_splits.split(",")) if args.seg_splits else None
    exclude = ()
    if args.calibration:
        from .calibration import load_calibration

        exclude = tuple(load_calibration(args.calibration)["calibration_ids"]) + tuple(
            m.loc[m.calibration, "image_id"])
    ids = None
    if args.image_ids_file:
        import pandas as pd

        ids = pd.read_csv(args.image_ids_file, dtype=str, keep_default_na=False).image_id.str.strip().tolist()
    if args.segmenter == "frangi":
        p = segrun.frangi_manifest(m, cfg, ROOT, args.out, splits, args.threshold_scale, ids, seg_splits, exclude)
    else:
        if not args.model:
            sys.exit("--model is required for the U-Net segmenter")
        p = segrun.unet_predict_manifest(m, cfg, ROOT, args.model, args.out, splits, args.threshold_scale, ids,
                                         seg_splits, exclude)
    run = json.loads((Path(args.out) / "segmentation_run.json").read_text())
    _print({"derived_manifest": str(p), "n": run["n"],
            "segmentation_config_sha256": run["segmentation_config_sha256"], "fi_computed": False})


def cmd_qc_sheet(args):
    from .experiments.exp_e import external_faz_spec
    from .qc import make_qc_sheet

    cfg = _cfg(args)
    sheet = make_qc_sheet(args.segmented, args.selection, cfg, ROOT, external_faz_spec(cfg), args.out,
                          overwrite_blank=args.overwrite_blank)
    _print({"qc_sheet": args.out, "n_rows": len(sheet),
            "segmentation_config_sha256": sheet.segmentation_config_sha256.iloc[0] if len(sheet) else None,
            "note": "blank decisions; to be completed by a named human reviewer"})


def cmd_qc_export(args):
    from .experiments.exp_e import external_faz_spec
    from .qc_export import export_review

    cfg = _cfg(args)
    s = export_review(args.segmented, args.selection, cfg, ROOT, args.out, external_faz_spec(cfg))
    _print({k: s[k] for k in ("n_images", "qc_sheet", "segmentation_config_sha256", "graph_review_sha256",
                              "human_qc_completed", "fi_computed", "data_provenance")})


def cmd_unet_train(args):
    from .calibration import load_calibration
    from .config import config_hash
    from .provenance import sha256_file
    from .segmentation import run as segrun

    over = {"segmentation": {"unet": {"epochs": args.epochs}}} if args.epochs else None
    cfg = _cfg(args, over)
    m = _manifest(args, cfg)
    cal = load_calibration(args.calibration)
    prov = {"manifest": {"path": args.manifest, "sha256": sha256_file(args.manifest)},
            "calibration_file": {"path": args.calibration, "sha256": sha256_file(args.calibration),
                                 "calibration_ids_sha256": cal["calibration_ids_sha256"]},
            "full_config_sha256": config_hash(cfg),
            "cli_overrides": {"epochs": args.epochs, "patches_per_epoch": args.patches_per_epoch}}
    meta = segrun.unet_train_manifest(m, cfg, ROOT, cal["calibration_ids"], args.out, args.patches_per_epoch,
                                      provenance=prov, reuse_identical=args.reuse_identical)
    _print({k: meta[k] for k in ("n_train", "n_optimizer_steps", "torch_num_threads", "elapsed_seconds",
                                 "model_sha256")} | {"final_loss": meta["history"][-1],
                                                     "reused_existing_model": meta.get("reused_existing_model", False)})


def cmd_seg_eval(args):
    from .calibration import load_calibration
    from .segmentation import run as segrun

    cfg = _cfg(args)
    m = _manifest(args, cfg)
    cal = load_calibration(args.calibration)
    from .provenance import now_utc, sha256_file, write_json

    meta = json.loads((Path(args.model) / "meta.json").read_text()) if args.model else None
    segmented = Path(args.segmented)
    derived = segmented / "manifest_segmented.csv" if segmented.is_dir() else segmented
    df = segrun.evaluate_against_reference(m, cfg, ROOT, derived, cal["calibration_ids"], meta)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    run_path = derived.parent / "segmentation_run.json"
    seg_run = json.loads(run_path.read_text()) if run_path.exists() else {}
    summary = {"created_utc": now_utc(), "table": args.out, "table_sha256": sha256_file(args.out),
               "derived_manifest": str(derived), "derived_manifest_sha256": sha256_file(derived),
               "segmenter": seg_run.get("segmenter"),
               "segmentation_config_sha256": seg_run.get("segmentation_config_sha256"),
               "model_sha256": (meta or {}).get("model_sha256") or seg_run.get("params", {}).get("model_sha256"),
               "calibration_ids_sha256": cal["calibration_ids_sha256"],
               "summaries": segrun.summarize_evaluation(df),
               "note": ("calibration eyes and U-Net training images/subjects excluded; descriptive only, "
                        "no model or threshold is selected from these values")}
    write_json(Path(args.out).with_suffix(".json"), summary)
    _print(summary)


def cmd_embed(args):
    from .embeddings import extract_manifest

    cfg = _cfg(args)
    m = _manifest(args, cfg)
    meta = extract_manifest(m, ROOT, args.model, cfg["embeddings"], args.out, tuple(args.splits.split(",")))
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    _print({k: v for k, v in meta.items() if k != "input_sha256"})


def cmd_exp_e(args):
    from .experiments import exp_e
    from .experiments.common import ReadinessError

    cfg = _cfg(args)
    m = _manifest(args, cfg)
    try:
        r = exp_e.run(cfg, m, args.segmenter, args.qc, args.pipeline, args.out)
    except ReadinessError as exc:
        _print({"experiment": "E", "ready": False, "reasons": exc.reasons,
                "report": str(Path(args.out) / "readiness.json")})
        sys.exit(3)
    _print({"experiment": "E", "data_provenance": r["data_provenance"], "cv": r["cv"]})


def cmd_exp_f(args):
    import numpy as np
    import pandas as pd

    from .experiments import exp_e, exp_f
    from .experiments.common import ReadinessError
    from .manifest import grouping_level, is_synthetic

    cfg = _cfg(args)
    m = _manifest(args, cfg)
    m = m[m.split.str.lower() == "train"]
    reasons, qc = exp_e.readiness(cfg, m, args.segmenter, args.qc, args.pipeline)
    npz = Path(args.embeddings)
    if not npz.exists():
        reasons.append(f"embeddings missing: {npz}")
    if reasons:
        _print({"experiment": "F", "ready": False, "reasons": reasons})
        sys.exit(3)
    d = np.load(npz, allow_pickle=False)
    meta_path = npz.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    feats = pd.read_csv(Path(args.pipeline) / "features.csv")
    table = m[["image_id", "group_key", "grade_num"]].merge(feats, on="image_id")
    table = table[~table.image_id.isin(qc["failed_image_ids"])]
    info = {"data_provenance": "SYNTHETIC - NOT CLINICAL" if is_synthetic(m) else "real",
            "grouping_level": grouping_level(m)}
    try:
        r = exp_f.run(cfg, table, d["image_id"], d["X"], meta, args.out, info)
    except ReadinessError as exc:
        _print({"experiment": "F", "ready": False, "reasons": exc.reasons})
        sys.exit(3)
    _print({"experiment": "F", "delta": r["delta_b_minus_a"]})


def cmd_ablate(args):
    from .ablations import run
    from .calibration import load_calibration

    cfg = _cfg(args)
    m = _manifest(args, cfg)
    cal = load_calibration(args.calibration)
    df = run(m, args.dataset or cfg["data"]["primary_dataset"], cal, cfg, ROOT, args.out,
             args.params.split(",") if args.params else None, args.workers)
    print(df.drop(columns=[c for c in ("traceback",) if c in df]).to_string())


def cmd_dropout(args):
    from .calibration import load_calibration
    from .experiments.exp0_dropout import patch_dropout
    from .manifest import analysis_rows

    cfg = _cfg(args, {"pipeline": {"overlays": False}})
    m = _manifest(args, cfg)
    cal = load_calibration(args.calibration)
    rows, _ = analysis_rows(m, cal["calibration_ids"], dataset=args.dataset or cfg["data"]["primary_dataset"])
    df = patch_dropout(rows, cfg, ROOT, cal["d_um"], args.out, args.n_images, args.n_patches, args.radius_um)
    print(df[["image_id"] + [c for c in df.columns if c.startswith("delta_")]].to_string())


def cmd_overlay_sheet(args):
    import pandas as pd

    from .experiments.exp0_dropout import overlay_sheet

    feats = pd.read_csv(Path(args.pipeline) / "features.csv")
    df = overlay_sheet(args.pipeline, feats.image_id.astype(str), args.out, args.n,
                       _cfg(args)["project"]["base_seed"])
    _print({"queued": len(df), "review_csv": str(Path(args.out) / "overlay_review.csv"),
            "note": "review queue only; not evidence of review"})


def cmd_octa500_manifest(args):
    import pandas as pd

    from .calibration import load_calibration
    from .manifest import load_manifest, validate_manifest
    from .octa500 import build_manifest, eligibility_audit, manifest_errors
    from .provenance import now_utc, sha256_file, write_json

    cfg = _cfg(args)
    catalog_path = ROOT / (args.catalog or cfg["octa500"]["source_catalog"])
    out = ROOT / (args.out or cfg["data"]["manifest"])
    audit_path = ROOT / args.audit
    cal = load_calibration(args.calibration) if args.calibration else None
    catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)
    manifest, audit = build_manifest(catalog, cfg, ROOT, calibration=cal)
    errors = manifest_errors(audit)
    if errors:
        audit.to_csv(audit_path, index=False)
        sys.exit(f"{len(errors)} catalog rows failed source checks; manifest not written. "
                 f"First: {errors[:3]} (audit: {audit_path})")
    previous = sha256_file(out) if out.exists() else None
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    loaded = load_manifest(out, cfg)  # the pipeline's own schema and label checks
    validation = validate_manifest(loaded, cfg, ROOT)
    audit = eligibility_audit(audit, manifest, cfg, validation)
    audit.to_csv(audit_path, index=False)
    counts = audit.groupby(["fov_mm", "disease"]).size().unstack(fill_value=0)
    summary = {
        "created_utc": now_utc(), "manifest": str(out.relative_to(ROOT)), "manifest_sha256": sha256_file(out),
        "previous_manifest_sha256": previous, "catalog": str(catalog_path.relative_to(ROOT)),
        "catalog_sha256": sha256_file(catalog_path), "readiness_audit": str(audit_path.relative_to(ROOT)),
        "readiness_audit_sha256": sha256_file(audit_path), "n_rows": len(manifest),
        "n_source_checks_passed": int(audit.source_checks_passed.sum()),
        "n_valid_inputs": validation["n_valid_inputs"], "all_inputs_valid": validation["all_inputs_valid"],
        "n_calibration_flagged": int(audit.calibration.sum()),
        "calibration_file": args.calibration,
        "calibration_ids_sha256": cal["calibration_ids_sha256"] if cal else None,
        "inventory_counts_fov_by_disease": {str(k): v for k, v in counts.T.to_dict().items()},
        "eligible_counts": {c.removeprefix("eligible_"): int(audit[c].sum())
                            for c in audit.columns if c.startswith("eligible_")},
        "seg_split_counts": audit.seg_split.value_counts().sort_index().to_dict(),
        "primary_projection": cfg["octa500"]["primary_projection"],
        "decision_record": "docs/OCTA500_PREANALYSIS_NOTE_2026-09-30.md",
        "rows_dropped": 0, "fi_computed": False, "clinical_statistics_computed": False,
    }
    if args.summary:
        write_json(ROOT / args.summary, summary)
    _print({k: summary[k] for k in ("manifest", "manifest_sha256", "n_rows", "n_valid_inputs",
                                    "n_calibration_flagged", "eligible_counts")})
    if not validation["all_inputs_valid"]:
        sys.exit(2)


def cmd_octa500_label_check(args):
    import numpy as np
    import pandas as pd

    from .io_masks import read_mask, read_plane
    from .octa500 import alignment_scores, cavf_consistency, summarize_alignment
    from .provenance import now_utc, sha256_file, write_json

    cfg = _cfg(args)
    o = cfg["octa500"]
    catalog_path = ROOT / (args.catalog or o["source_catalog"])
    cat = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)
    projections = [o["primary_projection"], *o["context_projections"]]
    binary = {"mode": "binary"}
    rows = []
    for r in cat.to_dict("records"):
        needed = projections + ["capillary", "large_vessel", "faz", "artery", "vein", "cavf"]
        arrays = {}
        for name in needed:
            path = ROOT / r[f"{name}_path"]
            if sha256_file(path) != r[f"{name}_sha256"]:
                sys.exit(f"{r['image_id']}: {name} hash differs from the catalog")
            arrays[name] = read_plane(path) if (name in projections or name == "cavf") else read_mask(path, binary)
        vessel = arrays["capillary"] | arrays["large_vessel"]
        rec = {"image_id": r["image_id"], "fov_mm": r["fov_mm"],
               **alignment_scores({p: arrays[p] for p in projections}, vessel),
               **cavf_consistency(arrays["capillary"], arrays["large_vessel"], arrays["faz"],
                                  arrays["artery"], arrays["vein"], arrays["cavf"])}
        rows.append(rec)
    table = pd.DataFrame(rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    summary = summarize_alignment(table, cfg)
    cav = {c: {"n_images_nonzero": int((table[c] > 0).sum()), "max_px": int(table[c].max()),
               "total_px": int(table[c].sum())}
           for c in table.columns if c.endswith("_px")}
    report = {"created_utc": now_utc(), "catalog_sha256": sha256_file(catalog_path),
              "table": str(out.relative_to(ROOT)), "table_sha256": sha256_file(out),
              "alignment": summary, "cavf_consistency": cav,
              "per_fov_median_identity_auroc": {
                  str(f): {p: float(np.median(g[f"auroc_{p}_identity"])) for p in projections}
                  for f, g in table.groupby("fov_mm")},
              "scope": "Label/projection pixel agreement only; no disease labels read, no graph or FI.",
              "prespecified_in": "docs/OCTA500_PREANALYSIS_NOTE_2026-09-30.md (D1)"}
    write_json(ROOT / args.report, report)
    _print(report["alignment"])
    if not summary["passed"]:
        sys.exit("prespecified projection-alignment criteria FAILED; do not use the manifest (note D1)")


def cmd_octa500_graph_qc(args):
    from .calibration import load_calibration, verify_calibration_inputs
    from .octa500 import export_graph_qc, select_graph_qc

    cfg = _cfg(args)
    m = _manifest(args, cfg)
    cal = load_calibration(args.calibration)
    problems = verify_calibration_inputs(cal, m, ROOT)
    if problems:
        sys.exit("frozen calibration inputs no longer match: " + "; ".join(problems[:10]))
    sel = select_graph_qc(m, cfg, cal["calibration_ids"])
    summary = export_graph_qc(m, sel, cfg, ROOT, ROOT / args.out, cal)
    _print({k: summary[k] for k in ("n_images", "n_per_fov", "graph_review_sha256",
                                    "pipeline_code_sha256", "fi_computed", "human_qc_completed")})


def cmd_report(args):
    from .reporting.core import ReportRefusal
    from .reporting.paper import build_report, clinical_layout, layout_from_json, synthetic_e2e_layout

    if args.layout == "file":
        if not args.layout_file:
            sys.exit("--layout file requires --layout-file")
        layout = layout_from_json(args.layout_file)
    elif args.layout == "synthetic-e2e":
        if not args.e2e_dir:
            sys.exit("--layout synthetic-e2e requires --e2e-dir")
        layout = synthetic_e2e_layout(args.e2e_dir)
    else:
        layout = clinical_layout()
    jobs = json.loads(Path(args.jobs).read_text()) if args.jobs else None
    try:
        idx = build_report(layout, args.mode, args.out, ROOT, jobs)
    except ReportRefusal as exc:
        _print({"command": "report", "mode": args.mode, "refused": True, "reasons": exc.reasons,
                "outputs_written": False})
        sys.exit(3)
    refused = sorted(k for k, v in idx["artifacts"].items() if v["status"] == "refused")
    _print({"command": "report", "mode": args.mode, "data_label": idx["data_label"], "out": args.out,
            "status_counts": idx["status_counts"], "refused_artifacts": refused, "flags": idx["flags"]})
    if refused:
        sys.exit(4)


def readiness_report(cfg) -> dict:
    from .qc import qc_status
    from .segmentation.unet import torch_status

    octa_manifest = ROOT / cfg["data"]["manifest"]
    cal = ROOT / cfg["calibration"]["frozen_file"]
    e = cfg["exp_e"]
    qc = qc_status(ROOT / "data/manifests/drac_qc_selection.csv", "frangi", int(e["min_qc_audited"]),
                   float(e["max_qc_fail_fraction"]))
    sheets = {}
    for sheet in sorted((ROOT / "outputs/drac_qc").glob("*/**/qc_sheet_*.csv")):
        st = qc_status(sheet, sheet.stem.replace("qc_sheet_", ""), int(e["min_qc_audited"]),
                       float(e["max_qc_fail_fraction"]))
        sheets[str(sheet.relative_to(ROOT))] = {"n_rows": st["n_rows"], "n_audited": st["n_audited"],
                                                "reasons": st["reasons"]}
    items = {
        "drac_hash_bound_qc_sheets": sheets,
        "octa500_archives_present": any((ROOT / "data/raw/octa500").glob("*.zip")),
        "octa500_manifest_present": octa_manifest.exists(),
        "calibration_frozen": cal.exists(),
        "drac_manifest_present": (ROOT / e["manifest"]).exists(),
        "drac_frangi_qc_ready": qc["ready"], "drac_frangi_qc_reasons": qc["reasons"],
        "torch": torch_status(),
    }
    blockers = []
    if not items["octa500_manifest_present"]:
        blockers.append(f"OCTA-500 manifest absent ({octa_manifest}); primary analysis (B) and A, C, D "
                        "cannot run on real data")
    if not items["calibration_frozen"]:
        blockers.append("no frozen OCTA-500 calibration; run `calibrate` once the OCTA-500 manifest exists")
    if not qc["ready"]:
        blockers.append("DRAC Experiment E/F blocked: " + "; ".join(qc["reasons"]))
    if not items["torch"]["available"]:
        blockers.append("torch not importable: U-Net and embeddings unavailable (" +
                        items["torch"].get("reason", "") + ")")
    items["blockers"] = blockers
    pending = []
    align = ROOT / "reports/octa500_label_alignment.json"
    items["octa500_label_alignment_passed"] = (
        json.loads(align.read_text())["alignment"]["passed"] if align.exists() else None)
    if items["octa500_label_alignment_passed"] is not True:
        blockers.append("OCTA-500 projection/label alignment check absent or failed "
                        "(`octa500-label-check`; note D1)")
    import pandas as pd

    graph_sheets = {}
    for sheet in sorted((ROOT / "outputs/octa500_graph_qc").glob("*/qc_sheet_octa500_graph.csv")):
        q = pd.read_csv(sheet, dtype=str, keep_default_na=False)
        graph_sheets[str(sheet.relative_to(ROOT))] = {
            "n_rows": len(q), "n_with_human_decision": int(q.decision.str.strip().ne("").sum())}
    items["octa500_graph_qc_sheets"] = graph_sheets
    if not any(v["n_with_human_decision"] for v in graph_sheets.values()):
        pending.append("OCTA-500 graph overlays await human review (not a coded gate; findings may "
                       "require pre-outcome code/parameter amendments)")
    items["pending_not_gated"] = pending
    items["clinical_results_available"] = False
    return items


def cmd_readiness(args):
    from .provenance import write_json

    rep = readiness_report(_cfg(args))
    if args.out:
        write_json(args.out, rep)
    _print(rep)


def cmd_doctor(args):
    root = Path(__file__).resolve().parents[1]
    needed = ["io_masks", "manifest", "octa500", "calibration", "biomarkers", "pipeline", "viz", "qc", "qc_export",
              "stats.primary", "stats.cv", "stats.effects", "experiments.exp_a", "experiments.exp_b",
              "experiments.exp_c", "experiments.exp_d", "experiments.exp_e", "experiments.exp_f",
              "experiments.secondary_summary", "segmentation.frangi", "segmentation.unet",
              "embeddings", "ablations"]
    availability = {}
    for name in needed:
        try:
            availability[name] = importlib.util.find_spec("retina_fragility." + name) is not None
        except ModuleNotFoundError:
            availability[name] = False
    rep = readiness_report(_cfg(args))
    _print({"project_root": str(root), "module_files_present": availability,
            "readiness": rep, "project_complete": False,
            "note": "Module presence is not proof of validation; see tests and reports."})


def cmd_synthetic_e2e(args):
    from .e2e import run_synthetic_e2e

    res = run_synthetic_e2e(args.out, _cfg(args), quick=args.quick, n_workers=args.workers)
    _print(res)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="retina-fragility", description=__doc__)
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synthetic", help="Experiment 0 synthetic graphs (not clinical)")
    s.add_argument("--quick", action="store_true")
    s.set_defaults(func=cmd_synthetic)
    sub.add_parser("inventory-drac", help="Validate DRAC grading files, build manifest").set_defaults(
        func=cmd_inventory_drac)
    sub.add_parser("doctor", help="Module availability + real-data readiness").set_defaults(func=cmd_doctor)
    s = sub.add_parser("readiness", help="Precise readiness/blocker report")
    s.add_argument("--out")
    s.set_defaults(func=cmd_readiness)

    s = sub.add_parser("make-fixtures", help="Write SYNTHETIC mask fixtures + manifest")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n-normal", type=int, default=22, help="per FOV")
    s.add_argument("--n-dr", type=int, default=12, help="per FOV")
    s.add_argument("--n-seeds", type=int, default=150)
    s.add_argument("--dropout-patches", type=int, default=2)
    s.set_defaults(func=cmd_make_fixtures)

    s = sub.add_parser("validate-manifest", help="Decode every input; report failures")
    s.add_argument("--manifest", required=True)
    s.add_argument("--no-reference-masks", action="store_true")
    s.add_argument("--out")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("calibrate", help="Select + freeze calibration IDs and d")
    s.add_argument("--manifest", required=True)
    s.add_argument("--dataset")
    s.add_argument("--out")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("process", help="Per-image pipeline (graphs, curves, features, overlays)")
    s.add_argument("--manifest", required=True)
    s.add_argument("--dataset")
    s.add_argument("--calibration")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int)
    s.add_argument("--rows", choices=["analysis", "all-nontest"], default="analysis")
    s.add_argument("--faz-mode", choices=["reference", "disk", "auto", "none"], default="reference")
    s.add_argument("--no-coverage", action="store_true")
    s.add_argument("--image-ids")
    s.set_defaults(func=cmd_process)

    s = sub.add_parser("analyze", help="Experiments A-D + Holm summary on pipeline features")
    s.add_argument("--manifest", required=True)
    s.add_argument("--dataset")
    s.add_argument("--calibration", required=True)
    s.add_argument("--pipeline", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--experiments", default="B,A,C,D")
    s.add_argument("--workers", type=int)
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("exp-c-plan", help="Freeze a sharded Experiment C plan (units, shards, identities)")
    s.add_argument("--manifest", required=True)
    s.add_argument("--dataset")
    s.add_argument("--calibration", required=True)
    s.add_argument("--pipeline", required=True)
    s.add_argument("--run-dir", required=True)
    s.add_argument("--n-shards", type=int, required=True)
    s.set_defaults(func=cmd_exp_c_plan)

    s = sub.add_parser("exp-c-shard", help="Run or resume one Experiment C shard (Slurm array task)")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--workers", type=int)
    s.add_argument("--max-new-units", type=int, help="stop after N new units (interruption drills)")
    s.set_defaults(func=cmd_exp_c_shard)

    s = sub.add_parser("exp-c-status", help="Progress of a sharded Experiment C run (no results)")
    s.add_argument("--run-dir", required=True)
    s.set_defaults(func=cmd_exp_c_status)

    s = sub.add_parser("exp-c-reduce", help="Validate all shards against the plan, write Experiment C results")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--out", required=True, help="Experiment C output directory, e.g. <analysis>/exp_c")
    s.set_defaults(func=cmd_exp_c_reduce)

    s = sub.add_parser("segment", help="Segment images (frangi | unet) -> derived manifest")
    s.add_argument("--segmenter", choices=["frangi", "unet"], required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model")
    s.add_argument("--splits", default="train", help="label-availability split(s) (OCTA-500: public)")
    s.add_argument("--seg-splits", help="segmentation split(s), e.g. val,test for held-out OCTA-500 rows")
    s.add_argument("--calibration", help="frozen calibration.json: its eyes are never segmented")
    s.add_argument("--threshold-scale", type=float)
    s.add_argument("--image-ids-file", help="CSV with an image_id column (e.g. the QC selection)")
    s.set_defaults(func=cmd_segment)

    s = sub.add_parser("qc-sheet", help="Hash-bound blank QC review sheet for a segmentation run")
    s.add_argument("--segmented", required=True, help="segmentation run directory (from `segment`)")
    s.add_argument("--selection", required=True, help="CSV with the image_id review selection")
    s.add_argument("--out", required=True)
    s.add_argument("--overwrite-blank", action="store_true",
                   help="replace an existing sheet only if it holds no review entries")
    s.set_defaults(func=cmd_qc_sheet)

    s = sub.add_parser("qc-export", help="Native-resolution review images + graphs + hash-bound sheet "
                                         "(no FI, no statistics)")
    s.add_argument("--segmented", required=True)
    s.add_argument("--selection", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_qc_export)

    s = sub.add_parser("unet-train", help="Train the U-Net (seg_split=train, no calibration rows)")
    s.add_argument("--manifest", required=True)
    s.add_argument("--calibration", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--epochs", type=int)
    s.add_argument("--patches-per-epoch", type=int)
    s.add_argument("--reuse-identical", action="store_true",
                   help="if --out already holds a model, reuse it only when its recorded config, inputs, labels, "
                        "calibration, manifest and code equal this request; otherwise refuse")
    s.set_defaults(func=cmd_unet_train)

    s = sub.add_parser("seg-eval", help="Dice vs reference on non-training, non-calibration rows")
    s.add_argument("--manifest", required=True)
    s.add_argument("--segmented", required=True, help="segmentation run directory or its derived manifest")
    s.add_argument("--calibration", required=True)
    s.add_argument("--model", help="default for U-Net runs: model_dir in segmentation_run.json")
    s.add_argument("--out", required=True, help="per-image CSV; a .json summary is written beside it")
    s.set_defaults(func=cmd_seg_eval)

    s = sub.add_parser("embed", help="Frozen embeddings (resnet50 | dinov2_vits14) to .npz")
    s.add_argument("--manifest", required=True)
    s.add_argument("--model", choices=["resnet50", "dinov2_vits14"], required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--splits", default="train")
    s.set_defaults(func=cmd_embed)

    s = sub.add_parser("exp-e", help="External DRAC analysis (QC-gated)")
    s.add_argument("--manifest", required=True, help="segmented DRAC manifest")
    s.add_argument("--segmenter", choices=["frangi", "unet"], required=True)
    s.add_argument("--qc", default="data/manifests/drac_qc_selection.csv",
                   help="hash-bound QC sheet (from qc-sheet/qc-export) for this segmentation run")
    s.add_argument("--pipeline", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_exp_e)

    s = sub.add_parser("exp-f", help="Embeddings vs embeddings + fragility (QC-gated, exploratory)")
    s.add_argument("--manifest", required=True, help="segmented DRAC manifest")
    s.add_argument("--segmenter", choices=["frangi", "unet"], default="frangi")
    s.add_argument("--qc", default="data/manifests/drac_qc_selection.csv",
                   help="hash-bound QC sheet (from qc-sheet/qc-export) for this segmentation run")
    s.add_argument("--pipeline", required=True)
    s.add_argument("--embeddings", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_exp_f)

    s = sub.add_parser("ablate", help="Parameterised pipeline ablations (+ B and A)")
    s.add_argument("--manifest", required=True)
    s.add_argument("--dataset")
    s.add_argument("--calibration", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--params")
    s.add_argument("--workers", type=int)
    s.set_defaults(func=cmd_ablate)

    s = sub.add_parser("dropout-check", help="Experiment 0 patch-dropout response on healthy images")
    s.add_argument("--manifest", required=True)
    s.add_argument("--dataset")
    s.add_argument("--calibration", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-images", type=int, default=10)
    s.add_argument("--n-patches", type=int, default=3)
    s.add_argument("--radius-um", type=float, default=250.0)
    s.set_defaults(func=cmd_dropout)

    s = sub.add_parser("overlay-sheet", help="Queue N overlays for human visual QC")
    s.add_argument("--pipeline", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=20)
    s.set_defaults(func=cmd_overlay_sheet)

    s = sub.add_parser("octa500-manifest", help="Validated OCTA-500 analysis manifest + readiness "
                                                 "audit from the source catalog (no rows dropped)")
    s.add_argument("--catalog", help="default: octa500.source_catalog")
    s.add_argument("--out", help="default: data.manifest")
    s.add_argument("--audit", default="data/manifests/octa500_readiness_audit.csv")
    s.add_argument("--summary", default="reports/octa500_manifest_summary.json")
    s.add_argument("--calibration", help="frozen calibration.json: flag its eyes calibration=true")
    s.set_defaults(func=cmd_octa500_manifest)

    s = sub.add_parser("octa500-label-check", help="Projection/label alignment and GT_CAVF consistency "
                                                    "(no disease labels, no graphs)")
    s.add_argument("--catalog")
    s.add_argument("--out", default="data/manifests/octa500_label_alignment.csv")
    s.add_argument("--report", default="reports/octa500_label_alignment.json")
    s.set_defaults(func=cmd_octa500_label_check)

    s = sub.add_parser("octa500-graph-qc", help="Outcome-blind graph review export on the primary "
                                                 "projection (no FI, blank hash-bound sheet)")
    s.add_argument("--manifest", required=True)
    s.add_argument("--calibration", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_octa500_graph_qc)

    s = sub.add_parser("synthetic-e2e", help="Whole workflow on SYNTHETIC fixtures (software check)")
    s.add_argument("--out", required=True)
    s.add_argument("--quick", action="store_true")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(func=cmd_synthetic_e2e)

    s = sub.add_parser("report", help="Render paper tables/figures + artifact index from saved analysis "
                                      "outputs (no new analysis)")
    s.add_argument("--mode", required=True, choices=["clinical", "fixture"],
                   help="clinical refuses synthetic inputs; fixture refuses real inputs and labels every "
                        "output SYNTHETIC — NOT CLINICAL")
    s.add_argument("--layout", required=True, choices=["clinical-default", "synthetic-e2e", "file"])
    s.add_argument("--layout-file", help="JSON Layout (see reporting.paper.Layout)")
    s.add_argument("--e2e-dir", help="synthetic-e2e output directory (with --layout synthetic-e2e)")
    s.add_argument("--jobs", help="JSON mapping input path prefixes to Slurm job IDs")
    s.add_argument("--out", required=True, help="new, empty output directory (never overwritten)")
    s.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    from .experiments.common import ReadinessError
    from .experiments.exp_c_shard import ShardError
    from .segmentation.unet import DependencyUnavailable

    try:
        args.func(args)
    except (DependencyUnavailable, ReadinessError, ShardError) as exc:
        _print({"command": args.command, "ready": False,
                "reasons": getattr(exc, "reasons", [str(exc)])})
        sys.exit(3)


if __name__ == "__main__":
    main()
