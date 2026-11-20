"""Render every planned paper table/figure that machine artifacts allow, plus an artifact index.

``build_report`` runs in three phases:

1. **Preflight (global).** Read every available input through the ledger and
   refuse the whole report if provenance contradicts the mode, or if the
   configuration, calibration, code or frozen-document identities disagree.
   In clinical mode the OCTA-500 graph-QC sheet must be complete, bound to the
   pipeline code fingerprint and dated before the processing run (gate G2).
2. **Artifacts.** Each artifact renders independently. Count or consistency
   failures refuse only that artifact (status ``refused``); absent inputs give
   ``missing_input``; an analysis readiness refusal gives ``not_run``; a
   non-estimable model is rendered as such.
3. **Index.** ``paper_artifact_index.json`` maps each artifact to its outputs,
   source hashes, config/calibration/code identities and job provenance.

Artifact IDs follow ``docs/manuscript/RESULTS_ARTIFACT_PLAN.md``.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import config_hash, load_config
from ..provenance import code_fingerprint, sha256_file
from . import cohort, external, maps, primary, requirements, secondary
from .core import (CLINICAL, FIXTURE, MISSING, NOT_ESTIMABLE, NOT_RUN, REFUSED, RENDERED,
                   ArtifactWriter, Ledger, ReportRefusal, label, markdown_table, prepare_output_root)

REPORT_ENTRY_MODULES = ("retina_fragility.reporting.paper",)
INDEX_FORMAT = "retina_fragility.paper_artifact_index/1"
FROZEN_SHA256_FILES = ("docs/PREREGISTRATION_2026-09-29.sha256", "docs/OCTA500_PREANALYSIS_NOTE_2026-09-30.sha256")


@dataclass
class ExternalInputs:
    results_dir: str
    pipeline: str | None = None
    segmented_manifest: str | None = None


@dataclass
class Layout:
    """Where the analysis artifacts live. Paths are relative to the project root unless absolute."""

    dataset: str
    manifest: str
    calibration: str
    pipeline: str
    analysis: str
    calibration_audit: str | None = None
    ablations: str | None = None
    exp_e: dict[str, ExternalInputs] = field(default_factory=dict)
    exp_f: str | None = None
    graph_qc_sheet: str | None = None
    graph_qc_disposition: str | None = None
    frozen_config: str | None = None
    notes: list[str] = field(default_factory=list)

    def resolve(self, root: Path) -> Layout:
        def r(p):
            return None if p is None else str((root / p).resolve() if not Path(p).is_absolute() else Path(p))

        return Layout(dataset=self.dataset, manifest=r(self.manifest), calibration=r(self.calibration),
                      pipeline=r(self.pipeline), analysis=r(self.analysis),
                      calibration_audit=r(self.calibration_audit), ablations=r(self.ablations),
                      exp_e={k: ExternalInputs(r(v.results_dir), r(v.pipeline), r(v.segmented_manifest))
                             for k, v in self.exp_e.items()},
                      exp_f=r(self.exp_f), graph_qc_sheet=r(self.graph_qc_sheet),
                      graph_qc_disposition=r(self.graph_qc_disposition),
                      frozen_config=r(self.frozen_config), notes=list(self.notes))


def clinical_layout() -> Layout:
    """Paths fixed in docs/CLI_USAGE.md; the U-Net Experiment E paths are not yet fixed there."""
    return Layout(
        dataset="OCTA500", manifest="data/manifests/octa500_manifest.csv",
        calibration="outputs/calibration/calibration.json",
        calibration_audit="outputs/calibration/calibration_selection_audit.csv",
        pipeline="outputs/octa500_reference", analysis="outputs/analysis", ablations="outputs/ablations",
        exp_e={"frangi": ExternalInputs("outputs/exp_e_frangi", "outputs/drac_frangi_pipeline",
                                        "outputs/drac_frangi/manifest_segmented.csv"),
               "unet": ExternalInputs("outputs/exp_e_unet", "outputs/drac_unet_pipeline",
                                      "outputs/drac_unet_octa500_v1/manifest_segmented.csv")},
        exp_f="outputs/exp_f_resnet50",
        graph_qc_sheet="outputs/octa500_graph_qc/v2_primary_20260930/qc_sheet_octa500_graph.csv",
        graph_qc_disposition="outputs/octa500_graph_qc/v2_primary_20260930_disposition.json",
        frozen_config="config.yaml",
        notes=["Experiment E U-Net paths follow the artifact plan's analogy and are not yet fixed in "
               "CLI_USAGE.md; supply --layout-file if the runs use other paths.",
               "The graph-QC disposition record path and schema (reporting/gates.py) are proposed; "
               "the study owners must confirm them when recording the G2 outcome."])


def synthetic_e2e_layout(e2e_dir: str) -> Layout:
    """Layout written by ``retina-fragility synthetic-e2e`` (SYNTHETIC fixtures only)."""
    from ..synthetic import SYNTHETIC_DATASET

    d = Path(e2e_dir)
    return Layout(
        dataset=SYNTHETIC_DATASET, manifest=str(d / "fixtures/manifest.csv"),
        calibration=str(d / "calibration.json"), pipeline=str(d / "pipeline"), analysis=str(d),
        ablations=str(d / "ablations"),
        exp_e={"frangi": ExternalInputs(str(d / "external/exp_e_frangi"), str(d / "external/frangi_pipeline"),
                                        str(d / "external/frangi/manifest_segmented.csv"))},
        exp_f=str(d / "external/exp_f"),
        notes=["synthetic-e2e layout: no calibration selection audit, no OCTA-500 graph-QC sheet, "
               "serial Experiment C and quick-mode settings (config_used.json)."])


def layout_from_json(path: str | Path) -> Layout:
    d = json.loads(Path(path).read_text())
    d["exp_e"] = {k: ExternalInputs(**v) for k, v in d.get("exp_e", {}).items()}
    return Layout(**d)


# ------------------------------------------------------------------ helpers
def _p(path: str | None, *parts: str) -> Path | None:
    return None if path is None else Path(path, *parts)


def _read_results(ledger: Ledger, d: Path | None) -> dict | None:
    if d is None or not (d / "results.json").is_file():
        return None
    r = ledger.json(d / "results.json")
    ledger.declare(d / "results.json", r.get("data_provenance"))
    return r


def _jobs_for(sources: list[str], jobs: dict[str, list[str]]) -> list[str]:
    found = set()
    for s in sources:
        for prefix, ids in jobs.items():
            if s.startswith(prefix.rstrip("/")):
                found.update(map(str, ids))
    return sorted(found)


def _frozen_docs_check(root: Path) -> list[str]:
    reasons = []
    for f in FROZEN_SHA256_FILES:
        p = root / f
        if not p.is_file():
            reasons.append(f"frozen-document hash file missing: {f}")
            continue
        proc = subprocess.run(["sha256sum", "-c", p.name], cwd=p.parent, capture_output=True, text=True)
        if proc.returncode != 0:
            reasons.append(f"frozen-document hash check failed: {f}: {proc.stdout.strip()} {proc.stderr.strip()}")
    return reasons


def _config_diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    keys = set(a) | set(b)
    out = []
    for k in sorted(keys, key=str):
        pa, pb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(pa, dict) and isinstance(pb, dict):
            out += _config_diff(pa, pb, f"{prefix}{k}.")
        elif json.dumps(pa, sort_keys=True, default=str) != json.dumps(pb, sort_keys=True, default=str):
            out.append(f"{prefix}{k}")
    return out


# ------------------------------------------------------------------ report
class Report:
    def __init__(self, layout: Layout, mode: str, out: str | Path, root: str | Path,
                 jobs: dict[str, list[str]] | None = None):
        self.root = Path(root).resolve()
        self.L = layout.resolve(self.root)
        self.mode = mode
        self.out_arg = out
        self.jobs = jobs or {}
        self.ledger = Ledger(mode, self.root)
        self.artifacts: dict[str, dict] = {}
        self.identity: dict = {}
        self.flags: list[str] = []

    # ---------------------------------------------------------- phase 1
    def preflight(self) -> None:
        L, led = self.L, self.ledger
        reasons: list[str] = []
        need = {"manifest": L.manifest, "calibration": L.calibration,
                "pipeline run.json": _p(L.pipeline, "run.json"), "pipeline config_used.json":
                _p(L.pipeline, "config_used.json")}
        absent = [k for k, v in need.items() if v is None or not Path(v).is_file()]
        if absent:
            raise ReportRefusal([f"required input missing: {a}" for a in absent])
        from ..calibration import load_calibration
        from ..manifest import is_synthetic, load_manifest

        self.cfg = led.json(_p(L.pipeline, "config_used.json"))
        self.run = led.json(_p(L.pipeline, "run.json"))
        led.declare(_p(L.pipeline, "run.json"), self.run.get("data_provenance"))
        led.record(Path(L.calibration))
        self.cal = load_calibration(L.calibration)
        led.declare(L.calibration, self.cal.get("data_provenance"))
        led.record(Path(L.manifest))
        self.manifest = load_manifest(L.manifest, self.cfg)
        led.declare(L.manifest, "SYNTHETIC - NOT CLINICAL" if is_synthetic(self.manifest) else "real")
        self.cal_audit = (led.csv(L.calibration_audit, dtype={"image_id": str})
                          if L.calibration_audit and Path(L.calibration_audit).is_file() else None)
        an = Path(L.analysis)
        self.results = {k: _read_results(led, an / f"exp_{k.lower()}") for k in "ABCD"}
        hp = an / "secondary_holm.json"
        self.holm = led.json(hp) if hp.is_file() else None
        if self.holm is not None:
            led.declare(hp, self.holm.get("data_provenance"))
        self.abl_summary, self.abl_variants, self.abl_cfgs = None, {}, {}
        if L.ablations and (Path(L.ablations) / "ablation_summary.json").is_file():
            self.abl_summary = led.json(Path(L.ablations) / "ablation_summary.json")
            for r in self.abl_summary.get("runs", []):
                tag = f"{r['parameter']}={r['value']}"
                d = Path(L.ablations) / tag
                self.abl_variants[tag] = _read_results(led, d / "exp_b")
                if (d / "pipeline/config_used.json").is_file():
                    self.abl_cfgs[tag] = led.json(d / "pipeline/config_used.json")
                if (d / "pipeline/run.json").is_file():
                    rr = led.json(d / "pipeline/run.json")
                    led.declare(d / "pipeline/run.json", rr.get("data_provenance"))
        self.ext = {}
        for seg, e in L.exp_e.items():
            d = Path(e.results_dir)
            rec = {"results": _read_results(led, d), "readiness": None, "run": None, "manifest": None}
            if rec["results"] is None and (d / "readiness.json").is_file():
                rec["readiness"] = led.json(d / "readiness.json")
            if e.pipeline and (Path(e.pipeline) / "run.json").is_file():
                rec["run"] = led.json(Path(e.pipeline) / "run.json")
                led.declare(Path(e.pipeline) / "run.json", rec["run"].get("data_provenance"))
            if e.segmented_manifest and Path(e.segmented_manifest).is_file():
                led.record(Path(e.segmented_manifest))
                rec["manifest"] = load_manifest(e.segmented_manifest, self.cfg)
                led.declare(e.segmented_manifest, "SYNTHETIC - NOT CLINICAL" if is_synthetic(rec["manifest"])
                            else "real")
            self.ext[seg] = rec
        self.rf = _read_results(led, _p(L.exp_f))
        dirs = [L.pipeline, L.analysis, *(Path(L.analysis) / f"exp_{k}" for k in "abcd"), L.ablations,
                L.exp_f, Path(L.calibration).parent, Path(L.manifest).parent]
        for e in L.exp_e.values():
            dirs += [e.results_dir, e.pipeline]
        led.scan_markers(d for d in dirs if d is not None)
        reasons += led.provenance_reasons()
        if reasons:
            raise ReportRefusal(reasons)

        # identities
        ch = config_hash(self.cfg)
        ident = {"config_sha256": ch, "calibration_ids_sha256": self.cal["calibration_ids_sha256"],
                 "calibration_d_um": self.cal["d_um"],
                 "pipeline_code_sha256_of_run": (self.run.get("code_fingerprint") or {}).get("sha256")}
        if self.run.get("full_config_sha256") != ch:
            reasons.append("pipeline run.json full_config_sha256 differs from its config_used.json")
        if self.run.get("calibration_ids_sha256") != self.cal["calibration_ids_sha256"]:
            reasons.append("pipeline run used a different frozen calibration (ID hash differs)")
        if self.run.get("d_um") is not None and not np.isclose(self.run["d_um"], self.cal["d_um"], rtol=1e-12):
            reasons.append("pipeline coverage distance differs from the frozen calibration d")
        for k, r in self.results.items():
            if r is None:
                continue
            if r.get("config_sha256") != ch:
                reasons.append(f"Experiment {k} config_sha256 differs from the pipeline configuration")
            if r.get("calibration_ids_sha256") != self.cal["calibration_ids_sha256"]:
                reasons.append(f"Experiment {k} calibration ID hash differs from the frozen record")
            if r.get("calibration_d_um") is None or not np.isclose(r["calibration_d_um"], self.cal["d_um"], rtol=1e-12):
                reasons.append(f"Experiment {k} calibration d differs from the frozen record")
        for seg, rec in self.ext.items():
            r = rec["results"]
            if r is not None and r.get("config_sha256") != ch:
                reasons.append(f"Experiment E ({seg}) config_sha256 differs from the analysis configuration")
        if self.rf is not None and self.rf.get("config_sha256") != ch:
            reasons.append("Experiment F config_sha256 differs from the analysis configuration")
        for tag, vcfg in self.abl_cfgs.items():
            param = tag.split("=", 1)[0]
            allowed = {".".join(primary.PRIMARY_SETTING_PATHS.get(param, ("", ""))), "pipeline.overlays"}
            diff = set(_config_diff(self.cfg, vcfg))
            if diff - allowed:
                reasons.append(f"ablation {tag} changes more than its declared parameter: {sorted(diff - allowed)}")
            vr = self.abl_variants.get(tag)
            if vr is not None and vr.get("config_sha256") != config_hash(vcfg):
                reasons.append(f"ablation {tag}: exp_b config_sha256 differs from its config_used.json")
            if vr is not None and vr.get("calibration_ids_sha256") != self.cal["calibration_ids_sha256"]:
                reasons.append(f"ablation {tag}: calibration ID hash differs from the frozen record")
        from ..pipeline import pipeline_code_fingerprint

        cur = pipeline_code_fingerprint()["sha256"]
        ident["pipeline_code_sha256_current"] = cur
        if cur != ident["pipeline_code_sha256_of_run"]:
            self.flags.append("per-image pipeline code changed since the processing run "
                              f"({str(ident['pipeline_code_sha256_of_run'])[:12]} → {cur[:12]})")
        if self.mode == CLINICAL:
            if not self.L.frozen_config or not Path(self.L.frozen_config).is_file():
                reasons.append("clinical mode requires the frozen config.yaml")
            else:
                led.record(Path(self.L.frozen_config))
                frozen = config_hash(load_config(self.L.frozen_config))
                ident["frozen_config_sha256"] = frozen
                if frozen != ch:
                    reasons.append("analysis configuration differs from the frozen config.yaml")
            reasons += _frozen_docs_check(self.root)
            for f in FROZEN_SHA256_FILES:
                if (self.root / f).is_file():
                    led.record(self.root / f)
            from .gates import graph_qc_reasons

            g2, g2_flags = graph_qc_reasons(_p(self.L.graph_qc_sheet), _p(self.L.graph_qc_disposition),
                                            self.manifest, self.cfg, self.cal["calibration_ids"], self.run,
                                            ledger=led)
            reasons += g2
            self.flags += g2_flags
        exc = (self.results.get("C") or {}).get("execution") or {}
        ident["exp_c_execution"] = {k: exc.get(k) for k in ("mode", "plan_sha256")}
        self.identity = ident
        if reasons:
            raise ReportRefusal(reasons)

    # ---------------------------------------------------------- phase 2
    def _artifact(self, aid: str, tag: str, title: str, fn, deps=()) -> None:
        """Render one artifact. ``deps`` are input files read in preflight that it uses."""
        from ..experiments.common import ReadinessError

        sources_before = set(self.ledger.files)
        try:
            w = ArtifactWriter(self.out, aid, self.mode)
            try:
                status, extra = fn(w)
            except ReadinessError as exc:
                raise ReportRefusal([f"analysis table not loadable: {r}" for r in exc.reasons]) from exc
            except (ReportRefusal, _Missing):
                raise
            except Exception as exc:  # recorded as a refusal (exit code 4), never silently dropped
                raise ReportRefusal([f"renderer error: {exc!r}"]) from exc
            entry = {"status": status, "tag": tag, "title": title, **extra}
            entry["outputs"] = w.hashes()
        except ReportRefusal as exc:
            import matplotlib.pyplot as plt

            plt.close("all")
            entry = {"status": REFUSED, "tag": tag, "title": title, "reasons": exc.reasons, "outputs": {}}
            for f in sorted((self.out / aid).glob("*")):
                if f.name != "SYNTHETIC_NOT_CLINICAL.txt":
                    f.unlink()
            (self.out / aid / "REFUSED.json").write_text(json.dumps(
                {"data_label": label(self.mode), "artifact": aid, "reasons": exc.reasons}, indent=2, ensure_ascii=False) + "\n")
        except _Missing as exc:
            entry = {"status": MISSING, "tag": tag, "title": title, "missing": exc.args[0], "outputs": {}}
            (self.out / aid / "MISSING_INPUT.json").write_text(json.dumps(
                {"data_label": label(self.mode), "artifact": aid, "missing": exc.args[0]}, indent=2, ensure_ascii=False) + "\n")
        used = set(self.ledger.files) - sources_before
        used |= {self.ledger.rel(p) for p in deps if p is not None}
        entry["sources"] = {s: self.ledger.files[s] for s in sorted(used) if s in self.ledger.files}
        entry["job_ids"] = _jobs_for(list(entry["sources"]), self.jobs)
        self.artifacts[aid] = entry

    def _analysis_table(self):
        if not hasattr(self, "_table"):
            from ..experiments.common import load_analysis_table

            self._table, self._info = load_analysis_table(self.L.pipeline, self.manifest, self.cal,
                                                          self.L.dataset)
            self.ledger.record(Path(self.L.pipeline) / "features.csv")
        return self._table, self._info

    def _note(self, text: str) -> str:
        return f"{label(self.mode)}. {text}"

    def render(self) -> None:
        cfg = self.cfg
        rb = self.results.get("B")
        po = cfg["stress"]["primary_order"]
        fi = cfg["stats"]["fi_column"]
        # Holm values are shown anywhere only after the summary validates against the saved results.
        holm_adj, holm_note = None, "unavailable: secondary_holm.json missing"
        if self.holm is not None:
            try:
                secondary.holm_table(self.holm, self.results, cfg)
                holm_adj, holm_note = secondary.holm_lookup(self.holm), None
            except ReportRefusal as exc:
                holm_note = "withheld: Holm summary failed validation (see M11)"
                self.flags.append("Holm summary failed validation; adjusted p-values withheld: "
                                  + "; ".join(exc.reasons))
        an, pl = Path(self.L.analysis), Path(self.L.pipeline)
        res = {k: an / f"exp_{k.lower()}" / "results.json" for k in "ABCD"}
        base = [self.L.manifest, self.L.calibration, pl / "run.json", pl / "config_used.json"]
        holm_p = an / "secondary_holm.json"

        # M1 + M2
        def m1(w):
            if rb is None:
                raise _Missing(["analysis/exp_b/results.json"])
            table, info = self._analysis_table()
            excl_p = Path(self.L.pipeline) / "exclusions.csv"
            excl = self.ledger.csv(excl_p, dtype={"image_id": str}) if excl_p.is_file() else pd.DataFrame(
                columns=["image_id", "stage", "reason", "details"])
            flow = cohort.cohort_flow(self.manifest, self.cal, self.cal_audit, table, self.run, excl,
                                      {k: v for k, v in self.results.items() if v is not None}, cfg,
                                      self.L.dataset)
            self._flow = flow
            wide = cohort.stage_wide(flow["stages"])
            w.write_csv("cohort_flow.csv", wide)
            w.write_csv("technical_exclusions.csv", flow["technical_exclusions"])
            body = (markdown_table(wide) + "\n\n## Technical exclusions by reason\n\n"
                    + (markdown_table(flow["technical_exclusions"]) if len(flow["technical_exclusions"])
                       else "None recorded.")
                    + "\n\n## Complete-case missingness (primary model columns)\n\n"
                    + markdown_table(pd.DataFrame([flow["complete_case_missing_by_column"]]))
                    + ("\n\n## Calibration pool by FOV\n\n" + markdown_table(pd.DataFrame(
                        [{"fov": k, **v} for k, v in flow["calibration_pool"].items()]))
                       if flow["calibration_pool"] else "\n\nCalibration selection audit: not supplied.")
                    + f"\n\nAll {len(flow['checks'])} reconciliation checks passed; see reconciliation.json.\n")
            w.write_markdown("cohort_flow.md", "M1 Cohort flow (QC/PROV)", body)
            w.write_json("reconciliation.json", {"checks": flow["checks"]})
            w.save_figure(_flow_figure(flow["stages"], self._note(
                "Counts recomputed from hashed inputs and reconciled with recorded analysis audits.")), "cohort_flow")
            return RENDERED, {"n_checks": len(flow["checks"])}

        self._artifact("M1_cohort_flow", "QC/PROV", "Cohort flow and count reconciliation", m1,
                       base + [self.L.calibration_audit, *res.values()])

        def m2(w):
            if self.artifacts["M1_cohort_flow"]["status"] != RENDERED:
                raise ReportRefusal(["Table 1 requires a reconciled cohort flow (M1)"])
            table, _ = self._analysis_table()
            t1 = cohort.table1(table, self._flow["stages"], po)
            w.write_csv("table1.csv", t1)
            w.write_markdown("table1.md", "M2 Table 1 (descriptive; no hypothesis tests)", markdown_table(t1)
                             + "\n\nCells: median [IQR] {missing}; sex: n (%) male [missing]. Columns are the "
                               "analysed images with features (cohort-flow stage 8). No p-values are "
                               "prespecified for Table 1, so none are shown.\n")
            return RENDERED, {}

        self._artifact("M2_table1", "QC/PROV", "Table 1 by group × FOV", m2, base + [res["B"], pl / "features.csv"])

        # M3/M4 (+ X1 forest)
        def m3(w):
            if rb is None:
                raise _Missing(["analysis/exp_b/results.json"])
            pt = primary.primary_tables(rb)
            w.write_csv("primary_models.csv", pt["models"])
            body = markdown_table(pt["models"])
            if len(pt["coefficients"]):
                w.write_csv("primary_coefficients.csv", pt["coefficients"])
                body += "\n\n## All coefficients (standardised continuous covariates)\n\n" + markdown_table(pt["coefficients"])
            if pt["fallback"] is not None and len(pt["fallback"]):
                w.write_csv("ridge_fallback_no_p_value.csv", pt["fallback"])
                body += ("\n\n## Ridge fallback (role fallback_no_p_value; penalised, no p-value)\n\n"
                         + markdown_table(pt["fallback"]))
            body += (f"\n\nPrimary status (verbatim): `{pt['meta']['status_verbatim']}`. Sensitivities are "
                     "`sensitivity_uncorrected` and never replace the primary result. A nonsignificant LRT "
                     "shows only that the data did not provide evidence of improved fit at the prespecified "
                     "α; it does not establish absence of an incremental association.\n")
            w.write_markdown("primary_table.md", "M3/M4 Primary H3 result and prespecified sensitivities", body)
            w.write_json("primary_meta.json", pt["meta"])
            abl = self._ablation_frame()
            w.save_figure(primary.forest_figure(primary.forest_entries(rb, abl, self.abl_variants), self._note(
                "Non-estimable models are listed without an estimate. Ablations and sensitivities are "
                "uncorrected and never replace the primary result.")), "forest_primary_sensitivity_ablation")
            status = RENDERED if rb["primary"].get("status") == "ok" else NOT_ESTIMABLE
            return status, {"primary_status": rb["primary"].get("status"),
                            "failure_reasons": rb["primary"].get("failure_reasons")}

        self._artifact("M3_M4_primary", "PRIM", "Primary H3 result, sensitivities, forest plot", m3,
                       base + [res["B"]] + self._ablation_sources())

        def m5(w):
            if rb is None:
                raise _Missing(["analysis/exp_b/results.json"])
            tab, cv = primary.prediction_table(rb)
            w.write_csv("prediction_cv.csv", tab)
            body = markdown_table(tab)
            if cv.get("status") == "ok":
                d = Path(self.L.analysis) / "exp_b"
                need = [d / "exp_b_oof_predictions.csv", d / "exp_b_auroc_by_repeat.csv"]
                if not all(p.is_file() for p in need):
                    raise _Missing([self.ledger.rel(p) for p in need if not p.is_file()])
                preds = self.ledger.csv(need[0])
                aucs = self.ledger.csv(need[1])
                checks = primary.check_oof(preds, aucs, cv)
                w.write_json("oof_checks.json", {"checks": checks})
                w.save_figure(primary.roc_figure(preds, cv, self._note(
                    "Out-of-fold predictions from repeated grouped cross-validation; the ΔAUROC interval is "
                    "conditional on the fitted fold models (not a refit bootstrap). No p-value.")), "roc_cross_validated")
                body += f"\n\nNote: {cv['paired_delta'].get('note')}\n"
            w.write_markdown("prediction_cv.md", "M5 Cross-validated prediction (PRIM companion, no p-value)", body)
            return (RENDERED if cv.get("status") == "ok" else NOT_RUN), {"cv_status": cv.get("status")}

        self._artifact("M5_prediction_cv", "PRIM-companion", "Cross-validated AUROC and ROC", m5, base + [res["B"]])

        def m6(w):
            d = Path(self.L.analysis) / "exp_b"
            vif, sp = d / "exp_b_vif_primary_sample.csv", d / "exp_b_spearman_primary_sample.csv"
            if rb is None:
                raise _Missing(["analysis/exp_b/results.json"])
            if not vif.is_file() or not sp.is_file():
                w.write_markdown("collinearity.md", "M6 Collinearity", "Not written by Experiment B (written only "
                                 "when the complete-case sample has n > p + 2).")
                return NOT_RUN, {}
            v, s = self.ledger.csv(vif), self.ledger.csv(sp, index_col=0)
            w.write_csv("vif.csv", v)
            w.write_csv("spearman.csv", s.reset_index().rename(columns={"index": "variable"}))
            w.write_markdown("collinearity.md", "M6 Collinearity on the primary complete-case sample",
                             markdown_table(v.round(3)) + "\n\n" +
                             markdown_table(s.round(3).reset_index().rename(columns={"index": "variable"})))
            return RENDERED, {}

        self._artifact("M6_collinearity", "PRIM-companion", "VIF and Spearman matrix", m6, base + [res["B"]])

        # M7 Experiment A
        def m7(w):
            ra = self.results.get("A")
            if ra is None:
                raise _Missing(["analysis/exp_a/results.json"])
            t = secondary.comparison_table(ra.get("comparisons", []), "A", cfg, holm_adj, holm_note=holm_note)
            w.write_csv("exp_a_comparisons.csv", t)
            w.write_markdown("exp_a_comparisons.md", "M7 Experiment A (H1)", markdown_table(t) +
                             "\n\nOnly the pooled Mann–Whitney p of primary-order FI is a Holm-family member "
                             "(H1_H2_secondary, m = 3); it is unadjusted for FOV and must be read with the "
                             "per-FOV and adjusted rows. All other p-values are exploratory_uncorrected.\n")
            table, _ = self._analysis_table()
            w.save_figure(secondary.distribution_figure(
                table, [(fi, "FI (adaptive betweenness)"), ("FI_random", "FI (random, mean of repetitions)")],
                "FI by group and FOV", self._note("Each point is one analysed image; boxes show median and IQR. "
                                                  "Descriptive display; tests are in the table.")), "fi_distribution")
            return RENDERED, {}

        self._artifact("M7_exp_a", "SEC", "Experiment A table and FI distribution", m7,
                       base + [res["A"], holm_p, pl / "features.csv"])

        def m8(w):
            ra = self.results.get("A")
            if ra is None:
                raise _Missing(["analysis/exp_a/results.json"])
            d = Path(self.L.analysis) / "exp_a"
            missing_curves = {k: len(v.get("missing_curves", [])) for k, v in (ra.get("curves") or {}).items()}
            files = sorted(d.glob("exp_a_curves_*.csv"))
            if not files:
                raise _Missing(["analysis/exp_a/exp_a_curves_*.csv"])
            frames = {f.stem.replace("exp_a_curves_", ""): self.ledger.csv(f) for f in files}
            for f in sorted(d.glob("exp_a_curves_*.png")):
                self.ledger.record(f)
            w.save_figure(_curves_figure(frames, self._note("Mean S(q) with 95% group-bootstrap bands, as "
                                                            "computed by Experiment A.")), "mean_curves")
            w.write_json("curves_index.json", {"curve_tables": sorted(frames), "missing_curves_count": missing_curves})
            return RENDERED, {"missing_curves_count": missing_curves}

        self._artifact("M8_curves", "SEC", "Mean S(q) curves by scope", m8, base + [res["A"]])

        def m9(w):
            rd = self.results.get("D")
            if rd is None:
                raise _Missing(["analysis/exp_d/results.json"])
            labels = {}
            q = int(round(cfg["stress"]["q_eval"] * 100))
            order_labels = rd.get("order_labels") or {}
            for o, lab in order_labels.items():
                short = "exploratory heuristic" if "EXPLORATORY" in lab.upper() else lab.split(",")[0].split(";")[0]
                labels[f"G{q}_{o}"] = short
                labels[f"qc_{o}"] = short
            t = secondary.comparison_table(rd.get("comparisons", []), "D", cfg, holm_adj, labels, holm_note)
            w.write_csv("exp_d_comparisons.csv", t)
            table, _ = self._analysis_table()
            ft = secondary.forman_tie_table(table)
            w.write_csv("forman_tie_diagnostics.csv", ft)
            w.write_markdown("exp_d.md", "M9 Experiment D (H2) with Forman-type tie diagnostics (X4)",
                             markdown_table(t) + "\n\n## X4 Forman-type tie diagnostics (median [IQR])\n\n"
                             + markdown_table(ft) + "\n\nForman-type rows are a degree-based heuristic, not "
                             "Ricci curvature, and are exploratory. Ties are broken by edge ID (approximately "
                             "raster order).\n\n## Order labels (verbatim from Experiment D)\n\n"
                             + "\n".join(f"- `{o}`: {lab}" for o, lab in order_labels.items()) + "\n")
            cols = [(f"G{q}_{o}", f"G{q} {o}" + (" (exploratory heuristic)" if o == "forman" else ""))
                    for o in cfg["exp_d"]["orders"]]
            w.save_figure(secondary.distribution_figure(table, cols, f"G{q} by removal order",
                                                        self._note("G10 = mean random S(0.10) − targeted S(0.10).")),
                          "g10_by_order")
            return RENDERED, {}

        self._artifact("M9_exp_d", "SEC/EXPL", "Experiment D table, figure and Forman ties", m9,
                       base + [res["D"], holm_p, pl / "features.csv"])

        def m10(w):
            rc = self.results.get("C")
            if rc is None:
                raise _Missing(["analysis/exp_c/results.json"])
            d = Path(self.L.analysis) / "exp_c"
            need = [d / "exp_c_per_dr.csv", d / "exp_c_matching_audit.csv", d / "exp_c_null_fi.csv"]
            if not all(p.is_file() for p in need):
                raise _Missing([self.ledger.rel(p) for p in need if not p.is_file()])
            per, aud = self.ledger.csv(need[0]), self.ledger.csv(need[1])
            self.ledger.record(need[2])
            ct = secondary.exp_c_tables(rc, per, aud, holm_adj, self.mode == CLINICAL, holm_note)
            w.write_csv("exp_c_summary.csv", ct["summary"])
            w.write_csv("exp_c_status_counts.csv", secondary.missing_z_reasons(per))
            w.write_markdown("exp_c.md", "M10 Experiment C (H4) density-matched thinning",
                             markdown_table(ct["summary"].T.reset_index().rename(columns={"index": "quantity", 0: "value"}))
                             + f"\n\nCaveats (verbatim): {ct['caveats']}\n\nFlags: "
                             + ("; ".join(ct["flags"]) or "none") + "\n")
            w.save_figure(secondary.exp_c_figure(per, aud, float(cfg["exp_c"]["tolerance_rel"]), self._note(
                "Null = uniform random whole-segment thinning of same-FOV NORMAL donors; z-scores share donors "
                "and are dependent; inference is conditional on the donor pool.")), "exp_c_z_residuals_null")
            self.flags += [f"M10: {f}" for f in ct["flags"]]
            return RENDERED, {"flags": ct["flags"], "execution": rc.get("execution", {}).get("mode")}

        self._artifact("M10_exp_c", "SEC", "Experiment C summary and figures", m10, base + [res["C"], holm_p])

        def m11(w):
            if self.holm is None:
                raise _Missing(["analysis/secondary_holm.json"])
            t, checks = secondary.holm_table(self.holm, self.results, cfg)
            w.write_csv("holm_families.csv", t)
            reg = Path(self.L.analysis) / "p_value_register.csv"
            if reg.is_file():
                self.ledger.record(reg)
            w.write_markdown("holm.md", "M11 Holm-adjusted secondary families", markdown_table(t) +
                             f"\n\n{self.holm.get('statement')}\n\nMissing-member policy: "
                             f"{self.holm.get('missing_member_policy')}\n")
            w.write_json("holm_checks.json", {"checks": checks, "register_counts": self.holm.get("register_counts")})
            return RENDERED, {}

        self._artifact("M11_holm", "SEC", "Holm families and p-value register", m11,
                       base + [holm_p, res["A"], res["C"], res["D"]])

        def m12(w):
            from ..manifest import analysis_rows

            rows, _ = analysis_rows(self.manifest, self.cal["calibration_ids"], dataset=self.L.dataset)
            sel, accounting = maps.select(rows, Path(self.L.pipeline), int(cfg["project"]["base_seed"]))
            w.write_csv("selection_accounting.csv", accounting)
            if not len(sel):
                raise _Missing(["per-image overlays (pipeline overlays disabled or absent)"])
            out_sel = sel[["cell", "rank_in_cell", "image_id", "selection_key"]]
            w.write_csv("selection.csv", out_sel)
            unfilled = accounting[accounting.status.ne("filled")].stratum.tolist()
            if unfilled:
                self.flags.append(f"M12: strata without an eligible image: {unfilled}")
            w.write_json("selection_rule.json", {
                "purpose": maps.PURPOSE, "rule": maps.__doc__.split("Rule", 1)[1].strip(),
                "strata": ["disease group (NORMAL/DR)", "FOV"], "n_per_cell": 1,
                "reads_fi_features_curves_or_association_results": False,
                "status": maps.RULE_STATUS})
            for c in [c for c, _ in maps.PANELS]:
                for p in sel[c]:
                    self.ledger.record(Path(p))
            w.save_figure(maps.panel_figure(sel, self._note(maps.ATTRIBUTION if self.mode == CLINICAL else
                                                            "Synthetic fixture graphs; no dataset image.")),
                          "fragility_map_panel")
            return RENDERED, {"n_selected": int(len(sel)), "unfilled_strata": unfilled,
                              "skipped_missing_overlays": int(accounting.skipped_missing_overlays.sum())}

        self._artifact("M12_fragility_maps", "QC/PROV", "FI-blind stratified representative fragility maps (rule fixed 2026-10-06)", m12, base)

        def x1(w):
            if self.abl_summary is None:
                raise _Missing(["ablations/ablation_summary.json"])
            t = self._ablation_frame().fillna("—")
            w.write_csv("ablation_summary_table.csv", t)
            w.write_markdown("ablations.md", "X1 Processing ablations (SENS, sensitivity_uncorrected)",
                             markdown_table(t) + "\n\nEach row re-runs the pipeline, the primary LRT and "
                             "Experiment A with one change. Rows equal to the primary setting duplicate it. "
                             "Failures and not-applicable settings are listed, not hidden.\n")
            return RENDERED, {}

        self._artifact("X1_ablations", "SENS", "Processing ablation summary", x1, base + self._ablation_sources())

        for seg, e in self.L.exp_e.items():
            self._artifact(f"E_{seg}", "EXT", f"Experiment E ({seg})", lambda w, s=seg: self._exp_e(w, s),
                           [self.L.manifest, Path(e.results_dir) / "results.json",
                            Path(e.results_dir) / "readiness.json", e.segmented_manifest,
                            e.pipeline and Path(e.pipeline) / "run.json"])
        if not self.L.exp_e:
            def no_e(w):
                raise _Missing(["no Experiment E inputs in the layout"])

            self._artifact("E_external", "EXT", "Experiment E", no_e)

        def f1(w):
            if self.rf is None:
                raise _Missing([self.L.exp_f or "exp_f results"])
            t = external.exp_f_table(self.rf)
            w.write_csv("exp_f.csv", t)
            w.write_markdown("exp_f.md", "F1 Experiment F (H5, exploratory)", markdown_table(t) +
                             f"\n\nEmbedding: {self.rf.get('embedding')} ({self.rf.get('embedding_weights')}); "
                             f"n = {self.rf.get('n')}; missing embeddings: {len(self.rf.get('missing_embeddings') or [])}. "
                             f"{self.rf.get('note')}\n")
            if self.rf.get("missing_embeddings"):
                self.flags.append("F1: some images lack embeddings")
            return RENDERED, {}

        self._artifact("F1_exp_f", "EXPL", "Experiment F embeddings vs embeddings + fragility", f1,
                       [_p(self.L.exp_f, "results.json")])

    def _ablation_sources(self) -> list:
        if not self.L.ablations:
            return []
        a = Path(self.L.ablations)
        out = [a / "ablation_summary.json"]
        for tag in self.abl_variants:
            out += [a / tag / "exp_b/results.json", a / tag / "pipeline/config_used.json",
                    a / tag / "pipeline/run.json"]
        return out

    def _ablation_frame(self):
        if self.abl_summary is None:
            return None
        return primary.ablation_rows(self.abl_summary, self.abl_variants, self.cfg)

    def _exp_e(self, w, seg):
        rec, e = self.ext[seg], self.L.exp_e[seg]
        if rec["results"] is None:
            if rec["readiness"] is not None:
                rd = rec["readiness"]
                w.write_markdown("readiness.md", f"Experiment E ({seg}) not run",
                                 "The analysis refused to compute any group statistic:\n\n"
                                 + "\n".join(f"- {r}" for r in rd.get("reasons", [])) + "\n")
                w.write_csv("qc.csv", external.qc_table(rd.get("qc") or {}))
                return NOT_RUN, {"reasons": rd.get("reasons")}
            raise _Missing([self.ledger.rel(Path(e.results_dir) / "results.json")])
        r = rec["results"]
        qc = r.get("qc") or {}
        if self.mode == CLINICAL and not (qc.get("ready") and qc.get("binding_verified")):
            raise ReportRefusal([f"Experiment E ({seg}) results lack a passed, bound human-QC gate"])
        flow, checks = external.flow_table(r, rec["manifest"], rec["run"])
        w.write_csv("flow.csv", flow)
        w.write_csv("qc.csv", external.qc_table(qc))
        tr, cvt = external.trend_table(r), external.cv_table(r)
        w.write_csv("trend.csv", tr)
        w.write_csv("cv.csv", cvt)
        body = ("## Flow\n\n" + markdown_table(flow.fillna("")) + "\n\n## E3 QC outcome\n\n"
                + markdown_table(external.qc_table(qc)) + "\n\n## Grade trend (exploratory_uncorrected)\n\n"
                + markdown_table(tr) + "\n\n## Cross-validated AUROC (no p-values)\n\n" + markdown_table(cvt)
                + f"\n\nBaseline used: {r.get('baseline_used')}; dropped: {r.get('baseline_dropped') or 'none'}. "
                  f"Grouping level: {r.get('grouping_level')}.\n\nLimitations (verbatim):\n"
                + "\n".join(f"- {x}" for x in r.get("limitations", [])) + "\n"
                + ("\nFixture QC decisions are automated SYNTHETIC entries accepted only for synthetic data; "
                   "they are not human review.\n" if self.mode == FIXTURE else ""))
        w.write_markdown(f"exp_e_{seg}.md", f"E Experiment E ({seg}; EXT)", body)
        w.write_json("checks.json", {"checks": checks})
        feats_p = Path(e.pipeline or "") / "features.csv"
        if rec["manifest"] is not None and feats_p.is_file():
            feats = self.ledger.csv(feats_p, dtype={"image_id": str})
            w.save_figure(external.grade_figure(feats, rec["manifest"], qc.get("failed_image_ids"),
                                                self.cfg["stats"]["fi_column"], seg, self._note(
                    "Image-level display; DRAC has no patient identifiers. QC-failed images excluded. "
                    "Raw FI is not comparable with OCTA-500 and is never pooled.")), "fi_by_grade")
        return RENDERED, {}

    # ---------------------------------------------------------- phase 3
    def index(self) -> dict:
        gen = code_fingerprint(REPORT_ENTRY_MODULES)
        statuses = {k: v["status"] for k, v in self.artifacts.items()}
        idx = {"format": INDEX_FORMAT, "data_label": label(self.mode), "mode": self.mode,
               "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "layout": asdict(self.L), "identity": self.identity,
               "report_code_fingerprint": {"sha256": gen["sha256"], "files": gen["files"]},
               "rendering_job": {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                 "host": platform.node(), "python": platform.python_version()},
               "job_provenance_supplied": bool(self.jobs), "job_map": self.jobs,
               "flags": self.flags, "status_counts": pd.Series(statuses).value_counts().to_dict(),
               "artifacts": self.artifacts, "plan_rows": requirements.plan_rows(self.artifacts),
               "inputs_sha256": dict(sorted(self.ledger.files.items())),
               "provenance_declarations": self.ledger.declarations,
               "statement": ("Rendering only: values are copied from saved analysis outputs; descriptive "
                             "counts/summaries are recomputed from hashed inputs and reconciled. No model was "
                             "fitted and no test was run by this report.")}
        if not self.jobs:
            idx["flags"] = idx["flags"] + ["no job provenance map supplied (--jobs)"]
        return idx

    def run_all(self) -> dict:
        self.out = None
        out = Path(self.out_arg).resolve()
        if out.exists() and any(out.iterdir()):
            raise ReportRefusal([f"output directory {out} exists and is not empty; reports are never overwritten"])
        self.preflight()
        inputs = [self.L.pipeline, self.L.analysis, self.L.manifest, self.L.calibration, self.L.ablations,
                  self.L.exp_f, *(e.results_dir for e in self.L.exp_e.values())]
        self.out = prepare_output_root(out, inputs, self.mode)
        if self.mode == FIXTURE:
            (self.out / "SYNTHETIC_NOT_CLINICAL.txt").write_text(
                "SYNTHETIC — NOT CLINICAL. Every file below was rendered from synthetic fixtures for "
                "software validation only.\n")
        self.render()
        idx = self.index()
        (self.out / "paper_artifact_index.json").write_text(json.dumps(_clean_json(idx), indent=2, default=str, ensure_ascii=False) + "\n")
        lines = [f"# Paper artifacts — {label(self.mode)}", "",
                 f"Mode: `{self.mode}`. Index: `paper_artifact_index.json` "
                 f"(sha256 {sha256_file(self.out / 'paper_artifact_index.json')}).", "",
                 "| Artifact | Tag | Status | Detail |", "|---|---|---|---|"]
        for k, v in self.artifacts.items():
            det = "; ".join(v.get("reasons", []) or v.get("missing", []) or [str(v.get("primary_status", ""))])
            lines.append(f"| {k} | {v['tag']} | {v['status']} | {det.replace('|', '/')} |")
        lines += ["", "Flags: " + ("; ".join(idx["flags"]) or "none"), ""]
        (self.out / "README.md").write_text("\n".join(lines))
        return idx


class _Missing(Exception):
    pass


def _clean_json(o):
    from .core import _clean

    return _clean(o)


def _flow_figure(stages: pd.DataFrame, note: str):
    import matplotlib.pyplot as plt

    tot = stages[stages.group.eq("all")].sort_values("order")
    fig, ax = plt.subplots(figsize=(8.4, 0.62 * len(tot) + 1.2))
    for i, r in enumerate(tot.itertuples()):
        y = len(tot) - i
        cells = stages[(stages.order == r.order) & ~stages.group.eq("all")]
        detail = ", ".join(f"{c.group} {c.fov}: {c.n}" for c in cells.itertuples())
        excluded = r.stage.startswith(("excluded", "technical", "primary model: missing"))
        ax.add_patch(plt.Rectangle((0.02 + 0.08 * excluded, y - 0.4), 0.9 - 0.08 * excluded, 0.8,
                                   fill=True, color="#f3e3dc" if excluded else "#e2ebf3", ec="#555"))
        ax.text(0.05 + 0.08 * excluded, y + 0.12, f"{r.stage}: n = {r.n}", fontsize=8, va="center")
        ax.text(0.05 + 0.08 * excluded, y - 0.18, detail, fontsize=6.5, va="center", color="#333")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, len(tot) + 0.7)
    ax.set_axis_off()
    ax.set_title("M1 cohort flow (counts by group × FOV)", fontsize=10)
    fig.text(0.5, 0.01, note, ha="center", fontsize=6.5)
    fig.subplots_adjust(top=0.93, bottom=0.06)
    return fig


def _curves_figure(frames: dict[str, pd.DataFrame], note: str):
    import matplotlib.pyplot as plt

    keys = sorted(frames)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.8), squeeze=False, sharey=True)
    for ax, k in zip(axes[0], keys):
        f = frames[k]
        for grp, colour in (("normal", "#4c6a92"), ("DR", "#b04b27")):
            if f"{grp}_mean" in f:
                ax.plot(f.q, f[f"{grp}_mean"], color=colour, label=grp)
                ax.fill_between(f.q, f[f"{grp}_lo"], f[f"{grp}_hi"], color=colour, alpha=0.25)
        ax.set_title(k.replace("_", " "), fontsize=8)
        ax.set_xlabel("fraction of edges removed q")
        ax.set_xlim(0, 1)
    axes[0][0].set_ylabel("S(q)")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.text(0.5, 0.01, note, ha="center", fontsize=6.5)
    fig.subplots_adjust(bottom=0.2, top=0.84, wspace=0.08)
    return fig


def build_report(layout: Layout, mode: str, out: str | Path, root: str | Path = ".",
                 jobs: dict[str, list[str]] | None = None) -> dict:
    return Report(layout, mode, out, root, jobs).run_all()
