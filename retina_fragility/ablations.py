"""Parameterised ablations of the reference-label pipeline (+ primary B and A).

Each (parameter, value) re-runs the per-image pipeline into its own directory
with one configuration change, then the primary LRT (B) and Experiment A. The
summary lists every run, including failures (exceptions are recorded, not
hidden). Segmentation-threshold ablations apply only to segmenter pipelines
(Experiment E) and are reported as not applicable here.

Ablations are sensitivity analyses; they never replace the primary result.
"""
from __future__ import annotations

import copy
import traceback
from pathlib import Path

import pandas as pd

from .calibration import d_for_percentile
from .experiments import exp_a, exp_b
from .experiments.common import load_analysis_table
from .manifest import analysis_rows
from .pipeline import run_manifest
from .provenance import write_json

PARAM_PATHS = {"step_frac": ("stress", "step_frac"),
               "spur_length_factor": ("graph", "spur_length_factor"),
               "min_fragment_length_um": ("graph", "min_fragment_length_um"),
               "crop_center_mm": ("pipeline", "crop_center_mm"),
               # centerline = the guide-compatible pixel-path coverage (secondary FIperf/P0 only)
               "coverage_geometry": ("stress", "coverage_geometry")}


def _variant(cfg: dict, param: str, value) -> dict:
    c = copy.deepcopy(cfg)
    c["pipeline"]["overlays"] = False  # figures are not needed for sensitivity runs
    if param in PARAM_PATHS:
        a, b = PARAM_PATHS[param]
        c[a][b] = value
    return c


def run(manifest: pd.DataFrame, dataset: str, calibration: dict, cfg: dict, root, out_root,
        params: list[str] | None = None, n_workers: int | None = None) -> pd.DataFrame:
    out_root = Path(out_root)
    params = params or list(cfg["ablations"])
    rows_in, _ = analysis_rows(manifest, calibration["calibration_ids"], dataset=dataset)
    summary = []
    for param in params:
        for value in cfg["ablations"][param]:
            tag = f"{param}={value}"
            rec = {"parameter": param, "value": value}
            if param == "seg_threshold_scale":
                rec.update(status="not_applicable",
                           reason="segmentation threshold ablation applies to Experiment E segmenters")
                summary.append(rec)
                continue
            try:
                c = _variant(cfg, param, value)
                d_um = calibration["d_um"]
                if param == "d_percentile":
                    d_um = d_for_percentile(calibration, float(value))
                rec["d_um"] = d_um
                pdir = out_root / tag / "pipeline"
                run = run_manifest(rows_in, c, root, pdir, d_um, n_workers=n_workers,
                                   calibration_record=calibration)
                table, info = load_analysis_table(pdir, manifest, calibration, dataset,
                                                  expected_d_um=d_um)
                rb = exp_b.run(table, info, c, out_root / tag / "exp_b", calibration)
                ra = exp_a.run(table, info, c, pdir, out_root / tag / "exp_a", calibration)
                prim = rb["primary"]
                a_fi = next((r for r in ra["comparisons"] if r.get("scope") == "pooled"
                             and r.get("feature") == c["stats"]["fi_column"]), {})
                a_perf = next((r for r in ra["comparisons"] if r.get("scope") == "pooled"
                               and r.get("feature") == "FIperf_adaptive_betweenness"), {})
                rec.update(status="ok", n_ok=run["n_ok"], n_excluded=run["n_excluded"],
                           primary_status=prim["status"], primary_failures=";".join(prim["failure_reasons"]),
                           primary_n=prim["n"], primary_lr=prim["lr_statistic"], primary_p=prim["p_value"],
                           primary_or=prim.get("odds_ratio_added"),
                           a_pooled_auroc_fi=a_fi.get("auroc"), a_pooled_mwu_p_fi=a_fi.get("mwu_p"),
                           a_pooled_auroc_fiperf=a_perf.get("auroc"),
                           p_value_role="sensitivity_uncorrected (ablation)",
                           data_provenance=info["data_provenance"])
            except Exception as exc:  # recorded failure
                rec.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            summary.append(rec)
    df = pd.DataFrame(summary)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "ablation_summary.csv", index=False)
    write_json(out_root / "ablation_summary.json", {"runs": summary,
                                                    "note": "Sensitivity analyses; the primary result "
                                                            "is the unablated configuration."})
    return df
