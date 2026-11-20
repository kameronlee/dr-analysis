"""Machine-readable accounting of every artifact-plan row (33 rows; RESULTS_ARTIFACT_PLAN.md §§2-5).

For each row: whether it is required, which generator produces it (a report
artifact ID or an existing writer outside the renderer), the gates that must
pass first, and why the clinical artifact is still unavailable. Software
availability is kept separate from clinical results and from human review:
``software`` describes code, ``clinical_result`` and ``human_review`` describe
what exists as of the stated date. Optional rows stay optional; no optional
analysis is adopted here.
"""
from __future__ import annotations

AS_OF = "2026-10-06"
NOT_RUN = "unavailable: clinical analysis not run"

PLAN_ROWS: dict[str, dict] = {
    "M1": {"required": "required", "report_artifact": "M1_cohort_flow", "gates": ["G4", "G5"],
           "software": "implemented (reporting/cohort.py; count reconciliation)", "clinical_result": NOT_RUN},
    "M2": {"required": "required", "report_artifact": "M2_table1", "gates": ["G4"],
           "software": "implemented (reporting/cohort.py)", "clinical_result": NOT_RUN},
    "M3": {"required": "required", "report_artifact": "M3_M4_primary", "gates": ["G0", "G1", "G2", "G3", "G4", "G5"],
           "software": "implemented (analysis: experiments/exp_b.py; table: reporting/primary.py)",
           "clinical_result": NOT_RUN},
    "M4": {"required": "required", "report_artifact": "M3_M4_primary", "gates": ["G5"],
           "software": "implemented", "clinical_result": NOT_RUN},
    "M5": {"required": "required", "report_artifact": "M5_prediction_cv", "gates": ["G5"],
           "software": "implemented (ROC from saved out-of-fold predictions)", "clinical_result": NOT_RUN},
    "M6": {"required": "required (supplement acceptable)", "report_artifact": "M6_collinearity", "gates": ["G5"],
           "software": "implemented", "clinical_result": NOT_RUN},
    "M7": {"required": "required", "report_artifact": "M7_exp_a", "gates": ["G5"],
           "software": "implemented", "clinical_result": NOT_RUN},
    "M8": {"required": "required", "report_artifact": "M8_curves", "gates": ["G5"],
           "software": "implemented (CSV/PNG written by experiments/exp_a.py; combined figure re-drawn "
                       "from those CSVs by the report)", "clinical_result": NOT_RUN},
    "M9": {"required": "required", "report_artifact": "M9_exp_d", "gates": ["G5"],
           "software": "implemented", "clinical_result": NOT_RUN},
    "M10": {"required": "required", "report_artifact": "M10_exp_c", "gates": ["G3", "G4", "G5", "G6"],
            "software": "implemented (sharded execution required as evidence; serial flagged)",
            "clinical_result": NOT_RUN},
    "M11": {"required": "required", "report_artifact": "M11_holm", "gates": ["G6"],
            "software": "implemented (bound to saved member p-values)", "clinical_result": NOT_RUN},
    "M12": {"required": "required (main or supplement)", "report_artifact": "M12_fragility_maps", "gates": ["G4"],
            "software": "implemented; FI-blind disease/FOV-stratified selection rule fixed 2026-10-06 before "
                        "clinical outcomes (automated implementation decision accepted"
                        "reports/m12_selection_rule_2026-10-06.json); skipped/unfilled strata recorded",
            "clinical_result": NOT_RUN, "pending": "final caption attribution review"},
    "E1": {"required": "required", "report_artifact": "E_frangi", "gates": ["G8", "G9"],
           "software": "implemented (readiness refusals reported as not_run)", "clinical_result": NOT_RUN,
           "human_review": "DRAC Frangi QC 0/30 decisions"},
    "E2": {"required": "required (planned second segmenter)", "report_artifact": "E_unet", "gates": ["G7", "G8", "G9"],
           "software": "implemented; output paths not yet fixed in CLI_USAGE.md (layout file supported)",
           "clinical_result": NOT_RUN, "human_review": "DRAC U-Net QC 0/30 decisions"},
    "E3": {"required": "required", "report_artifact": "E_<segmenter> (qc.csv)", "gates": ["G8"],
           "software": "implemented (aggregate fail reasons only)", "clinical_result": NOT_RUN,
           "human_review": "0/60 baseline DRAC decisions"},
    "F1": {"required": "required as exploratory", "report_artifact": "F1_exp_f", "gates": ["G8", "G9"],
           "software": "implemented", "clinical_result": NOT_RUN},
    "F2": {"required": "optional", "report_artifact": None, "gates": [],
           "software": "embedding code exists; weights availability not established",
           "clinical_result": "not planned", "pending": "optional; not adopted"},
    "S1": {"required": "required", "report_artifact": None, "gates": ["G7"],
           "software": "existing writer segmentation/run.py (job 1467330)",
           "clinical_result": "exists: outputs/unet/octa500_v1/model/meta.json (training record, not a clinical result)"},
    "S2": {"required": "required", "report_artifact": None, "gates": ["G7"],
           "software": "scripts/render_unet_heldout_report.py --display identity_repair_2026-10-06",
           "clinical_result": "exists: descriptive segmentation agreement (SEG), not a disease result; "
                              "reports/manuscript/unet_octa500_v1_identity_repair_2026-10-06/"},
    "S3": {"required": "required", "report_artifact": None, "gates": ["G7"],
           "software": "existing qc-export (job 1467330)",
           "clinical_result": "exists: blank 30-row package outputs/drac_qc/unet_octa500_v1/review_package/",
           "human_review": "0/30 decisions"},
    "S4": {"required": "optional (not prespecified)", "report_artifact": None, "gates": [],
           "software": "implemented, not planned", "clinical_result": "not planned", "pending": "optional; not adopted"},
    "S5": {"required": "required", "report_artifact": None, "gates": ["G2"],
           "software": "sheet exists; clinical report enforces completeness/binding/disposition "
                       "(reporting/gates.py)", "clinical_result": "not applicable",
           "human_review": "OCTA-500 graph QC 0/20 decisions; disposition record not yet created"},
    "S6": {"required": "required by the task contract", "report_artifact": None, "gates": ["G4"],
           "software": "existing overlay-sheet command (experiments/exp0_dropout.overlay_sheet)",
           "clinical_result": NOT_RUN, "human_review": "20-image overlay review not started (needs G4 outputs)"},
    "V1": {"required": "required (supplement)", "report_artifact": None, "gates": [],
           "software": "scripts/run_synthetic.py", "clinical_result": "not applicable (SYNTHETIC; reports/experiment0/ exists)"},
    "V2": {"required": "required", "report_artifact": None, "gates": [],
           "software": "pytest suite", "clinical_result": "not applicable (SYNTHETIC)",
           "pending": "re-run at the final code freeze"},
    "V3": {"required": "required", "report_artifact": None, "gates": [],
           "software": "synthetic-e2e", "clinical_result": "not applicable (SYNTHETIC)",
           "pending": "re-run at the final code freeze"},
    "V4": {"required": "required (supplement citation)", "report_artifact": None, "gates": [],
           "software": "tests/test_exp_c_shards.py; Slurm 1460984/1460985",
           "clinical_result": "not applicable (SYNTHETIC)"},
    "V5": {"required": "required (Experiment 0 extension)", "report_artifact": None, "gates": ["G2"],
           "software": "existing dropout-check command", "clinical_result": "unavailable: not run (must follow G2)"},
    "X1": {"required": "required", "report_artifact": "X1_ablations (+ forest in M3_M4_primary)", "gates": ["G4"],
           "software": "implemented", "clinical_result": NOT_RUN},
    "X2": {"required": "prespecified in the task", "report_artifact": "E_<segmenter> via a layout file per variant",
           "gates": ["G8 per variant"],
           "software": "segment --threshold-scale; review inputs prepared(job 1767791: 4 variants x 30 "
                       "blank hash-bound sheets)",
           "clinical_result": NOT_RUN, "human_review": "0/120 variant decisions (fresh review required; no "
                                                       "amendment waives it)"},
    "X3": {"required": "analysis unspecified", "report_artifact": None, "gates": ["before G4"],
           "software": "values saved per image; no analysis consumes them", "clinical_result": NOT_RUN,
           "pending": "owner decision (recommendation: descriptive agreement table only)"},
    "X4": {"required": "required whenever Forman rows are shown", "report_artifact": "M9_exp_d (forman_tie_diagnostics.csv)",
           "gates": ["G4"], "software": "implemented", "clinical_result": NOT_RUN},
    "X5": {"required": "optional (not prespecified)", "report_artifact": None, "gates": ["before G4/G6"],
           "software": "not implemented", "clinical_result": "not planned",
           "pending": "owner decision; none adopted (PREOUTCOME_REPORTING_DECISIONS_2026-10-06.md §2)"},
}


def plan_rows(artifacts: dict[str, dict] | None = None) -> dict[str, dict]:
    """Plan rows with this run's render status attached where a report artifact exists."""
    out = {}
    for rid, row in PLAN_ROWS.items():
        r = dict(row, as_of=AS_OF)
        aid = (row.get("report_artifact") or "").split(" ")[0]
        if artifacts is not None and aid in artifacts:
            r["render_status_this_run"] = artifacts[aid]["status"]
        out[rid] = r
    return out
