"""Independent post hoc checks of cross-tracer agreement and field-specific AUROC.

Run in the project's CPU environment from any directory, supplying --root and --output.
The only file written is the aggregate JSON selected by --output.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


AUDIT = Path("outputs/exploratory_20261116_step0")
PRIMARY = Path("outputs/exploratory_20261009")
TRACERS = ("adaptive", "frangi_capillary")


def eye_table(path):
    table = pd.read_csv(path).set_index("image_id")
    if not table.index.is_unique:
        raise ValueError(f"Duplicate image identifiers in {path.name}")
    return table


def interior_agreement(root, cohort, config):
    from retina_fragility.io_masks import read_mask
    from scripts.endnode_mechanism import location_parts
    from scripts.step0_null_controls import graph_path

    original = eye_table(root / AUDIT / "endnode_mechanism/end_node_parts.csv")
    primary = eye_table(root / PRIMARY / "pipeline/features.csv")
    result = {}
    for tracer in TRACERS:
        folder = root / AUDIT / "label_sources_v2" / tracer / "pipeline"
        features = eye_table(folder / "features.csv")
        digest = hashlib.sha256()
        rows = []
        for eye, case in cohort.iterrows():
            faz = read_mask(root / case.faz_mask_path, config["data"]["label_maps"]["faz"])
            raw = graph_path(folder, eye).read_bytes()
            digest.update(eye.encode() + hashlib.sha256(faz.tobytes()).digest() + hashlib.sha256(raw).digest())
            graph = json.loads(raw)
            feature = features.loc[eye]
            if graph["meta"]["image_id"] != eye or tuple(graph["shape"]) != faz.shape:
                raise ValueError(f"Invalid graph identifier or dimensions for {eye} and {tracer}")
            if not np.isclose(primary.at[eye, "analysis_area_mm2"], feature.analysis_area_mm2, atol=1e-12, rtol=0):
                raise ValueError(f"Area mismatch for {eye} and {tracer}")
            counts = location_parts(graph["nodes"], graph["edges"], graph["shape"], faz)
            if sum(counts.values()) != feature.n_endpoints_post:
                raise ValueError(f"Endpoint partition failed for {eye} and {tracer}")
            rows.append({
                "fov": int(case.fov_mm),
                "dr": int(case.disease == "DR"),
                "reference": original.at[eye, "end_loc_interior"],
                "tracer": counts["end_loc_interior"] / feature.analysis_area_mm2,
                "length_density": feature.graph_length_density_mm_per_mm2,
            })
        df = pd.DataFrame(rows)
        df["wide"] = df.fov.eq(6).astype(int)
        outcome = df.groupby("fov").tracer.transform(lambda s: (s - s.mean()) / s.std())
        design = sm.add_constant(df[["dr", "wide", "length_density"]])
        model = sm.OLS(outcome, design, missing="raise").fit(cov_type="HC3")
        result[tracer] = {
            "graph_and_faz_digest_sorted_by_image_id": digest.hexdigest(),
            "adjusted_dr_minus_healthy_in_tracer_own_field_sd": float(model.params["dr"]),
            "nominal_hc3_ci95": [float(v) for v in model.conf_int().loc["dr"]],
            "by_field_mm": {
                str(field): {
                    "eyes": len(group),
                    "dr_eyes": int(group.dr.sum()),
                    "spearman_reference_vs_tracer": float(spearmanr(group.reference, group.tracer).statistic),
                    "median_tracer_minus_reference_endpoints_per_mm2": float((group.tracer - group.reference).median()),
                }
                for field, group in df.groupby("fov")
            },
        }
    return result


def discrimination(root, cohort):
    pred = pd.read_csv(root / PRIMARY / "analysis/exp_b/exp_b_oof_predictions.csv")
    if set(pred.image_id) != set(cohort.index) or set(pred.model) != {"baseline", "baseline_plus_FI"}:
        raise ValueError("Prediction cohort or model keys changed")
    data = pred.merge(cohort[["disease", "fov_mm"]], left_on="image_id", right_index=True, validate="many_to_one")
    if not np.array_equal(data.y, data.disease.eq("DR").astype(int)):
        raise ValueError("Prediction diagnoses differ from cohort")
    if data.duplicated(["repeat", "model", "image_id"]).any() or data.repeat.nunique() != 10:
        raise ValueError("Invalid held-out prediction structure")
    if not data.groupby(["repeat", "model"]).size().eq(len(cohort)).all() or len(data) != len(cohort) * 20:
        raise ValueError("Incomplete held-out predictions")
    result = {}
    for field, rows in data.groupby("fov_mm"):
        fields = {}
        for (repeat, model), group in rows.groupby(["repeat", "model"]):
            fields.setdefault(repeat, {})[model] = roc_auc_score(group.y, group.prob)
        scores = pd.DataFrame.from_dict(fields, orient="index")
        result[str(int(field))] = {
            "eyes": rows.image_id.nunique(),
            "dr_eyes": rows.loc[rows.disease.eq("DR"), "image_id"].nunique(),
            "baseline_mean_auroc": float(scores.baseline.mean()),
            "augmented_mean_auroc": float(scores.baseline_plus_FI.mean()),
            "mean_paired_delta": float((scores.baseline_plus_FI - scores.baseline).mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from retina_fragility.config import load_config

    config = load_config(root / "config.yaml")
    manifest = eye_table(root / config["data"]["manifest"])
    cohort = manifest.loc[manifest.disease.isin(["DR", "NORMAL"]) & ~manifest.calibration].sort_index()
    if len(cohort) != 295 or cohort.subject_id.nunique() != 295:
        raise ValueError("Eligible cohort changed")
    results = {
        "status": "Independent, post hoc review diagnostics. All comparisons reuse the discovery cohort.",
        "endpoint": "Degree-one nodes more than five native pixels from the FAZ and image edge, divided by original analysis area in square millimeters. Rank correlation compares paired eyes within each field. Adjusted models use each tracer's own field-specific outcome SD and length density, with a field indicator and common length slope.",
        "discrimination": "AUROC computed by scan field within each of ten repeats on stored predictions from the original pooled cross-validated models. Models were not refitted by field.",
        "cross_tracer_interior_endpoints": interior_agreement(root, cohort, config),
        "existing_model_discrimination_by_field_mm": discrimination(root, cohort),
    }
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
