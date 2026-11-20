"""Check the workshop manuscript's statistical results against the project archive.

Run with the project's CPU environment, from any directory. All three tables
and the five crude and adjusted estimates in Results are read from paper.tex.
The numerical graph example uses the study's implementation. For the graph-level
Table 2 inputs, first run
interior_perturbation.py as documented in that script.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

from retina_fragility.graph.betweenness import edge_betweenness
from retina_fragility.graph.model import graph_from_edge_list
from retina_fragility.graph.stress import run_removal, summarize_curve


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUTS = ROOT / "outputs"
PILOT = OUTPUTS / "exploratory_20261009"
AUDIT = OUTPUTS / "exploratory_20261116_step0"
LENGTH = "graph_length_density_mm_per_mm2"
INDEX = "FI_adaptive_betweenness"
INTERIOR = "end_loc_interior"
SET_A = ["q_signal", "q_cnr", "q_sharp", "q_stripe"]
SET_B = ["q_faz_floor", "q_faz_noise", "q_lv_signal"]


def read_json(path):
    return json.loads(path.read_text())


def check(label, observed, printed):
    """Use the decimal precision printed for each value, including integers."""
    fields = printed.split()
    values = np.atleast_1d(observed)
    if len(values) != len(fields):
        raise ValueError(f"{label}: found {len(values)} values versus {len(fields)} expected")
    for value, field in zip(values, fields):
        decimals = len(field.partition(".")[2])
        if not np.isclose(value, float(field), rtol=0, atol=0.5 * 10**-decimals - 1e-14):
            raise ValueError(f"{label}: {value} disagrees with manuscript value {field}")


def table_row(tex, label):
    start = tex.index("\n" + label + " &") + len(label) + 3
    cells = tex[start:tex.index(r"\\", start)]
    return " ".join(re.findall(r"[-+]?\d+(?:\.\d+)?", cells))


def prose_interval(tex, marker):
    """Read the estimate and two CI limits from the indicated Results sentence."""
    results = tex.split(r"\section{Results}", 1)[1].split(r"\section{Discussion}", 1)[0]
    statement = re.split(r"\.(?:\s|$)", results.split(marker, 1)[1], maxsplit=1)[0]
    values = re.findall(r"[-+]?\d+\.\d+", statement)
    if len(values) != 3:
        raise ValueError(f"Cannot read estimate and interval after {marker!r}")
    return " ".join(values)


def checked_sentence(tex, marker, label, observed):
    """Compare numbers printed in a sentence, ignoring the exponent in mm²."""
    statement = tex.split(marker, 1)[1]
    statement = re.split(r"\.(?:\s|$)", statement, maxsplit=1)[0]
    statement = statement.replace("$^2$", "²").replace("one DR eye", "1 DR eye")
    check(label, observed, " ".join(re.findall(r"[-+]?\d+(?:\.\d+)?", statement)))


def cohort_and_tables(tex):
    manifest = pd.read_csv(ROOT / "data/manifests/octa500_manifest.csv")
    check("cohort flow", [len(manifest), manifest.calibration.sum(),
          (~manifest.disease.isin(["DR", "NORMAL"])).sum()], "500 20 185")
    eligible = manifest[manifest.disease.isin(["DR", "NORMAL"]) & ~manifest.calibration]
    used = ["image_id", LENGTH, INDEX, "vessel_density", "faz_area_mm2", "loop_density_per_mm2",
            "fractal_dimension", "end_density_per_mm2_pre"]
    features = pd.read_csv(PILOT / "pipeline/features.csv", usecols=used)
    d = eligible.merge(features, on="image_id", validate="one_to_one")
    check("complete eyes and participants", [len(d), d.subject_id.nunique()], "295 295")
    d = d.assign(dr=d.disease.eq("DR").astype(int), wide=d.fov_mm.eq(6).astype(int),
                 male=d.sex.eq("M").astype(int))
    groups = d.groupby(["fov_mm", "dr"], sort=True)
    for label, values in [
        ("Eyes", groups.size()), ("Age (years)", groups.age.mean()),
        (r"\quad SD", groups.age.std()), (r"Male (\%)", 100 * groups.male.mean()),
        ("Vessel length density", groups[LENGTH].mean()), ("FAZ area (mm$^2$)", groups.faz_area_mm2.mean()),
    ]:
        check("Table 1 " + label, values, table_row(tex, label))
    new = read_json(HERE / "interior_perturbation_results.json")
    check("paired graphs", [new["n_common"], new["endpoint_partition_checks_passed"]], "295 1180")
    if new["original_max_abs_discrepancy_vs_archive_per_mm2"] > 1e-10:
        raise ValueError("Original interior endpoints disagree with archive")
    for key, label in [("original", "Original"), ("dropout1", r"1\% dropout"),
                       ("boundary10", "Boundary flips"), ("dilate1", "1-pixel dilation")]:
        result = new["conditions"][key]
        values = [result["dr_coefficient_sd"], *result["hc3_ci95"]]
        for metric in ["median_coverage_percent", "median_paired_change_interior_per_mm2"]:
            values.extend(result["by_field"][fov][metric] for fov in ("3", "6"))
        if key != "original":
            values.extend(result["spearman_by_fov_mm"][str(fov)] for fov in (3, 6))
        check("Table 2 " + label, values, table_row(tex, label))
    return d


def revised_endpoint_checks(tex):
    review = HERE
    sensitivity = read_json(review / "endpoint_sensitivity.json")
    paired = read_json(HERE / "paired_interior_dilation.json")["fields_mm"]
    width = read_json(review / "mask_width_results.json")
    audit = read_json(HERE / "interior_perturbation_results.json")["conditions"]
    fields = ("3", "6")
    results = tex.split(r"\section{Results}", 1)[1].split(r"\section{Discussion}", 1)[0]

    checked_sentence(tex, "one horizontal or vertical pixel (", "physical radius by field",
                     [paired["3"]["one_pixel_um"], 3, paired["6"]["one_pixel_um"], 6])
    checked_sentence(tex, "One-pixel dilation raised median vessel coverage from ",
                     "abstract coverage and rank", [audit["original"]["by_field"]["3"]["median_coverage_percent"],
                     audit["dilate1"]["by_field"]["3"]["median_coverage_percent"], 3,
                     audit["dilate1"]["spearman_by_fov_mm"]["3"]])
    quartiles = sensitivity["fields"]["3"]["full_sample_linear_contrasts"]
    by_quantile = {q["field_pooled_percentile"]: q for q in quartiles}
    contrasts = []
    for quantile in (0.25, 0.75):
        effect = by_quantile[quantile]["dr_minus_healthy_sd"]
        contrasts.extend([effect["estimate"], *effect["hc3_ci95"]])
    checked_sentence(results, "With separate slopes in the 3-mm field,", "3-mm quartile contrasts", contrasts)
    upper = by_quantile[0.75]
    nearby = upper["eyes_within_one_length_unit"]
    checked_sentence(results, " at the upper quartile. Only ", "upper quartile and local support",
                     [nearby["DR"], nearby["healthy"], 1, upper["length_mm_per_mm2"]])

    slopes, changes, coverage, losses, traced_length = [], [], [], [], []
    for field in fields:
        fov = int(field)
        spline = sensitivity["fields"][field]["natural_spline_sensitivity"]
        effect = spline["DR_minus_healthy_slope_beyond_common_curve"]
        slopes.extend([effect["estimate"], *effect["hc3_ci95"], fov])
        shift = paired[field]["at_common_original_length"]["change"]
        changes.extend([shift["dr_minus_healthy"], *shift["hc3_ci95"], fov])
        coverage.extend([audit[c]["by_field"][field]["median_coverage_percent"] for c in ("original", "dilate1")])
        coverage.append(fov)
        losses.extend([-paired[field]["overall_median_per_mm2"]["change"], fov])
        traced_length.append(width[field + "mm"]["median_within_eye_relative_change_percent"]["total_length_mm_post"])
    checked_sentence(results, "Allowing curvature,", "diagnosis-specific slopes with curvature", slopes)
    pooled = sensitivity["pooled_models"]
    joint = pooled["joint_hull_quality_adjusted_dr"]
    checked_sentence(results, "Requiring joint overlap of length density and contrast retained ",
                     "joint-hull overlap and estimate", [pooled["joint_hull_n"]["DR"], pooled["joint_hull_n"]["healthy"],
                     joint["estimate"], *joint["hc3_ci95"]])
    checked_sentence(results, "Dilation raised median vessel coverage from ", "mask coverage", coverage)
    checked_sentence(results, "The median within-eye changes in total traced length were ",
                     "paired traced lengths", traced_length)
    checked_sentence(results, "The median within-eye loss was ", "paired interior endpoint losses", losses)
    checked_sentence(results, "Holding original length density fixed,", "dilation shifts in original units", changes)


def physical_radius_checks(tex, cohort):
    fixed = read_json(HERE / "physical_radius_results.json")
    by_field = fixed["by_field"]
    radius = fixed["radius_um"]
    validation = fixed["validation"]
    check("paired 15-µm cohort", [sum(g["eyes"] for g in by_field.values()),
          validation["cross_dilated_masks_pixel_identical_to_archive"]], f"{len(cohort)} {len(cohort)}")
    neighbors = []
    for field, eyes in cohort.groupby("fov_mm"):
        key = str(int(field))
        record = by_field[key]
        check(f"{key}-mm eyes and DR cases in the 15-µm analysis", [record["eyes"], record["dr"]],
              f"{len(eyes)} {eyes.dr.sum()}")
        pitch = eyes.pixel_size_um.to_numpy()
        reported_kernel = fixed["kernel_by_field"][key]
        np.testing.assert_allclose(pitch, reported_kernel["pitch_um"], rtol=0, atol=1e-10)
        steps = int(np.floor(radius / pitch[0]))
        offset = np.arange(-steps, steps + 1) * pitch[0]
        expected = np.hypot(offset[:, None], offset) <= radius + 1e-12
        np.testing.assert_array_equal(reported_kernel["grid"], expected)
        neighbors.extend([expected.sum() - 1, field])
    checked_sentence(tex, "A post hoc dilation used a common ", "physical dilation cutoff", [radius])
    checked_sentence(tex.replace("eight neighboring", "8 neighboring").replace("four at", "4 at"),
                     "This included ", "native neighbors inside the physical cutoff", neighbors)
    checked_sentence(tex, "With the common ", "15-µm coverage",
                     [radius, by_field["3"]["median_dilated_coverage_percent"], 3,
                      by_field["6"]["median_dilated_coverage_percent"], 6])
    checked_sentence(tex, "The rank correlations with the original tracing were ", "15-µm rank correlations",
                     [by_field[k]["spearman_original_vs_15um_interiors"] for k in ("3", "6")])

    cross = read_json(HERE / "interior_perturbation_results.json")["conditions"]["dilate1"]
    check("independent 6-mm graph matches", validation["cross_dilation_graphs_checked_all_six_mm"],
          str(by_field["6"]["eyes"]))
    wide = by_field["6"]
    np.testing.assert_allclose(
        [wide["median_dilated_coverage_percent"], wide["median_paired_change_interior_per_mm2"],
         wide["spearman_original_vs_15um_interiors"]],
        [cross["by_field"]["6"]["median_coverage_percent"], cross["by_field"]["6"]["median_paired_change_interior_per_mm2"],
         cross["spearman_by_fov_mm"]["6"]], rtol=0, atol=1e-10)


def fragility(d, tex):
    baseline = ["wide", "age", "male", "vessel_density", "faz_area_mm2"]
    standardized = ((d[INDEX] - d[INDEX].mean()) / d[INDEX].std()).rename("F")
    crude = sm.Logit(d.dr, sm.add_constant(standardized)).fit(disp=False)
    check("crude fragility odds ratio", np.exp([crude.params.F, *crude.conf_int().loc["F"]]),
          prose_interval(tex, "The crude odds ratio for DR per SD of fragility was "))
    if d[["sex", "age", "fov_mm", INDEX, "vessel_density", "faz_area_mm2", "loop_density_per_mm2"]].isna().any().any():
        raise ValueError("Missing values in primary model")
    for term, shown in [("loop_density_per_mm2", "0.69 0.25 1.93 0.475"),
                        ("fractal_dimension", "2.56 0.996 6.59 0.0499")]:
        x = sm.add_constant(d[baseline + [term]].copy())
        m0 = sm.Logit(d.dr, x).fit(disp=False)
        x["F"] = standardized
        m1 = sm.Logit(d.dr, x).fit(disp=False)
        lr = 2 * (m1.llf - m0.llf)
        check(term, [np.exp(m1.params["F"]), *np.exp(m1.conf_int().loc["F"]), chi2.sf(lr, 1)], shown)
        if term == "loop_density_per_mm2":
            likelihoods = re.search(
                r"The log likelihood rose from \$(-[\d.]+)\$ to \$(-[\d.]+)\$, "
                r"giving a likelihood-ratio statistic of ([\d.]+)", tex)
            if likelihoods is None:
                raise ValueError("Cannot locate the likelihood comparison in Results")
            check("primary log likelihoods and LR", [m0.llf, m1.llf, lr], " ".join(likelihoods.groups()))
            check("adjusted fragility odds ratio", np.exp([m1.params.F, *m1.conf_int().loc["F"]]),
                  prose_interval(tex, "With the baseline covariates it was "))
    for field, shown in [(None, "-0.90"), (3, "-0.85"), (6, "-0.90")]:
        subset = d if field is None else d[d.fov_mm == field]
        check("fragility/area rank correlation", spearmanr(subset[INDEX], subset.vessel_density).statistic, shown)
    exp_b = PILOT / "analysis/exp_b"
    fit = read_json(exp_b / "results.json")
    failure = fit["sensitivities"]["fractal_added"]
    if failure["status"] != "not_estimable" or not all("at_boundary" in r for r in failure["failure_reasons"]):
        raise ValueError("Joint sensitivity disposition changed")
    pred = pd.read_csv(exp_b / "exp_b_oof_predictions.csv")
    if pred.groupby(["repeat", "group"]).fold.nunique().max() != 1:
        raise ValueError("Participants assigned to multiple folds in a repeat")
    auc = {}
    for model, rows in pred.groupby("model"):
        scores = [roc_auc_score(s.y, s.prob) for _, s in rows.groupby("repeat")]
        check("OOF size and repeats", [len(rows), len(scores)], "2950 10")
        auc[model] = np.mean(scores)
    a, b = auc["baseline"], auc["baseline_plus_FI"]
    check("paired OOF AUROC", [a, b, b - a, *fit["prediction_cv"]["paired_delta"]["ci"]],
          "0.901 0.898 -0.0028 -0.0044 -0.0014")


def worked_graph():
    graph = graph_from_edge_list(6, [(0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 4, 1), (1, 5, 1)])
    arrays = graph.compact()
    scores = edge_betweenness(arrays["n"], arrays["eu"], arrays["ev"], arrays["length"])
    np.testing.assert_array_equal(scores, [9, 5, 5, 5, 5])
    curve = run_removal(graph, "adaptive_betweenness")
    np.testing.assert_allclose(curve.S, [1, 2 / 5, 2 / 5, 1 / 5, 1 / 5, 0])
    check("worked example fragility", summarize_curve(curve)["FI"], "0.52")


def ols(data, outcome, terms):
    fit = sm.OLS(data[outcome], sm.add_constant(data[terms])).fit(cov_type="HC3")
    return [fit.params.dr, *fit.conf_int().loc["dr"]]


def endpoints(d, tex):
    data = d.merge(pd.read_csv(AUDIT / "endnode_mechanism/end_node_parts.csv"), on="image_id", validate="one_to_one")
    for group, name in [(SET_A, "quality_specificity/quality_proxies.csv"), (SET_B, "metric_audit/quality_proxies_b.csv")]:
        proxies = pd.read_csv(AUDIT / name, usecols=["image_id", *group])
        data = data.merge(proxies, on="image_id", validate="one_to_one")
    for col in [INTERIOR, "end_density_per_mm2_pre", "faz_area_mm2", *SET_A, *SET_B]:
        parts = data.groupby("fov_mm")[col]
        data[col + "_z"] = (data[col] - parts.transform("mean")) / parts.transform("std")
    base = ["dr", LENGTH, "wide"]
    first = base + [p + "_z" for p in SET_A]
    second = base + [p + "_z" for p in SET_B]
    check("raw endpoints", ols(data, "end_density_per_mm2_pre_z", base), "-1.22 -1.54 -0.91")
    check("FAZ", ols(data, "faz_area_mm2_z", base), "0.95 0.65 1.26")
    z = INTERIOR + "_z"
    check("crude interior endpoints", ols(data, z, ["dr"]), prose_interval(tex, "On the original cleaned graphs,"))
    check("field-adjusted interior endpoints", ols(data, z, ["dr", "wide"]),
          prose_interval(tex, "Adjusting for field alone gave "))
    check("interior endpoints", ols(data, z, base), prose_interval(tex, "Adding length density reversed the sign to "))
    check("quality set A", ols(data, z, first), "-0.34 -0.55 -0.14")
    check("quality set B", ols(data, z, second), "-0.58 -0.80 -0.36")
    check("additional age/sex adjustment", ols(data, z, first + ["age", "male"])[0], "-0.345")
    in_range = []
    for fov, rows in data.groupby("fov_mm"):
        healthy, dr = rows[rows.dr == 0], rows[rows.dr == 1]
        check("median interior endpoints/mm, DR and healthy",
              [(g[INTERIOR] / g[LENGTH]).median() for g in (dr, healthy)],
              "1.05 0.62" if fov == 3 else "1.71 1.21")
        check("healthy endpoint-density association", spearmanr(healthy[INTERIOR], healthy[LENGTH]).statistic,
              "-0.93" if fov == 3 else "-0.94")
        check("field-specific quality model", ols(rows, z, [p for p in first if p != "wide"]),
              "-0.45 -0.75 -0.15" if fov == 3 else "-0.28 -0.57 +0.01")
        keep = pd.Series(True, index=rows.index)
        for variable in [LENGTH, "q_cnr"]:
            keep &= rows[variable].between(max(dr[variable].min(), healthy[variable].min()),
                                           min(dr[variable].max(), healthy[variable].max()))
        if fov == 3:
            check("high contrast, outside shared ranges", [(dr.q_cnr > healthy.q_cnr.max()).sum(),
                  (~keep.loc[dr.index]).sum()], "13 14")
            check("raw endpoints removed by spur pruning, healthy and DR",
                  [100 * g.n_end_spur.sum() / g.n_end_raw.sum() for g in (healthy, dr)], "63 50")
        in_range.append(rows[keep])
    selected = pd.concat(in_range)
    check("shared ranges DR and healthy", [selected.dr.sum(), (selected.dr == 0).sum()], "47 95")
    check("shared ranges model", ols(selected, z, first), "-0.39 -0.59 -0.19")
    audit = pd.read_csv(AUDIT / "metric_audit/audit_table.csv").set_index("metric")
    check("audited measures and qualifying vessel metrics", [len(audit), audit.drop("faz_area_mm2").robust_candidate.sum()], "11 0")
    rho = [audit.at["end_density_per_mm2_pre", f"spearman_{change}_{field}mm"]
           for field in [3, 6] for change in ["dropout1", "boundary10", "dilate1"]]
    check("minimum raw endpoint rank correlation", min(rho), "0.48")


def tracers_and_review(tex):
    original = read_json(AUDIT / "endnode_mechanism/m4_results.json")["sources"]
    calibrated = read_json(AUDIT / "label_sources_v2/results.json")["tracers"]
    packet = OUTPUTS / "expert_review_2026-11-17"
    key = pd.read_csv(packet / "key_NOT_FOR_REVIEWER.csv")
    review_totals = {"Reference mask": [], "Capillary-scale Frangi": [], "Adaptive threshold": []}
    for part in ("A", "B"):
        reply = pd.read_csv(packet / f"scored_official/part_{part}.csv")
        joined = reply.merge(key, left_on="image", right_on="label", validate="one_to_one")
        check("reviewed eyes", joined.fov_mm.value_counts().sort_index(), "10 10")
        for field, g in joined.groupby("fov_mm"):
            if part == "A":
                check("reference graph verdicts", [g.iloc[:, 1].eq(k).sum() for k in ("pass", "fail", "uncertain")],
                      "10 0 0" if field == 3 else "4 1 5")
                review_totals["Reference mask"].extend([g.iloc[:, 1].eq("pass").sum(), len(g)])
            else:
                passes = [sum(((g["tracer_" + side] == tracer) & (g.iloc[:, j] == "pass")).sum()
                              for side, j in [("X", 1), ("Y", 2)]) for tracer in ("capillary-scale Frangi", "adaptive threshold")]
                check("classical graph passes", passes, "10 9" if field == 3 else "5 1")
                for label, passed in zip(["Capillary-scale Frangi", "Adaptive threshold"], passes):
                    review_totals[label].extend([passed, len(g)])
    raw = read_json(AUDIT / "metric_audit/results.json")["metrics"]["end_density_per_mm2_pre"]["adjusted"]["pooled"]
    check("Table of tracers: Reference mask", [raw["n"], 64, 1, 1, raw["coef"], *raw["ci"],
          *review_totals["Reference mask"]], table_row(tex, "Reference mask"))
    for label, block in [("Uncalibrated Frangi", original["frangi"]),
                         ("Capillary-scale Frangi", calibrated["frangi_capillary"]),
                         ("Adaptive threshold", calibrated["adaptive"]),
                         ("Held-out U-Net", original["unet_heldout"])]:
        model = block["audit_model"]["end_density_per_mm2_pre"]["pooled"]
        ratio = block["density_ratio_median"]
        values = [block["n"], block["n_dr"], ratio["fov_3mm"], ratio["fov_6mm"],
                  model["coef"], *model["ci"], *review_totals.get(label, [])]
        check("Table of tracers: " + label, values, table_row(tex, label))
    mix = read_json(packet / "scored_official/summary.json")["part_A"]["disease_mix"]
    check("reviewed DR cases", [mix[k]["DR"] for k in ("fov_3mm", "fov_6mm")], "1 4")


def degraded_images(tex):
    degradation = read_json(AUDIT / "degradation/results.json")
    versions = pd.read_csv(AUDIT / "degradation/versions.csv")
    original = versions[versions.kind == "original"].set_index("image_id")
    healthy = original[original.y == 0]
    altered = versions[(versions.y == 0) & (versions.kind != "original")]
    retention = {}
    for condition in ("blur", "noise", "dim"):
        effect = degradation["T1"][condition]
        severe = altered[(altered.kind == condition) & (altered.level == 3)].set_index("image_id")
        ratios = severe.density / healthy.density
        increases = severe.ends_per_mm - healthy.ends_per_mm
        retention[condition] = ratios.median() * 100
        if not (ratios.count() == increases.count() == 62 and ratios.median() < 1 and increases.median() > 0
                and np.isclose(ratios.median(), effect["median_density_ratio"])):
            raise ValueError(f"Strongest {condition} effects differ from the reported paired medians")
    printed = re.search(r"blur and dimming retained only ([0-9.]+)\\% and ([0-9.]+)\\%", tex)
    if printed is None:
        raise ValueError("Cannot locate the severe blur and dimming values in the manuscript")
    check("severe blur and dimming: percentage retained", [retention["blur"], retention["dim"]], " ".join(printed.groups()))
    matched = []
    for eye in original.itertuples():
        donors = altered[(altered.image_id != eye.Index) & (altered.fov_mm == eye.fov_mm)
                         & ((altered.density / eye.density - 1).abs() <= 0.05)]
        donor_means = donors.groupby("image_id").ends_per_mm.mean().dropna()
        if len(donor_means) >= 3:
            matched.append((eye.y, eye.fov_mm, eye.ends_per_mm - donor_means.mean()))
    paired = pd.DataFrame(matched, columns=["dr", "field", "difference"])
    for dr, phase, printed in [(0, "T2", "62 59 -0.12 -0.15 -0.11"), (1, "T3", "15 12 -0.95 -1.52 -0.50")]:
        pooled = degradation[phase]["ends_per_mm"]["pooled"]
        diffs = paired[paired.dr == dr].difference
        if not np.isclose(diffs.median(), pooled["median"]):
            raise ValueError(f"Independent {phase} reconstruction disagrees with archived median")
        check(phase + " degraded-image comparison", [len(original[original.y == dr]), len(diffs), diffs.median(),
              *pooled["ci"]], printed)
    check("6-mm DR with degraded matches", ((paired.dr == 1) & (paired.field == 6)).sum(), "5")


if __name__ == "__main__":
    tex = (HERE.parent / "paper.tex").read_text()
    cohort = cohort_and_tables(tex)
    worked_graph()
    fragility(cohort, tex)
    endpoints(cohort, tex)
    revised_endpoint_checks(tex)
    physical_radius_checks(tex, cohort)
    tracers_and_review(tex)
    degraded_images(tex)
    print("Verified all three tables, the crude and adjusted fits, the worked graph, and the sensitivity analyses.")
