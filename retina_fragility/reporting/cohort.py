"""M1 cohort flow with count reconciliation, and M2 Table 1 (descriptive, no p-values).

Counts are recomputed from the hashed manifest, frozen calibration record and
pipeline outputs with the same selection functions the analysis uses
(``manifest.analysis_rows``, ``experiments.common.load_analysis_table``,
``stats.primary.complete_cases``). Every recomputed count must equal the count
recorded by the analysis (``analysis_table_audit``, ``run.json``,
``primary.complete_case_audit``); any difference refuses both artifacts.
Outputs contain aggregate counts only, never image identifiers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..manifest import analysis_rows
from ..provenance import sha256_strings
from ..stats.primary import complete_cases
from .core import ReportRefusal, fmt

GROUP_LABEL = {"normal": "NORMAL", "dr": "DR"}


def _group(series: pd.Series) -> pd.Series:
    return series.map(GROUP_LABEL).fillna("other")


def _fov(series: pd.Series) -> pd.Series:
    return series.map(lambda f: f"{float(f):g} mm" if pd.notna(f) else "unknown")


def _counts(frame: pd.DataFrame, stage: str, order: int, group: pd.Series | None = None) -> list[dict]:
    g = _group(frame.analysis_group) if group is None else group
    f = _fov(frame.fov_mm)
    out = []
    for (gg, ff), n in pd.Series(1, index=pd.MultiIndex.from_arrays([g, f])).groupby(level=[0, 1]).sum().items():
        out.append({"order": order, "stage": stage, "group": gg, "fov": ff, "n": int(n)})
    out.append({"order": order, "stage": stage, "group": "all", "fov": "all", "n": int(len(frame))})
    return out


def check(checks: list[dict], name: str, expected, observed) -> None:
    ok = bool(expected == observed)
    checks.append({"check": name, "expected": expected, "observed": observed, "ok": ok})


def cohort_flow(manifest: pd.DataFrame, calibration: dict, calibration_audit: pd.DataFrame | None,
                table: pd.DataFrame, run: dict, exclusions: pd.DataFrame, results: dict[str, dict],
                cfg: dict, dataset: str) -> dict:
    """Stages, technical exclusions by reason and the reconciliation record.

    ``results`` maps experiment letter -> results.json for the OCTA-500
    experiments present (B required). Raises ReportRefusal when counts differ.
    """
    checks: list[dict] = []
    st = cfg["stats"]
    rb = results["B"]
    info = rb["analysis_table_audit"]
    prim = rb["primary"]
    ds = manifest[manifest.dataset == dataset]
    cal_ids = list(calibration["calibration_ids"])
    rows, audit = analysis_rows(manifest, cal_ids, dataset=dataset)
    audit = audit.set_index("image_id")
    ds_reasons = audit.loc[ds.image_id, "reasons"]

    stages: list[dict] = []
    stages += _counts(ds, f"{dataset} manifest rows", 1)
    other = ds[~ds.analysis_group.isin(["normal", "dr"])]
    stages += _counts(other, "excluded: disease label not in the primary analysis", 2)
    eligible = ds[ds.analysis_group.isin(["normal", "dr"])]
    stages += _counts(eligible, "primary-eligible by disease label (NORMAL or DR)", 3)
    cal_rows = eligible[eligible.image_id.isin(cal_ids)]
    stages += _counts(cal_rows, "excluded: frozen calibration eyes", 4)
    held = eligible[~eligible.image_id.isin(cal_ids) & ds_reasons.loc[eligible.image_id].ne("").to_numpy()]
    if len(held):
        stages += _counts(held, "excluded: other manifest reasons (held-out test/different dataset)", 5)
    stages += _counts(rows, "analysis rows (primary-eligible, non-calibration)", 6)

    feat_ids = set(table.image_id.astype(str))
    without = rows[~rows.image_id.isin(feat_ids)]
    excl = exclusions.copy()
    excl["image_id"] = excl.image_id.astype(str)
    reason_of = excl.set_index("image_id").reason.to_dict() if len(excl) else {}
    tech = without.assign(reason=without.image_id.map(lambda i: reason_of.get(i) or "not_processed"))
    reason_rows = []
    for (r, g, f), n in (tech.assign(g=_group(tech.analysis_group), f=_fov(tech.fov_mm))
                         .groupby(["reason", "g", "f"]).size().items()):
        reason_rows.append({"reason": r, "group": g, "fov": f, "n": int(n)})
    stages += _counts(without, "technical exclusions (pipeline; reasons below)", 7)
    stages += _counts(rows[rows.image_id.isin(feat_ids)], "analysed images with features", 8)

    fi = st["fi_column"]
    base = list(st["baseline_primary"])
    cc, cc_audit = complete_cases(table, base + [fi], "y")
    dropped = table[~table.image_id.isin(set(cc.image_id))]
    stages += _counts(dropped, "primary model: missing covariate or FI (complete-case exclusion)", 9)
    stages += _counts(cc, "primary model complete cases", 10)

    # ---- reconciliation against what the analysis recorded
    n_ds = len(ds)
    n_other_reason = int(ds_reasons.ne("").sum())
    check(checks, "manifest rows = excluded by manifest reasons + analysis rows", n_ds,
          n_other_reason + len(rows))
    check(checks, "analysis rows recorded by Experiment B = recomputed", info.get("n_analysis_rows"),
          int(len(rows)))
    recorded_excl = {k: int(v) for k, v in (info.get("manifest_exclusions") or {}).items()}
    recomputed_excl = {k: int(v) for k, v in audit[~audit.included].reasons.value_counts().items()}
    check(checks, "manifest exclusion reasons recorded = recomputed", recorded_excl, recomputed_excl)
    check(checks, "calibration exclusions recorded = frozen calibration IDs",
          info.get("n_calibration_excluded"), len(cal_ids))
    check(checks, "frozen calibration IDs are NORMAL rows of this dataset", len(cal_ids),
          int(ds[ds.image_id.isin(cal_ids)].analysis_group.eq("normal").sum()))
    check(checks, "calibration ID hash matches the frozen record", calibration["calibration_ids_sha256"],
          sha256_strings(cal_ids))
    if calibration_audit is not None:
        sel = calibration_audit[calibration_audit.selected.astype(str).str.lower().eq("true")]
        check(checks, "calibration selection audit selects exactly the frozen IDs",
              calibration["calibration_ids_sha256"], sha256_strings(sel.image_id.astype(str)))
    check(checks, "images with features recorded = recomputed", info.get("n_with_features"), int(len(table)))
    n_without_rec = len(info.get("rows_without_features") or [])
    check(checks, "analysis rows = with features + without features (recorded)", int(len(rows)),
          int(info.get("n_with_features", -1)) + n_without_rec)
    check(checks, "pipeline processed exactly the analysis rows (run.json n_rows)", int(len(rows)),
          run.get("n_rows"))
    check(checks, "pipeline n_ok = images with features", run.get("n_ok"), int(len(table)))
    check(checks, "pipeline n_excluded = technical exclusions", run.get("n_excluded"), int(len(without)))
    rec_reasons = {k: int(v) for k, v in (run.get("exclusion_reasons") or {}).items()}
    check(checks, "pipeline exclusion reasons = technical exclusions by reason", rec_reasons,
          {k: int(v) for k, v in tech.reason.value_counts().items()})
    check(checks, "every technical exclusion has a recorded pipeline reason", 0,
          int(tech.reason.eq("not_processed").sum()))
    check(checks, "no feature row outside the analysis rows", 0,
          len(feat_ids - set(rows.image_id.astype(str))))
    check(checks, "no calibration eye among analysed features", 0, len(feat_ids & set(cal_ids)))
    cca = prim.get("complete_case_audit") or {}
    check(checks, "primary complete-case input = images with features", cca.get("n_input"), int(len(table)))
    check(checks, "primary complete cases recorded = recomputed", cca.get("n_complete"), int(len(cc)))
    check(checks, "primary n = complete cases", prim.get("n"), int(len(cc)))
    check(checks, "complete + dropped = input", cca.get("n_input"),
          (cca.get("n_complete") or 0) + (cca.get("n_dropped") or 0))
    check(checks, "primary events/non-events = recomputed DR/NORMAL complete cases",
          [prim.get("n_events"), prim.get("n_nonevents")],
          [int((cc.y == 1).sum()), int((cc.y == 0).sum())])
    for letter, res in sorted(results.items()):
        if letter == "B" or res is None:
            continue
        a = res.get("analysis_table_audit") or {}
        check(checks, f"Experiment {letter} analysed the same table as Experiment B",
              [info.get("n_analysis_rows"), info.get("n_with_features"), n_without_rec],
              [a.get("n_analysis_rows"), a.get("n_with_features"), len(a.get("rows_without_features") or [])])
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise ReportRefusal([f"count reconciliation failed: {c['check']} (expected {c['expected']}, "
                             f"observed {c['observed']})" for c in failed])
    pool = None
    if calibration_audit is not None:
        ca = calibration_audit
        pool = {fk: {"eligible": int(ca[ca.fov_key.astype(str).eq(fk)].eligible.astype(str).str.lower()
                                     .eq("true").sum()),
                     "selected": int(ca[ca.fov_key.astype(str).eq(fk)].selected.astype(str).str.lower()
                                     .eq("true").sum())}
                for fk in sorted(set(ca.fov_key.astype(str)) - {""})}
    return {"stages": pd.DataFrame(stages), "technical_exclusions": pd.DataFrame(
                reason_rows, columns=["reason", "group", "fov", "n"]),
            "checks": checks, "calibration_pool": pool,
            "complete_case_missing_by_column": cc_audit["missing_by_column"]}


def stage_wide(stages: pd.DataFrame) -> pd.DataFrame:
    """One row per stage; columns group × FOV plus the total."""
    s = stages.copy()
    s["cell"] = np.where(s.group.eq("all"), "total", s.group + " " + s.fov)
    w = s.pivot_table(index=["order", "stage"], columns="cell", values="n", aggfunc="sum",
                      fill_value=0).reset_index()
    cells = sorted(c for c in w.columns if c not in ("order", "stage", "total"))
    return w[["stage"] + cells + ["total"]]


# ------------------------------------------------------------------ Table 1
TABLE1_VARIABLES = [  # (column, label, scale, decimals)
    ("age", "Age, years", 1.0, 1),
    ("vessel_density", "Vessel density (fraction of analysis area)", 1.0, 3),
    ("faz_area_mm2", "FAZ area, mm²", 1.0, 3),
    ("loop_density_per_mm2", "Loop density, per mm²", 1.0, 2),
    ("fractal_dimension", "Fractal dimension (box counting)", 1.0, 3),
    ("L0_um", "Total graph length L0, mm", 1e-3, 1),
    ("S0_{order}", "Baseline connectivity S0 (largest-component length share)", 1.0, 3),
    ("n_components_post", "Graph components after cleanup", 1.0, 1),
]


def _summary(x: pd.Series, scale: float, digits: int = 3) -> tuple[str, int]:
    v = pd.to_numeric(x, errors="coerce").to_numpy(float) * scale
    ok = np.isfinite(v)
    if not ok.any():
        return "unavailable", int((~ok).sum())
    q1, med, q3 = np.quantile(v[ok], [0.25, 0.5, 0.75])
    return f"{fmt(med, digits)} [{fmt(q1, digits)}, {fmt(q3, digits)}]", int((~ok).sum())


def table1(table: pd.DataFrame, flow_stages: pd.DataFrame, primary_order: str) -> pd.DataFrame:
    """Median [IQR] and missing counts by group × FOV; n per column matches cohort-flow stage 8."""
    t = table.assign(_g=_group(table.analysis_group), _f=_fov(table.fov_mm))
    cols = [(g, f) for g in ("NORMAL", "DR") for f in sorted(t._f.unique())]
    cols += [(g, "all FOV") for g in ("NORMAL", "DR")]
    stage8 = flow_stages[flow_stages.order.eq(8)]
    rows = []

    def sub(g, f):
        m = t._g.eq(g) if f == "all FOV" else (t._g.eq(g) & t._f.eq(f))
        return t[m]

    nrow = {"variable": "Images, n"}
    for g, f in cols:
        n = len(sub(g, f))
        if f != "all FOV":
            want = stage8[(stage8.group == g) & (stage8.fov == f)].n.sum()
            if int(want) != n:
                raise ReportRefusal([f"Table 1 cell {g} {f}: n = {n} differs from cohort flow ({want})"])
        nrow[f"{g} {f}"] = str(n)
    rows.append(nrow)
    sex = {"variable": "Male sex, n (%) [missing]"}
    for g, f in cols:
        s = pd.to_numeric(sub(g, f).sex_male, errors="coerce")
        k, n = int((s == 1).sum()), int(s.notna().sum())
        sex[f"{g} {f}"] = (f"{k} ({100 * k / n:.1f}%) [{int(s.isna().sum())}]" if n
                           else f"unavailable [{int(s.isna().sum())}]")
    rows.append(sex)
    for col, name, scale, digits in TABLE1_VARIABLES:
        col = col.format(order=primary_order)
        r = {"variable": f"{name}: median [IQR] {{missing}}"}
        for g, f in cols:
            if col not in t.columns:
                r[f"{g} {f}"] = "column absent"
                continue
            text, miss = _summary(sub(g, f)[col], scale, digits)
            r[f"{g} {f}"] = f"{text} {{{miss}}}"
        rows.append(r)
    return pd.DataFrame(rows)
