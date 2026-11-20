"""Step 0 null-model controls: healthy-target control and patchy thinning families.

Post-outcome exploratory analysis. Authority and decision rules:
``docs/STEP0_NULL_CONTROLS_AMENDMENT_2026-11-16.md``. Nothing in the package,
the configuration or any saved result is modified; this script only reads the
2026-10-09 exploratory outputs and writes under a new output root.

Subcommands::

    record   write the dated machine record (timestamp + input hashes); once only
    saved    analyses that need no new simulation (0a, approximate 0b, 5-vs-50 check)
    plan     freeze targets, donors, units and shard assignment
    shard    run one shard (resumable; one fsync'd JSON line per finished unit)
    status   progress of a run
    reduce   refuse unless complete, then compute every summary and the decision

Work unit = (family, healthy donor, repetition). The ``uniform`` family uses the
Experiment C permutation and seed keys, so its rows for DR targets must equal
the saved Experiment C rows; ``reduce`` checks this.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retina_fragility.config import config_hash, load_config  # noqa: E402
from retina_fragility.graph.model import VesselGraph  # noqa: E402
from retina_fragility.graph.stress import EmptyGraphError, run_removal, summarize_curve  # noqa: E402
from retina_fragility.graph.thinning import nearest_prefix, thinned_graph  # noqa: E402
from retina_fragility.provenance import (  # noqa: E402
    code_fingerprint,
    environment,
    now_utc,
    numerical_library_versions,
    read_json,
    sha256_file,
    write_json,
)
from retina_fragility.seeds import rng_for  # noqa: E402
from retina_fragility.stats.effects import holm  # noqa: E402

AMENDMENT = "docs/STEP0_NULL_CONTROLS_AMENDMENT_2026-11-16.md"
RECORD = "reports/step0_null_controls_amendment_2026-11-16.json"
SRC_RUN = "outputs/exploratory_20261009"
OUT_ROOT = "outputs/exploratory_20261116_step0"
STATUS_LABEL = "post_outcome_exploratory_reference_label_analysis"

FAMILIES = ("uniform", "patchy64", "patchy16", "patchy4", "patchy1")
N_SEEDS = {"patchy64": 64, "patchy16": 16, "patchy4": 4, "patchy1": 1}
MAX_SEEDS = 64
RANK = {f: i for i, f in enumerate(FAMILIES)}
REPETITIONS = 5
CALIPER_REL = 0.05
MIN_DONORS = 10
N_BOOT = 2000
DENSITY_COL = "graph_length_density_mm_per_mm2"
FI_COL = "FI_adaptive_betweenness"
ORDER = "adaptive_betweenness"
SEC_PER_ROW = {3.0: 3.5, 6.0: 5.3}  # measured in the 2026-10-09 Experiment C shards
CODE_MODULES = ("retina_fragility.graph.thinning", "retina_fragility.graph.stress",
                "retina_fragility.graph.model", "retina_fragility.seeds")
PLAN_FORMAT = "step0_null_controls.plan/1"


class Step0Error(RuntimeError):
    """Refusal: missing record, changed input, or inconsistent run state."""


# ------------------------------------------------------------------ geometry / orders
def edge_centroids(g: VesselGraph, c: dict) -> np.ndarray:
    """(m, 2) centroid (y, x) in pixels of each edge: both end nodes plus the interior path."""
    out = np.empty((len(c["edge_ids"]), 2), float)
    for i, eid in enumerate(c["edge_ids"]):
        e = g.edges[int(eid)]
        nu, nv = g.nodes[e.u], g.nodes[e.v]
        pts = [(nu.y, nu.x), *[(float(p[0]), float(p[1])) for p in e.path], (nv.y, nv.x)]
        out[i] = np.mean(np.asarray(pts, float), axis=0)
    return out


def image_extent(g: VesselGraph, cent: np.ndarray) -> np.ndarray:
    if g.shape is not None:
        return np.asarray(g.shape, float)
    return cent.max(axis=0) + 1.0


def removal_order(family: str, m: int, cent: np.ndarray, extent: np.ndarray, base_seed: int,
                  donor_id: str, rep: int) -> np.ndarray:
    """Removal order of the m compact edge indices for (family, donor, repetition).

    ``uniform`` is exactly the Experiment C permutation. A patchy family removes
    edges in ascending distance from their centroid to the nearest of the first
    S nested seed points (ties by edge index), i.e. S discs growing together.
    """
    if family == "uniform":
        return rng_for(base_seed, "exp_c_perm", donor_id, rep).permutation(m)
    if family not in N_SEEDS:
        raise ValueError(f"unknown family {family}")
    seeds = rng_for(base_seed, "step0_patchy_seeds", donor_id, rep).uniform(size=(MAX_SEEDS, 2)) * extent
    s = seeds[:N_SEEDS[family]]
    d = np.sqrt(((cent[:, None, :] - s[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return np.lexsort((np.arange(m), d))


def unit_rows(g: VesselGraph, c: dict, cent: np.ndarray, donor_id: str, rep: int, family: str,
              area_mm2: float, targets, cfg: dict) -> tuple[list[dict], list[dict]]:
    """Matching rows (one per target) and null-FI rows (one per distinct in-tolerance prefix)."""
    base_seed = cfg["project"]["base_seed"]
    tol = float(cfg["exp_c"]["tolerance_rel"])
    minfrag = float(cfg["graph"]["min_fragment_length_um"])
    scfg = cfg["stress"]
    m = len(c["length"])
    perm = removal_order(family, m, cent, image_extent(g, cent), base_seed, donor_id, rep)
    audit, need = [], set()
    for tid, target in targets:
        k, achieved = nearest_prefix(perm, c["n"], c["eu"], c["ev"], c["length"], target, minfrag, area_mm2)
        resid = (achieved - target) / target
        ok = bool(abs(resid) <= tol)
        audit.append({"target_id": tid, "k": int(k), "achieved_density": float(achieved),
                      "residual_rel": float(resid), "within_tolerance": ok})
        if ok:
            need.add(int(k))
    null = []
    for k in sorted(need):
        t = thinned_graph(g, c["edge_ids"][perm[:k]].tolist(), minfrag)
        try:
            cv = run_removal(t, ORDER, scfg["step_frac"],
                             rng=rng_for(base_seed, "exp_c_random", donor_id, rep, k),
                             backend=scfg["betweenness_backend"], rel_quantum=scfg["tie_rel_quantum"],
                             keep_batches=False)
            s = summarize_curve(cv, scfg["q_eval"], scfg["s_threshold"])
            null.append({"k": k, "FI_null": float(s["FI"]), "m_thinned": int(t.n_edges), "status": "ok"})
        except EmptyGraphError:
            null.append({"k": k, "FI_null": None, "m_thinned": 0, "status": "empty_after_thinning"})
    return audit, null


# ------------------------------------------------------------------------- statistics
def median_ci(x, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "median": None, "ci": None}
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    meds = np.median(x[idx], axis=1)
    return {"n": int(x.size), "median": float(np.median(x)),
            "ci": [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))],
            "fraction_negative": float(np.mean(x < 0))}


def matched_contrast(df: pd.DataFrame, col: str, caliper: float = CALIPER_REL) -> pd.DataFrame:
    """Per DR eye: value minus the mean value of healthy eyes of the same FOV within the density caliper.

    ``df`` needs image_id, y, fov_mm, density and ``col``. Rows whose ``col`` is
    missing are not used on either side.
    """
    ok = df[np.isfinite(df[col].astype(float))]
    healthy = ok[ok.y == 0]
    rows = []
    for r in ok[ok.y == 1].itertuples():
        pool = healthy[(healthy.fov_mm == r.fov_mm) & ((healthy.density / r.density - 1.0).abs() <= caliper)]
        v = float(getattr(r, col))
        mm = float(pool[col].mean()) if len(pool) else np.nan
        rows.append({"image_id": r.image_id, "fov_mm": r.fov_mm, "density": r.density, "value": v,
                     "n_matched": int(len(pool)), "matched_mean": mm, "d": v - mm})
    return pd.DataFrame(rows, columns=["image_id", "fov_mm", "density", "value", "n_matched", "matched_mean", "d"])


def contrast_summary(m: pd.DataFrame, n_dr_total: int, rng: np.random.Generator) -> dict:
    d = m.d.to_numpy(float) if len(m) else np.array([])
    d = d[np.isfinite(d)]
    out = {"n_dr_total": int(n_dr_total), "n_dr_with_value": int(len(m)), "n_dr_matched": int(d.size),
           "overlap_fraction": float(d.size / n_dr_total) if n_dr_total else None}
    out.update({k: v for k, v in median_ci(d, rng).items() if k != "n"})
    if d.size >= 1 and np.any(d != 0):
        w = sps.wilcoxon(d, alternative="two-sided")
        out.update(wilcoxon_statistic=float(w.statistic), wilcoxon_p_two_sided=float(w.pvalue))
    else:
        out.update(wilcoxon_statistic=None, wilcoxon_p_two_sided=None)
    return out


def contrast_by_scope(df: pd.DataFrame, col: str, base_seed: int, label: str) -> dict:
    out = {}
    for scope, sub in (("pooled", df), ("fov_3mm", df[df.fov_mm == 3.0]), ("fov_6mm", df[df.fov_mm == 6.0])):
        m = matched_contrast(sub, col)
        n_dr = int((sub.y == 1).sum())
        out[scope] = contrast_summary(m, n_dr, rng_for(base_seed, "step0_boot", label, scope))
    return out


def ols_group_effect(df: pd.DataFrame, col: str) -> dict:
    """Sensitivity: OLS of ``col`` on DR, density and FOV with HC3 standard errors."""
    import statsmodels.api as sm

    d = df[np.isfinite(df[col].astype(float))]
    X = pd.DataFrame({"const": 1.0, "dr": d.y.astype(float), "density": d.density.astype(float),
                      "fov_6mm": (d.fov_mm == 6.0).astype(float)})
    if d.fov_mm.nunique() < 2:
        X = X.drop(columns="fov_6mm")
    if len(d) <= X.shape[1] + 1 or d.y.nunique() < 2:
        return {"status": "not_estimable", "n": int(len(d))}
    fit = sm.OLS(d[col].astype(float).to_numpy(), X.to_numpy()).fit(cov_type="HC3")
    j = list(X.columns).index("dr")
    ci = fit.conf_int()[j]
    return {"status": "ok", "n": int(len(d)), "coef_dr": float(fit.params[j]), "se_hc3": float(fit.bse[j]),
            "ci": [float(ci[0]), float(ci[1])], "p": float(fit.pvalues[j]), "columns": list(X.columns)}


def decide(z_contrast: dict, rank_contrast: dict, healthy_uniform: dict) -> dict:
    """Decision rules of the amendment, applied to the pooled contrasts."""
    pz, pr = z_contrast.get("wilcoxon_p_two_sided"), rank_contrast.get("wilcoxon_p_two_sided")
    adj = holm({"z_uniform": pz, "patchiness_rank": pr})
    mz, mr = z_contrast.get("median"), rank_contrast.get("median")
    z_sup = adj["z_uniform"] is not None and adj["z_uniform"] < 0.05 and mz is not None and mz < 0
    r_sup = adj["patchiness_rank"] is not None and adj["patchiness_rank"] < 0.05 and mr is not None and mr > 0
    z_opp = adj["z_uniform"] is not None and adj["z_uniform"] < 0.05 and mz is not None and mz > 0
    r_opp = adj["patchiness_rank"] is not None and adj["patchiness_rank"] < 0.05 and mr is not None and mr < 0
    ov = [c.get("overlap_fraction") for c in (z_contrast, rank_contrast)]
    ci = healthy_uniform.get("ci")
    healthy_below = ci is not None and ci[1] < 0
    if any(o is None or o < 0.5 for o in ov):
        verdict = "insufficient_overlap"
    elif z_sup or r_sup:
        verdict = "disease_reading_supported"
    elif healthy_below and not (z_opp or r_opp):
        verdict = "null_model_reading_supported"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "holm_adjusted_p": adj, "z_contrast_supports": bool(z_sup),
            "rank_contrast_supports": bool(r_sup), "opposite_direction_significant": bool(z_opp or r_opp),
            "healthy_uniform_median_z_below_zero": bool(healthy_below), "overlap_fractions": ov}


# ---------------------------------------------------------------------- inputs / guard
def input_files(cfg: dict) -> dict[str, str]:
    files = [AMENDMENT, "config.yaml", cfg["data"]["manifest"], cfg["calibration"]["frozen_file"],
             f"{SRC_RUN}/pipeline/features.csv", f"{SRC_RUN}/pipeline/run.json",
             f"{SRC_RUN}/analysis/exp_c/results.json", f"{SRC_RUN}/analysis/exp_c/exp_c_per_dr.csv",
             f"{SRC_RUN}/analysis/exp_c/exp_c_matching_audit.csv", f"{SRC_RUN}/analysis/exp_b/results.json"]
    return {f: sha256_file(ROOT / f) for f in files}


def check_record(cfg: dict) -> dict:
    p = ROOT / RECORD
    if not p.exists():
        raise Step0Error(f"no dated record at {RECORD}; run 'record' before any analysis")
    rec = read_json(p)
    now = input_files(cfg)
    changed = [f for f, h in rec["files_sha256"].items() if now.get(f) != h]
    if changed:
        raise Step0Error("inputs changed since the dated record: " + ", ".join(changed))
    return rec


def load_table(cfg: dict) -> tuple[pd.DataFrame, dict]:
    from retina_fragility.calibration import load_calibration
    from retina_fragility.experiments.common import load_analysis_table
    from retina_fragility.manifest import load_manifest

    m = load_manifest(ROOT / cfg["data"]["manifest"], cfg)
    cal = load_calibration(ROOT / cfg["calibration"]["frozen_file"])
    table, info = load_analysis_table(ROOT / SRC_RUN / "pipeline", m, cal, cfg["data"]["primary_dataset"])
    table = table.copy()
    table["density"] = table[DENSITY_COL].astype(float)
    table["FI"] = table[FI_COL].astype(float)
    table["fov_mm"] = table.fov_mm.astype(float)
    return table, info


def graph_path(pipeline_root: Path, image_id: str) -> Path:
    from retina_fragility.pipeline import safe_name

    return Path(pipeline_root) / "images" / safe_name(image_id) / "graph.json"


def script_identity(cfg: dict) -> dict:
    return {"script_sha256": sha256_file(Path(__file__)), "config_sha256": config_hash(cfg),
            "code_sha256": code_fingerprint(CODE_MODULES)["sha256"],
            "numerical_libraries": numerical_library_versions()}


# ------------------------------------------------------------------------------- plan
def plan_targets(table: pd.DataFrame, available) -> tuple[list[dict], dict[str, list]]:
    """Target rows and, per healthy donor, its targets [(image_id, density)] in table order."""
    available = set(available)
    t = table[np.isfinite(table.density) & np.isfinite(table.FI)]
    donors = t[(t.y == 0) & t.image_id.isin(available)]
    targets, donor_targets = [], {}
    for r in t.itertuples():
        elig = donors[(donors.fov_mm == r.fov_mm) & (donors.density >= r.density) & (donors.image_id != r.image_id)]
        targets.append({"image_id": r.image_id, "y": int(r.y), "fov_mm": float(r.fov_mm),
                        "density": float(r.density), "FI": float(r.FI), "n_eligible_donors": int(len(elig))})
        for d in elig.image_id:
            donor_targets.setdefault(d, []).append((r.image_id, float(r.density)))
    return targets, donor_targets


def make_plan(table: pd.DataFrame, cfg: dict, pipeline_root: Path, n_shards: int, identity: dict,
              extra: dict | None = None) -> dict:
    avail = [i for i in table.image_id if graph_path(pipeline_root, i).exists()]
    targets, donor_targets = plan_targets(table, avail)
    fov = dict(zip(table.image_id, table.fov_mm.astype(float)))
    area = dict(zip(table.image_id, table.analysis_area_mm2.astype(float)))
    units = [(f, d, r) for d in sorted(donor_targets) for r in range(REPETITIONS) for f in FAMILIES]
    cost = {u: len(donor_targets[u[1]]) * SEC_PER_ROW.get(fov[u[1]], 5.0) for u in units}
    load = [0.0] * n_shards
    shards: list[list] = [[] for _ in range(n_shards)]
    for u in sorted(units, key=lambda u: (-cost[u], u[1], u[2], RANK[u[0]])):  # deterministic LPT
        i = int(np.argmin(load))
        shards[i].append(list(u))
        load[i] += cost[u]
    for s in shards:
        s.sort(key=lambda u: (u[1], u[2], RANK[u[0]]))
    body = {"format": PLAN_FORMAT, "analysis_status": STATUS_LABEL, "identity": identity,
            "settings": {"families": list(FAMILIES), "repetitions": REPETITIONS, "order": ORDER,
                         "tolerance_rel": float(cfg["exp_c"]["tolerance_rel"]),
                         "min_fragment_length_um": float(cfg["graph"]["min_fragment_length_um"]),
                         "density_metric": DENSITY_COL, "fi_column": FI_COL},
            "pipeline_root": str(pipeline_root), "targets": targets,
            "donors": {d: {"area_mm2": area[d], "fov_mm": fov[d], "targets": donor_targets[d],
                           "graph_sha256": sha256_file(graph_path(pipeline_root, d))}
                       for d in sorted(donor_targets)},
            "shards": shards, "n_units": len(units),
            "estimated_core_hours": float(sum(cost.values()) / 3600.0), **(extra or {})}
    body["plan_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    return body


# ------------------------------------------------------------------------------ shard
def _shard_file(run_dir: Path, i: int) -> Path:
    return Path(run_dir) / "shards" / f"shard_{i:05d}.jsonl"


def read_shard(path: Path, repair: bool = False) -> list[dict]:
    """Complete JSON lines of a shard file. A torn final line (preemption) is dropped;
    with ``repair`` the file is truncated to the last complete line."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    good_end, rows = 0, []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
        good_end += len(line)
    if repair and good_end != len(raw):
        with open(path, "r+b") as fh:
            fh.truncate(good_end)
            fh.flush()
            os.fsync(fh.fileno())
    return rows


def run_shard(run_dir: Path, shard: int, cfg: dict, identity: dict, max_units: int | None = None) -> dict:
    run_dir = Path(run_dir)
    plan = read_json(run_dir / "plan.json")
    if plan["identity"] != identity:
        raise Step0Error("script, config, package code or libraries differ from the frozen plan")
    units = [tuple(u) for u in plan["shards"][shard]]
    path = _shard_file(run_dir, shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    done_rows = read_shard(path, repair=True)
    for r in done_rows:
        if r["plan_sha256"] != plan["plan_sha256"]:
            raise Step0Error(f"{path} holds rows of another plan")
    done = {tuple(r["unit"]) for r in done_rows}
    todo = [u for u in units if u not in done]
    proot = Path(plan["pipeline_root"])
    cache: dict = {}
    n_run = 0
    with open(path, "ab") as fh:
        for fam, donor, rep in todo:
            if max_units is not None and n_run >= max_units:
                break
            if donor not in cache:
                cache.clear()
                gp = graph_path(proot, donor)
                if sha256_file(gp) != plan["donors"][donor]["graph_sha256"]:
                    raise Step0Error(f"donor graph changed since the plan: {donor}")
                g = VesselGraph.load(gp)
                c = g.compact()
                cache[donor] = (g, c, edge_centroids(g, c))
            g, c, cent = cache[donor]
            info = plan["donors"][donor]
            t0 = time.monotonic()
            audit, null = unit_rows(g, c, cent, donor, int(rep), fam, float(info["area_mm2"]),
                                    [tuple(t) for t in info["targets"]], cfg)
            row = {"plan_sha256": plan["plan_sha256"], "unit": [fam, donor, int(rep)], "audit": audit,
                   "null": null, "elapsed_s": round(time.monotonic() - t0, 3)}
            fh.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())
            fh.flush()
            os.fsync(fh.fileno())
            n_run += 1
    n_done = len(done) + n_run
    return {"shard": shard, "n_units": len(units), "n_done": n_done,
            "status": "complete" if n_done == len(units) else "partial"}


def run_status(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    plan = read_json(run_dir / "plan.json")
    done, secs, incomplete = 0, 0.0, []
    for i, units in enumerate(plan["shards"]):
        rows = read_shard(_shard_file(run_dir, i))
        done += len(rows)
        secs += sum(r["elapsed_s"] for r in rows)
        if len(rows) != len(units):
            incomplete.append(i)
    return {"plan_sha256": plan["plan_sha256"], "n_units": plan["n_units"], "n_units_done": done,
            "core_hours_done": secs / 3600.0, "n_shards": len(plan["shards"]),
            "incomplete_shards": incomplete}


# ----------------------------------------------------------------------------- reduce
def collect(run_dir: Path) -> tuple[dict, pd.DataFrame, float]:
    """Strictly validated audit frame (one row per family, donor, repetition, target)."""
    run_dir = Path(run_dir)
    plan = read_json(run_dir / "plan.json")
    planned = {tuple(u) for s in plan["shards"] for u in s}
    seen, frames, secs = set(), [], 0.0
    for i in range(len(plan["shards"])):
        for r in read_shard(_shard_file(run_dir, i)):
            if r["plan_sha256"] != plan["plan_sha256"]:
                raise Step0Error(f"shard {i} holds rows of another plan")
            u = tuple(r["unit"])
            if u in seen:
                raise Step0Error(f"duplicate unit {u}")
            if u not in planned or list(u) not in plan["shards"][i]:
                raise Step0Error(f"unit {u} does not belong to shard {i}")
            seen.add(u)
            secs += r["elapsed_s"]
            want = [t[0] for t in plan["donors"][u[1]]["targets"]]
            if [a["target_id"] for a in r["audit"]] != want:
                raise Step0Error(f"unit {u}: matching rows do not cover its targets")
            ks = {n["k"] for n in r["null"]}
            if ks != {a["k"] for a in r["audit"] if a["within_tolerance"]}:
                raise Step0Error(f"unit {u}: null rows do not cover the in-tolerance prefixes")
            a = pd.DataFrame(r["audit"])
            n = pd.DataFrame(r["null"], columns=["k", "FI_null", "m_thinned", "status"])
            a = a.merge(n, on="k", how="left")
            a.insert(0, "rep", u[2])
            a.insert(0, "donor_id", u[1])
            a.insert(0, "family", u[0])
            frames.append(a)
    missing = planned - seen
    if missing:
        raise Step0Error(f"{len(missing)} of {len(planned)} units missing; run is incomplete")
    audit = pd.concat(frames, ignore_index=True)
    audit["FI_null"] = audit.FI_null.astype(float)
    return plan, audit, secs


def per_target(plan: dict, audit: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (family, target) with the null mean, SD, z and delta."""
    targets = pd.DataFrame(plan["targets"])
    valid = audit[audit.within_tolerance & audit.status.eq("ok")]
    v = valid.groupby(["family", "target_id"]).FI_null.agg(null_mean="mean", null_sd=lambda x: x.std(ddof=1),
                                                           n_valid="size").reset_index()
    a = audit.groupby(["family", "target_id"]).agg(
        n_reps_total=("rep", "size"), n_out_of_tolerance=("within_tolerance", lambda x: int((~x).sum())),
        median_abs_residual=("residual_rel", lambda x: float(x.abs().median()))).reset_index()
    grid = pd.MultiIndex.from_product([FAMILIES, targets.image_id], names=["family", "target_id"]).to_frame(index=False)
    out = (grid.merge(targets, left_on="target_id", right_on="image_id").drop(columns="image_id")
           .merge(a, on=["family", "target_id"], how="left").merge(v, on=["family", "target_id"], how="left"))
    out["n_valid"] = out.n_valid.fillna(0).astype(int)
    out["delta"] = out.FI - out.null_mean
    defined = (out.n_valid >= 2) & (out.null_sd > 0)
    out["z"] = np.where(defined, out.delta / out.null_sd.where(out.null_sd > 0), np.nan)
    out["status"] = np.select([out.n_eligible_donors == 0, out.n_valid < 2, ~(out.null_sd > 0)],
                              ["no_eligible_donor", "no_valid_repetitions", "null_sd_zero_z_undefined"], "ok")
    return out


def wide_table(per: pd.DataFrame) -> pd.DataFrame:
    base = per[per.family == "uniform"][["target_id", "y", "fov_mm", "density", "FI", "n_eligible_donors"]]
    w = base.rename(columns={"target_id": "image_id"}).reset_index(drop=True)
    for f in FAMILIES:
        s = per[per.family == f].set_index("target_id")
        w[f"z_{f}"] = w.image_id.map(s.z)
        w[f"delta_{f}"] = w.image_id.map(s.delta)
    d = np.abs(w[[f"delta_{f}" for f in FAMILIES]].to_numpy(float))
    ok = np.isfinite(d).all(axis=1)
    rank = np.full(len(w), np.nan)
    rank[ok] = np.argmin(d[ok], axis=1)  # first minimum: ties go to the less patchy family
    w["patchiness_rank"] = rank
    return w


def exp_c_reproduction(audit: pd.DataFrame) -> dict:
    """Uniform rows for DR targets must equal the saved Experiment C rows (repetitions 0-4)."""
    saved = pd.read_csv(ROOT / SRC_RUN / "analysis/exp_c/exp_c_matching_audit.csv",
                        usecols=["dr_image_id", "donor_id", "rep", "k", "FI_null", "within_tolerance"],
                        dtype={"dr_image_id": str, "donor_id": str})
    saved = saved[saved.rep < REPETITIONS]
    mine = audit[audit.family == "uniform"][["target_id", "donor_id", "rep", "k", "FI_null", "within_tolerance"]]
    j = saved.merge(mine, left_on=["dr_image_id", "donor_id", "rep"], right_on=["target_id", "donor_id", "rep"],
                    suffixes=("_saved", "_step0"))
    both = j[j.FI_null_saved.notna() & j.FI_null_step0.notna()]
    return {"n_saved_rows": int(len(saved)), "n_compared": int(len(j)),
            "n_k_mismatch": int((j.k_saved != j.k_step0).sum()),
            "n_tolerance_flag_mismatch": int((j.within_tolerance_saved != j.within_tolerance_step0).sum()),
            "n_fi_compared": int(len(both)),
            "max_abs_fi_difference": float((both.FI_null_saved - both.FI_null_step0).abs().max()) if len(both) else None,
            "exact": bool(len(j) == len(saved) and (j.k_saved == j.k_step0).all() and len(both) > 0
                          and (both.FI_null_saved - both.FI_null_step0).abs().max() <= 1e-12)}


def summarise(plan: dict, audit: pd.DataFrame, cfg: dict, reproduction: dict | None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    base_seed = cfg["project"]["base_seed"]
    per = per_target(plan, audit)
    wide = wide_table(per)
    main = wide[wide.n_eligible_donors >= MIN_DONORS]
    res: dict = {"analysis_status": STATUS_LABEL, "human_qc_satisfied": False,
                 "plan_sha256": plan["plan_sha256"],
                 "n_targets": {"dr": int((wide.y == 1).sum()), "healthy": int((wide.y == 0).sum()),
                               "dr_min_donors": int((main.y == 1).sum()),
                               "healthy_min_donors": int((main.y == 0).sum())},
                 "status_counts": per.groupby("family").status.value_counts().unstack(fill_value=0).to_dict("index"),
                 "families": {}, "exp_c_reproduction": reproduction}
    for label, frame in (("min_donors", main), ("all_defined", wide)):
        fam = {}
        for f in FAMILIES:
            fam[f] = {g: {"z": median_ci(frame.loc[frame.y == yv, f"z_{f}"], rng_for(base_seed, "step0_boot", label, f, g, "z")),
                          "delta": median_ci(frame.loc[frame.y == yv, f"delta_{f}"], rng_for(base_seed, "step0_boot", label, f, g, "d"))}
                      for g, yv in (("dr", 1), ("healthy", 0))}
            fam[f]["matched_contrast_z"] = contrast_by_scope(frame, f"z_{f}", base_seed, f"{label}|z|{f}")
            fam[f]["ols_z"] = ols_group_effect(frame, f"z_{f}")
        ranks = {g: frame.loc[frame.y == yv, "patchiness_rank"].dropna().astype(int).value_counts().sort_index()
                 for g, yv in (("dr", 1), ("healthy", 0))}
        rank_block = {"distribution": {g: {FAMILIES[i]: int(n) for i, n in r.items()} for g, r in ranks.items()},
                      "matched_contrast": contrast_by_scope(frame, "patchiness_rank", base_seed, f"{label}|rank"),
                      "ols": ols_group_effect(frame, "patchiness_rank")}
        res["families" if label == "min_donors" else "sensitivity_all_defined"] = (
            {**fam, "patchiness_rank": rank_block})
    block = res["families"]
    res["decision"] = decide(block["uniform"]["matched_contrast_z"]["pooled"],
                             block["patchiness_rank"]["matched_contrast"]["pooled"],
                             block["uniform"]["healthy"]["z"])
    return res, per, wide


# ----------------------------------------------------------------- saved-data analyses
def saved_analyses(table: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    base_seed = cfg["project"]["base_seed"]
    t = table[["image_id", "y", "fov_mm", "density", "FI"]].copy()
    res: dict = {"analysis_status": STATUS_LABEL, "human_qc_satisfied": False,
                 "0a_matched_fi_contrast": contrast_by_scope(t, "FI", base_seed, "saved|FI"),
                 "0a_ols_fi": ols_group_effect(t, "FI")}
    # 5 versus 50 repetitions, from the saved Experiment C rows only
    audit = pd.read_csv(ROOT / SRC_RUN / "analysis/exp_c/exp_c_matching_audit.csv",
                        dtype={"dr_image_id": str, "donor_id": str})
    per = pd.read_csv(ROOT / SRC_RUN / "analysis/exp_c/exp_c_per_dr.csv", dtype={"image_id": str})
    valid = audit[audit.within_tolerance & audit.status.eq("ok")]
    fi_dr = per.set_index("image_id").FI_DR

    def z_of(v):
        g = v.groupby("dr_image_id").FI_null.agg(["mean", "std", "size"])
        return ((fi_dr.reindex(g.index) - g["mean"]) / g["std"]).where((g["size"] >= 2) & (g["std"] > 0))

    z50, z5 = z_of(valid), z_of(valid[valid.rep < REPETITIONS])
    j = pd.concat([z50.rename("z50"), z5.rename("z5")], axis=1).dropna()
    res["repetition_check_5_vs_50"] = {
        "n": int(len(j)), "spearman": float(sps.spearmanr(j.z50, j.z5).statistic),
        "max_abs_difference": float((j.z50 - j.z5).abs().max()),
        "median_abs_difference": float((j.z50 - j.z5).abs().median()),
        "median_z50": float(j.z50.median()), "median_z5": float(j.z5.median()),
        "saved_z_reproduced_max_abs_error": float((z50 - per.set_index("image_id").z.reindex(z50.index)).abs().max())}
    # approximate 0b: interpolate the saved null mean and SD over target density within FOV
    ok = per[per.status.eq("ok")]
    t["z_uniform_approx"] = np.nan
    for fov, sub in ok.groupby("fov_mm"):
        s = sub.sort_values("target_density").drop_duplicates("target_density")
        sel = (t.fov_mm == float(fov)) & (t.y == 0) & t.density.between(s.target_density.min(), s.target_density.max())
        mu = np.interp(t.loc[sel, "density"], s.target_density, s.null_mean)
        sd = np.interp(t.loc[sel, "density"], s.target_density, s.null_sd)
        t.loc[sel, "z_uniform_approx"] = (t.loc[sel, "FI"] - mu) / sd
    t.loc[t.y == 1, "z_uniform_approx"] = t.loc[t.y == 1, "image_id"].map(ok.set_index("image_id").z)
    h = t[(t.y == 0)]
    res["0b_approximation"] = {
        "note": ("APPROXIMATION from saved Experiment C nulls (interpolated over target density within FOV; "
                 "the healthy eye itself is not removed from the donor pool). The simulation is the result of record."),
        "n_healthy_in_dr_density_range": int(h.z_uniform_approx.notna().sum()), "n_healthy": int(len(h)),
        "healthy_z": median_ci(h.z_uniform_approx, rng_for(base_seed, "step0_boot", "saved|approx|healthy")),
        "dr_z": median_ci(t.loc[t.y == 1, "z_uniform_approx"], rng_for(base_seed, "step0_boot", "saved|approx|dr")),
        "matched_contrast_z": contrast_by_scope(t, "z_uniform_approx", base_seed, "saved|approx|z")}
    return res, t


# -------------------------------------------------------------------------------- text
def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _mc(c: dict) -> str:
    ci = c.get("ci")
    return (f"median difference {_fmt(c.get('median'))} (95% interval "
            f"{_fmt(ci[0]) if ci else 'n/a'} to {_fmt(ci[1]) if ci else 'n/a'}), "
            f"{c['n_dr_matched']} of {c['n_dr_total']} DR eyes matched, "
            f"Wilcoxon p = {_fmt(c.get('wilcoxon_p_two_sided'), 4)}")


def saved_text(res: dict) -> str:
    a, r, b = res["0a_matched_fi_contrast"], res["repetition_check_5_vs_50"], res["0b_approximation"]
    L = ["STEP 0, saved-data analyses (post-outcome exploratory; human QC unmet)", "",
         "0a. FI of DR eyes minus FI of healthy eyes of the same FOV and density (within 5%):"]
    L += [f"  {s}: {_mc(a[s])}" for s in ("pooled", "fov_3mm", "fov_6mm")]
    o = res["0a_ols_fi"]
    if o["status"] == "ok":
        L.append(f"  OLS sensitivity (FI on DR, density, FOV; HC3): DR coefficient {_fmt(o['coef_dr'])} "
                 f"(95% interval {_fmt(o['ci'][0])} to {_fmt(o['ci'][1])}), p = {_fmt(o['p'], 4)}")
    L += ["", "Repetition check from saved Experiment C rows (5 versus 50 repetitions per donor):",
          f"  n = {r['n']}, Spearman {_fmt(r['spearman'])}, median z {_fmt(r['median_z5'])} versus "
          f"{_fmt(r['median_z50'])}, largest difference {_fmt(r['max_abs_difference'])}", "",
          "0b, APPROXIMATION ONLY (interpolated from saved nulls; simulation is the result of record):",
          f"  healthy eyes in the DR density range: {b['n_healthy_in_dr_density_range']} of {b['n_healthy']}",
          f"  median z, healthy: {_fmt(b['healthy_z']['median'])}; DR: {_fmt(b['dr_z']['median'])}",
          f"  matched contrast of z, pooled: {_mc(b['matched_contrast_z']['pooled'])}"]
    return "\n".join(L) + "\n"


def reduce_text(res: dict) -> str:
    F, d = res["families"], res["decision"]
    L = ["STEP 0, simulation results (post-outcome exploratory; human QC unmet)", "",
         f"Targets with at least {MIN_DONORS} donors: {res['n_targets']['dr_min_donors']} DR, "
         f"{res['n_targets']['healthy_min_donors']} healthy.", "",
         "Median z by thinning family (negative = real graph less fragile than the thinned healthy graphs):"]
    for f in FAMILIES:
        dz, hz = F[f]["dr"]["z"], F[f]["healthy"]["z"]
        L.append(f"  {f:9s} DR {_fmt(dz['median'])} [{_fmt(dz['ci'][0])}, {_fmt(dz['ci'][1])}]   "
                 f"healthy {_fmt(hz['median'])} [{_fmt(hz['ci'][0])}, {_fmt(hz['ci'][1])}]")
    L += ["", "DR minus matched healthy, uniform z: " + _mc(F["uniform"]["matched_contrast_z"]["pooled"]),
          "DR minus matched healthy, patchiness rank: " + _mc(F["patchiness_rank"]["matched_contrast"]["pooled"]),
          "Best-matching family counts: " + json.dumps(F["patchiness_rank"]["distribution"]), "",
          f"Holm-adjusted p: {json.dumps(d['holm_adjusted_p'])}", f"DECISION (amendment rules): {d['verdict']}"]
    rp = res.get("exp_c_reproduction")
    if rp:
        L += ["", f"Experiment C reproduction check (uniform, DR targets, repetitions 0-4): exact = {rp['exact']}, "
                  f"{rp['n_compared']} rows compared, largest FI difference {rp['max_abs_fi_difference']}"]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------- cli
def _provenance(cfg: dict) -> dict:
    return {"created_utc": now_utc(), "performed_by": "",
            "amendment": AMENDMENT, "record": RECORD, "identity": script_identity(cfg),
            "environment": environment(), "inputs_sha256": input_files(cfg)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record")
    sub.add_parser("saved")
    p = sub.add_parser("plan")
    p.add_argument("--n-shards", type=int, default=300)
    p = sub.add_parser("shard")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--max-units", type=int, default=None)
    sub.add_parser("status")
    sub.add_parser("reduce")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    cfg = load_config(ROOT / "config.yaml")
    out, run_dir = ROOT / OUT_ROOT, ROOT / OUT_ROOT / "run"

    if args.cmd == "record":
        if (ROOT / RECORD).exists():
            raise Step0Error(f"{RECORD} already exists; it is written once")
        write_json(ROOT / RECORD, {
            "adopted_utc": now_utc(), "decided_by": "",
            "instructed_by": "user, in session, 2026-11-16", "is_human_review": False,
            "human_approval_claimed": False, "analysis_status": STATUS_LABEL,
            "outcomes_available_at_adoption": True, "original_human_gates_satisfied": False,
            "step0_results_available_at_adoption": False, "files_sha256": input_files(cfg),
            "identity_at_adoption": script_identity(cfg)})
        print(json.dumps(read_json(ROOT / RECORD), indent=1))
        return 0

    check_record(cfg)
    if args.cmd == "saved":
        table, info = load_table(cfg)
        res, eyes = saved_analyses(table, cfg)
        res["provenance"], res["analysis_table_audit"] = _provenance(cfg), info
        (out / "saved").mkdir(parents=True, exist_ok=True)
        write_json(out / "saved" / "results.json", res)
        eyes.to_csv(out / "saved" / "per_eye.csv", index=False)
        (out / "saved" / "summary.txt").write_text(saved_text(res))
        print(saved_text(res))
    elif args.cmd == "plan":
        if (run_dir / "plan.json").exists():
            raise Step0Error("a plan already exists; it is frozen")
        table, info = load_table(cfg)
        plan = make_plan(table, cfg, Path(SRC_RUN) / "pipeline", args.n_shards, script_identity(cfg),
                         {"created_utc": now_utc(), "inputs_sha256": input_files(cfg), "analysis_table_audit": info})
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "plan.json", plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "n_units": plan["n_units"],
                          "n_shards": len(plan["shards"]), "n_donors": len(plan["donors"]),
                          "n_targets": len(plan["targets"]),
                          "estimated_core_hours": plan["estimated_core_hours"]}, indent=1))
    elif args.cmd == "shard":
        print(json.dumps(run_shard(run_dir, args.shard, cfg, script_identity(cfg), args.max_units)))
    elif args.cmd == "status":
        print(json.dumps(run_status(run_dir), indent=1))
    elif args.cmd == "reduce":
        plan, audit, secs = collect(run_dir)
        if plan["identity"] != script_identity(cfg):
            raise Step0Error("script, config, package code or libraries differ from the frozen plan")
        res, per, wide = summarise(plan, audit, cfg, exp_c_reproduction(audit))
        res["provenance"], res["core_hours"] = _provenance(cfg), secs / 3600.0
        a = out / "analysis"
        a.mkdir(parents=True, exist_ok=True)
        audit.to_csv(a / "matching_audit.csv.gz", index=False)
        per.to_csv(a / "per_target_long.csv", index=False)
        wide.to_csv(a / "per_target_wide.csv", index=False)
        write_json(a / "results.json", res)
        (a / "summary.txt").write_text(reduce_text(res))
        print(reduce_text(res))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Step0Error as exc:
        sys.exit(f"REFUSED: {exc}")
