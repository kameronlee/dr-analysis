"""Experiment C (key secondary, H4): DR vs density-matched randomly thinned healthy graphs.

For every DR image (target density = its graph length density) and every
eligible healthy donor (same FOV, not calibration, starting density >=
target), ``exp_c.repetitions`` random segment-removal orders are drawn. The
order for (donor, repetition) is shared by all targets (common random
numbers). The nearest-feasible prefix is removed, topology-only cleanup is
re-applied, and the achieved residual is logged. Repetitions outside the
prespecified relative tolerance are dropped and counted. FI of each thinned
graph (primary order) forms the null; z = (FI_DR - mean) / sd, undefined when
sd = 0 or no valid repetition exists. One-sided Wilcoxon signed-rank for
z > 0 and a group-bootstrap CI of the median z.

Caveats written to every output: donors are shared across DR images, so z
scores are dependent; inference is conditional on the donor pool.

The work decomposes into independent (donor, repetition) units: the
permutation depends only on (donor, repetition), and every DR target of the
donor is matched against it. ``run`` executes all units in one process pool;
``exp_c_shard`` executes the same units as resumable Slurm shards. Both
assemble rows in the same canonical order and call ``finalize``, so the two
paths produce identical outputs.
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..graph.model import VesselGraph
from ..graph.stress import EmptyGraphError, run_removal, summarize_curve
from ..graph.thinning import nearest_prefix, thinned_graph
from ..seeds import rng_for
from ..stats.effects import group_bootstrap_weights, percentile_ci
from .common import header, write_results

CAVEATS = ("Healthy donors are shared across DR images, so per-DR z-scores are dependent; "
           "inference is conditional on the donor pool. Thinning removes whole segments without "
           "rewiring; the density match is nearest-feasible, residuals are logged, and "
           "repetitions outside the prespecified tolerance are excluded and counted.")


def _graph_path(pipeline_root: Path, image_id: str) -> Path:
    from ..pipeline import safe_name

    return pipeline_root / "images" / safe_name(image_id) / "graph.json"


def _min_fragment(cfg: dict) -> float:
    return float(cfg["graph"]["min_fragment_length_um"])


NULL_COLUMNS = ["donor_id", "rep", "k", "FI_null", "m_thinned", "status"]
# Code whose source determines Experiment C results (serial and sharded paths).
EXP_C_ENTRY_MODULES = ("retina_fragility.experiments.exp_c", "retina_fragility.experiments.exp_c_shard")


def _fi_for_prefixes(g: VesselGraph, c: dict, donor_id: str, rep: int, ks, cfg: dict) -> list[dict]:
    """FI (primary order) of the donor graph thinned by each prefix k of permutation (donor, rep)."""
    out = []
    order = cfg["exp_c"]["fi_order"]
    scfg = cfg["stress"]
    # identical permutation to the one used for matching (common random numbers)
    perm = rng_for(cfg["project"]["base_seed"], "exp_c_perm", donor_id, rep).permutation(len(c["length"]))
    for k in ks:
        t = thinned_graph(g, c["edge_ids"][perm[:k]].tolist(), _min_fragment(cfg))
        try:
            cv = run_removal(t, order, scfg["step_frac"],
                             rng=rng_for(cfg["project"]["base_seed"], "exp_c_random", donor_id, rep, k),
                             backend=scfg["betweenness_backend"], rel_quantum=scfg["tie_rel_quantum"],
                             keep_batches=False)
            s = summarize_curve(cv, scfg["q_eval"], scfg["s_threshold"])
            out.append({"donor_id": donor_id, "rep": rep, "k": k, "FI_null": s["FI"],
                        "m_thinned": t.n_edges, "status": "ok"})
        except EmptyGraphError:
            out.append({"donor_id": donor_id, "rep": rep, "k": k, "FI_null": np.nan,
                        "m_thinned": 0, "status": "empty_after_thinning"})
    return out


def match_unit(c: dict, donor_id: str, rep: int, area_mm2: float, targets, cfg: dict) -> tuple[list[dict], list[int]]:
    """Nearest-feasible prefix of permutation (donor, rep) for each DR target of the donor.

    ``targets`` is a sequence of (dr_image_id, target_density). Returns the
    matching-audit rows and the sorted distinct prefixes k within tolerance.
    """
    base_seed = cfg["project"]["base_seed"]
    tol = float(cfg["exp_c"]["tolerance_rel"])
    perm = rng_for(base_seed, "exp_c_perm", donor_id, rep).permutation(len(c["length"]))
    rows, need = [], set()
    for dr_id, target in targets:
        k, achieved = nearest_prefix(perm, c["n"], c["eu"], c["ev"], c["length"], target,
                                     _min_fragment(cfg), area_mm2)
        resid = (achieved - target) / target
        ok = abs(resid) <= tol
        rows.append({"dr_image_id": dr_id, "donor_id": donor_id, "rep": rep, "k": k,
                     "m_donor": len(c["length"]), "target_density": target,
                     "achieved_density": achieved, "residual_rel": resid,
                     "within_tolerance": ok})
        if ok:
            need.add(int(k))
    return rows, sorted(need)


def compute_unit(g: VesselGraph, c: dict, donor_id: str, rep: int, area_mm2: float, targets,
                 cfg: dict) -> tuple[list[dict], list[dict]]:
    """One (donor, repetition) unit: matching-audit rows and null-FI rows."""
    audit, ks = match_unit(c, donor_id, rep, area_mm2, targets, cfg)
    return audit, _fi_for_prefixes(g, c, donor_id, rep, ks, cfg)


def _donor_worker(args) -> tuple[list[dict], list[dict]]:
    graph_path, donor_id, area_mm2, targets, cfg = args
    g = VesselGraph.load(graph_path)
    c = g.compact()
    audit, null = [], []
    for r in range(int(cfg["exp_c"]["repetitions"])):
        a, n = compute_unit(g, c, donor_id, r, area_mm2, targets, cfg)
        audit.extend(a)
        null.extend(n)
    return audit, null


def _donor_subset(donors: list[str], cap: int | None, base_seed: int, target_id: str) -> list[str]:
    if cap is None or len(donors) <= cap:
        return donors
    key = {d: hashlib.sha256(f"{base_seed}|exp_c_donor|{target_id}|{d}".encode()).hexdigest() for d in donors}
    return sorted(donors, key=key.get)[:cap]


def deviations_for(cfg: dict) -> list[str]:
    ccfg = cfg["exp_c"]
    if ccfg["max_donors"] is not None:
        return [f"max_donors cap = {ccfg['max_donors']} (deviation from 'all eligible donors')"]
    return []


def plan_targets(table: pd.DataFrame, cfg: dict, available_donors) -> tuple[list[dict], dict[str, list]]:
    """Per-DR rows (terminal status or ``pending`` with its ordered donor list) and,
    per donor, its DR targets [(dr_image_id, target_density)] in DR-table order.

    Donors are healthy rows of the same FOV with a processed graph
    (``available_donors``) and starting density >= target.
    """
    ccfg = cfg["exp_c"]
    base_seed = cfg["project"]["base_seed"]
    dens_col = ccfg["density_metric"]
    fi_col = f"FI_{ccfg['fi_order']}"
    normal = table[table.y == 0]
    dr = table[table.y == 1]
    available = list(available_donors)
    per_dr: list[dict] = []
    donor_targets: dict[str, list] = {}
    for t in dr.itertuples():
        target = getattr(t, dens_col)
        fi_t = getattr(t, fi_col)
        row = {"image_id": t.image_id, "fov_mm": t.fov_mm, "target_density": target, "FI_DR": fi_t,
               "group_key": t.group_key}
        if not (np.isfinite(target) and np.isfinite(fi_t)):
            row["status"] = "dr_feature_missing"
            per_dr.append(row)
            continue
        elig = normal[(normal.fov_mm == t.fov_mm) & (normal[dens_col] >= target)
                      & normal.image_id.isin(available)].image_id.tolist()
        elig = _donor_subset(sorted(elig), ccfg["max_donors"], base_seed, t.image_id)
        row["n_donors"] = len(elig)
        if not elig:
            row["status"] = "no_eligible_donor"
            per_dr.append(row)
            continue
        for d in elig:
            donor_targets.setdefault(d, []).append((t.image_id, float(target)))
        row["status"] = "pending"
        row["donors"] = list(elig)
        per_dr.append(row)
    return per_dr, donor_targets


def canonical_order(per_dr: list[dict], audit_rows: list[dict], null_rows: list[dict]):
    """Audit rows ordered (DR table order, donor order of that DR, rep); null rows by (donor, rep, k)."""
    dr_pos = {r["image_id"]: i for i, r in enumerate(per_dr)}
    donor_pos = {(r["image_id"], d): j for r in per_dr for j, d in enumerate(r.get("donors", []))}
    audit_rows = sorted(audit_rows, key=lambda a: (dr_pos[a["dr_image_id"]],
                                                   donor_pos[(a["dr_image_id"], a["donor_id"])], a["rep"]))
    null_rows = sorted(null_rows, key=lambda n: (n["donor_id"], n["rep"], n["k"]))
    return audit_rows, null_rows


def run(table: pd.DataFrame, info: dict, cfg: dict, pipeline_root: str | Path, out_dir: str | Path,
        calibration: dict | None, n_workers: int | None = None) -> dict:
    t0 = time.monotonic()
    pipeline_root, out_dir = Path(pipeline_root), Path(out_dir)
    normal = table[table.y == 0]
    available = [iid for iid in normal.image_id if _graph_path(pipeline_root, iid).exists()]
    per_dr, donor_targets = plan_targets(table, cfg, available)
    area = dict(zip(normal.image_id, normal.analysis_area_mm2.astype(float)))
    jobs = [(str(_graph_path(pipeline_root, d)), d, area[d], donor_targets[d], cfg)
            for d in sorted(donor_targets)]
    nw = int(n_workers or cfg["project"]["n_workers"])
    if nw > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=nw) as pool:
            chunks = list(pool.map(_donor_worker, jobs, chunksize=1))
    else:
        chunks = [_donor_worker(j) for j in jobs]
    audit, null = canonical_order(per_dr, [a for ch in chunks for a in ch[0]],
                                  [n for ch in chunks for n in ch[1]])
    return finalize(per_dr, audit, null, int((table.y == 1).sum()), info, cfg, calibration, out_dir,
                    t0, {"mode": "serial", "n_workers": nw})


def finalize(per_dr: list[dict], audit: list[dict], null_rows: list[dict], n_dr: int, info: dict, cfg: dict,
             calibration: dict | None, out_dir: str | Path, t0: float, execution: dict) -> dict:
    """Per-DR z-scores, Wilcoxon test, bootstrap CI and all output files (serial and sharded)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ccfg, st = cfg["exp_c"], cfg["stats"]
    base_seed = cfg["project"]["base_seed"]
    dens_col = ccfg["density_metric"]
    R, tol = int(ccfg["repetitions"]), float(ccfg["tolerance_rel"])
    deviations = deviations_for(cfg)
    per_dr = [{k: v for k, v in r.items() if k != "donors"} for r in per_dr]
    audit_df = pd.DataFrame(audit)
    null = pd.DataFrame(null_rows, columns=NULL_COLUMNS)
    null.to_csv(out_dir / "exp_c_null_fi.csv", index=False)
    if len(audit_df):
        audit_df = audit_df.merge(null[["donor_id", "rep", "k", "FI_null", "status"]],
                                  on=["donor_id", "rep", "k"], how="left")
    audit_df.to_csv(out_dir / "exp_c_matching_audit.csv", index=False)
    for row in per_dr:
        if row["status"] != "pending":
            continue
        a = audit_df[audit_df.dr_image_id == row["image_id"]]
        valid = a[a.within_tolerance & a.status.eq("ok")]
        row.update(n_reps_total=len(a), n_reps_out_of_tolerance=int((~a.within_tolerance).sum()),
                   n_reps_empty=int(a.status.eq("empty_after_thinning").sum()),
                   n_reps_valid=len(valid),
                   median_abs_residual=float(a.residual_rel.abs().median()) if len(a) else np.nan)
        if len(valid) < 2:
            row.update(status="no_valid_repetitions", z=np.nan)
            continue
        mu, sd = float(valid.FI_null.mean()), float(valid.FI_null.std(ddof=1))
        row.update(null_mean=mu, null_sd=sd)
        if not sd > 0:
            row.update(status="null_sd_zero_z_undefined", z=np.nan)
        else:
            row.update(status="ok", z=(row["FI_DR"] - mu) / sd)
    per = pd.DataFrame(per_dr)
    per.to_csv(out_dir / "exp_c_per_dr.csv", index=False)
    zs = (per.loc[per.status.eq("ok"), ["z", "group_key"]] if "z" in per.columns
          else pd.DataFrame(columns=["z", "group_key"]))
    summary: dict = {"n_dr": int(n_dr), "n_z_defined": len(zs),
                     "status_counts": per.status.value_counts().to_dict()}
    if len(zs) >= 1 and np.any(zs.z.to_numpy() != 0):
        w = sps.wilcoxon(zs.z.to_numpy(), alternative="greater")
        rng = rng_for(base_seed, "exp_c_median_z_bootstrap")
        W = group_bootstrap_weights(zs.group_key.astype(str).to_numpy(), None, int(st["n_boot"]), rng)
        z = zs.z.to_numpy()
        meds = []
        for wb in W:
            rep = np.repeat(z, wb.astype(int))
            meds.append(np.median(rep) if rep.size else np.nan)
        summary.update(wilcoxon_statistic=float(w.statistic), wilcoxon_p_greater=float(w.pvalue),
                       median_z=float(np.median(z)), median_z_ci=percentile_ci(np.array(meds), float(st["alpha"])),
                       fraction_z_positive=float(np.mean(z > 0)))
    else:
        summary.update(wilcoxon_p_greater=None, reason="no defined non-zero z-scores")
    from ..stats import multiplicity

    summary["wilcoxon_p_role"] = multiplicity.role(cfg, *multiplicity.C_KEY)
    res = header(cfg, "C", "key secondary (H4)", info, calibration)
    res.update(summary=summary, caveats=CAVEATS, deviations=deviations,
               settings={"repetitions": R, "tolerance_rel": tol, "density_metric": dens_col,
                         "fi_order": ccfg["fi_order"], "rethinning_cleanup":
                             "fragment rule + isolated nodes + degree-2 contraction; no spur pruning",
                         "min_fragment_length_um": _min_fragment(cfg)},
               execution=execution, elapsed_seconds=time.monotonic() - t0)
    write_results(out_dir, res)
    return res
