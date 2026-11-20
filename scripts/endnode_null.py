"""Thinning null for end nodes: how was the density lost, one segment at a time or in patches?

Post-outcome exploratory. Authority and fixed readings:
``docs/ENDNODE_NULL_AMENDMENT_2026-11-16.md``. Uses the pipeline's own cleanup
functions; modifies nothing in the package, the configuration or saved results.

Subcommands::

    record   dated machine record; once only
    plan     re-clean every eye once, freeze targets, donors and shards
    shard    run one shard (resumable; one fsync'd JSON line per finished unit)
    status   progress
    reduce   refuse unless complete, then every summary and reading

Work unit = (family, healthy donor). For each of the donor's targets the donor
graph is thinned in family order until its density after full re-cleaning
matches the target's, and its end nodes are counted.
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
from scipy import ndimage as ndi
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retina_fragility.config import config_hash, load_config  # noqa: E402
from retina_fragility.graph.cleanup import contract_degree2, drop_small_fragments, prune_spurs  # noqa: E402
from retina_fragility.graph.model import VesselGraph  # noqa: E402
from retina_fragility.graph.thinning import cleaned_length, nearest_prefix  # noqa: E402
from retina_fragility.io_masks import read_mask  # noqa: E402
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
from scripts import step0_null_controls as s0  # noqa: E402
from scripts import step0_quality_specificity as qs  # noqa: E402

AMENDMENT = "docs/ENDNODE_NULL_AMENDMENT_2026-11-16.md"
RECORD = "reports/endnode_null_amendment_2026-11-16.json"
OUT = "outputs/exploratory_20261116_step0/endnode_null"
FAMILIES = s0.FAMILIES
RANK = {f: i for i, f in enumerate(FAMILIES)}
REP = 0
TOL = 0.02
CONVERGED = 0.002
MAX_EVAL = 4
MIN_DONORS = 10
BAND_PX = 5
CODE_MODULES = ("retina_fragility.graph.cleanup", "retina_fragility.graph.thinning",
                "retina_fragility.graph.model", "retina_fragility.seeds")
PLAN_FORMAT = "endnode_null.plan/1"


class NullError(RuntimeError):
    pass


# ------------------------------------------------------------------------------ core
def reclean(graph: VesselGraph, removed_ids, gcfg: dict) -> VesselGraph:
    """Remove edges, then the pipeline's cleanup stages 3-7 with the frozen parameters."""
    g = graph.subgraph_without(removed_ids)
    g.drop_isolated_nodes()
    contract_degree2(g)
    prune_spurs(g, float(gcfg.get("spur_length_factor", 2.0)), int(gcfg.get("spur_prune_passes", 1)))
    g.drop_isolated_nodes()
    contract_degree2(g)
    drop_small_fragments(g, float(gcfg["min_fragment_length_um"]))
    g.drop_isolated_nodes()
    return g


def faz_distance(faz: np.ndarray | None) -> np.ndarray | None:
    if faz is None or not np.any(faz):
        return None
    return ndi.distance_transform_edt(~np.asarray(faz, bool))


def end_counts(g: VesselGraph, dist: np.ndarray | None, band: int = BAND_PX) -> dict:
    """Degree-one nodes by location (same rule as the M2 location split)."""
    H, W = g.shape
    out = {"total": 0, "border": 0, "perifaz": 0, "interior": 0}
    for nid, d in g.degrees().items():
        if d != 1:
            continue
        nd = g.nodes[nid]
        y, x = float(nd.y), float(nd.x)
        out["total"] += 1
        if min(y, x, H - 1 - y, W - 1 - x) <= band:
            out["border"] += 1
        elif dist is not None and dist[int(round(min(max(y, 0), H - 1))), int(round(min(max(x, 0), W - 1)))] <= band:
            out["perifaz"] += 1
        else:
            out["interior"] += 1
    return out


def match_final(g: VesselGraph, c: dict, perm: np.ndarray, rho: float, area_mm2: float, gcfg: dict):
    """Prefix of ``perm`` whose density after full re-cleaning is nearest ``rho``.

    The fast fragment-only rule proposes a prefix; each full re-cleaning gives
    the true density and a correction factor for the next proposal. At most
    ``MAX_EVAL`` graphs are built. Returns (k, graph, final density, evaluations).
    """
    minfrag = float(gcfg["min_fragment_length_um"])
    ids, m = c["edge_ids"], len(c["length"])

    def fast(k: int) -> float:
        active = np.ones(m, bool)
        active[perm[:k]] = False
        return cleaned_length(c["n"], c["eu"], c["ev"], c["length"], active, minfrag) / 1000.0 / area_mm2

    seen: dict[int, tuple] = {}
    best = None
    k, _ = nearest_prefix(perm, c["n"], c["eu"], c["ev"], c["length"], rho, minfrag, area_mm2)
    for _ in range(MAX_EVAL):
        gk = reclean(g, ids[perm[:k]].tolist(), gcfg)
        fin = gk.total_length() / 1000.0 / area_mm2
        seen[k] = fin
        if best is None or abs(fin - rho) < abs(best[2] - rho):
            best = (k, gk, fin)
        if abs(fin - rho) <= CONVERGED * rho:
            break
        df = fast(k)
        ratio = fin / df if df > 0 else 1.0
        k_new, _ = nearest_prefix(perm, c["n"], c["eu"], c["ev"], c["length"],
                                  rho / ratio if ratio > 0 else rho, minfrag, area_mm2)
        if k_new in seen:
            break
        k = int(k_new)
    return int(best[0]), best[1], float(best[2]), len(seen)


def load_eye(info: dict, cfg: dict):
    g = VesselGraph.load(ROOT / info["graph_path"]) if not Path(info["graph_path"]).is_absolute() \
        else VesselGraph.load(info["graph_path"])
    faz = None
    if info.get("faz_path"):
        p = Path(info["faz_path"])
        faz = read_mask(p if p.is_absolute() else ROOT / p, cfg["data"]["label_maps"]["faz"])
    return g, faz_distance(faz)


def eye_stats(info: dict, cfg: dict) -> dict:
    """Statistics of one eye after one re-cleaning with nothing removed."""
    g, dist = load_eye(info, cfg)
    g0 = reclean(g, [], cfg["graph"])
    e = end_counts(g0, dist)
    a = float(info["area_mm2"])
    return {"density_r": g0.total_length() / 1000.0 / a, "end_interior_r": e["interior"] / a,
            "end_total_r": e["total"] / a, "n_edges_r": int(g0.n_edges)}


def unit_rows(family: str, donor_id: str, dinfo: dict, targets, cfg: dict) -> list[list]:
    g, dist = load_eye(dinfo, cfg)
    c = g.compact()
    cent = s0.edge_centroids(g, c)
    perm = s0.removal_order(family, len(c["length"]), cent, s0.image_extent(g, cent),
                            cfg["project"]["base_seed"], donor_id, REP)
    a = float(dinfo["area_mm2"])
    rows = []
    for tid, rho in targets:
        k, gk, fin, n_eval = match_final(g, c, perm, float(rho), a, cfg["graph"])
        e = end_counts(gk, dist)
        rows.append([tid, k, fin, (fin - rho) / rho, n_eval, e["total"], e["interior"]])
    return rows


# ------------------------------------------------------------------------------ plan
def script_identity(cfg: dict) -> dict:
    return {"script_sha256": sha256_file(Path(__file__)), "step0_script_sha256": sha256_file(Path(s0.__file__)),
            "config_sha256": config_hash(cfg), "code_sha256": code_fingerprint(CODE_MODULES)["sha256"],
            "numerical_libraries": numerical_library_versions()}


def gather_eyes(cfg: dict) -> dict:
    """Every processed eye: group, FOV, area, graph and FAZ paths (no statistics yet)."""
    man = pd.read_csv(ROOT / cfg["data"]["manifest"], dtype=str, keep_default_na=False).set_index("image_id")
    table, _ = s0.load_table(cfg)
    eyes = {}
    for r in table.itertuples():
        eyes[r.image_id] = {"group": "DR" if r.y == 1 else "NORMAL", "donor_pool": bool(r.y == 0),
                            "fov_mm": float(r.fov_mm), "area_mm2": float(r.analysis_area_mm2),
                            "graph_path": str(s0.graph_path(Path(s0.SRC_RUN) / "pipeline", r.image_id)),
                            "faz_path": man.loc[r.image_id, "faz_mask_path"]}
    f = pd.read_csv(ROOT / qs.PIPE_OTHER / "features.csv", dtype={"image_id": str})
    for r in f.itertuples():
        eyes[r.image_id] = {"group": man.loc[r.image_id, "disease"], "donor_pool": False,
                            "fov_mm": float(man.loc[r.image_id, "fov_mm"]), "area_mm2": float(r.analysis_area_mm2),
                            "graph_path": str(s0.graph_path(Path(qs.PIPE_OTHER), r.image_id)),
                            "faz_path": man.loc[r.image_id, "faz_mask_path"]}
    return eyes


def make_plan(eyes: dict, cfg: dict, n_shards: int, identity: dict, extra: dict | None = None) -> dict:
    for iid, info in eyes.items():
        info.update(eye_stats(info, cfg))
    donors: dict[str, list] = {}
    n_el: dict[str, int] = {}
    for tid, t in eyes.items():
        el = [d for d, di in eyes.items() if di["donor_pool"] and d != tid and di["fov_mm"] == t["fov_mm"]
              and di["density_r"] >= t["density_r"]]
        n_el[tid] = len(el)
        for d in el:
            donors.setdefault(d, []).append([tid, t["density_r"]])
    for tid in eyes:
        eyes[tid]["n_eligible_donors"] = n_el[tid]
    units = [(f, d) for d in sorted(donors) for f in FAMILIES]
    cost = {u: len(donors[u[1]]) for u in units}
    load = [0.0] * n_shards
    shards: list[list] = [[] for _ in range(n_shards)]
    for u in sorted(units, key=lambda u: (-cost[u], u[1], RANK[u[0]])):
        i = int(np.argmin(load))
        shards[i].append(list(u))
        load[i] += cost[u]
    for s in shards:
        s.sort(key=lambda u: (u[1], RANK[u[0]]))
    body = {"format": PLAN_FORMAT, "analysis_status": s0.STATUS_LABEL, "identity": identity,
            "settings": {"families": list(FAMILIES), "repetition": REP, "tolerance_rel": TOL,
                         "max_evaluations": MAX_EVAL, "band_px": BAND_PX,
                         "graph_cleanup": {k: cfg["graph"][k] for k in ("spur_length_factor", "spur_prune_passes",
                                                                        "min_fragment_length_um")}},
            "eyes": eyes, "donors": donors, "shards": shards, "n_units": len(units),
            "n_pairs_per_family": int(sum(len(v) for v in donors.values())), **(extra or {})}
    body["plan_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    return body


# ----------------------------------------------------------------------------- shard
def _shard_file(run_dir: Path, i: int) -> Path:
    return Path(run_dir) / "shards" / f"shard_{i:05d}.jsonl"


def run_shard(run_dir: Path, shard: int, cfg: dict, identity: dict, max_units: int | None = None) -> dict:
    run_dir = Path(run_dir)
    plan = read_json(run_dir / "plan.json")
    if plan["identity"] != identity:
        raise NullError("script, config, package code or libraries differ from the frozen plan")
    units = [tuple(u) for u in plan["shards"][shard]]
    path = _shard_file(run_dir, shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    done_rows = s0.read_shard(path, repair=True)
    if any(r["plan_sha256"] != plan["plan_sha256"] for r in done_rows):
        raise NullError(f"{path} holds rows of another plan")
    done = {tuple(r["unit"]) for r in done_rows}
    n_run = 0
    with open(path, "ab") as fh:
        for fam, donor in units:
            if (fam, donor) in done:
                continue
            if max_units is not None and n_run >= max_units:
                break
            t0 = time.monotonic()
            rows = unit_rows(fam, donor, plan["eyes"][donor], plan["donors"][donor], cfg)
            line = {"plan_sha256": plan["plan_sha256"], "unit": [fam, donor], "rows": rows,
                    "elapsed_s": round(time.monotonic() - t0, 3)}
            fh.write((json.dumps(line, separators=(",", ":"), allow_nan=False) + "\n").encode())
            fh.flush()
            os.fsync(fh.fileno())
            n_run += 1
    n_done = len(done) + n_run
    return {"shard": shard, "n_units": len(units), "n_done": n_done,
            "status": "complete" if n_done == len(units) else "partial"}


def run_status(run_dir: Path) -> dict:
    plan = read_json(Path(run_dir) / "plan.json")
    done, secs, inc = 0, 0.0, []
    for i, units in enumerate(plan["shards"]):
        rows = s0.read_shard(_shard_file(run_dir, i))
        done += len(rows)
        secs += sum(r["elapsed_s"] for r in rows)
        if len(rows) != len(units):
            inc.append(i)
    return {"n_units": plan["n_units"], "n_units_done": done, "core_hours_done": secs / 3600.0,
            "incomplete_shards": inc}


# ---------------------------------------------------------------------------- reduce
def collect(run_dir: Path) -> tuple[dict, pd.DataFrame, float]:
    run_dir = Path(run_dir)
    plan = read_json(run_dir / "plan.json")
    planned = {tuple(u) for s in plan["shards"] for u in s}
    seen, frames, secs = set(), [], 0.0
    for i in range(len(plan["shards"])):
        for r in s0.read_shard(_shard_file(run_dir, i)):
            if r["plan_sha256"] != plan["plan_sha256"]:
                raise NullError(f"shard {i} holds rows of another plan")
            u = tuple(r["unit"])
            if u in seen:
                raise NullError(f"duplicate unit {u}")
            if u not in planned:
                raise NullError(f"unit {u} is not in the plan")
            if [x[0] for x in r["rows"]] != [t[0] for t in plan["donors"][u[1]]]:
                raise NullError(f"unit {u}: rows do not cover its targets")
            seen.add(u)
            secs += r["elapsed_s"]
            a = pd.DataFrame(r["rows"], columns=["target_id", "k", "final_density", "residual_rel", "n_eval",
                                                 "n_end_total", "n_end_interior"])
            a.insert(0, "donor_id", u[1])
            a.insert(0, "family", u[0])
            frames.append(a)
    if planned - seen:
        raise NullError(f"{len(planned - seen)} of {len(planned)} units missing; run is incomplete")
    rows = pd.concat(frames, ignore_index=True)
    area = {k: v["area_mm2"] for k, v in plan["eyes"].items()}
    rows["end_interior"] = rows.n_end_interior / rows.donor_id.map(area)
    rows["end_total"] = rows.n_end_total / rows.donor_id.map(area)
    rows["valid"] = rows.residual_rel.abs() <= TOL
    return plan, rows, secs


def per_target(plan: dict, rows: pd.DataFrame) -> pd.DataFrame:
    eyes = pd.DataFrame.from_dict(plan["eyes"], orient="index").rename_axis("image_id").reset_index()
    w = eyes[["image_id", "group", "fov_mm", "density_r", "end_interior_r", "end_total_r", "n_eligible_donors"]].copy()
    v = rows[rows.valid]
    usable = pd.Series(True, index=w.image_id)
    for stat, col in (("interior", "end_interior"), ("total", "end_total")):
        g = v.groupby(["family", "target_id"])[col].agg(["mean", "std", "size"])
        for f in FAMILIES:
            s = g.xs(f, level="family") if f in g.index.get_level_values("family") else g.iloc[0:0]
            mu, sd, n = w.image_id.map(s["mean"]), w.image_id.map(s["std"]), w.image_id.map(s["size"]).fillna(0)
            obs = w[f"end_{stat}_r"]
            w[f"delta_{stat}_{f}"] = obs - mu
            w[f"z_{stat}_{f}"] = np.where((n >= 2) & (sd > 0), (obs - mu) / sd.where(sd > 0), np.nan)
            if stat == "interior":
                w[f"null_mean_{f}"], w[f"n_valid_{f}"] = mu, n.astype(int)
                usable &= (n.to_numpy() >= MIN_DONORS)
    w["usable"] = usable.to_numpy()
    d = np.abs(w[[f"delta_interior_{f}" for f in FAMILIES]].to_numpy(float))
    ok = np.isfinite(d).all(axis=1)
    rank = np.full(len(w), np.nan)
    rank[ok] = np.argmin(d[ok], axis=1)
    w["patchiness_rank"] = rank
    return w


def summarise(plan: dict, rows: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    base_seed = cfg["project"]["base_seed"]
    w = per_target(plan, rows)
    u = w[w.usable].copy()
    res: dict = {"analysis_status": s0.STATUS_LABEL, "plan_sha256": plan["plan_sha256"],
                 "n_rows": int(len(rows)), "fraction_within_tolerance": float(rows.valid.mean()),
                 "within_tolerance_by_family": rows.groupby("family").valid.mean().to_dict(),
                 "median_evaluations": float(rows.n_eval.median()),
                 "n_targets": w.group.value_counts().to_dict(), "n_usable": u.group.value_counts().to_dict()}

    def med(frame, col, label):
        return s0.median_ci(frame[col], rng_for(base_seed, "endnull_boot", label))

    A = {}
    for f in FAMILIES:
        A[f] = {"pooled": med(u[u.group == "DR"], f"z_interior_{f}", f"A|{f}"),
                **{n: med(u[(u.group == "DR") & (u.fov_mm == fov)], f"z_interior_{f}", f"A|{f}|{n}")
                   for fov, n in ((3.0, "fov_3mm"), (6.0, "fov_6mm"))},
                "pooled_total_end_nodes": med(u[u.group == "DR"], f"z_total_{f}", f"A|{f}|total")}
    res["A_dr_against_each_null"] = A

    def below(c):
        return c.get("ci") is not None and c["ci"][1] < 0

    def includes(c):
        return c.get("ci") is not None and c["ci"][0] <= 0 <= c["ci"][1]

    coh = u[u.group.isin(["DR", "NORMAL"])].assign(y=lambda d: (d.group == "DR").astype(int), density=lambda d: d.density_r)
    B = {"z_uniform": s0.contrast_by_scope(coh, "z_interior_uniform", base_seed, "endnull|B|z"),
         "patchiness_rank": s0.contrast_by_scope(coh, "patchiness_rank", base_seed, "endnull|B|rank")}
    adj = holm({"z_uniform": B["z_uniform"]["pooled"].get("wilcoxon_p_two_sided"),
                "patchiness_rank": B["patchiness_rank"]["pooled"].get("wilcoxon_p_two_sided")})
    B["holm_adjusted_p"] = adj
    res["B_dr_against_matched_healthy"] = B
    C: dict = {"by_group": {}, "healthy_by_density_tertile": {}}
    for grp, s in u.groupby("group"):
        C["by_group"][grp] = {"n": int(len(s)),
                              "rank_counts": {FAMILIES[int(k)]: int(v) for k, v in
                                              s.patchiness_rank.dropna().astype(int).value_counts().sort_index().items()},
                              **{f: med(s, f"z_interior_{f}", f"C|{grp}|{f}") for f in FAMILIES}}
    h = u[u.group == "NORMAL"].copy()
    for fov, n in ((3.0, "fov_3mm"), (6.0, "fov_6mm")):
        s = h[h.fov_mm == fov]
        if len(s) >= 6:
            t = pd.qcut(s.density_r, 3, labels=["low", "mid", "high"])
            for lab in ("low", "mid", "high"):
                ss = s[t == lab]
                C["healthy_by_density_tertile"][f"{n}|{lab}"] = {
                    "n": int(len(ss)), "z_uniform": med(ss, "z_interior_uniform", f"C|h|{n}|{lab}"),
                    "z_patchy1": med(ss, "z_interior_patchy1", f"C|h|{n}|{lab}|p1"),
                    "median_rank": float(ss.patchiness_rank.median()) if len(ss) else None}
    res["C_descriptive"] = C
    zc, rc = B["z_uniform"]["pooled"], B["patchiness_rank"]["pooled"]
    res["readings"] = {
        "dr_below_uniform_null": bool(below(A["uniform"]["pooled"])),
        "patchy_families_reproducing_dr": [f for f in FAMILIES[1:] if includes(A[f]["pooled"])],
        "dr_below_every_null_mask_texture": bool(all(below(A[f]["pooled"]) for f in FAMILIES)),
        "dr_patchier_than_matched_healthy": bool(
            (adj["z_uniform"] is not None and adj["z_uniform"] < 0.05 and (zc.get("median") or 0) < 0)
            or (adj["patchiness_rank"] is not None and adj["patchiness_rank"] < 0.05 and (rc.get("median") or 0) > 0))}
    return res, w


def text(res: dict) -> str:
    def m(c):
        if c.get("median") is None:
            return "n/a"
        return f"{c['median']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] (n={c['n']})"

    A, B, C = res["A_dr_against_each_null"], res["B_dr_against_matched_healthy"], res["C_descriptive"]
    shown = [FAMILIES[0], FAMILIES[len(FAMILIES) // 2], FAMILIES[-1]]
    L = ["THINNING NULL FOR END NODES (post-outcome exploratory)", "",
         f"Rows within the 2% density tolerance: {res['fraction_within_tolerance']:.1%}; usable targets: {res['n_usable']}",
         "", "A. DR eyes against thinned healthy graphs: median z of interior end-node density",
         "   (negative = fewer end nodes than the thinned healthy graphs)"]
    for f in FAMILIES:
        L.append(f"   {f:9s} pooled {m(A[f]['pooled'])}; 3 mm {m(A[f]['fov_3mm'])}; 6 mm {m(A[f]['fov_6mm'])}")
    L += ["", "B. DR minus healthy eyes of the same density:",
          "   uniform z:       " + s0._mc(B["z_uniform"]["pooled"]),
          "   patchiness rank: " + s0._mc(B["patchiness_rank"]["pooled"]),
          f"   Holm-adjusted p: {json.dumps(B['holm_adjusted_p'])}", "",
          f"C. Median z by group ({' | '.join(shown)}) and best-matching family counts:"]
    for g, v in C["by_group"].items():
        L.append(f"   {g:7s} n={v['n']:3d}  " + " | ".join(m(v[f]) for f in shown) + f"  {v['rank_counts']}")
    L.append("   Healthy eyes by density tertile (uniform z | single-patch z | median rank):")
    for k, v in C["healthy_by_density_tertile"].items():
        L.append(f"   {k:14s} n={v['n']:3d}  {m(v['z_uniform'])} | {m(v['z_patchy1'])} | {v['median_rank']}")
    L += ["", "READINGS: " + json.dumps(res["readings"])]
    return "\n".join(L) + "\n"


def input_files(cfg: dict) -> dict[str, str]:
    files = [AMENDMENT, "config.yaml", cfg["data"]["manifest"], f"{s0.SRC_RUN}/pipeline/features.csv",
             f"{qs.PIPE_OTHER}/features.csv"]
    return {f: sha256_file(ROOT / f) for f in files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record")
    p = sub.add_parser("plan")
    p.add_argument("--n-shards", type=int, default=48)
    p = sub.add_parser("shard")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--max-units", type=int, default=None)
    sub.add_parser("status")
    sub.add_parser("reduce")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    cfg = load_config(ROOT / "config.yaml")
    out, run_dir = ROOT / OUT, ROOT / OUT / "run"
    if args.cmd == "record":
        if (ROOT / RECORD).exists() or out.exists():
            raise NullError("record or outputs already exist; the record is written once, first")
        write_json(ROOT / RECORD, {
            "adopted_utc": now_utc(), "decided_by": "",
            "instructed_by": "user, in session, 2026-11-16", "is_human_review": False,
            "analysis_status": s0.STATUS_LABEL, "results_available_at_adoption": False,
            "prototype": "two calibration eyes only (excluded from every analysis)",
            "identity_at_adoption": script_identity(cfg), "files_sha256": input_files(cfg)})
        print(json.dumps(read_json(ROOT / RECORD), indent=1)[:400])
        return 0
    if not (ROOT / RECORD).exists():
        raise NullError("no dated record; run 'record' first")
    rec = read_json(ROOT / RECORD)
    bad = [f for f, h in rec["files_sha256"].items() if input_files(cfg).get(f) != h]
    if bad:
        raise NullError("inputs changed since the dated record: " + ", ".join(bad))
    if args.cmd == "plan":
        if (run_dir / "plan.json").exists():
            raise NullError("a plan already exists; it is frozen")
        plan = make_plan(gather_eyes(cfg), cfg, args.n_shards, script_identity(cfg),
                         {"created_utc": now_utc(), "inputs_sha256": input_files(cfg)})
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "plan.json", plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "n_units": plan["n_units"],
                          "n_eyes": len(plan["eyes"]), "n_donors": len(plan["donors"]),
                          "n_pairs_per_family": plan["n_pairs_per_family"]}, indent=1))
    elif args.cmd == "shard":
        print(json.dumps(run_shard(run_dir, args.shard, cfg, script_identity(cfg), args.max_units)))
    elif args.cmd == "status":
        print(json.dumps(run_status(run_dir), indent=1))
    elif args.cmd == "reduce":
        plan, rows, secs = collect(run_dir)
        if plan["identity"] != script_identity(cfg):
            raise NullError("script, config, package code or libraries differ from the frozen plan")
        res, w = summarise(plan, rows, cfg)
        res["core_hours"] = secs / 3600.0
        res["provenance"] = {"created_utc": now_utc(), "performed_by": "",
                             "amendment": AMENDMENT, "record": RECORD, "identity": script_identity(cfg),
                             "environment": environment()}
        a = out / "analysis"
        a.mkdir(parents=True, exist_ok=True)
        rows.to_csv(a / "null_rows.csv.gz", index=False)
        w.to_csv(a / "per_target.csv", index=False)
        write_json(a / "results.json", res)
        (a / "summary.txt").write_text(text(res))
        print(text(res))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (NullError, s0.Step0Error) as exc:
        sys.exit(f"REFUSED: {exc}")
