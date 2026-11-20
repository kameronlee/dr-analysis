"""Experiment C as deterministic, resumable shards with a strict reducer (Slurm arrays).

Work unit = (healthy donor, repetition). A unit matches every DR target of the
donor against permutation (donor, repetition) and computes the null FI of each
distinct within-tolerance prefix, which is exactly the work the serial ``exp_c.run``
does for that pair (same eligibility, same local seeds, same functions).

Workflow::

    plan    freeze the targets, donors, units and shard assignment, together with
            every identity the results depend on (config hash, Experiment C code
            fingerprint, numerical library versions, hashes of the pipeline
            features/run record, donor graphs, manifest and calibration file).
    shard   run the units of one shard (Slurm array task). Each finished unit is
            appended to a checkpoint (JSON lines, fsync'd), so a preempted or
            timed-out task resumes where it stopped. Complete shards are published
            atomically as ``shard_NNNNN.json``.
    reduce  refuse unless every shard is complete, bound to this plan and covers
            exactly its units (no missing, duplicate or foreign unit; matching
            rows cover every DR target of each unit; FI rows cover exactly the
            within-tolerance prefixes). Rows are then put in the serial canonical
            order and passed to ``exp_c.finalize``: the outputs equal the serial run.

Nothing here writes results.json except ``reduce``. Shards never produce a partial
summary. Any mismatch in config, code, libraries or inputs is refused, never mixed.
"""
from __future__ import annotations

import contextlib
import fcntl
import functools
import hashlib
import json
import os
import shutil
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import config_hash
from ..graph.model import VesselGraph
from ..provenance import (
    _default,
    _sanitize,
    code_fingerprint,
    environment,
    now_utc,
    numerical_library_versions,
    read_json,
    sha256_file,
    write_json,
)
from . import exp_c

PLAN_FORMAT = "retina_fragility.exp_c_plan/1"
SHARD_FORMAT = "retina_fragility.exp_c_shard/1"
FLOAT_FIELDS = {"target_density", "achieved_density", "residual_rel", "FI_null", "FI_DR", "fov_mm"}


class ShardError(RuntimeError):
    """Refusal: inconsistent, incomplete or mixed Experiment C shard state."""


# ----------------------------------------------------------------- identity / io
def current_identity(cfg: dict) -> dict:
    return {"config_sha256": config_hash(cfg),
            "code_sha256": code_fingerprint(exp_c.EXP_C_ENTRY_MODULES)["sha256"],
            "numerical_libraries": numerical_library_versions()}


def _dumps(obj) -> str:
    return json.dumps(_sanitize(obj), default=_default, allow_nan=False, separators=(",", ":"))


def _plain(row: dict) -> dict:
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}


def _restore(row: dict) -> dict:
    """JSON null -> NaN for float fields (write_json stores non-finite floats as null)."""
    return {k: (np.nan if (v is None and k in FLOAT_FIELDS) else v) for k, v in row.items()}


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _hash_inputs(files: dict) -> dict:
    out = {}
    for name, path in sorted((files or {}).items()):
        if path is None:
            continue
        p = Path(path)
        if not p.is_file():
            raise ShardError(f"plan input {name} not found: {p}")
        out[name] = {"path": str(p), "sha256": sha256_file(p)}
    return out


def plan_sha256(identity: dict) -> str:
    return config_hash(identity)


def _shard_name(i: int) -> str:
    return f"shard_{i:05d}"


# ----------------------------------------------------------------------- plan
def make_plan(table: pd.DataFrame, info: dict, cfg: dict, pipeline_root: str | Path,
              calibration: dict | None, n_shards: int, input_files: dict | None = None) -> dict:
    """Deterministic plan: same eligibility and donor order as ``exp_c.run``."""
    n_shards = int(n_shards)
    if n_shards < 1:
        raise ShardError("n_shards must be >= 1")
    pipeline_root = Path(pipeline_root)
    normal = table[table.y == 0]
    available = [iid for iid in normal.image_id if exp_c._graph_path(pipeline_root, iid).exists()]
    per_dr, donor_targets = exp_c.plan_targets(table, cfg, available)
    ndon = normal.set_index("image_id")
    dens_col = cfg["exp_c"]["density_metric"]
    donors = {}
    for d in sorted(donor_targets):
        gp = exp_c._graph_path(pipeline_root, d)
        donors[d] = {"graph_path": str(gp), "graph_sha256": sha256_file(gp),
                     "graph_bytes": gp.stat().st_size,
                     "analysis_area_mm2": float(ndon.loc[d, "analysis_area_mm2"]),
                     "fov_mm": float(ndon.loc[d, "fov_mm"]), "density": float(ndon.loc[d, dens_col]),
                     "targets": [[t, float(v)] for t, v in donor_targets[d]]}
    R = int(cfg["exp_c"]["repetitions"])
    units = [[d, r] for d in sorted(donors) for r in range(R)]
    files = dict(input_files or {})
    files.setdefault("pipeline_features", pipeline_root / "features.csv")
    if (pipeline_root / "run.json").exists():
        files.setdefault("pipeline_run", pipeline_root / "run.json")
    identity = {
        "format": PLAN_FORMAT, **current_identity(cfg),
        "pipeline_root": str(pipeline_root),
        "inputs": _hash_inputs(files),
        "calibration": {"calibration_ids_sha256": (calibration or {}).get("calibration_ids_sha256"),
                        "d_um": (calibration or {}).get("d_um")},
        "n_dr": int((table.y == 1).sum()),
        "per_dr": [_plain(r) for r in per_dr],
        "donors": donors,
        "units": units,
        "n_shards": n_shards,
        "shard_assignment": "units sorted by (donor_id, rep); unit i -> shard i mod n_shards",
    }
    identity = json.loads(_dumps(identity))  # exactly what is stored and re-hashed later
    return {"plan_sha256": plan_sha256(identity), "created_utc": now_utc(),
            "info": json.loads(_dumps(info)), "identity": identity,
            "summary": {"n_units": len(units), "n_donors": len(donors), "n_shards": n_shards,
                        "units_per_shard_max": -(-len(units) // n_shards),
                        "n_dr_pending": sum(r["status"] == "pending" for r in per_dr),
                        "n_matchings": sum(len(v["targets"]) for v in donors.values()) * R,
                        "deviations": exp_c.deviations_for(cfg)}}


def write_plan(plan: dict, run_dir: str | Path) -> Path:
    """Write plan.json once. An identical plan is a no-op; a different one is refused."""
    run_dir = Path(run_dir)
    path = run_dir / "plan.json"
    if path.exists():
        old = load_plan(run_dir)
        if old["plan_sha256"] != plan["plan_sha256"]:
            raise ShardError(f"{path} holds a different plan ({old['plan_sha256'][:12]}); inputs, "
                             "config or code changed. Use a new --run-dir; shards are never mixed.")
        return path
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, _dumps(plan))
    return path


def load_plan(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "plan.json"
    if not path.exists():
        raise ShardError(f"no plan at {path}")
    plan = read_json(path)
    ident = plan.get("identity", {})
    if ident.get("format") != PLAN_FORMAT or plan.get("plan_sha256") != plan_sha256(ident):
        raise ShardError(f"{path} is corrupt or edited (plan hash does not match its content)")
    return plan


def verify_current(plan: dict, cfg: dict, check_inputs: bool = True) -> None:
    """Refuse when the current config, code, libraries or inputs differ from the plan."""
    ident = plan["identity"]
    cur = current_identity(cfg)
    diffs = [f"{k}: plan {ident[k]} != current {cur[k]}" for k in cur if ident[k] != cur[k]]
    if check_inputs:
        for name, rec in ident["inputs"].items():
            p = Path(rec["path"])
            if not p.is_file() or sha256_file(p) != rec["sha256"]:
                diffs.append(f"input {name} ({p}) changed or missing since planning")
    if diffs:
        raise ShardError("Experiment C state differs from the plan; refusing to mix: " + "; ".join(diffs))


def verify_donor_graphs(plan: dict, donors=None) -> None:
    """Refuse unless each donor graph (all, or ``donors``) still has its planned hash.

    Checked before new work, before a completed shard is reused and before
    reduction, so results computed from one graph are never combined with a
    pipeline that now holds another. A restored file with the planned hash passes.
    """
    ident = plan["identity"]["donors"]
    bad = []
    for d in sorted(ident if donors is None else set(donors)):
        p = Path(ident[d]["graph_path"])
        if not p.is_file():
            bad.append(f"{d}: {p} missing")
        elif sha256_file(p) != ident[d]["graph_sha256"]:
            bad.append(f"{d}: {p} differs from the planned hash")
    if bad:
        raise ShardError(f"{len(bad)} donor graph(s) changed or missing since planning; refusing to mix: "
                         + "; ".join(bad[:10]))


def shard_units(plan: dict, shard: int) -> list[list]:
    n = plan["identity"]["n_shards"]
    if not 0 <= int(shard) < n:
        raise ShardError(f"shard {shard} outside 0..{n - 1}")
    return plan["identity"]["units"][int(shard)::n]


# ---------------------------------------------------------------------- shard
@functools.lru_cache(maxsize=2)
def _donor_graph(path: str, sha256: str):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ShardError(f"donor graph {path} differs from the planned hash; refusing")
    g = VesselGraph.from_dict(json.loads(data))
    return g, g.compact()


def _unit_job(args) -> dict:
    donor, rep, dinfo, cfg = args
    t0 = time.monotonic()
    g, c = _donor_graph(dinfo["graph_path"], dinfo["graph_sha256"])
    targets = [(t, float(v)) for t, v in dinfo["targets"]]
    audit, null = exp_c.compute_unit(g, c, donor, int(rep), float(dinfo["analysis_area_mm2"]), targets, cfg)
    return {"unit": [donor, int(rep)], "audit": [_plain(a) for a in audit],
            "null": [_plain(n) for n in null], "elapsed_s": round(time.monotonic() - t0, 3)}


def _shard_header(plan: dict, shard: int) -> dict:
    units = shard_units(plan, shard)
    return {"format": SHARD_FORMAT, "plan_sha256": plan["plan_sha256"], "shard": int(shard),
            "n_shards": plan["identity"]["n_shards"], "units_sha256": config_hash(units),
            "config_sha256": plan["identity"]["config_sha256"],
            "code_sha256": plan["identity"]["code_sha256"],
            "numerical_libraries": plan["identity"]["numerical_libraries"]}


def _read_checkpoint(path: Path, header: dict, expected: set) -> dict:
    """Completed unit records from a checkpoint; a torn final line is dropped (and trimmed)."""
    if not path.exists():
        return {}
    text = path.read_text()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        return {}
    try:
        head = json.loads(lines[0])
    except json.JSONDecodeError:
        head = None
    if head != header:
        raise ShardError(f"checkpoint {path} belongs to a different plan/shard/code; delete it or use "
                         "the matching run directory (records are never mixed)")
    done: dict = {}
    keep = [lines[0]]
    for i, line in enumerate(lines[1:], start=1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:  # torn write at preemption: the unit is simply redone
                break
            raise ShardError(f"checkpoint {path} is corrupt at line {i + 1}") from None
        key = tuple(rec["unit"])
        if key not in expected:
            raise ShardError(f"checkpoint {path} holds unit {key} that is not assigned to this shard")
        if key in done:
            raise ShardError(f"checkpoint {path} holds unit {key} twice (concurrent writers?)")
        done[key] = rec
        keep.append(line)
    if len(keep) != len(lines) or not text.endswith("\n"):
        # drop a torn line, and terminate a complete last record whose newline was lost,
        # so the next append starts on its own line
        _atomic_write_text(path, "\n".join(keep) + "\n")
    return done


@contextlib.contextmanager
def _exclusive(lock_path: Path):
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ShardError(f"another process is running this shard ({lock_path})") from None
        yield
    finally:
        os.close(fd)


def _validate_shard_file(plan: dict, shard: int, path: Path) -> dict:
    rec = read_json(path)
    header = _shard_header(plan, shard)
    if {k: rec.get(k) for k in header} != header:
        raise ShardError(f"{path} is not a shard of this plan/code/config; refusing")
    expected = [tuple(u) for u in shard_units(plan, shard)]
    got = [tuple(u["unit"]) for u in rec["units"]]
    if got != expected:
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        dup = len(got) - len(set(got))
        raise ShardError(f"{path}: unit coverage differs from the plan (missing {missing[:5]}, "
                         f"foreign {extra[:5]}, duplicates {dup})")
    donors = plan["identity"]["donors"]
    for u in rec["units"]:
        d, r = u["unit"]
        want_dr = [t for t, _ in donors[d]["targets"]]
        if [a["dr_image_id"] for a in u["audit"]] != want_dr or any(
                a["donor_id"] != d or a["rep"] != r for a in u["audit"]):
            raise ShardError(f"{path}: matching rows of unit {(d, r)} do not cover its planned DR targets")
        ks = sorted({int(a["k"]) for a in u["audit"] if a["within_tolerance"]})
        if [n["k"] for n in u["null"]] != ks or any(n["donor_id"] != d or n["rep"] != r for n in u["null"]):
            raise ShardError(f"{path}: null-FI rows of unit {(d, r)} do not equal its within-tolerance prefixes")
    return rec


def run_shard(run_dir: str | Path, shard: int, cfg: dict, workers: int = 1,
              max_new_units: int | None = None) -> dict:
    """Run (or resume) one shard. ``max_new_units`` stops early (tests/interruption drills)."""
    run_dir = Path(run_dir)
    plan = load_plan(run_dir)
    verify_current(plan, cfg)
    shards_dir = run_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    name = _shard_name(shard)
    final = shards_dir / f"{name}.json"
    units = shard_units(plan, shard)
    verify_donor_graphs(plan, {u[0] for u in units})
    if final.exists():
        _validate_shard_file(plan, shard, final)
        return {"shard": shard, "status": "complete", "n_units": len(units), "n_new": 0}
    header = _shard_header(plan, shard)
    ckpt = shards_dir / f"{name}.partial.jsonl"
    t0 = time.monotonic()
    with _exclusive(shards_dir / f"{name}.lock"):
        expected = {tuple(u) for u in units}
        done = _read_checkpoint(ckpt, header, expected)
        if not done:  # new, empty or torn-header-free checkpoint: start it with the binding header
            _atomic_write_text(ckpt, _dumps(header) + "\n")
        pending = [u for u in units if tuple(u) not in done]
        if max_new_units is not None:
            pending = pending[:max_new_units]
        donors = plan["identity"]["donors"]
        jobs = [(d, r, donors[d], cfg) for d, r in pending]
        n_new = 0
        with open(ckpt, "a") as fh:
            def append(rec):
                fh.write(_dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                done[tuple(rec["unit"])] = rec

            if workers > 1 and len(jobs) > 1:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    for fut in as_completed([pool.submit(_unit_job, j) for j in jobs]):
                        append(fut.result())
                        n_new += 1
            else:
                for j in jobs:
                    append(_unit_job(j))
                    n_new += 1
        if len(done) < len(units):
            return {"shard": shard, "status": "incomplete", "n_units": len(units),
                    "n_done": len(done), "n_new": n_new}
        rec = {**header, "created_utc": now_utc(), "environment": environment(),
               "elapsed_s_this_invocation": round(time.monotonic() - t0, 3),
               "units": [done[tuple(u)] for u in units]}
        _atomic_write_text(final, _dumps(rec))
        _validate_shard_file(plan, shard, final)
        ckpt.unlink()
    return {"shard": shard, "status": "complete", "n_units": len(units), "n_new": n_new}


# --------------------------------------------------------------------- status
def shard_status(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    plan = load_plan(run_dir)
    n = plan["identity"]["n_shards"]
    rows, unit_seconds = [], []
    for i in range(n):
        name = _shard_name(i)
        final = run_dir / "shards" / f"{name}.json"
        ckpt = run_dir / "shards" / f"{name}.partial.jsonl"
        n_units = len(shard_units(plan, i))
        if final.exists():
            unit_seconds += [u.get("elapsed_s", np.nan) for u in read_json(final)["units"]]
            rows.append({"shard": i, "state": "complete", "n_done": n_units, "n_units": n_units})
        elif ckpt.exists():
            k = 0
            for line in ckpt.read_text().splitlines()[1:]:
                with contextlib.suppress(json.JSONDecodeError):
                    unit_seconds.append(json.loads(line).get("elapsed_s", np.nan))
                    k += 1
            rows.append({"shard": i, "state": "partial", "n_done": k, "n_units": n_units})
        else:
            rows.append({"shard": i, "state": "not_started", "n_done": 0, "n_units": n_units})
    df = pd.DataFrame(rows)
    secs = np.asarray(unit_seconds, float)
    n_units_all = len(plan["identity"]["units"])
    timing = {}
    if secs.size:
        timing = {"units_timed": int(secs.size), "unit_seconds_median": float(np.nanmedian(secs)),
                  "unit_seconds_max": float(np.nanmax(secs)),
                  "cpu_hours_done": float(np.nansum(secs) / 3600),
                  "cpu_hours_projected_total": float(np.nanmean(secs) * n_units_all / 3600),
                  "projection_note": "mean over completed units x all units; units differ in cost"}
    return {"plan_sha256": plan["plan_sha256"], "n_shards": n,
            "n_units": n_units_all, "n_units_done": int(df.n_done.sum()),
            "state_counts": df.state.value_counts().to_dict(),
            "incomplete_shards": df.loc[df.state != "complete", "shard"].tolist(),
            "timing": timing,
            "note": "partial counts are checkpoint lines; only reduce validates them"}


# --------------------------------------------------------------------- reduce
def reduce(run_dir: str | Path, out_dir: str | Path, cfg: dict) -> dict:
    """Validate every shard against the plan, then write the Experiment C outputs.

    Refuses (writing nothing) when any shard is missing or incomplete, bound to
    another plan, or its unit coverage differs from the plan.
    """
    t0 = time.monotonic()
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    plan = load_plan(run_dir)
    verify_current(plan, cfg)
    verify_donor_graphs(plan)
    ident = plan["identity"]
    st = shard_status(run_dir)
    if st["incomplete_shards"]:
        raise ShardError(f"{len(st['incomplete_shards'])} of {st['n_shards']} shards are not complete "
                         f"(first: {st['incomplete_shards'][:10]}); {st['n_units_done']}/{st['n_units']} "
                         "units done. No result was written.")
    audit, null, shard_hashes = [], [], {}
    for i in range(ident["n_shards"]):
        path = run_dir / "shards" / f"{_shard_name(i)}.json"
        rec = _validate_shard_file(plan, i, path)
        shard_hashes[_shard_name(i)] = sha256_file(path)
        for u in rec["units"]:
            audit.extend(_restore(a) for a in u["audit"])
            null.extend(_restore(n) for n in u["null"])
    per_dr = [_restore(r) for r in ident["per_dr"]]
    audit, null = exp_c.canonical_order(per_dr, audit, null)
    if out_dir.exists():
        prev = out_dir / "results.json"
        if not prev.exists() or read_json(prev).get("execution", {}).get("plan_sha256") != plan["plan_sha256"]:
            raise ShardError(f"{out_dir} exists and does not hold a result of this plan; choose a new --out")
    staging = out_dir.with_name(f".{out_dir.name}.staging-{uuid.uuid4().hex[:8]}")
    execution = {"mode": "sharded", "plan_sha256": plan["plan_sha256"], "run_dir": str(run_dir),
                 "n_shards": ident["n_shards"], "n_units": len(ident["units"]),
                 "shard_files_sha256": shard_hashes, "inputs": ident["inputs"],
                 "code_sha256": ident["code_sha256"]}
    try:
        res = exp_c.finalize(per_dr, audit, null, ident["n_dr"], plan["info"], cfg, ident["calibration"],
                             staging, t0, execution)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    old = None
    if out_dir.exists():
        old = out_dir.with_name(f".{out_dir.name}.replaced-{uuid.uuid4().hex[:8]}")
        out_dir.rename(old)
    staging.rename(out_dir)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    write_json(run_dir / "reduce_record.json", {"created_utc": now_utc(), "out_dir": str(out_dir),
                                                  "plan_sha256": plan["plan_sha256"],
                                                  "results_sha256": sha256_file(out_dir / "results.json")})
    return res
