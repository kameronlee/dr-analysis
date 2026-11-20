"""Edge-removal stress tests, S(q), P(q), and FI summaries (review section 4).

The input graph is never modified: all bookkeeping uses an ``active`` mask on
compact arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .betweenness import edge_betweenness
from .coverage import CoverageTracker
from .model import VesselGraph

ADAPTIVE_ORDERS = {"adaptive_betweenness", "forman"}
STATIC_ORDERS = {"static_betweenness", "width", "random"}
ALL_ORDERS = ADAPTIVE_ORDERS | STATIC_ORDERS

# Report labels. The config key "forman" is kept for output-column stability, but the
# score is NOT a validated Forman-Ricci curvature of the vessel multigraph
# (literature_appraisal.md 4.4): it applies Sreejith et al. (2016)'s
# unweighted simple-graph formula 4 - deg(u) - deg(v) under a project-defined
# convention for parallel edges and self-loops, and ties (most edges) are broken by
# ascending edge id, which follows the raster order of extraction. All its results
# are exploratory and uncorrected.
ORDER_LABELS = {
    "adaptive_betweenness": "adaptive length-weighted edge betweenness (primary order)",
    "random": "uniform random edge order (mean over repetitions)",
    "static_betweenness": "static length-weighted edge betweenness (secondary)",
    "width": "descending vessel width (secondary)",
    "forman": ("degree-based Forman-type heuristic, EXPLORATORY: Sreejith 2016 unweighted "
               "formula; multigraph and self-loop convention project-defined (self-loop: "
               "4 - 2 deg(u), an ad hoc extension); ties broken by edge id (raster order); "
               "not a Ricci-curvature claim"),
}

FI_CAVEATS = (
    "FI = 1 - 2R is not clipped. With edge-count q, unequal edge lengths and a "
    "discrete batch grid, R may exceed 0.5 (FI < 0) or the starter-convention mean "
    "may differ from the actual-q integral; the bounds stated in the project prose "
    "do not hold in general. R_mean excludes S0 and includes the final S=0 sample; "
    "the final batch may be partial."
)


class EmptyGraphError(ValueError):
    """Raised when a stress test is requested on a graph without edges/length."""


def batch_size(m: int, step_frac: float) -> int:
    return max(1, int(round(step_frac * m)))


def largest_component_length(n: int, eu, ev, length, active) -> tuple[float, int]:
    """(largest component total edge length, number of components incl. isolated nodes)."""
    idx = np.flatnonzero(active)
    if n == 0:
        return 0.0, 0
    if idx.size == 0:
        return 0.0, n
    a = coo_matrix((np.ones(idx.size), (eu[idx], ev[idx])), shape=(n, n))
    ncomp, lab = connected_components(a, directed=False)
    comp_len = np.bincount(lab[eu[idx]], weights=length[idx], minlength=ncomp)
    return float(comp_len.max()), int(ncomp)


def _ordered(score: np.ndarray, edge_ids: np.ndarray, candidates: np.ndarray,
             rel_quantum: float) -> np.ndarray:
    """Candidates sorted by descending score (quantised), ties by ascending edge id."""
    s = score[candidates]
    scale = rel_quantum * max(1.0, float(np.max(np.abs(s))) if s.size else 1.0)
    qs = np.round(s / scale)
    order = np.lexsort((edge_ids[candidates], -qs))
    return candidates[order]


def forman_curvature(n: int, eu, ev, active) -> np.ndarray:
    """Degree-based Forman-type score F(e) = 4 - deg(u) - deg(v) on active edges.

    Sreejith et al. (2016) unweighted simple-graph formula. On this multigraph,
    degrees count every parallel edge, and a self-loop contributes 2 to its
    node's degree (u = v, so F = 4 - 2 deg(u)). Both are project conventions
    that the source does not cover. See ``ORDER_LABELS["forman"]``.
    """
    idx = np.flatnonzero(active)
    deg = np.bincount(eu[idx], minlength=n) + np.bincount(ev[idx], minlength=n)
    return 4.0 - deg[eu] - deg[ev]


@dataclass
class Curve:
    order: str
    q_edge: np.ndarray
    q_len: np.ndarray
    S: np.ndarray
    n_components: np.ndarray
    P: np.ndarray | None = None
    removal_batches: list[list[int]] = field(default_factory=list)  # edge ids per batch

    def to_frame(self):
        import pandas as pd

        d = {"step": np.arange(len(self.S)), "q_edge": self.q_edge, "q_len": self.q_len,
             "S": self.S, "n_components": self.n_components}
        if self.P is not None:
            d["P"] = self.P
        df = pd.DataFrame(d)
        df.insert(0, "order", self.order)
        return df


def run_removal(graph: VesselGraph, order: str, step_frac: float = 0.01,
                rng: np.random.Generator | None = None, backend: str = "igraph",
                coverage: CoverageTracker | None = None, rel_quantum: float = 1e-9,
                keep_batches: bool = True) -> Curve:
    """Remove edges in batches according to ``order`` until none remain."""
    if order not in ALL_ORDERS:
        raise ValueError(f"unknown order {order}")
    c = graph.compact()
    n, eu, ev, length, width, eids = c["n"], c["eu"], c["ev"], c["length"], c["width"], c["edge_ids"]
    m = len(length)
    L0 = float(length.sum())
    if m == 0 or L0 <= 0:
        raise EmptyGraphError("graph has no edges / zero total length")
    b = batch_size(m, step_frac)
    active = np.ones(m, dtype=bool)
    if coverage is not None:
        coverage.reset()
        P0 = coverage.fraction()

    static_seq = None
    if order == "random":
        if rng is None:
            raise ValueError("random order requires an rng")
        static_seq = rng.permutation(m)
    elif order == "width":
        static_seq = _ordered(width, eids, np.arange(m), rel_quantum)
    elif order == "static_betweenness":
        bt = edge_betweenness(n, eu, ev, length, active, backend)
        static_seq = _ordered(bt, eids, np.arange(m), rel_quantum)

    lcc, nc = largest_component_length(n, eu, ev, length, active)
    q_edge, q_len, S, NC = [0.0], [0.0], [lcc / L0], [nc]
    P = [P0] if coverage is not None else None
    batches: list[list[int]] = []
    removed_n, removed_len, pos = 0, 0.0, 0
    while removed_n < m:
        if static_seq is not None:
            sel = static_seq[pos:pos + b]
            pos += b
        else:
            cand = np.flatnonzero(active)
            if order == "adaptive_betweenness":
                score = edge_betweenness(n, eu, ev, length, active, backend)
            else:  # forman: most negative curvature first
                score = -forman_curvature(n, eu, ev, active)
            sel = _ordered(score, eids, cand, rel_quantum)[:b]
        active[sel] = False
        removed_n += len(sel)
        removed_len += float(length[sel].sum())
        if coverage is not None:
            coverage.remove(sel)
            P.append(coverage.fraction())
        lcc, nc = largest_component_length(n, eu, ev, length, active)
        q_edge.append(removed_n / m)
        q_len.append(removed_len / L0)
        S.append(lcc / L0)
        NC.append(nc)
        if keep_batches:
            batches.append(eids[sel].tolist())
    q_len[-1] = 1.0  # exact, avoids float drift
    return Curve(order, np.array(q_edge), np.array(q_len), np.array(S), np.array(NC),
                 None if P is None else np.array(P), batches)


def interp_at(q: np.ndarray, y: np.ndarray, q0: float) -> float:
    return float(np.interp(q0, q, y))


def first_crossing(q: np.ndarray, S: np.ndarray, thr: float = 0.5) -> float:
    """First recorded q with S < thr (0 if S0 < thr)."""
    below = np.flatnonzero(S < thr)
    return float(q[below[0]]) if below.size else float("nan")


def _trapz(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def summarize_curve(curve: Curve, q_eval: float = 0.10, thr: float = 0.5) -> dict:
    S, qe, ql = curve.S, curve.q_edge, curve.q_len
    post = S[1:]
    R_mean = float(post.mean())
    R_right = float(np.sum((qe[1:] - qe[:-1]) * post))
    R_trap = _trapz(S, qe)
    R_trap_len = _trapz(S, ql)
    out = {
        "S0": float(S[0]),
        "n_batches": int(len(post)),
        "R_mean": R_mean, "FI": 1.0 - 2.0 * R_mean,
        "R_right": R_right, "FI_right": 1.0 - 2.0 * R_right,
        "R_trap": R_trap, "FI_trap": 1.0 - 2.0 * R_trap,
        "R_trap_len": R_trap_len, "FI_trap_len": 1.0 - 2.0 * R_trap_len,
        f"S_q{int(round(q_eval * 100)):02d}": interp_at(qe, S, q_eval),
        "qc": first_crossing(qe, S, thr),
        "qc_len": first_crossing(ql, S, thr),
    }
    if curve.P is not None:
        P0 = float(curve.P[0])
        if P0 > 0:
            out["P0"] = P0
            out["FIperf"] = 1.0 - float(np.mean(curve.P[1:] / P0))
            out[f"P_q{int(round(q_eval * 100)):02d}"] = interp_at(qe, curve.P, q_eval)
        else:
            out["P0"] = P0
            out["FIperf"] = float("nan")
            out["FIperf_undefined_reason"] = "P0_zero"
    return out


def mean_curve(curves: list[Curve]) -> Curve:
    """Pointwise mean over repetitions sharing the same q grid."""
    q = curves[0].q_edge
    for c in curves:
        if len(c.q_edge) != len(q) or not np.allclose(c.q_edge, q):
            raise ValueError("repetitions do not share a q grid")
    S = np.mean([c.S for c in curves], axis=0)
    ql = np.mean([c.q_len for c in curves], axis=0)
    nc = np.mean([c.n_components for c in curves], axis=0)
    P = None
    if all(c.P is not None for c in curves):
        P = np.mean([c.P for c in curves], axis=0)
    return Curve(curves[0].order, q.copy(), ql, S, nc, P, [])


def stress_suite(graph: VesselGraph, orders, step_frac: float, n_random: int,
                 rng_factory, backend: str = "igraph", coverage: CoverageTracker | None = None,
                 coverage_orders=(), q_eval: float = 0.10, thr: float = 0.5,
                 rel_quantum: float = 1e-9) -> dict:
    """Run all orders; returns {"curves": {order: Curve}, "features": {...}, "random_reps": [...]}.

    ``rng_factory(rep)`` must return a fresh deterministic Generator per repetition.
    """
    curves: dict[str, Curve] = {}
    feats: dict[str, float] = {}
    rand_rows = []
    for order in orders:
        cov = coverage if (coverage is not None and order in coverage_orders) else None
        if order == "random":
            reps = []
            for r in range(n_random):
                cv = run_removal(graph, "random", step_frac, rng_factory(r), backend, cov,
                                 rel_quantum, keep_batches=False)
                s = summarize_curve(cv, q_eval, thr)
                s["rep"] = r
                rand_rows.append(s)
                reps.append(cv)
            mc = mean_curve(reps)
            curves["random"] = mc
            ms = summarize_curve(mc, q_eval, thr)
            # On the shared edge-count grid, FI, FI_right, FI_trap and S(q) of the mean
            # curve equal the means over repetitions (linear functionals of S). The
            # length axis differs between repetitions, so FI_trap_len is averaged
            # per repetition instead of being integrated on the mean curve.
            for k in ("FI", "FI_right", "FI_trap", "S0"):
                feats[f"{k}_random"] = ms[k]
            feats["FI_trap_len_random"] = float(np.mean([s["FI_trap_len"] for s in rand_rows]))
            key = f"S_q{int(round(q_eval * 100)):02d}"
            feats[f"{key}_random"] = ms[key]
            # Prespecified (resume_review.md B6): qc_random is the first
            # crossing of the mean random curve; the per-repetition median is a
            # diagnostic only.
            feats["qc_random"] = ms["qc"]
            feats["qc_random_meancurve"] = ms["qc"]
            feats["qc_random_median_of_reps"] = float(np.nanmedian([s["qc"] for s in rand_rows]))
            feats["FI_random_sd_reps"] = float(np.std([s["FI"] for s in rand_rows], ddof=1)) \
                if n_random > 1 else float("nan")
            if "FIperf" in ms:
                feats["FIperf_random"] = ms["FIperf"]
        else:
            cv = run_removal(graph, order, step_frac, None, backend, cov, rel_quantum)
            curves[order] = cv
            s = summarize_curve(cv, q_eval, thr)
            for k, v in s.items():
                if k != "n_batches":
                    feats[f"{k}_{order}"] = v
    key = f"S_q{int(round(q_eval * 100)):02d}"
    for order in orders:
        if order != "random" and "random" in curves:
            feats[f"G{int(round(q_eval * 100))}_{order}"] = (
                feats[f"{key}_random"] - feats[f"{key}_{order}"])
    return {"curves": curves, "features": feats, "random_reps": rand_rows}
