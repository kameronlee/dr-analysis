"""Run the already implemented graph core against the guide's Experiment 0.

Run: .venv/bin/python -m scripts.run_synthetic [--quick]
These are software-validation fixtures, not retinal disease observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retina_fragility.config import config_hash, dump_config, load_config
from retina_fragility.graph.stress import run_removal, mean_curve, summarize_curve
from retina_fragility.synthetic import grid_graph, binary_tree_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    path = Path("config.yaml")
    cfg = load_config(path if path.exists() else None)
    if not path.exists():
        dump_config(cfg, path)
    spec = cfg["experiment0"]["quick" if args.quick else "full"]
    out = Path("reports/experiment0_quick" if args.quick else "reports/experiment0")
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out / "config.snapshot.yaml")
    graphs = {"grid": grid_graph(spec["grid_n"]), "tree": binary_tree_graph(spec["tree_depth"])}
    rows, reps, all_curves = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    t0 = time.monotonic()
    for ax, (name, graph) in zip(axes, graphs.items()):
        n_edges = len(graph.edges)
        random_curves = []
        for rep in range(spec["seeds"]):
            seed = cfg["project"]["base_seed"] + rep
            c = run_removal(graph, "random", cfg["stress"]["step_frac"], np.random.default_rng(seed))
            random_curves.append(c)
            reps.append({"network": name, "repetition": rep, "seed": seed, **summarize_curve(c)})
            frame = c.to_frame()
            frame.insert(0, "network", name)
            frame.insert(1, "repetition", rep)
            all_curves.append(frame)
        random_mean = mean_curve(random_curves)
        targeted = run_removal(graph, "adaptive_betweenness", cfg["stress"]["step_frac"],
                               backend=cfg["stress"]["betweenness_backend"])
        for label, c in [("random_mean", random_mean), ("adaptive_betweenness", targeted)]:
            assert np.all(np.diff(c.S) <= 1e-12), "Connectivity increased after edge removal"
            assert c.q_edge[0] == 0 and c.q_edge[-1] == 1 and c.S[-1] == 0
            assert len(graph.edges) == n_edges, "Stress test modified the input graph"
            summary = {"network": name, "order": label, "nodes": len(graph.nodes),
                       "edges": n_edges, **summarize_curve(c)}
            rows.append(summary)
            ax.plot(c.q_edge, c.S, label=label.replace("_", " "))
            if label == "adaptive_betweenness":
                frame = c.to_frame()
                frame.insert(0, "network", name)
                frame.insert(1, "repetition", -1)
                all_curves.append(frame)
            print(json.dumps(summary), flush=True)
        ax.set(title=f"Synthetic {name}", xlabel="Fraction of edges removed", ylabel="Largest-component length / original length",
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(reps).to_csv(out / "random_repetitions.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(out / "curves.csv", index=False)
    fig.suptitle("Experiment 0: synthetic software validation — not clinical results")
    fig.savefig(out / "collapse_curves.png", dpi=180)
    fig.savefig(out / "collapse_curves.pdf")
    plt.close(fig)
    checks = {}
    for order in ["random_mean", "adaptive_betweenness"]:
        s = summary[summary.order == order].set_index("network")
        checks[order + "_tree_more_fragile"] = bool(s.loc["tree", "FI"] > s.loc["grid", "FI"])
    checks["tree_targeted_qc_earlier_than_grid"] = bool(
        summary.query("network == 'tree' and order == 'adaptive_betweenness'").iloc[0].qc <
        summary.query("network == 'grid' and order == 'adaptive_betweenness'").iloc[0].qc)
    report = {"data_provenance": "SYNTHETIC - NOT CLINICAL", "config_hash": config_hash(cfg),
              "mode": "quick" if args.quick else "full", "python": platform.python_version(),
              "checks": checks, "elapsed_seconds": time.monotonic() - t0,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "limitations": "Only graph-level synthetic validation; image extraction and statistics remain unvalidated."}
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not all(checks.values()):
        raise SystemExit("One or more qualitative guide sanity checks failed; see saved report")


if __name__ == "__main__":
    main()
