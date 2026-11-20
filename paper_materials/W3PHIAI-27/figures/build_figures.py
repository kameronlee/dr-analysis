"""Render the metric audit, retinal tracing and endpoint overlap from packaged inputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from make_retinal_tracing import tracing_figure
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "plot_data"
BLUE = "#20658b"
ORANGE = "#a64912"
INK = "#20242a"


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", metadata={"Title": name.replace("_", " "),
                                               "CreationDate": None, "ModDate": None})
    plt.close(fig)


def audit_figure():
    table = pd.read_csv(DATA / "metric_audit.csv").sort_values("row")
    fig, (effects, ranks) = plt.subplots(1, 2, figsize=(7, 3.15), sharey=True,
                                         gridspec_kw={"width_ratios": [1.8, 1]})
    fig.subplots_adjust(left=.30, right=.986, bottom=.21, top=.88, wspace=.18)
    for prefix, offset in [("", -.115), ("quality_", .115)]:
        base = prefix == ""
        value = table[prefix + "difference"].to_numpy()
        errors = np.stack([value - table[prefix + "lo"], table[prefix + "hi"] - value])
        effects.errorbar(value, table.row + offset, xerr=errors, fmt="o" if base else "s",
                         markersize=3.1, color=INK if base else BLUE,
                         markerfacecolor=INK if base else "white", markeredgewidth=.78,
                         elinewidth=.85, capsize=0, zorder=3)

    ranking = table.dropna(subset=["min_rank"])
    ranks.scatter(ranking.min_rank, ranking.row, s=11.5, color=INK, zorder=2)
    ranks.text(.455, 10, "Mask held fixed", va="center", color="#4a4a4a")
    effects.set_yticks(range(11), ["Endpoint density, raw", "Endpoint density, cleaned (all)",
                                   "Fragility, random removal", "Fractal dimension", "Fragility, targeted removal",
                                   "Junction density, raw", "Junction density, cleaned", "Vessel area density",
                                   "Loop density", "Coverage fragility", "FAZ area (control)"])
    for ax in (effects, ranks):
        ax.set_ylim(10.54, -.55)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", length=2.6, pad=3)
        ax.spines["left"].set_visible(False)
        ax.axhline(9.55, color="#adb1b5", linewidth=.6)
        for row in range(10):
            ax.axhline(row, color="#e7e9eb", linewidth=.45, zorder=0)
    effects.axvline(0, color="#777b80", linewidth=.7, zorder=1)
    effects.set_xlim(-1.68, 1.38)
    effects.set_xticks([-1.5, -1, -.5, 0, .5, 1])
    effects.set_title("A   Adjusted DR–healthy difference", loc="left", fontweight="bold", pad=6)
    effects.set_xlabel("Difference (within-field SD)")
    ranks.vlines(.8, -.55, 9.55, color="#777b80", linestyle=(0, (3, 2)), linewidth=.8, zorder=1)
    ranks.set_xlim(.42, 1.025)
    ranks.set_xticks([.5, .6, .7, .8, .9, 1.0])
    ranks.set_title("B   Rank stability", loc="left", fontweight="bold", pad=6)
    ranks.set_xlabel("Minimum Spearman ρ")
    handles = [Line2D([], [], color=INK, marker="o", linestyle="-", linewidth=.8,
                      markersize=3.5, label="Length density + scan size"),
               Line2D([], [], color=BLUE, marker="s", markerfacecolor="white", linewidth=.8,
                      markersize=3.5, label="Additional image-quality adjustment")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .006), ncol=2,
               handlelength=1.8, columnspacing=2.4, handletextpad=.55, borderpad=0)
    return fig


def endpoint_overlap_figure():
    eyes = pd.read_csv(DATA / "endpoint_overlap.csv", float_precision="round_trip")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.65))
    fig.subplots_adjust(left=.091, right=.984, top=.875, bottom=.32, wspace=.21)
    band_color = "#e4e7e8"
    groups = [("NORMAL", "Healthy", BLUE, "o", "white", "-"),
              ("DR", "Diabetic retinopathy", ORANGE, "^", ORANGE, "--")]
    for field, letter, ax in zip((3, 6), ("A", "B"), axes):
        field_eyes = eyes.loc[eyes.fov_mm.eq(field)]
        upper_quartile = field_eyes.length_mm_per_mm2.quantile(.75)
        ax.axvspan(upper_quartile - 1, upper_quartile + 1, color=band_color, linewidth=0, zorder=0)
        for diagnosis, _, color, marker, face, line in groups:
            observed = field_eyes.loc[field_eyes.disease.eq(diagnosis)]
            x = observed.length_mm_per_mm2.to_numpy()
            y = observed.interior_endpoints_per_mm2.to_numpy()
            ax.scatter(x, y, marker=marker, s=10.5 if diagnosis == "NORMAL" else 14,
                       facecolors=face, edgecolors=color, linewidths=.6, alpha=.80, zorder=2)
            fit = np.polynomial.Polynomial.fit(x, y, 1)
            limits = np.array([x.min(), x.max()])
            ax.plot(limits, fit(limits), color=color, linestyle=line, linewidth=1.05, zorder=3)
        n = field_eyes.disease.value_counts()
        ax.set_title(f"{letter}   {field} × {field} mm", loc="left", fontweight="bold", pad=5)
        ax.set_title(f"{n['NORMAL']} healthy, {n['DR']} DR", loc="right", fontsize=8.6, pad=5)
        ax.grid(axis="y", color="#e0e3e4", linewidth=.45)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.6, pad=2.5)
        if field == 3:
            ax.set(xlim=(10, 34), xticks=[10, 15, 20, 25, 30], ylim=(0, 64), yticks=[0, 20, 40, 60])
        else:
            ax.set(xlim=(6, 21), xticks=[6, 9, 12, 15, 18, 21], ylim=(0, 34), yticks=[0, 10, 20, 30])
    axes[0].set_ylabel("Interior endpoints per mm²")
    fig.text(.535, .145, "Traced vessel length density (mm/mm²)", ha="center")
    handles = [Line2D([], [], marker=marker, markerfacecolor=face, markeredgecolor=color,
                      markersize=4, linestyle=line, color=color, linewidth=1.05, label=label)
               for _, label, color, marker, face, line in groups]
    handles.append(Patch(facecolor=band_color, edgecolor="none", label="Upper-quartile window"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.535, .012), ncol=3,
               handlelength=1.7, columnspacing=1.8, handletextpad=.5, borderpad=0)
    return fig


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9.25, "axes.labelsize": 9.25,
                         "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "axes.edgecolor": "#62666a", "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                         "legend.frameon": False, "legend.fontsize": 9.25, "pdf.fonttype": 42})
    save(audit_figure(), "metric_audit")
    save(tracing_figure(), "retinal_tracing")
    save(endpoint_overlap_figure(), "endpoint_overlap")
