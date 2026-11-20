"""Illustrate one graph deletion and plot the cohort's targeted-removal results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "plot_data"
INDEX = "FI_adaptive_betweenness"


def draw_example(ax, x_center, bridge):
    """Draw two junctions with two terminal branches each, all edges equally long."""
    scale = .46
    x_left, x_right = x_center - scale / 2, x_center + scale / 2
    y_center = .56
    ink = "#283c46"
    node_x, node_y = [], []
    for x, direction in [(x_left, -1), (x_right, 1)]:
        node_x.append(x)
        node_y.append(y_center)
        for side in (-1, 1):
            end_x = x + direction * .8 * scale
            end_y = y_center + side * .6 * scale
            ax.plot([x, end_x], [y_center, end_y], color=ink, lw=1.65, zorder=1)
            node_x.append(end_x)
            node_y.append(end_y)
    if bridge:
        ax.plot([x_left, x_right], [y_center, y_center], color="#a24a21", lw=3, zorder=1)
    ax.scatter(node_x, node_y, s=17, facecolors="white", edgecolors=ink, linewidths=1, zorder=2)


def main():
    eyes = pd.read_csv(DATA / "fragility_points.csv", float_precision="round_trip")
    curve = pd.read_csv(DATA / "fragility_curve.csv", float_precision="round_trip")
    grid = curve.edges_removed_fraction.to_numpy()
    low, median, high = curve[["q25", "median", "q75"]].to_numpy().T

    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 9.25, "axes.titlesize": 10,
        "axes.labelsize": 9.25, "pdf.fonttype": 42, "legend.fontsize": 9.25,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
        "axes.edgecolor": "#62666a", "axes.labelcolor": "#20242a", "text.color": "#20242a",
    })
    fig, (trajectory, density) = plt.subplots(1, 2, figsize=(7, 3.56))
    fig.subplots_adjust(left=.105, right=.986, top=.548, bottom=.146, wspace=.355)
    example = fig.add_axes([.04, .648, .93, .35])
    example.set(xlim=(0, 6.51), ylim=(0, 1.24), aspect="equal")
    example.axis("off")
    example.text(0, 1.20, "A   Five equal-length vessel segments", va="top",
                 fontsize=10, fontweight="bold")
    draw_example(example, .92, True)
    draw_example(example, 3.0, False)
    example.annotate("", xy=(2.18, .56), xytext=(1.74, .56),
                     arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#62666a"})
    example.text(.92, .06, "Intact graph", ha="center", va="bottom")
    example.text(3.0, .06, "Bridge removed", ha="center", va="bottom")
    example.text(4.0, .81, "9 shortest paths cross the bridge.", va="center")
    example.text(4.0, .56, "5 cross each terminal branch.", va="center")
    example.text(4.0, .25, "Largest remaining share = 2/5", va="center", fontweight="bold")
    for ax in (trajectory, density):
        ax.grid(axis="y", color="#e2e4e6", linewidth=.45)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.6, pad=3)

    band = trajectory.fill_between(100 * grid, low, high, color="#b9cbd3", linewidth=0)
    line, = trajectory.plot(100 * grid, median, color="#25383f", linewidth=1.5)
    trajectory.set_title("B   Targeted edge removal", loc="left", fontweight="bold", pad=6)
    trajectory.set_xlabel("Original edges removed (%)")
    trajectory.set_ylabel("Fraction of initial length\nin largest component")
    trajectory.set_xlim(0, 30)
    trajectory.set_ylim(0, 1.02)
    trajectory.set_xticks(range(0, 31, 5))
    trajectory.set_yticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1"])
    trajectory.legend([line, band], ["Median", "Interquartile range"], loc="upper right",
                      frameon=False, borderpad=0, handlelength=1.65, handletextpad=.5)

    for field, marker, face, edge in [(3, "o", "white", "#3d4b56"), (6, "^", "#79618d", "white")]:
        group = eyes[eyes.fov_mm == field]
        density.scatter(100 * group.vessel_density, group[INDEX], s=12.5, marker=marker,
                        facecolors=face, edgecolors=edge, linewidths=.58, label=f"{field} mm", zorder=2)
        rho = group.vessel_density.rank().corr(group[INDEX].rank())
        print(f"{field} mm: n={len(group)}, Spearman rho={rho:.5f}")
    rho = eyes.vessel_density.rank().corr(eyes[INDEX].rank())
    print(f"Pooled: n={len(eyes)}, Spearman rho={rho:.5f}")
    density.set_title("C   Fragility and vessel area", loc="left", fontweight="bold", pad=6)
    density.set_xlabel("Vessel area density (%)")
    density.set_ylabel("Fragility index")
    density.set_ylim(.71, .98)
    density.set_yticks([.75, .80, .85, .90, .95])
    density.set_xlim(17, 54)
    density.set_xticks([20, 30, 40, 50])
    density.legend(loc="upper right", frameon=False, borderpad=0,
                   handlelength=.9, handletextpad=.4, labelspacing=.23)
    fig.savefig(HERE / "fragility.pdf", metadata={"Title": "Targeted edge removal and vessel area density",
                                                "CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
