"""Render the scan and registered graph geometry in the retinal tracing figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.patheffects as pe
import numpy as np


HERE = Path(__file__).resolve().parent
MASK = "#e9e9e9"
CENTERLINE = "#74dce6"
ENDPOINT = "#ffbd55"
INK = "#20242a"


def mask_edges(mask):
    """Boundary segments on the exact outlines of the foreground pixel squares."""
    segments = []
    padded = np.pad(mask, 1)
    for y, x in zip(*np.nonzero(mask)):
        for dy, dx, start, stop in (
            (-1, 0, (-.5, -.5), (.5, -.5)), (1, 0, (-.5, .5), (.5, .5)),
            (0, -1, (-.5, -.5), (-.5, .5)), (0, 1, (.5, -.5), (.5, .5)),
        ):
            if not padded[y + 1 + dy, x + 1 + dx]:
                segments.append(((x + start[0], y + start[1]), (x + stop[0], y + stop[1])))
    return segments


def scale(ax, label, length, shape):
    height, width = shape
    x, y = .052 * width - .5, .935 * height - .5
    border = [pe.withStroke(linewidth=2.9, foreground="#181818")]
    ax.plot([x, x + length], [y, y], lw=1.6, color="white", solid_capstyle="butt",
            path_effects=border)
    ax.annotate(label, (x, y), xytext=(0, 3), textcoords="offset points",
                color="white", fontsize=8.5, va="bottom",
                path_effects=[pe.withStroke(linewidth=1.55, foreground="#202020")])


def tracing_figure():
    with np.load(HERE / "plot_data/retinal_tracing.npz") as data:
        scan = data["scan"]
        top, left = int(data["top"]), int(data["left"])
        pixel_size = float(data["pixel_size_um"])
        h, w = data["reference_mask"].shape
        crop = scan[top:top + h, left:left + w]
        # Each box shows the exact field enlarged in the next panel.
        context = scan[top - h:top + 2*h, left - w:left + 2*w]
        fig, axes = plt.subplots(1, 4, figsize=(7, 2.24))
        fig.subplots_adjust(left=.004, right=.996, top=.88, bottom=.125, wspace=.035)
        labels = ["Macular angiogram", "Magnified region", "Reference mask", "One-pixel dilation"]
        for ax, letter, label in zip(axes, "ABCD", labels):
            ax.set_axis_off()
            ax.set_title(f"{letter}  {label}", loc="left", color=INK, fontweight="bold",
                         fontsize=9.4, pad=4)
        for ax, arr in zip(axes, (scan, context, crop, crop)):
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255, interpolation="none")
        for ax, (x, y, width, height) in zip(axes[:2], [
            (left - w, top - h, 3*w, 3*h), (w, h, w, h),
        ]):
            ax.add_patch(Rectangle((x - .5, y - .5), width, height, fill=False, edgecolor="white",
                                   linewidth=.9, path_effects=[pe.withStroke(linewidth=2.1,
                                                                             foreground="#242424")]))
        scale(axes[0], "500 µm", 500 / pixel_size, scan.shape)
        scale(axes[1], "100 µm", 100 / pixel_size, context.shape)

        for ax, condition in zip(axes[2:], ["reference", "dilated"]):
            mask = data[f"{condition}_mask"]
            ax.add_collection(LineCollection(mask_edges(mask), colors=MASK, linewidths=.52))
            skeleton = data[f"{condition}_centerline"]
            ax.imshow(np.ma.masked_array(np.ones_like(skeleton), mask=~skeleton),
                      cmap=ListedColormap([CENTERLINE]), interpolation="none", alpha=.74)
            ends = data[f"{condition}_endpoints"]
            if len(ends):
                circles = ax.scatter(ends[:, 0], ends[:, 1], facecolors="none", edgecolors=ENDPOINT,
                                     s=37, linewidths=1.3, zorder=6)
                circles.set_path_effects([pe.withStroke(linewidth=2.4, foreground="#393022")])
            # Preserve the original pixel extent after adding vector overlays.
            ax.set_xlim(-.5, w - .5)
            ax.set_ylim(h - .5, -.5)
        scale(axes[2], "50 µm", 50 / pixel_size, crop.shape)

        handles = [Line2D([], [], color=MASK, linewidth=.88, label="Vessel-mask boundary",
                          path_effects=[pe.withStroke(linewidth=2.1, foreground="#595959")]),
                   Patch(facecolor=CENTERLINE, edgecolor="#477e84", linewidth=.6,
                         label="Cleaned centerline pixels"),
                   Line2D([], [], color=ENDPOINT, marker="o", markeredgewidth=1.25, markerfacecolor="none",
                          linestyle="none", markersize=5.5, label="Interior endpoint",
                          path_effects=[pe.withStroke(linewidth=1.95, foreground="#393022")])]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 0), ncol=3,
                   borderaxespad=.05, frameon=False, handlelength=1.45,
                   handletextpad=.5, columnspacing=2, fontsize=9.25)
    return fig


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "Liberation Sans", "pdf.fonttype": 42})
    fig = tracing_figure()
    fig.savefig(HERE / "retinal_tracing.pdf", dpi=300,
                metadata={"Title": "Retinal tracing before and after mask dilation",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)
