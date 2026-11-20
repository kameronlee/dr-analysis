"""M12: FI-blind, disease/FOV-stratified selection rule and panel of representative fragility maps.

Rule (fixed 2026-10-06 before any clinical outcome, as an automated implementation
decision accepted, ``reports/m12_selection_rule_2026-10-06.json``; not a
human review): within each prespecified disease-group (NORMAL/DR) × FOV stratum
of the analysis rows (calibration eyes excluded), order images by
``sha256(base_seed|paper_fragility_map_panel_v1|image_id)`` and take the first
``n_per_cell`` (= 1) images whose three per-image overlays exist. The disease
label is used only to define the strata. Within strata the rule is blind to FI,
every other feature, removal curves and association results; overlay existence
depends only on whether processing produced a graph. Images skipped for missing
overlays and strata without an eligible image are recorded; no substitute is
chosen by FI, association results or visual appeal. Selection is recorded
before the panel is drawn.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .cohort import _fov, _group

PURPOSE = "paper_fragility_map_panel_v1"
RULE_STATUS = ("fixed 2026-10-06 before clinical outcomes as an automated implementation decision accepted"
               "(reports/m12_selection_rule_2026-10-06.json); not human review; caption attribution pending")
PANELS = (("overlay_bridge_cutoff", "Bridge cut-off length (µm)"),
          ("overlay_unique_coverage", "Unique coverage (10⁻³ mm²)"),
          ("overlay_primary_removal_q", "Removal fraction q (primary order)"))
ATTRIBUTION = ("Background projections and labels: OCTA-500 (Li et al., Med Image Anal 2024;93:103092, "
               "doi:10.1016/j.media.2024.103092), CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); "
               "graph extraction, overlays and colour maps by this project. Final caption attribution "
               "review is pending (docs/DATASET_TERMS_2026-09-30.md).")


def selection_key(base_seed: int, image_id: str) -> str:
    return hashlib.sha256(f"{base_seed}|{PURPOSE}|{image_id}".encode()).hexdigest()


def overlay_files(image_dir: Path) -> dict[str, Path] | None:
    out = {}
    for stem, _ in PANELS:
        p = image_dir / f"{stem}.png"
        if not p.is_file():
            p = image_dir / f"{stem}_on_mask.png"
        if not p.is_file():
            return None
        out[stem] = p
    return out


def select(rows: pd.DataFrame, pipeline_root: Path, base_seed: int,
           n_per_cell: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(selected images, per-stratum accounting of skipped and unfilled positions)."""
    from ..pipeline import safe_name

    t = rows.assign(group=_group(rows.analysis_group), fov=_fov(rows.fov_mm),
                    key=rows.image_id.astype(str).map(lambda i: selection_key(base_seed, i)))
    chosen, accounting = [], []
    for g in ("NORMAL", "DR"):
        for f in sorted(t.fov.unique()):
            sub = t[(t.group == g) & (t.fov == f)]
            taken = skipped = 0
            for r in sub.sort_values("key").itertuples():
                files = overlay_files(pipeline_root / "images" / safe_name(r.image_id))
                if files is None:
                    skipped += 1
                    continue
                chosen.append({"cell": f"{g} {f}", "rank_in_cell": taken + 1, "image_id": r.image_id,
                               "selection_key": r.key, **{k: str(v) for k, v in files.items()}})
                taken += 1
                if taken == n_per_cell:
                    break
            accounting.append({"stratum": f"{g} {f}", "eligible_rows": int(len(sub)), "selected": taken,
                               "skipped_missing_overlays": skipped,
                               "status": "filled" if taken == n_per_cell else "unfilled: no eligible image"})
    return pd.DataFrame(chosen), pd.DataFrame(accounting)


def panel_figure(selection: pd.DataFrame, note: str):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    n = max(len(selection), 1)
    fig, axes = plt.subplots(n, len(PANELS), figsize=(3.4 * len(PANELS), 3.3 * n + 0.9), squeeze=False)
    for i, r in enumerate(selection.itertuples()):
        for j, (stem, title) in enumerate(PANELS):
            ax = axes[i][j]
            ax.imshow(mpimg.imread(getattr(r, stem)))
            ax.set_axis_off()
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.text(-0.04, 0.5, f"{r.cell} #{r.rank_in_cell}", transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=8)
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=6.5, wrap=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.07, wspace=0.05, hspace=0.12)
    return fig
