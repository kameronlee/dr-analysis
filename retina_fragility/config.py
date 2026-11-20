"""Configuration: one YAML file deep-merged over documented defaults.

Every key that the package reads has a default here. ``config.yaml`` at the
project root is the single user-editable file; unknown keys are rejected so
that typos cannot silently fall back to defaults.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "project": {
        "base_seed": 20260929,
        "output_root": "outputs",
        "n_workers": 1,
    },
    "data": {
        "manifest": "data/manifests/octa500_manifest.csv",
        # Input roots are treated as read-only; outputs may not live inside them.
        "input_roots": ["data/raw"],
        # Explicit label mappings. mode "labels": pixel is foreground iff value in
        # `values`; every other value must be listed in `background_values` or the
        # mask is rejected. mode "binary": mask must contain exactly two values
        # {0, x}; x is foreground.
        "label_maps": {
            "vessel": {"mode": "binary", "values": [], "background_values": [0]},
            "large_vessel": {"mode": "binary", "values": [], "background_values": [0]},
            "faz": {"mode": "binary", "values": [], "background_values": [0]},
        },
        "disease_map": {"NORMAL": "normal", "DR": "dr"},
        # Explicit sex coding -> sex_male indicator. Unmapped non-empty values are errors.
        "sex_map": {"M": 1, "F": 0, "Male": 1, "Female": 0, "male": 1, "female": 0},
        # Dataset label of the primary cohort (manifest `dataset` column).
        "primary_dataset": "OCTA500",
        "subject_id_policy": "image_is_subject",  # or "require"
        "expected_geometry": {
            "3": {"shape": [304, 304], "pixel_size_um": 3000.0 / 304.0},
            "6": {"shape": [400, 400], "pixel_size_um": 15.0},
        },
        "pixel_size_rel_tolerance": 0.005,
    },
    "calibration": {
        "n_per_fov": {"3": 10, "6": 10},
        "eligible_group": "normal",
        "percentile": 95.0,
        "weighting": "equal_image",  # or "pixel"
        "frozen_file": "outputs/calibration/calibration.json",
    },
    "qc": {
        "octa500_graph_n_per_fov": {"3": 10, "6": 10},
        # Outcome-blind ordering key purpose: sha256(base_seed|purpose|image_id).
        "octa500_graph_selection_purpose": "OCTA_GRAPH_QC",
    },
    # OCTA-500 data contract (docs/OCTA500_PREANALYSIS_NOTE_2026-09-30.md). Column
    # names refer to data/manifests/octa500_source_catalog.csv (`<name>_path`/`_sha256`).
    "octa500": {
        "source_catalog": "data/manifests/octa500_source_catalog.csv",
        # Paper B5 = OCTA maximum projection ILM-OPL: the plane the labels were drawn on.
        "primary_projection": "octa_ilm_opl",
        "context_projections": ["octa_full", "octa_opl_bm"],
        "vessel_label": "capillary",
        "large_vessel_label": "large_vessel",
        "faz_label": "faz",
        # Label-availability split: every OCTA-500 disease label is public.
        "label_split": "public",
        # Official segmentation split (paper section 5.1; inclusive subject-number ranges).
        "seg_split_ranges": {
            "train": [[10001, 10240], [10301, 10440]],
            "val": [[10241, 10250], [10441, 10450]],
            "test": [[10251, 10300], [10451, 10500]],
        },
        # Prespecified confirmatory alignment check (note D1).
        "alignment_min_identity_fraction": 0.95,
    },
    "analysis_mask": {
        # Band (in µm) along the image border excluded from the analysis area.
        "border_band_um": 50.0,
        "exclude_faz": True,
    },
    "graph": {
        "junction_merge_distance_px": 2.0,  # strictly closer than this -> merged
        "force_2x2_blocks_as_junctions": True,
        "border_band_px": 1,  # edges with any pixel within this ring are dropped
        "faz_inside_fraction": 0.5,
        "spur_length_factor": 2.0,
        "spur_prune_passes": 1,
        "min_fragment_length_um": 50.0,
        # Pixel-unit mode (external data of unknown scale): used instead of the
        # µm threshold when pixel_size_um is None.
        "min_fragment_length_px": None,
    },
    "stress": {
        "step_frac": 0.01,
        "random_repetitions": 100,
        "orders": ["adaptive_betweenness", "random", "static_betweenness", "width", "forman"],
        "primary_order": "adaptive_betweenness",
        "betweenness_backend": "igraph",  # or "reference"
        "tie_rel_quantum": 1e-9,
        "q_eval": 0.10,
        "s_threshold": 0.5,
        "compute_coverage": True,
        "coverage_orders": ["adaptive_betweenness", "random"],
        # vessel_territory: edge owns the vessel-mask pixels nearest to its skeleton
        # pixels (consistent with the calibration distance to vessel pixels);
        # centerline: edge pixel path only (labelled sensitivity).
        "coverage_geometry": "vessel_territory",
    },
    "pipeline": {
        # Central square crop (mm) applied to masks before processing (FOV ablation).
        "crop_center_mm": None,
        "save_raw_graph": False,
        "overlays": True,
        "skip_existing": True,
    },
    "stats": {
        "n_boot": 2000,
        "alpha": 0.05,
        "fi_column": "FI_adaptive_betweenness",
        "baseline_primary": ["fov_6mm", "age", "sex_male", "vessel_density", "faz_area_mm2",
                             "loop_density_per_mm2"],
        "sensitivity_sets": {
            "fractal_substituted": ["fov_6mm", "age", "sex_male", "vessel_density",
                                    "faz_area_mm2", "fractal_dimension"],
            "fractal_added": ["fov_6mm", "age", "sex_male", "vessel_density", "faz_area_mm2",
                              "loop_density_per_mm2", "fractal_dimension"],
        },
        "standardize_continuous": True,
        "min_events_per_class": 10,
        "separation_prob_tol": 1e-8,
        "max_abs_coef_sd": 15.0,
        "cv": {"n_splits": 5, "n_repeats": 10, "C": 1.0},
        "holm_families": {
            "H1_H2_secondary": ["A:FI_adaptive_betweenness", "D:G10", "D:qc_adaptive_betweenness"],
            "H4": ["C:wilcoxon_z_gt_0"],
        },
    },
    "exp_a": {
        "features": ["FI_adaptive_betweenness", "FI_random", "FIperf_adaptive_betweenness",
                     "vessel_density", "loop_density_per_mm2", "fractal_dimension"],
        "curve_grid": 101,
    },
    "exp_c": {
        "repetitions": 50,
        "density_metric": "graph_length_density_mm_per_mm2",
        "tolerance_rel": 0.02,
        "max_donors": None,
        "fi_order": "adaptive_betweenness",
    },
    "exp_d": {"orders": ["adaptive_betweenness", "static_betweenness", "width", "forman"]},
    "exp_e": {
        "manifest": "data/manifests/drac_manifest.csv",
        "labels_corrected_version": False,  # must be set True explicitly
        "pixel_size_um": None,  # None -> pixel-unit mode; never assumed = OCTA-500
        "fov_mm": None,
        "geometry_source": None,
        "min_fragment_length_px": 5.0,
        "faz": {"mode": "disk", "radius_px": 20, "center": "image_center"},
        "min_qc_audited": 30,
        # Prespecified before any QC decision exists (resume_review.md B4).
        "max_qc_fail_fraction": 0.2,
        "segmenters": ["frangi", "unet"],
        # DRAC has no age/sex: image biomarkers only. Constant columns (e.g. FAZ
        # area under a fixed disk) are dropped with a logged reason.
        "baseline": ["vessel_density", "faz_area_mm2", "loop_density_per_mm2"],
        "n_permutations": 10000,
        "grouping_level": "image",
        "cv": {"n_splits": 5, "n_repeats": 10, "C": 1.0},
    },
    "exp_f": {
        "embedding_models": ["resnet50", "dinov2_vits14"],
        "n_splits": 5,
        "C": 1.0,
        "fragility_features": ["FI_adaptive_betweenness", "FI_random",
                               "G10_adaptive_betweenness", "qc_adaptive_betweenness"],
        "segmenter": "frangi",
    },
    "embeddings": {
        "image_size": 224,
        "batch_size": 16,
        # "imagenet" requires locally available torchvision weights; nothing is
        # downloaded unless allow_download is true.
        "weights": "imagenet",
        "allow_download": False,
        "device": "cpu",
    },
    "segmentation": {
        # Pre-outcome amendment 2026-09-30 (resume_review.md D6): sigma²-normalised
        # Frangi (1998) with one structuredness constant across scales, sigmas up to 6 px
        # (DRAC major vessels ~6-12 px wide; sigma ~ w / (2 sqrt 3)). The pre-amendment
        # v1 masks used implementation skimage_unnormalized with sigmas [1, 1.5, 2, 3].
        "frangi": {"implementation": "frangi1998_normalized",
                   "sigmas": [1.0, 1.5, 2.0, 3.0, 4.0, 6.0], "beta": 0.5, "black_ridges": False,
                   "threshold": "otsu", "threshold_scale": 1.0, "min_object_px": 20},
        "unet": {"base_channels": 16, "depth": 4, "epochs": 60, "lr": 1e-3,
                 "batch_size": 4, "patch_size": 256, "threshold": 0.5,
                 "threshold_scale": 1.0, "device": "cpu", "seed_purpose": "unet"},
    },
    "experiment0": {
        "full": {"grid_n": 30, "tree_depth": 9, "seeds": 20},
        "quick": {"grid_n": 12, "tree_depth": 6, "seeds": 5},
        "random_repetitions_per_seed": 1,
    },
    "ablations": {
        "step_frac": [0.01, 0.02, 0.05],
        "spur_length_factor": [0.0, 1.0, 2.0],
        "min_fragment_length_um": [0.0, 50.0, 100.0],
        "d_percentile": [50.0, 75.0, 95.0],
        "seg_threshold_scale": [0.9, 1.0, 1.1],
        "crop_center_mm": [None, 3.0],
        "coverage_geometry": ["vessel_territory", "centerline"],
    },
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        full = f"{path}.{key}" if path else str(key)
        if key not in base:
            # free-form maps whose keys are data-defined
            if path.split(".")[-1] in {"label_maps", "disease_map", "expected_geometry",
                                       "n_per_fov", "sensitivity_sets", "holm_families",
                                       "sex_map"}:
                out[key] = copy.deepcopy(val)
                continue
            raise ConfigError(f"unknown config key: {full}")
        if isinstance(base[key], dict) and isinstance(val, dict):
            out[key] = _merge(base[key], val, full)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load config.yaml (if given) over DEFAULTS, then apply ``overrides``."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path) as fh:
            user = yaml.safe_load(fh) or {}
        cfg = _merge(cfg, user)
    if overrides:
        cfg = _merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    s = cfg["stress"]
    if not (0 < s["step_frac"] <= 1):
        raise ConfigError("stress.step_frac must be in (0, 1]")
    if s["primary_order"] not in s["orders"]:
        raise ConfigError("stress.primary_order must be one of stress.orders")
    if s["betweenness_backend"] not in {"igraph", "reference"}:
        raise ConfigError("stress.betweenness_backend must be igraph or reference")
    if cfg["data"]["subject_id_policy"] not in {"image_is_subject", "require"}:
        raise ConfigError("data.subject_id_policy must be image_is_subject or require")
    g = cfg["graph"]
    if g["spur_length_factor"] < 0 or g["min_fragment_length_um"] < 0:
        raise ConfigError("graph thresholds must be non-negative")
    if s["coverage_geometry"] not in {"vessel_territory", "centerline"}:
        raise ConfigError("stress.coverage_geometry must be vessel_territory or centerline")
    unknown = set(s["orders"]) - {"adaptive_betweenness", "random", "static_betweenness",
                                  "width", "forman"}
    if unknown:
        raise ConfigError(f"unknown stress orders: {sorted(unknown)}")
    for k, v in cfg["data"]["sex_map"].items():
        if v not in (0, 1):
            raise ConfigError(f"data.sex_map values must be 0/1, got {k}: {v}")
    if not (0 < cfg["calibration"]["percentile"] < 100):
        raise ConfigError("calibration.percentile must be in (0, 100)")
    _validate_octa500(cfg)
    if cfg["segmentation"]["frangi"]["implementation"] not in {"frangi1998_normalized",
                                                               "skimage_unnormalized"}:
        raise ConfigError("segmentation.frangi.implementation must be frangi1998_normalized or "
                          "skimage_unnormalized")


OCTA500_PROJECTIONS = ("oct_full", "oct_ilm_opl", "oct_opl_bm", "octa_full", "octa_ilm_opl",
                       "octa_opl_bm")
OCTA500_BINARY_LABELS = ("artery", "capillary", "faz", "large_vessel", "vein")


def _validate_octa500(cfg: dict) -> None:
    o = cfg["octa500"]
    projections = [o["primary_projection"], *o["context_projections"]]
    bad = sorted(set(projections) - set(OCTA500_PROJECTIONS))
    if bad or len(set(projections)) != len(projections):
        raise ConfigError(f"octa500 projections must be distinct members of {OCTA500_PROJECTIONS}")
    for key in ("vessel_label", "large_vessel_label", "faz_label"):
        if o[key] is not None and o[key] not in OCTA500_BINARY_LABELS:
            raise ConfigError(f"octa500.{key} must be one of {OCTA500_BINARY_LABELS}")
    if o["vessel_label"] is None or o["faz_label"] is None:
        raise ConfigError("octa500.vessel_label and octa500.faz_label are required")
    if str(o["label_split"]).strip().lower() == "test":
        raise ConfigError("octa500.label_split may not be 'test': OCTA-500 disease labels are public")
    ranges = o["seg_split_ranges"]
    if set(ranges) != {"train", "val", "test"}:
        raise ConfigError("octa500.seg_split_ranges needs exactly train, val and test")
    spans = []
    for name, items in ranges.items():
        for item in items:
            if len(item) != 2 or int(item[0]) > int(item[1]):
                raise ConfigError(f"octa500.seg_split_ranges.{name}: invalid range {item}")
            spans.append((int(item[0]), int(item[1]), name))
    spans.sort()
    for (a0, a1, an), (b0, b1, bn) in itertools.pairwise(spans):
        if b0 <= a1:
            raise ConfigError(f"octa500.seg_split_ranges overlap: {an} {a0}-{a1} and {bn} {b0}-{b1}")
    if not (0 < float(o["alignment_min_identity_fraction"]) <= 1):
        raise ConfigError("octa500.alignment_min_identity_fraction must be in (0, 1]")


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def dump_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
