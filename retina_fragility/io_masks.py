"""Read-only image inputs and explicit mask decoding from the reviewed protocol."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


class MaskError(ValueError):
    """An input cannot be interpreted without an undeclared assumption."""


def read_plane(path: str | Path) -> np.ndarray:
    """Read a single 2D image without silently converting RGB or image stacks."""
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise MaskError(f"Expected one 2D frame: {path}")
        # np.asarray preserves label values, including indexed palette labels.
        array = np.array(image)
    if array.ndim != 2 or not all(array.shape):
        raise MaskError(f"Expected a nonempty 2D plane, got {array.shape}: {path}")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise MaskError(f"Unsupported pixel dtype {array.dtype}: {path}")
    if not np.isfinite(array).all():
        raise MaskError(f"Nonfinite input pixels: {path}")
    return array


def decode_mask(array: np.ndarray, mapping: Mapping) -> np.ndarray:
    """Decode binary values or an exhaustive, explicitly declared label map.

    A binary image may be all background or all foreground, but has at most
    one positive foreground value. Multiple positive values require a label
    map. An all-zero FAZ remains empty; it is not silently replaced by a disk.
    """
    array = np.asarray(array)
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise MaskError("Mask must be a nonempty, finite 2D plane")
    present = set(np.unique(array).tolist())
    mode = mapping.get("mode")
    if mode == "binary":
        foreground = present - {0}
        if len(foreground) > 1 or any(v <= 0 for v in foreground):
            raise MaskError(f"Binary mask has undeclared label values: {sorted(present)}")
        return array != 0
    if mode != "labels":
        raise MaskError(f"Unknown label mapping mode: {mode!r}")
    foreground = set(mapping.get("values", []))
    background = set(mapping.get("background_values", []))
    if not foreground or foreground & background:
        raise MaskError("Label map needs foreground values disjoint from background values")
    unknown = present - foreground - background
    if unknown:
        raise MaskError(f"Mask contains unmapped values: {sorted(unknown)}")
    return np.isin(array, list(foreground))


def read_mask(path: str | Path, mapping: Mapping) -> np.ndarray:
    return decode_mask(read_plane(path), mapping)


def load_reference_masks(
    vessel_path: str | Path,
    faz_path: str | Path,
    label_maps: Mapping,
    large_vessel_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Union distinct capillary/large-vessel inputs; require reference FAZ."""
    vessel = read_mask(vessel_path, label_maps["vessel"])
    faz = read_mask(faz_path, label_maps["faz"])
    if faz.shape != vessel.shape:
        raise MaskError("Vessel and FAZ mask dimensions differ")
    if large_vessel_path is not None:
        large = read_mask(large_vessel_path, label_maps["large_vessel"])
        if large.shape != vessel.shape:
            raise MaskError("Large-vessel and capillary mask dimensions differ")
        vessel = vessel | large
    return vessel, faz


def read_intensity(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read an OCTA projection as float32 in [0, 1] using the dtype's nominal range.

    No per-image min/max stretching is applied, so thresholds stay comparable
    between images. Returns (image, info) where info records the source dtype.
    """
    array = read_plane(path)
    info = {"dtype": str(array.dtype), "shape": list(array.shape)}
    if array.dtype == np.bool_:
        out = array.astype(np.float32)
    elif np.issubdtype(array.dtype, np.integer):
        top = float(np.iinfo(array.dtype).max)
        if array.min() < 0:
            raise MaskError(f"Signed intensity image with negative values: {path}")
        out = array.astype(np.float32) / top
    else:
        lo, hi = float(array.min()), float(array.max())
        if lo < 0 or hi > 1:
            raise MaskError(f"Floating intensity image outside [0, 1]: {path}")
        out = array.astype(np.float32)
    info["nominal_range_scaled"] = True
    return out, info


def write_mask_png(path: str | Path, mask: np.ndarray) -> None:
    """Write a boolean mask as an 8-bit 0/255 PNG (decodable in binary mode)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(np.asarray(mask, bool), 255, 0).astype(np.uint8)).save(path)


def faz_disk_mask(shape: tuple[int, int], radius_px: float,
                  center: str | tuple[float, float] = "image_center") -> np.ndarray:
    """Fixed-disk FAZ used for external data without FAZ labels (configured, not inferred)."""
    H, W = shape
    if center == "image_center":
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    else:
        cy, cx = float(center[0]), float(center[1])
    if not (radius_px >= 0 and np.isfinite(radius_px)):
        raise MaskError("FAZ disk radius must be finite and non-negative")
    yy, xx = np.mgrid[:H, :W]
    return np.hypot(yy - cy, xx - cx) <= float(radius_px)


def faz_auto_mask(vessel: np.ndarray, closing_radius_px: int = 3,
                  max_center_distance_frac: float = 0.15,
                  max_area_frac: float = 0.05) -> tuple[np.ndarray, dict]:
    """Automatic FAZ: vessel-free component (after closing) nearest the image centre.

    Returns (mask, info). An implausible candidate (too far from the centre or
    too large) yields an empty mask plus a reason, never a silent substitute.
    """
    from skimage.morphology import binary_closing, disk

    v = np.asarray(vessel, bool)
    H, W = v.shape
    closed = binary_closing(v, disk(int(closing_radius_px))) if closing_radius_px > 0 else v
    lab, n = ndi.label(~closed)
    info = {"method": "auto", "closing_radius_px": int(closing_radius_px)}
    if n == 0:
        info["reason"] = "no_vessel_free_region"
        return np.zeros_like(v), info
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    idx = np.arange(1, n + 1)
    com = ndi.center_of_mass(np.ones_like(v, dtype=float), lab, idx)
    dist = np.array([np.hypot(a - cy, b - cx) for a, b in com])
    best = int(idx[int(np.argmin(dist))])
    mask = lab == best
    info.update(center_distance_px=float(dist.min()), area_px=int(mask.sum()))
    if dist.min() > max_center_distance_frac * min(H, W) or mask.sum() > max_area_frac * H * W:
        info["reason"] = "implausible_candidate"
        return np.zeros_like(v), info
    info["reason"] = None
    return mask, info


def make_analysis_mask(
    shape: tuple[int, int],
    pixel_size_um: float,
    faz: np.ndarray | None,
    settings: Mapping,
) -> np.ndarray:
    """Exclude the configured physical border band and optional FAZ mask.

    Border widths are rounded up to whole pixels. The outermost pixel row
    and column are inside any positive border band.
    """
    if len(shape) != 2 or min(shape) <= 0:
        raise MaskError("Invalid image shape")
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise MaskError("Physical analysis mask requires a positive pixel size")
    band_um = float(settings["border_band_um"])
    if not np.isfinite(band_um) or band_um < 0:
        raise MaskError("Border band must be finite and nonnegative")
    mask = np.ones(shape, dtype=bool)
    width = int(np.ceil(band_um / pixel_size_um))
    if width:
        mask[:width, :] = False
        mask[-width:, :] = False
        mask[:, :width] = False
        mask[:, -width:] = False
    if settings["exclude_faz"]:
        if faz is None or faz.shape != shape or faz.dtype != np.bool_:
            raise MaskError("FAZ exclusion requires a decoded boolean mask of matching shape")
        mask &= ~faz
    if not mask.any():
        raise MaskError("Analysis area is empty after border and FAZ exclusion")
    return mask
