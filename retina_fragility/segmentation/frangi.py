"""Frangi vesselness segmentation (training-free baseline segmenter for Experiment E).

OCTA vessels are bright on a dark background (``black_ridges: false``). The
response is thresholded with Otsu's method computed on the response itself,
scaled by ``threshold_scale`` (ablation ±10 %), then objects smaller than
``min_object_px`` are removed. No per-image manual tuning is possible.

Two implementations exist (``segmentation.frangi.implementation``):

* ``frangi1998_normalized`` (default since the 2026-09-30 pre-outcome
  amendment): Hessians from Gaussian derivatives are multiplied by sigma²
  (γ-normalised second derivatives, Frangi et al. 1998, Sec. 2), so responses are
  comparable across scales. The structuredness constant c is half of the maximum
  Hessian norm over **all** scales, which is one common value. It is not
  re-estimated per scale. β = ``beta`` (0.5). A bright ridge needs λ₂ < 0. The
  vesselness is the maximum over scales.
* ``skimage_unnormalized``: ``skimage.filters.frangi`` (0.26). Its Hessian is
  not scale-normalised, and c is fixed from the *first* sigma, so the maximum
  over scales is dominated by the smallest sigma. Broad vessels are then missed.
  This was the pre-amendment behaviour, and it reproduces the v1 QC masks
  exactly. Use it only to reproduce those masks.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import remove_small_objects

IMPLEMENTATIONS = ("frangi1998_normalized", "skimage_unnormalized")


def _normalized_frangi(img: np.ndarray, sigmas, beta: float, black_ridges: bool) -> tuple[np.ndarray, dict]:
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    per_scale = []
    for s in sigmas:
        H = hessian_matrix(img, float(s), mode="reflect", use_gaussian_derivatives=True)
        ev = hessian_matrix_eigvals([h * float(s) ** 2 for h in H])
        ev = np.take_along_axis(ev, np.abs(ev).argsort(0), 0)  # |λ1| <= |λ2|
        per_scale.append(ev)
    norms = [np.sqrt(ev[0] ** 2 + ev[1] ** 2) for ev in per_scale]
    c = 0.5 * max(float(n.max()) for n in norms)
    if not c > 0:
        return np.zeros(img.shape), {"c": c, "argmax_scale_counts": None}
    out = np.zeros(img.shape)
    arg = np.full(img.shape, -1, dtype=int)
    for k, (ev, S) in enumerate(zip(per_scale, norms)):
        l1, l2 = ev
        rb = np.abs(l1) / np.maximum(np.abs(l2), 1e-12)
        v = np.exp(-(rb ** 2) / (2.0 * beta ** 2)) * (1.0 - np.exp(-(S ** 2) / (2.0 * c ** 2)))
        # bright ridge: strongly negative curvature across the vessel; dark ridge: positive
        v[(l2 > 0) if not black_ridges else (l2 < 0)] = 0.0
        better = v > out
        arg[better] = k
        out = np.maximum(out, v)
    return out, {"c": c, "argmax_scale": arg}


def frangi_response(image: np.ndarray, sigmas, black_ridges: bool = False,
                    implementation: str = "frangi1998_normalized", beta: float = 0.5) -> np.ndarray:
    return frangi_response_info(image, sigmas, black_ridges, implementation, beta)[0]


def frangi_response_info(image: np.ndarray, sigmas, black_ridges: bool = False,
                         implementation: str = "frangi1998_normalized",
                         beta: float = 0.5) -> tuple[np.ndarray, dict]:
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("Frangi segmentation expects a 2-D projection image")
    if implementation == "skimage_unnormalized":
        return frangi(img, sigmas=list(sigmas), black_ridges=bool(black_ridges), beta=beta), {}
    if implementation == "frangi1998_normalized":
        return _normalized_frangi(img, list(sigmas), float(beta), bool(black_ridges))
    raise ValueError(f"unknown Frangi implementation {implementation!r}; use one of {IMPLEMENTATIONS}")


def segment(image: np.ndarray, fcfg: dict, threshold_scale: float | None = None) -> tuple[np.ndarray, dict]:
    impl = fcfg.get("implementation", "frangi1998_normalized")
    resp, rinfo = frangi_response_info(image, fcfg["sigmas"], fcfg["black_ridges"], impl,
                                       float(fcfg.get("beta", 0.5)))
    scale = float(fcfg["threshold_scale"] if threshold_scale is None else threshold_scale)
    if fcfg["threshold"] != "otsu":
        raise ValueError(f"unsupported Frangi threshold rule {fcfg['threshold']}")
    finite = resp[np.isfinite(resp)]
    if finite.size == 0 or float(finite.max()) <= float(finite.min()):
        return np.zeros(resp.shape, bool), {"threshold": None, "reason": "constant_response",
                                            "implementation": impl}
    thr = float(threshold_otsu(finite)) * scale
    mask = resp > thr
    min_px = int(fcfg["min_object_px"])
    if min_px > 0:
        mask = remove_small_objects(mask, max_size=min_px - 1) if _has_max_size() else \
            remove_small_objects(mask, min_size=min_px)
    info = {"threshold": thr, "threshold_scale": scale, "otsu_on": "frangi_response",
            "sigmas": list(fcfg["sigmas"]), "implementation": impl,
            "vessel_fraction": float(mask.mean())}
    if "c" in rinfo:
        info["structuredness_c"] = float(rinfo["c"])
    arg = rinfo.get("argmax_scale")
    if arg is not None:
        # which scale won inside the final mask (diagnostic of multiscale coverage)
        counts = np.bincount(arg[mask & (arg >= 0)], minlength=len(fcfg["sigmas"]))
        info["mask_px_by_sigma"] = {f"{s:g}": int(n) for s, n in zip(fcfg["sigmas"], counts)}
    return mask, info


def _has_max_size() -> bool:
    import inspect

    return "max_size" in inspect.signature(remove_small_objects).parameters
