"""Frozen image embeddings for exploratory Experiment F (optional torch/torchvision).

Models: ``resnet50`` (torchvision ImageNet weights, final fc removed -> 2048-d)
and ``dinov2_vits14`` (torch.hub, CLS token -> 384-d). Pretrained weights must
already be in the local torch cache unless ``embeddings.allow_download`` is
true. Missing torch or weights raise ``DependencyUnavailable`` with the exact
reason; random-initialised networks are never substituted in real runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .provenance import now_utc, sha256_file
from .segmentation.unet import DependencyUnavailable, _torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _resnet50(ecfg: dict, _random_init_for_tests: bool):
    _torch()
    import torch
    import torchvision

    if _random_init_for_tests:
        net = torchvision.models.resnet50(weights=None)
        wname = "RANDOM_INIT_TEST_ONLY"
    else:
        if ecfg["weights"] != "imagenet":
            raise DependencyUnavailable("only ImageNet-pretrained ResNet50 is supported for Experiment F")
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        fname = Path(w.url).name
        cached = Path(torch.hub.get_dir()) / "checkpoints" / fname
        if not cached.exists() and not ecfg["allow_download"]:
            raise DependencyUnavailable(f"ResNet50 ImageNet weights not in local cache ({cached}); "
                                        "set embeddings.allow_download: true to fetch them")
        net = torchvision.models.resnet50(weights=w)
        wname = f"torchvision {w}"
    net.fc = torch.nn.Identity()
    return net.eval(), wname


def _dinov2(ecfg: dict, _random_init_for_tests: bool):
    _torch()
    import torch

    if _random_init_for_tests:
        raise DependencyUnavailable("DINOv2 has no test-only random initialisation path")
    hub = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not hub.exists() and not ecfg["allow_download"]:
        raise DependencyUnavailable(f"DINOv2 hub code/weights not cached ({hub}); set "
                                    "embeddings.allow_download: true to fetch from torch.hub")
    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    return net.eval(), "torch.hub facebookresearch/dinov2 dinov2_vits14"


def load_model(name: str, ecfg: dict, _random_init_for_tests: bool = False):
    if name == "resnet50":
        return _resnet50(ecfg, _random_init_for_tests)
    if name == "dinov2_vits14":
        return _dinov2(ecfg, _random_init_for_tests)
    raise ValueError(f"unknown embedding model {name}")


def preprocess(image: np.ndarray, size: int, patch_multiple: int = 1) -> np.ndarray:
    """Grayscale [0,1] -> resized 3-channel ImageNet-normalised CHW float32."""
    from skimage.transform import resize

    s = size - size % patch_multiple if patch_multiple > 1 else size
    img = resize(np.asarray(image, np.float32), (s, s), order=1, anti_aliasing=True,
                 preserve_range=True).astype(np.float32)
    rgb = np.repeat(img[None], 3, axis=0)
    return (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


def extract(images: list[np.ndarray], name: str, ecfg: dict,
            _random_init_for_tests: bool = False) -> tuple[np.ndarray, dict]:
    torch = _torch()
    net, wname = load_model(name, ecfg, _random_init_for_tests)
    mult = 14 if name.startswith("dinov2") else 1
    feats = []
    bs = int(ecfg["batch_size"])
    with torch.no_grad():
        for i in range(0, len(images), bs):
            x = np.stack([preprocess(im, int(ecfg["image_size"]), mult) for im in images[i:i + bs]])
            out = net(torch.from_numpy(x))
            feats.append(out.reshape(out.shape[0], -1).numpy())
    X = np.concatenate(feats) if feats else np.zeros((0, 0))
    return X, {"model": name, "weights": wname, "image_size": int(ecfg["image_size"]),
               "torch_version": torch.__version__, "created_utc": now_utc(),
               "random_init_test_only": bool(_random_init_for_tests)}


def extract_manifest(frame, root, name: str, ecfg: dict, out_path: str | Path,
                     splits=("train",)) -> dict:
    """Extract embeddings for manifest rows (DRAC training split by default) to an .npz."""
    from .io_masks import read_intensity
    from .manifest import resolve_input

    load_model(name, ecfg)  # fail fast (missing torch/weights) before reading any image
    rows = frame[frame.split.str.lower().isin([s.lower() for s in splits])]
    if (rows.split.str.lower() == "test").any():
        raise ValueError("hidden-label test rows are not embedded for Experiment F")
    paths = [resolve_input(p, root) for p in rows.image_path]
    images = [read_intensity(p)[0] for p in paths]
    X, meta = extract(images, name, ecfg)
    meta["input_sha256"] = {str(p): sha256_file(p) for p in paths}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, image_id=np.array(rows.image_id.tolist()), X=X)
    return meta
