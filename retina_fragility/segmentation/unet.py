"""Modest 2-D U-Net vessel segmenter (optional dependency: torch).

Training uses OCTA-500 rows with ``seg_split == train`` only, never calibration
rows, and the reference vessel labels. The saved model directory records the
training image IDs and group keys; prediction refuses any row whose image ID
or group key was used for training (segmentation leakage guard).

If torch is not importable every entry point raises ``DependencyUnavailable``
with the exact reason; nothing falls back to random weights.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from ..provenance import now_utc, sha256_file


class DependencyUnavailable(RuntimeError):
    pass


class LeakageError(RuntimeError):
    pass


def torch_status() -> dict:
    try:
        import torch

        return {"available": True, "version": torch.__version__}
    except Exception as exc:  # ImportError or broken shared libraries
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _torch():
    st = torch_status()
    if not st["available"]:
        raise DependencyUnavailable("torch is not importable (" + st["reason"] + "); install the "
                                    "optional 'deep' extra to train or apply the U-Net")
    import torch

    return torch


def build_unet(base_channels: int = 16, depth: int = 4):
    torch = _torch()
    nn = torch.nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [base_channels * 2 ** i for i in range(depth + 1)]
            self.down = nn.ModuleList([block(1, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(depth)])
            self.pool = nn.MaxPool2d(2)
            self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2)
                                     for i in reversed(range(depth))])
            self.dec = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
            self.head = nn.Conv2d(ch[0], 1, 1)
            self.depth = depth

        def forward(self, x):
            skips = []
            for i, d in enumerate(self.down):
                x = d(x)
                if i < self.depth:
                    skips.append(x)
                    x = self.pool(x)
            for up, dec in zip(self.up, self.dec):
                x = up(x)
                x = dec(torch.cat([x, skips.pop()], dim=1))
            return self.head(x)

    return UNet()


def _pad_to(img: np.ndarray, mult: int) -> tuple[np.ndarray, tuple[int, int]]:
    H, W = img.shape
    ph, pw = (-H) % mult, (-W) % mult
    return np.pad(img, ((0, ph), (0, pw)), mode="reflect" if min(H, W) > 1 else "constant"), (H, W)


def _sample_patch(img, mask, size, rng):
    H, W = img.shape
    if H < size or W < size:
        ph, pw = max(0, size - H), max(0, size - W)
        img = np.pad(img, ((0, ph), (0, pw)), mode="reflect")
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
        H, W = img.shape
    y, x = int(rng.integers(0, H - size + 1)), int(rng.integers(0, W - size + 1))
    a, b = img[y:y + size, x:x + size], mask[y:y + size, x:x + size]
    k = int(rng.integers(0, 4))
    a, b = np.rot90(a, k), np.rot90(b, k)
    if rng.random() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def train(images: list[np.ndarray], masks: list[np.ndarray], train_ids: list[str],
          train_groups: list[str], ucfg: dict, seed: int, out_dir: str | Path,
          input_paths: list[str] | None = None, patches_per_epoch: int | None = None,
          extra_meta: dict | None = None) -> dict:
    """Train on the given (already leakage-filtered) images; save model + metadata.

    Refuses to overwrite an existing model. ``model.pt`` and then ``meta.json``
    are published atomically, so an interrupted run never leaves a model that
    looks complete. ``extra_meta`` (input/label hashes, manifest, calibration
    and code identities) is stored in ``meta.json``.
    """
    out = Path(out_dir)
    if (out / "model.pt").exists() or (out / "meta.json").exists():
        raise FileExistsError(f"{out} already holds a trained U-Net; choose a new output directory")
    torch = _torch()
    torch.manual_seed(seed % (2**63 - 1))
    rng = np.random.default_rng(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device(ucfg["device"])
    net = build_unet(int(ucfg["base_channels"]), int(ucfg["depth"])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=float(ucfg["lr"]))
    size = int(ucfg["patch_size"])
    size -= size % (2 ** int(ucfg["depth"]))
    bs = int(ucfg["batch_size"])
    n_patches = int(patches_per_epoch or max(bs, len(images)))
    history = []
    t0 = time.monotonic()
    for ep in range(int(ucfg["epochs"])):
        net.train()
        losses = []
        for _ in range(max(1, n_patches // bs)):
            batch = [_sample_patch(images[j], masks[j], size, rng)
                     for j in rng.integers(0, len(images), size=bs)]
            x = torch.from_numpy(np.stack([b[0] for b in batch])[:, None].astype(np.float32)).to(dev)
            y = torch.from_numpy(np.stack([b[1] for b in batch])[:, None].astype(np.float32)).to(dev)
            logits = net(x)
            p = torch.sigmoid(logits)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            loss = bce + dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        history.append({"epoch": ep, "loss": float(np.mean(losses))})
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".model.pt.tmp-{os.getpid()}"
    torch.save(net.state_dict(), tmp)
    os.replace(tmp, out / "model.pt")
    steps = max(1, n_patches // bs)
    meta = {"format": "retina_fragility.unet/1", "created_utc": now_utc(), "config": ucfg,
            "seed": int(seed), "train_image_ids": list(train_ids), "train_group_keys": list(train_groups),
            "n_train": len(images), "history": history, "elapsed_seconds": time.monotonic() - t0,
            "torch_version": torch.__version__, "torch_num_threads": int(torch.get_num_threads()),
            "device": str(dev), "effective_patch_size": size,
            "patches_per_epoch": n_patches, "steps_per_epoch": steps,
            "n_optimizer_steps": steps * int(ucfg["epochs"]),
            "model_selection": "none: weights after the last configured epoch (no validation-based choice)",
            "train_input_sha256": {p: sha256_file(p) for p in (input_paths or [])},
            **(extra_meta or {}),
            "model_sha256": sha256_file(out / "model.pt")}
    tmp = out / f".meta.json.tmp-{os.getpid()}"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, out / "meta.json")
    return meta


def load_model(model_dir: str | Path):
    torch = _torch()
    meta = json.loads((Path(model_dir) / "meta.json").read_text())
    if sha256_file(Path(model_dir) / "model.pt") != meta["model_sha256"]:
        raise RuntimeError("U-Net weights do not match the recorded hash")
    u = meta["config"]
    net = build_unet(int(u["base_channels"]), int(u["depth"]))
    net.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location="cpu"))
    net.eval()
    return net, meta


def check_leakage(meta: dict, image_ids, group_keys) -> None:
    ids = set(map(str, image_ids)) & set(meta["train_image_ids"])
    grp = set(map(str, group_keys)) & set(meta["train_group_keys"])
    if ids or grp:
        raise LeakageError(f"rows used for U-Net training requested for prediction/evaluation: "
                           f"ids={sorted(ids)[:5]}, groups={sorted(grp)[:5]}")


def predict_prob(net, meta: dict, image: np.ndarray) -> np.ndarray:
    torch = _torch()
    mult = 2 ** int(meta["config"]["depth"])
    padded, (H, W) = _pad_to(np.asarray(image, np.float32), mult)
    with torch.no_grad():
        x = torch.from_numpy(padded[None, None].astype(np.float32))
        p = torch.sigmoid(net(x))[0, 0].numpy()
    return p[:H, :W]


def segment(prob: np.ndarray, ucfg: dict, threshold_scale: float | None = None) -> np.ndarray:
    scale = float(ucfg["threshold_scale"] if threshold_scale is None else threshold_scale)
    return np.asarray(prob) > float(ucfg["threshold"]) * scale


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def cl_dice(pred: np.ndarray, ref: np.ndarray) -> float:
    """Centerline Dice (Shit et al., CVPR 2021): harmonic mean of topology precision
    |skel(pred) & ref| / |skel(pred)| and sensitivity |skel(ref) & pred| / |skel(ref)|.
    Descriptive only; never used to select a model or threshold.

    Empty-mask convention (same as ``dice``): exactly one empty mask is a complete
    miss and scores 0 (the defined factor of the harmonic mean is 0, so the
    undefined one cannot raise it); both empty is undefined (NaN), because there
    is no vessel to find. ``summarize_evaluation`` reports these counts separately.
    """
    from skimage.morphology import skeletonize

    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    if not pred.any() and not ref.any():
        return float("nan")
    if not pred.any() or not ref.any():
        return 0.0
    sp, sr = skeletonize(pred), skeletonize(ref)
    sp, sr = (sp if sp.any() else pred), (sr if sr.any() else ref)  # defensive; skeletons of non-empty masks are non-empty
    tprec = (sp & ref).sum() / sp.sum()
    tsens = (sr & pred).sum() / sr.sum()
    return float(2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) else 0.0
