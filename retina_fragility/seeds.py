"""Deterministic local seeds.

A seed is a pure function of the base seed and a tuple of keys, so results do
not depend on processing order, worker count, or which other images ran.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(base_seed: int, *keys: object) -> int:
    text = "|".join([str(int(base_seed))] + [str(k) for k in keys])
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "little")


def rng_for(base_seed: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, *keys))


def stable_rank_key(base_seed: int, item: object) -> str:
    """Hex digest used to order items deterministically independent of input order."""
    return hashlib.sha256(f"{int(base_seed)}|{item}".encode()).hexdigest()
