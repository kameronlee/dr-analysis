"""Provenance: hashes, environment, run records."""

from __future__ import annotations

import ast
import datetime as _dt
import hashlib
import json
import platform
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__

SYNTHETIC_BANNER = "SYNTHETIC - NOT CLINICAL"
PACKAGE = "retina_fragility"
PACKAGE_ROOT = Path(__file__).resolve().parent
# Numerical libraries whose version can change per-image outputs; part of cache keys.
NUMERICAL_LIBRARIES = ("numpy", "scipy", "skimage", "igraph", "networkx")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_strings(items: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(i) for i in items)).encode()).hexdigest()


def environment() -> dict[str, Any]:
    versions = {}
    for mod in ["numpy", "scipy", "pandas", "networkx", "igraph", "skimage", "sklearn",
                "statsmodels", "matplotlib", "yaml"]:
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional
            versions[mod] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_version": __version__,
        "libraries": versions,
    }


def numerical_library_versions() -> dict[str, str | None]:
    libs = environment()["libraries"]
    return {k: libs.get(k) for k in NUMERICAL_LIBRARIES}


# ------------------------------------------------------------------ code fingerprint
def _module_file(module: str) -> Path | None:
    parts = module.split(".")
    if parts[0] != PACKAGE:
        return None
    base = PACKAGE_ROOT.joinpath(*parts[1:])
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def module_import_closure(entry_modules: Iterable[str]) -> list[str]:
    """Package modules reachable from ``entry_modules`` through static imports.

    Imports are read from the source AST, including imports inside functions,
    so lazily imported modules (e.g. ``from . import viz``) are included. Parent
    package ``__init__`` modules are included because importing a submodule runs
    them. The result does not depend on what happens to be in ``sys.modules``.
    """
    seen: set[str] = set()
    stack = list(entry_modules)
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        path = _module_file(mod)
        if path is None:
            continue
        seen.add(mod)
        parts = mod.split(".")
        for i in range(1, len(parts)):
            stack.append(".".join(parts[:i]))
        is_pkg = path.name == "__init__.py"
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    pkg = parts if is_pkg else parts[:-1]
                    pkg = pkg[:len(pkg) - (node.level - 1)]
                    base = ".".join(pkg + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                if base.split(".")[0] != PACKAGE:
                    continue
                stack.append(base)
                stack.extend(f"{base}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                stack.extend(a.name for a in node.names if a.name.split(".")[0] == PACKAGE)
    return sorted(seen)


def code_fingerprint(entry_modules: Iterable[str]) -> dict[str, Any]:
    """SHA-256 over the source files of the static import closure of ``entry_modules``.

    Any edit to a module that can affect the outputs of the entry modules changes
    the fingerprint, even while ``PIPELINE_VERSION`` is unchanged. Files are hashed
    by content and keyed by package-relative path, so the fingerprint is
    reproducible across checkouts.
    """
    files = {}
    for mod in module_import_closure(entry_modules):
        p = _module_file(mod)
        files[p.relative_to(PACKAGE_ROOT.parent).as_posix()] = sha256_file(p)
    blob = "\n".join(f"{k}:{v}" for k, v in sorted(files.items()))
    return {"sha256": hashlib.sha256(blob.encode()).hexdigest(), "files": files,
            "entry_modules": sorted(entry_modules)}


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def run_record(cfg_hash: str, **extra: Any) -> dict[str, Any]:
    rec = {"created_utc": now_utc(), "config_sha256": cfg_hash, "environment": environment()}
    rec.update(extra)
    return rec


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _sanitize(o: Any) -> Any:
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def write_json(path: str | Path, obj: Any) -> None:
    """Write JSON; non-finite floats become null (never the string 'NaN')."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(_sanitize(obj), fh, indent=2, default=_default, allow_nan=False)


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def assert_outside_inputs(out_dir: str | Path, input_roots: Iterable[str | Path]) -> None:
    """Refuse to write outputs inside a read-only input root."""
    out = Path(out_dir).resolve()
    for root in input_roots:
        r = Path(root).resolve()
        if out == r or r in out.parents:
            raise PermissionError(f"output directory {out} lies inside read-only input root {r}")
