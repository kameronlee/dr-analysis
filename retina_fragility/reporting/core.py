"""Report modes, provenance refusal, input ledger and labelled writers.

Two explicit modes, never inferred:

* ``clinical`` — every input must declare ``data_provenance = "real"``. Any
  synthetic marker (a SYNTHETIC ``data_provenance``, a synthetic manifest
  dataset or a ``SYNTHETIC_NOT_CLINICAL.txt`` file in an input directory)
  refuses the whole report. There is no override.
* ``fixture`` — every input must be SYNTHETIC. Real inputs are refused, so a
  clinical result can never be rendered under a fixture label. Every table,
  figure, caption and record is labelled ``SYNTHETIC — NOT CLINICAL``.

Inputs whose provenance cannot be established are refused in both modes.
Every file read is hashed into the ledger that the artifact index records.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..provenance import SYNTHETIC_BANNER, sha256_file

CLINICAL, FIXTURE = "clinical", "fixture"
MODES = (CLINICAL, FIXTURE)
FIXTURE_LABEL = "SYNTHETIC — NOT CLINICAL"
CLINICAL_LABEL = "Real data (clinical mode)"
MARKER_FILE = "SYNTHETIC_NOT_CLINICAL.txt"

# Artifact status vocabulary of the paper-artifact index.
RENDERED = "rendered"
NOT_ESTIMABLE = "rendered_not_estimable"   # rendered; the analysis reported a non-estimable result
NOT_RUN = "not_run"                        # the analysis recorded a refusal/readiness failure
MISSING = "missing_input"                  # required machine artifact absent
REFUSED = "refused"                        # identity/count/provenance check failed; nothing rendered
UNAVAILABLE_TEXT = "unavailable"


class ReportRefusal(RuntimeError):
    """Raised with explicit reasons; the caller must not render anything for that scope."""

    def __init__(self, reasons: Iterable[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


def classify_provenance(value: Any) -> str:
    """'synthetic', 'real' or 'unknown' for a data_provenance value (str or list)."""
    if isinstance(value, (list, tuple, set)):
        kinds = {classify_provenance(v) for v in value}
        if not kinds:
            return "unknown"
        return kinds.pop() if len(kinds) == 1 else "mixed"
    if not isinstance(value, str) or not value:
        return "unknown"
    if value.upper().startswith("SYNTHETIC"):
        return "synthetic"
    return "real" if value == "real" else "unknown"


class Ledger:
    """Hashes every input that is read and collects provenance declarations."""

    def __init__(self, mode: str, root: str | Path):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        self.root = Path(root).resolve()
        self.files: dict[str, str] = {}
        self.declarations: list[dict] = []

    def rel(self, path: str | Path) -> str:
        p = Path(path).resolve()
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return p.as_posix()

    def record(self, path: str | Path) -> None:
        self.files[self.rel(path)] = sha256_file(path)

    def exists(self, path: str | Path | None) -> bool:
        return path is not None and Path(path).is_file()

    def json(self, path: str | Path) -> dict:
        p = Path(path)
        self.record(p)
        return json.loads(p.read_text())

    def csv(self, path: str | Path, **kw) -> pd.DataFrame:
        p = Path(path)
        self.record(p)
        return pd.read_csv(p, **kw)

    def text(self, path: str | Path) -> str:
        p = Path(path)
        self.record(p)
        return p.read_text()

    def declare(self, source: str | Path, value: Any) -> None:
        self.declarations.append({"source": self.rel(source), "data_provenance": value,
                                  "kind": classify_provenance(value)})

    def scan_markers(self, directories: Iterable[str | Path]) -> None:
        """Record SYNTHETIC marker files directly inside the given input directories."""
        for d in directories:
            if d is None:
                continue
            m = Path(d) / MARKER_FILE
            if m.is_file():
                self.declare(m, SYNTHETIC_BANNER)

    def provenance_reasons(self) -> list[str]:
        """Refusal reasons for the declared provenance under the ledger's mode."""
        reasons = []
        if not self.declarations:
            return ["no input declares its data provenance; refusing to render"]
        kinds = {d["kind"] for d in self.declarations}
        unknown = [d["source"] for d in self.declarations if d["kind"] in ("unknown", "mixed")]
        if unknown:
            reasons.append(f"provenance unknown or mixed in {unknown[:5]}; refusing to render")
        if self.mode == CLINICAL and "synthetic" in kinds:
            syn = [d["source"] for d in self.declarations if d["kind"] == "synthetic"]
            reasons.append("clinical mode refuses SYNTHETIC inputs: "
                           f"{syn[:5]} (fixtures are never rendered as clinical results)")
        if self.mode == FIXTURE and "real" in kinds:
            real = [d["source"] for d in self.declarations if d["kind"] == "real"]
            reasons.append("fixture mode refuses real inputs: "
                           f"{real[:5]} (render real analyses only in clinical mode)")
        return reasons


# ------------------------------------------------------------------ formatting
def label(mode: str) -> str:
    return FIXTURE_LABEL if mode == FIXTURE else CLINICAL_LABEL


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return not math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fmt(x: Any, digits: int = 3) -> str:
    """Number formatting that never hides missing values."""
    if x is None:
        return UNAVAILABLE_TEXT
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(f):
        return UNAVAILABLE_TEXT
    if f != 0 and (abs(f) < 10 ** -(digits) or abs(f) >= 10 ** 6):
        return f"{f:.{digits}e}"
    return f"{f:.{digits}f}"


def fmt_p(p: Any) -> str:
    if is_missing(p):
        return "no p-value"
    p = float(p)
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def fmt_ci(ci: Any, digits: int = 3) -> str:
    if not isinstance(ci, (list, tuple)) or len(ci) != 2 or any(is_missing(v) for v in ci):
        return UNAVAILABLE_TEXT
    return f"{fmt(ci[0], digits)} to {fmt(ci[1], digits)}"


def markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = [str(v).replace("|", "\\|").replace("\n", " ") for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ writing
class ArtifactWriter:
    """Writes one artifact's files into its own sub-directory with mode labels."""

    def __init__(self, out_root: Path, artifact_id: str, mode: str):
        self.dir = out_root / artifact_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.mode = mode
        self.outputs: list[str] = []
        if mode == FIXTURE:
            self.write_text(MARKER_FILE, f"{FIXTURE_LABEL}. Rendered from synthetic fixtures for "
                                         "software validation only; not a clinical finding.\n")

    def _done(self, name: str) -> None:
        self.outputs.append(name)

    def write_text(self, name: str, text: str) -> None:
        (self.dir / name).write_text(text)
        self._done(name)

    def write_csv(self, name: str, df: pd.DataFrame) -> None:
        df = df.copy() if len(df) else pd.DataFrame([{"note": "none recorded"}])
        df.insert(0, "data_label", label(self.mode))
        df.to_csv(self.dir / name, index=False)
        self._done(name)

    def write_markdown(self, name: str, title: str, body: str) -> None:
        head = f"# {title}\n\n**{label(self.mode)}**\n\n"
        self.write_text(name, head + body.rstrip() + "\n")

    def write_json(self, name: str, obj: dict) -> None:
        obj = {"data_label": label(self.mode), **obj}
        (self.dir / name).write_text(json.dumps(_clean(obj), indent=2, default=str, ensure_ascii=False) + "\n")
        self._done(name)

    def save_figure(self, fig, stem: str) -> None:
        """PNG, PDF and SVG with the mode label in the figure itself; deterministic metadata."""
        import matplotlib.pyplot as plt

        if self.mode == FIXTURE:
            fig.text(0.5, 0.995, FIXTURE_LABEL, ha="center", va="top", fontsize=11,
                     color="#b00020", fontweight="bold")
        # text stays text in PDF/SVG, so labels remain searchable and editable
        with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "svg.hashsalt": "retina"}):
            for suffix, meta in (("png", {"Software": None}), ("pdf", {"CreationDate": None}),
                                 ("svg", {"Date": None})):
                name = f"{stem}.{suffix}"
                fig.savefig(self.dir / name, dpi=200, metadata=meta)
                self._done(name)
        plt.close(fig)

    def hashes(self) -> dict[str, str]:
        return {f"{self.dir.name}/{n}": sha256_file(self.dir / n) for n in self.outputs}


def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def prepare_output_root(out: str | Path, input_paths: Iterable[str | Path | None], mode: str) -> Path:
    """Refuse non-empty outputs, outputs inside inputs, and fixture/clinical path mixing."""
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ReportRefusal([f"output directory {out} exists and is not empty; reports are never "
                             "overwritten (use a new dated directory)"])
    for p in input_paths:
        if p is None:
            continue
        r = Path(p).resolve()
        rdir = r if r.is_dir() else r.parent
        if out == rdir or rdir in out.parents:
            raise ReportRefusal([f"output directory {out} lies inside input location {rdir}"])
    lowered = out.as_posix().lower()
    if mode == CLINICAL and any(t in lowered for t in ("synthetic", "fixture")):
        raise ReportRefusal([f"clinical output path {out} names synthetic/fixture data; keep "
                             "clinical outputs separate from fixture renders"])
    if mode == FIXTURE and not any(t in lowered for t in ("synthetic", "fixture")):
        raise ReportRefusal([f"fixture output path {out} must contain 'synthetic' or 'fixture' so "
                             "fixture renders stay separate from clinical outputs"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
