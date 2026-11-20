"""Clinical reporting gate G2: the OCTA-500 graph-QC export contract and its disposition.

Refusal reasons (no clinical report renders) when any of these fails:

* the sheet has every ``octa500.QC_SHEET_COLUMNS`` column and exactly the
  review/image pairs that ``octa500.select_graph_qc`` selects for the frozen
  calibration (no missing, extra, duplicate or truncated rows), matching the
  export's ``review_inventory.json``;
* every identity column is non-empty and bound: image/mask hashes equal the
  manifest, ``graph_sha256`` equals the exported ``graph.json``,
  ``graph_review_sha256`` equals the current graph-construction identity and
  the inventory, ``pipeline_code_sha256`` equals the processing run;
* every row has a named, non-synthetic reviewer, an ISO date not in the
  future, a ``pass``/``fail`` decision and a reason for every ``fail``;
* a dated disposition record exists (artifact plan S5: amendment list or an
  explicit "none" statement) that is dated on or after the last review and on
  or before the processing run; recorded failures need an explicit failure
  disposition. No failure is approved automatically and no pass-rate rule is
  invented here.

Records carry dates only. A review, disposition and processing run on the same
calendar day cannot be ordered; that case is reported as a flag, not as
established chronology.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from ..provenance import sha256_file, sha256_strings

DISPOSITION_KEYS = ("decided_on", "decided_by", "amendments", "statement")


def _iso(value) -> date | None:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


def graph_qc_reasons(sheet: Path | None, disposition: Path | None, manifest: pd.DataFrame, cfg: dict,
                     calibration_ids, run: dict, ledger=None, today: date | None = None
                     ) -> tuple[list[str], list[str]]:
    """(refusal reasons, flags) for clinical gate G2."""
    from ..octa500 import QC_SHEET_COLUMNS, select_graph_qc
    from ..qc import graph_review_identity

    today = today or date.today()
    if sheet is None or not Path(sheet).is_file():
        return ["OCTA-500 graph-QC sheet (gate G2) missing"], []
    sheet = Path(sheet)
    read = ledger.csv if ledger is not None else pd.read_csv
    q = read(sheet, dtype=str, keep_default_na=False)
    reasons, flags = [], []
    missing_cols = [c for c in QC_SHEET_COLUMNS if c not in q.columns]
    if missing_cols:
        return [f"graph-QC sheet lacks required columns {missing_cols}"], []
    q = q.apply(lambda c: c.str.strip())

    # --- selection: exactly the prespecified outcome-blind review set
    expected = select_graph_qc(manifest, cfg, list(calibration_ids))
    want = set(zip(expected.review_id, expected.image_id.astype(str)))
    got = list(zip(q.review_id, q.image_id))
    if len(got) != len(set(got)) or q.review_id.duplicated().any() or q.image_id.duplicated().any():
        reasons.append("graph-QC sheet has duplicate review or image rows")
    if set(got) != want:
        reasons.append(f"graph-QC sheet rows differ from the prespecified selection ({len(set(got) & want)}/"
                       f"{len(want)} expected rows present, {len(set(got) - want)} unexpected)")
    inv_p = sheet.parent / "review_inventory.json"
    inv = None
    if not inv_p.is_file():
        reasons.append("graph-QC export review_inventory.json missing")
    else:
        inv = ledger.json(inv_p) if ledger is not None else json.loads(inv_p.read_text())
        if inv.get("selection_ids_sha256") != sha256_strings(q.image_id):
            reasons.append("graph-QC sheet image set differs from the export inventory")

    # --- identities
    idx = manifest.set_index("image_id", drop=False)
    for col in ("image_sha256", "vessel_mask_sha256", "faz_mask_sha256", "graph_sha256",
                "graph_review_sha256", "pipeline_code_sha256"):
        if q[col].eq("").any():
            reasons.append(f"graph-QC sheet has blank {col} values")
    mism = []
    for r in q.itertuples():
        if r.image_id not in idx.index:
            mism.append(r.review_id)
            continue
        m = idx.loc[r.image_id]
        for col in ("image_sha256", "vessel_mask_sha256", "large_vessel_mask_sha256", "faz_mask_sha256"):
            if col in m and str(m[col]) != getattr(r, col):
                mism.append(r.review_id)
        g = sheet.parent / r.review_id / "graph.json"
        if not g.is_file() or sha256_file(g) != r.graph_sha256:
            mism.append(r.review_id)
    if mism:
        reasons.append(f"graph-QC identities do not match the manifest/exported graphs for "
                       f"{len(set(mism))} rows")
    gid = graph_review_identity(cfg, {"mode": "reference"})["graph_review_sha256"]
    if set(q.graph_review_sha256) != {gid} or (inv is not None and inv.get("graph_review_sha256") != gid):
        reasons.append("graph-QC sheet graph_review_sha256 differs from the current graph construction")
    run_fp = (run.get("code_fingerprint") or {}).get("sha256")
    if set(q.pipeline_code_sha256) != {run_fp} or (inv is not None and inv.get("pipeline_code_sha256") != run_fp):
        reasons.append("pipeline code fingerprint bound in the graph-QC sheet differs from the processing "
                       "run (a dated amendment is required before reporting)")

    # --- decisions
    dec = q.decision.str.lower()
    blank = q[q.reviewer.eq("") | q.review_date.eq("") | ~dec.isin(["pass", "fail"])]
    if len(blank):
        reasons.append(f"OCTA-500 graph QC incomplete: {len(blank)}/{len(q)} rows lack a named reviewer, "
                       "date or pass/fail decision (gate G2 precedes any clinical report)")
    if q.reviewer.str.upper().str.contains("SYNTHETIC").any():
        reasons.append("graph-QC sheet contains non-human SYNTHETIC reviewer entries")
    dates = q.review_date.map(_iso)
    bad_dates = int((q.review_date.ne("") & dates.isna()).sum())
    if bad_dates:
        reasons.append(f"{bad_dates} graph-QC review dates are not ISO dates (YYYY-MM-DD)")
    if dates.dropna().gt(today).any():
        reasons.append("graph-QC review dates lie in the future")
    fails = q[dec.eq("fail")]
    if (fails.fail_reason == "").any():
        reasons.append(f"{int((fails.fail_reason == '').sum())} graph-QC failures have no fail_reason")

    # --- disposition (amendments or an explicit none statement), dated before processing
    run_day = _iso(str(run.get("created_utc", ""))[:10])
    last_review = max(dates.dropna()) if dates.notna().any() else None
    if disposition is None or not Path(disposition).is_file():
        reasons.append("graph-QC disposition record missing (artifact plan S5: dated amendment list or "
                       "'none' statement before processing)")
    else:
        d = ledger.json(disposition) if ledger is not None else json.loads(Path(disposition).read_text())
        absent = [k for k in DISPOSITION_KEYS if k not in d]
        decided = _iso(d.get("decided_on", ""))
        if absent:
            reasons.append(f"graph-QC disposition lacks {absent}")
        if not str(d.get("decided_by", "")).strip() or "SYNTHETIC" in str(d.get("decided_by", "")).upper():
            reasons.append("graph-QC disposition has no named human decision-maker")
        if decided is None:
            reasons.append("graph-QC disposition decided_on is not an ISO date")
        elif last_review is not None and decided < last_review:
            reasons.append("graph-QC disposition predates the last review decision")
        elif run_day is not None and decided > run_day:
            reasons.append("graph-QC disposition postdates the processing run (G2 must precede G4)")
        amend = d.get("amendments")
        if not isinstance(amend, list):
            reasons.append("graph-QC disposition 'amendments' must be a list (empty for none)")
        elif any(_iso((a or {}).get("date", "")) is None or (run_day and _iso(a["date"]) > run_day)
                 for a in amend):
            reasons.append("every amendment needs an ISO date on or before the processing run")
        if len(fails) and not str(d.get("failure_disposition", "")).strip():
            reasons.append(f"{len(fails)} graph-QC failures recorded without a failure_disposition in the "
                           "disposition record (failures are never approved automatically)")
        if decided is not None and run_day is not None and decided == run_day:
            flags.append("graph-QC disposition and processing run share a calendar day; date-only records "
                         "cannot establish their order")
    if run_day is None:
        reasons.append("processing run has no creation date")
    elif last_review is not None and last_review > run_day:
        reasons.append("processing run predates the last graph-QC decision (G2 must precede G4)")
    elif last_review is not None and last_review == run_day:
        flags.append("last graph-QC review and processing run share a calendar day; date-only records "
                     "cannot establish their order")
    return reasons, flags
