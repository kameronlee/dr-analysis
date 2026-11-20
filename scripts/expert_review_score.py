"""Score an expert's verdicts on the blinded graph-review packet (plan: docs/EXPERT_REVIEW_SCORING_AMENDMENT_2026-11-17.md).

The expert's reply is transcribed, keeping the reviewer's own verdict words, into copies of the
packet's answer sheets (CSV) by someone who has not opened the key. This script maps the words
through a fixed table (anything else is refused: ask the reviewer), unblinds with the key kept
outside the packet, and reports what the plan fixed before any reply existed: verdict counts with
exact intervals per scan size, agreement with Kameron's earlier verdicts on the same eyes, Part B
verdicts per classical tracer, and the two pre-specified readings.

Usage::

    python scripts/expert_review_score.py --part-a <filled answer_sheet_part_A.csv> \
        [--part-b <filled answer_sheet_part_B.csv>] --original <the reply as received (.txt/.eml/.pdf)> \
        --received YYYY-MM-DD --qualification "<role, e.g. retina specialist>" \
        --entered-by "<who transcribed the reply; must not have opened the key>" \
        [--octa-years N] [--device "<device they usually read>"] [--minutes N] \
        [--named "<name, only if they agreed to be thanked by name>"] [--tag expert1]

Output: outputs/expert_review_2026-11-17/scored_<received>_<tag>/ (refuses if it already has files).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retina_fragility.provenance import environment, now_utc, read_json, sha256_file, write_json  # noqa: E402

PLAN = "docs/EXPERT_REVIEW_SCORING_AMENDMENT_2026-11-17.md"
RECORD = "reports/expert_review_scoring_amendment_2026-11-17.json"
PACKET_DIR = "outputs/expert_review_2026-11-17"
KEY = f"{PACKET_DIR}/key_NOT_FOR_REVIEWER.csv"
EARLIER = "outputs/octa500_graph_qc/v2_primary_20260930/human_review_2026-11-16/decisions_parsed.csv"
MANIFEST = "data/manifests/octa500_manifest.csv"
VERDICTS = ("pass", "fail", "uncertain")
FOVS = (3.0, 6.0)
# The only words accepted for a verdict, fixed before any reply (scoring-plan review, issue 4).
WORDS = {"pass": "pass", "ok": "pass", "okay": "pass", "acceptable": "pass", "good": "pass", "accept": "pass",
         "fail": "fail", "failed": "fail", "bad": "fail", "reject": "fail", "rejected": "fail", "unacceptable": "fail",
         "uncertain": "uncertain", "unsure": "uncertain", "not sure": "uncertain", "can't tell": "uncertain",
         "cannot tell": "uncertain", "can not tell": "uncertain", "don't know": "uncertain"}
# Every file the scoring depends on, including what the expert receives, must be hashed in the lock record.
REQUIRED = (
    "scripts/expert_review_score.py",
    "scripts/expert_review_packet.py",
    "retina_fragility/provenance.py",
    KEY,
    EARLIER,
    MANIFEST,
    f"{PACKET_DIR}/expert_review_packet.zip",
    f"{PACKET_DIR}/packet_record.json",
    f"{PACKET_DIR}/packet/INSTRUCTIONS.txt",
    f"{PACKET_DIR}/packet/answer_sheet_part_A.csv",
    f"{PACKET_DIR}/packet/answer_sheet_part_B.csv",
)


def lock_problems(rec: dict) -> list[str]:
    locked = {**rec.get("files_sha256", {}), **rec.get("script_sha256_at_adoption", {})}
    missing = [f for f in REQUIRED if f not in locked]
    if missing:
        return ["the lock record does not cover: " + ", ".join(missing)]
    return ["locked file changed: " + f for f, h in locked.items() if sha256_file(ROOT / f) != h]


def norm(v: str | None) -> str:
    """The reviewer's verdict word mapped through WORDS; blank is "missing"; anything else raises."""
    s = " ".join((v or "").strip().lower().replace("’", "'").split()).strip(" .;:-!")
    if s == "":
        return "missing"
    if s in WORDS:
        return WORDS[s]
    raise ValueError(f"unreadable verdict {v!r}: not in the fixed word list; ask the reviewer")


def read_sheet(path: Path, part: str) -> list[dict]:
    with open(ROOT / PACKET_DIR / "packet" / f"answer_sheet_part_{part}.csv", newline="") as fh:
        blank_head = next(csv.reader(fh))
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], [r for r in rows[1:] if r and r[0].strip()]
    if [h.strip() for h in head] != blank_head:
        raise ValueError(f"part {part} header differs from the blank answer sheet: {head}")
    out, errors = [], []
    for r in body:
        r = r + [""] * (len(head) - len(r))
        try:
            if part == "A":
                out.append({"label": r[0].strip(), "verdict": norm(r[1]), "word": r[1].strip(), "reason": r[2].strip()})
            else:
                out.append({"label": r[0].strip(), "verdict_X": norm(r[1]), "verdict_Y": norm(r[2]),
                            "word_X": r[1].strip(), "word_Y": r[2].strip(), "reason": r[3].strip()})
        except ValueError as e:
            errors.append(f"{r[0]}: {e}")
    if errors:
        raise ValueError("; ".join(errors))
    labels = [o["label"] for o in out]
    want = [f"{part}{i:02d}" for i in range(1, 21)]
    if sorted(labels) != want:
        raise ValueError(f"part {part} sheet must list {want[0]}..{want[-1]} exactly once each")
    return out


def exact_ci(k: int, n: int) -> list[float] | None:
    """Clopper-Pearson 95% interval for k of n."""
    if n == 0:
        return None
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def counts(vs: list[str]) -> dict:
    c = {v: vs.count(v) for v in (*VERDICTS, "missing")}
    c.update(n=len(vs), complete=c["missing"] == 0,
             pass_share_of_all=(c["pass"] / len(vs)) if vs else None, pass_ci_of_all=exact_ci(c["pass"], len(vs)),
             pass_share_excluding_uncertain=(c["pass"] / (c["pass"] + c["fail"])) if (c["pass"] + c["fail"]) else None,
             pass_ci_excluding_uncertain=exact_ci(c["pass"], c["pass"] + c["fail"]))
    return c


def table(a: list[str], b: list[str], cats: tuple[str, ...]) -> dict:
    return {f"{x}|{y}": sum(1 for p, q in zip(a, b) if (p, q) == (x, y)) for x in cats for y in cats}


def kappa(a: list[str], b: list[str], cats: tuple[str, ...]):
    """Unweighted Cohen's kappa on pairs where both ratings are in ``cats``; None if undefined."""
    pairs = [(x, y) for x, y in zip(a, b) if x in cats and y in cats]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == c for x, _ in pairs) / n) * (sum(y == c for _, y in pairs) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def two_way(v: str) -> str:
    return "pass" if v == "pass" else ("not_pass" if v in ("fail", "uncertain") else "missing")


def agreement(ex: list[str], ea: list[str], with_kappa: bool) -> dict:
    """Raw agreement and cross tables; positive/negative agreement on pass vs not; kappa only where asked."""
    out = {}
    pairs3 = [(x, y) for x, y in zip(ex, ea) if x in VERDICTS and y in VERDICTS]
    out["three_way"] = {"n": len(pairs3), "agreement": (sum(x == y for x, y in pairs3) / len(pairs3)) if pairs3 else None,
                        "table_expert|earlier": table(ex, ea, VERDICTS)}
    e2, k2 = [two_way(v) for v in ex], [two_way(v) for v in ea]
    pairs2 = [(x, y) for x, y in zip(e2, k2) if "missing" not in (x, y)]
    a_ = sum(1 for p in pairs2 if p == ("pass", "pass"))
    d_ = sum(1 for p in pairs2 if p == ("not_pass", "not_pass"))
    bc = len(pairs2) - a_ - d_
    out["pass_vs_not"] = {"n": len(pairs2), "agreement": ((a_ + d_) / len(pairs2)) if pairs2 else None,
                          "positive_agreement": (2 * a_ / (2 * a_ + bc)) if (2 * a_ + bc) else None,
                          "negative_agreement": (2 * d_ / (2 * d_ + bc)) if (2 * d_ + bc) else None,
                          "table_expert|earlier": table(e2, k2, ("pass", "not_pass"))}
    if with_kappa:
        out["three_way"]["kappa"] = kappa(ex, ea, VERDICTS)
        out["pass_vs_not"]["kappa"] = kappa(e2, k2, ("pass", "not_pass"))
    return out


def score_a(rows: list[dict], key: dict, earlier: dict, disease: dict) -> dict:
    res = {"by_fov": {}, "agreement_with_earlier": {}, "disease_mix": {}}
    for scope, fov in (("all", None), ("fov_3mm", 3.0), ("fov_6mm", 6.0)):
        sel = [r for r in rows if fov is None or key[r["label"]]["fov_mm"] == fov]
        ex = [r["verdict"] for r in sel]
        ea = [earlier[key[r["label"]]["earlier_review_id"]] for r in sel]
        res["by_fov"][scope] = {"expert": counts(ex), "earlier": counts(ea)}
        # kappa only at 6 mm: at 3 mm the earlier reviewer passed all 10 (kappa 0 or undefined), and pooling
        # mixes the scan-size difference into the agreement (scoring-plan review, issue 5)
        res["agreement_with_earlier"][scope] = agreement(ex, ea, with_kappa=(scope == "fov_6mm"))
        mix: dict[str, int] = {}
        for r in sel:
            g = disease.get(key[r["label"]]["image_id"], "unknown")
            mix[g] = mix.get(g, 0) + 1
        res["disease_mix"][scope] = mix
    return res


def score_b(rows: list[dict], key: dict) -> dict:
    per_tracer: dict[str, dict[str, list[str]]] = {}
    paired = []
    for r in rows:
        k = key[r["label"]]
        vx, vy = r["verdict_X"], r["verdict_Y"]
        for tracer, v in ((k["tracer_X"], vx), (k["tracer_Y"], vy)):
            per_tracer.setdefault(tracer, {"all": [], "fov_3mm": [], "fov_6mm": []})
            per_tracer[tracer]["all"].append(v)
            per_tracer[tracer]["fov_3mm" if k["fov_mm"] == 3.0 else "fov_6mm"].append(v)
        paired.append({k["tracer_X"]: two_way(vx), k["tracer_Y"]: two_way(vy)})
    res = {"per_tracer": {t: {s: counts(v) for s, v in d.items()} for t, d in per_tracer.items()},
           "note": "Part B shows the same scans as Part A, so the two parts are not independent"}
    names = sorted(per_tracer)
    if len(names) == 2:
        a, b = names
        only_a = sum(1 for p in paired if p[a] == "pass" and p[b] == "not_pass")
        only_b = sum(1 for p in paired if p[b] == "pass" and p[a] == "not_pass")
        from scipy.stats import binomtest
        m = only_a + only_b
        res["paired_pass_vs_not"] = {"tracers": [a, b], f"pass_only_{a}": only_a, f"pass_only_{b}": only_b,
                                     "exact_mcnemar_p": None if m == 0 else float(binomtest(only_a, m, 0.5).pvalue)}
    return res


def readings(a: dict) -> dict:
    """Pre-specified readings; a scan size with any blank verdict is not read (scoring-plan review, issue 2)."""
    c3, c6 = a["by_fov"]["fov_3mm"]["expert"], a["by_fov"]["fov_6mm"]["expert"]
    if not c3["complete"]:
        q1 = "not_read_incomplete"
    else:
        q1 = "acceptable" if c3["pass"] >= 8 else "doubtful" if c3["pass"] <= 5 else "mixed"
    if not c6["complete"]:
        q2 = "not_read_incomplete"
    else:
        q2 = "confirmed" if c6["fail"] >= 3 else "not_confirmed" if (c6["fail"] + c6["uncertain"]) <= 1 else "mixed"
    return {"q1_3mm_graphs": q1, "q2_6mm_caveat": q2}


def load_key() -> dict:
    with open(ROOT / KEY, newline="") as fh:
        return {r["label"]: {**r, "fov_mm": float(r["fov_mm"])} for r in csv.DictReader(fh)}


def load_earlier() -> dict:
    with open(ROOT / EARLIER, newline="") as fh:
        return {r["review_id"]: norm(r["verdict"]) for r in csv.DictReader(fh)}


def load_disease() -> dict:
    with open(ROOT / MANIFEST, newline="") as fh:
        return {r["image_id"]: r["disease"] for r in csv.DictReader(fh)}


def received_date(s: str, adopted_utc: str) -> str:
    try:
        d = dt.date.fromisoformat(s)
    except ValueError:
        raise SystemExit(f"REFUSED: --received {s!r} is not a YYYY-MM-DD date") from None
    if d < dt.date.fromisoformat(adopted_utc[:10]):
        raise SystemExit("REFUSED: --received is earlier than the lock of the scoring plan")
    return d.isoformat()


def text(res: dict) -> str:
    def pc(c):
        ci = c["pass_ci_of_all"]
        return (f"{c['pass']} pass, {c['fail']} fail, {c['uncertain']} uncertain, {c['missing']} missing of {c['n']}"
                + (f" (pass share {c['pass_share_of_all']:.2f}, exact 95% CI {ci[0]:.2f} to {ci[1]:.2f})" if ci else ""))
    r = res["reviewer"]
    L = [f"Expert review scored under {PLAN}. Human review, not automated.",
         f"Reviewer: {r['qualification']}; OCTA years {r['octa_years']}; device {r['device']}; minutes {r['minutes']}; "
         f"named: {r['named'] or 'anonymous'}; saw earlier verdicts: no (blinded packet). Received {res['received']}; "
         f"entered by {res['entered_by']}.", "", "Part A (dataset-label graphs):"]
    for s, d in res["part_A"]["by_fov"].items():
        L.append(f"  {s:8s} expert:  {pc(d['expert'])}   eyes: {res['part_A']['disease_mix'][s]}")
        L.append(f"  {s:8s} earlier: {pc(d['earlier'])}")
        ag = res["part_A"]["agreement_with_earlier"][s]
        g3, g2 = ag["three_way"], ag["pass_vs_not"]
        if g3["agreement"] is not None:
            extra = ""
            if "kappa" in g2:
                k3 = "undefined" if g3["kappa"] is None else f"{g3['kappa']:.2f}"
                k2 = "undefined" if g2["kappa"] is None else f"{g2['kappa']:.2f}"
                extra = f"; kappa three-way {k3}, pass vs not {k2}"
            pa = "n/a" if g2["positive_agreement"] is None else f"{g2['positive_agreement']:.2f}"
            na = "n/a" if g2["negative_agreement"] is None else f"{g2['negative_agreement']:.2f}"
            L.append(f"           agreement with earlier: three-way {g3['agreement']:.2f}, pass vs not "
                     f"{g2['agreement']:.2f} (positive {pa}, negative {na}){extra}")
    if res.get("part_B"):
        L += ["", "Part B (classical tracers; same scans as Part A, not independent):"]
        for t, d in res["part_B"]["per_tracer"].items():
            for s, c in d.items():
                L.append(f"  {t:24s} {s:8s} {pc(c)}")
        if "paired_pass_vs_not" in res["part_B"]:
            L.append(f"  paired: {res['part_B']['paired_pass_vs_not']}")
    L += ["", f"Readings (plan): {res['readings']}",
          "Every fail counts whatever its reason; reasons are reported verbatim and never used to re-classify."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part-a", required=True)
    ap.add_argument("--part-b")
    ap.add_argument("--original", required=True)
    ap.add_argument("--received", required=True)
    ap.add_argument("--qualification", required=True)
    ap.add_argument("--entered-by", required=True)
    ap.add_argument("--octa-years")
    ap.add_argument("--device")
    ap.add_argument("--minutes")
    ap.add_argument("--named")
    ap.add_argument("--tag", default="expert1", help="short folder tag, one per expert, e.g. expert1, expert2")
    a = ap.parse_args(argv)
    if not (ROOT / RECORD).exists():
        sys.exit("REFUSED: the scoring plan is not locked (no ledger record)")
    rec = read_json(ROOT / RECORD)
    bad = lock_problems(rec)
    if bad:
        sys.exit("REFUSED: " + "; ".join(bad))
    received = received_date(a.received, rec["adopted_utc"])
    if not a.tag.replace("-", "").replace("_", "").isalnum():
        sys.exit("REFUSED: --tag must be letters, digits, '-' or '_'")
    out = ROOT / PACKET_DIR / f"scored_{received}_{a.tag}"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"REFUSED: {out.relative_to(ROOT)} already has files; use a new tag")
    key, earlier, disease = load_key(), load_earlier(), load_disease()
    rows_a = read_sheet(Path(a.part_a), "A")
    rows_b = read_sheet(Path(a.part_b), "B") if a.part_b else None
    res = {"plan": PLAN, "record": RECORD, "received": received, "entered_by": a.entered_by, "is_human_review": True,
           "reviewer": {"qualification": a.qualification, "octa_years": a.octa_years, "device": a.device,
                        "minutes": a.minutes, "named": a.named, "saw_earlier_verdicts": False,
                        "did_part_b": bool(rows_b)},
           "part_A": score_a(rows_a, key, earlier, disease), "part_B": score_b(rows_b, key) if rows_b else None}
    res["readings"] = readings(res["part_A"])
    out.mkdir(parents=True, exist_ok=True)
    inputs = {"part_A.csv": a.part_a, **({"part_B.csv": a.part_b} if a.part_b else {}),
              "original_reply" + Path(a.original).suffix: a.original}
    for name, src in inputs.items():
        shutil.copyfile(src, out / name)
    unblinded = [{**r, **{k: key[r["label"]][k] for k in ("image_id", "fov_mm", "earlier_review_id")},
                  "earlier_verdict": earlier[key[r["label"]]["earlier_review_id"]]} for r in rows_a]
    if rows_b:
        unblinded += [{**r, **{k: key[r["label"]][k] for k in ("image_id", "fov_mm", "tracer_X", "tracer_Y")}}
                      for r in rows_b]
    cols = sorted({c for u in unblinded for c in u})
    with open(out / "unblinded.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(unblinded)
    res["provenance"] = {"created_utc": now_utc(), "script_sha256": sha256_file(Path(__file__)),
                         "inputs_sha256": {n: sha256_file(out / n) for n in inputs}, "environment": environment()}
    write_json(out / "summary.json", res)
    (out / "summary.txt").write_text(text(res))
    print(text(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
