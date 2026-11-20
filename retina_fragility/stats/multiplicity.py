"""Explicit multiplicity role for every reported p-value.

Each p-value an experiment reports has exactly one role:

* ``primary`` — the single prespecified H3 LRT (Experiment B). It stands
  alone and is never part of a Holm family.
* ``holm:<family>`` — a member of a prespecified secondary family in
  ``stats.holm_families``. The Holm-adjusted value is reported only in
  ``secondary_holm.json``. The family size stays fixed when a member is
  unavailable (``stats.effects.holm``).
* ``sensitivity_uncorrected`` — a labelled sensitivity of the primary model.
  It receives no multiplicity correction and never replaces the primary result.
* ``exploratory_uncorrected`` — every other p-value (per-FOV scopes, other
  features and orders, adjusted-logistic LRTs in A/D, E trend tests). It is
  NOT Holm-corrected and must not be read as confirmatory.

Family members are written ``EXP:feature``. For A and D the member is the
pooled two-sided Mann-Whitney p of that feature, and ``D:G10`` means G10 of the
primary order. For C the member is the one-sided Wilcoxon p over all DR images.
"""
from __future__ import annotations

PRIMARY = "primary"
SENSITIVITY = "sensitivity_uncorrected"
EXPLORATORY = "exploratory_uncorrected"
C_KEY = ("C", "all_dr", "median_z", "wilcoxon_p_greater")


def family_members(cfg: dict) -> dict[tuple[str, str, str, str], tuple[str, str]]:
    """(experiment, scope, feature, test) -> (family, member label). Validates the config."""
    q = int(round(cfg["stress"]["q_eval"] * 100))
    po = cfg["stress"]["primary_order"]
    out: dict[tuple[str, str, str, str], tuple[str, str]] = {}
    for fam, members in cfg["stats"]["holm_families"].items():
        for mem in members:
            exp, feat = mem.split(":", 1)
            if exp in ("A", "D"):
                f = f"G{q}_{po}" if feat == f"G{q}" else feat
                key = (exp, "pooled", f, "mwu_p")
            elif exp == "C" and feat == "wilcoxon_z_gt_0":
                key = C_KEY
            else:
                raise ValueError(f"unsupported Holm family member {mem!r}")
            if key in out:
                raise ValueError(f"{mem!r} is in two Holm families ({out[key][0]}, {fam})")
            out[key] = (fam, mem)
    return out


def role(cfg: dict, experiment: str, scope: str, feature: str, test: str) -> str:
    fam = family_members(cfg).get((experiment, scope, feature, test))
    return f"holm:{fam[0]}" if fam else EXPLORATORY


def annotate_comparisons(rows: list[dict], experiment: str, cfg: dict) -> list[dict]:
    """Add ``mwu_p_role`` and ``adjusted_lr_p_role`` to Experiment A/D comparison rows."""
    members = family_members(cfg)
    for r in rows:
        key = (experiment, r.get("scope"), r.get("feature"), "mwu_p")
        fam = members.get(key)
        r["mwu_p_role"] = f"holm:{fam[0]}" if fam else EXPLORATORY
        r["adjusted_lr_p_role"] = EXPLORATORY
    return rows


def register(cfg: dict, results: dict[str, dict | None]) -> list[dict]:
    """Every p-value reported by A, C and D with its role (the multiplicity register)."""
    members = family_members(cfg)
    out = []
    for exp in ("A", "D"):
        res = results.get(exp)
        if not res:
            continue
        for r in res.get("comparisons", []):
            for test in ("mwu_p", "adjusted_lr_p"):
                if test not in r and r.get("status") != "ok":
                    continue
                fam = members.get((exp, r.get("scope"), r.get("feature"), test))
                out.append({"experiment": exp, "scope": r.get("scope"), "feature": r.get("feature"),
                            "test": test, "p": r.get(test),
                            "role": f"holm:{fam[0]}" if fam else EXPLORATORY,
                            "family": fam[0] if fam else None, "member": fam[1] if fam else None})
    res = results.get("C")
    if res:
        fam = members.get(C_KEY)
        out.append({"experiment": "C", "scope": "all_dr", "feature": "median_z",
                    "test": "wilcoxon_p_greater", "p": res.get("summary", {}).get("wilcoxon_p_greater"),
                    "role": f"holm:{fam[0]}" if fam else EXPLORATORY,
                    "family": fam[0] if fam else None, "member": fam[1] if fam else None})
    return out
