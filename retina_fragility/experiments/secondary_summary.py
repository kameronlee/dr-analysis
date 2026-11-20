"""Holm correction within the prespecified secondary families, plus a p-value register.

Family members are "EXP:feature" keys, e.g. "A:FI_adaptive_betweenness",
"D:G10", "D:qc_adaptive_betweenness", "C:wilcoxon_z_gt_0". For A/D members the
p-value is the pooled two-sided Mann-Whitney p (prespecified). "D:G10" refers
to G10 of the primary order. The single primary H3 test is never included.

The family size is fixed at the planned number of members. An unavailable
member is reported and counted as p = 1 (``stats.effects.holm``).

``p_value_register`` lists EVERY p-value that A, C and D report, each with its
role (``stats.multiplicity``). Only the rows with a ``holm:<family>`` role
receive a Holm correction. All other rows are exploratory and uncorrected. Per-FOV
scopes, the other features and orders, and the age/sex(/FOV)-adjusted LRTs are
in that uncorrected group.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..provenance import SYNTHETIC_BANNER, read_json, write_json
from ..stats import multiplicity
from ..stats.effects import holm


def run(cfg: dict, exp_dirs: dict[str, str | Path], out_path: str | Path) -> dict:
    loaded = {}
    for k, d in exp_dirs.items():
        p = Path(d) / "results.json"
        loaded[k] = read_json(p) if p.exists() else None
    members = multiplicity.family_members(cfg)
    reg = multiplicity.register(cfg, loaded)
    by_member = {r["member"]: r for r in reg if r["member"]}
    families = {}
    for fam, planned in cfg["stats"]["holm_families"].items():
        pvals, notes = {}, {}
        for m in planned:
            exp = m.split(":", 1)[0]
            r = by_member.get(m)
            if loaded.get(exp) is None:
                pvals[m], notes[m] = None, f"experiment {exp} results missing; counted as p = 1 in m"
            elif r is None or r["p"] is None:
                pvals[m] = None
                notes[m] = "p-value unavailable (not estimable or not run); counted as p = 1 in m"
            else:
                pvals[m] = r["p"]
        families[fam] = {"planned_members": list(planned), "family_size_m": len(planned),
                         "raw_p": pvals, "holm_adjusted_p": holm(pvals), "notes": notes}
    adj = {m: families[f]["holm_adjusted_p"][m] for f in families for m in families[f]["raw_p"]}
    for r in reg:
        r["holm_adjusted_p"] = adj.get(r["member"]) if r["member"] else None
    provenance = sorted({r.get("data_provenance") for r in loaded.values() if r} - {None})
    n_corr = sum(1 for r in reg if r["family"])
    out = {"families": families, "primary_excluded": True,
           "missing_member_policy": ("fixed family: an unavailable planned member keeps its place in m "
                                     "and is treated as p = 1 (never rejected; adjusted p = null)"),
           "p_value_source": "pooled two-sided Mann-Whitney for A/D members; one-sided Wilcoxon for C",
           "p_value_register": reg,
           "register_counts": {"holm_corrected": n_corr, "exploratory_uncorrected": len(reg) - n_corr},
           "statement": (f"Only {n_corr} of the {len(reg)} p-values reported by A, C and D belong to a "
                         "prespecified Holm family. All others are exploratory and uncorrected."),
           "data_provenance": provenance,
           "family_member_keys": {v[1]: list(k) for k, v in members.items()}}
    write_json(out_path, out)
    out_path = Path(out_path)
    pd.DataFrame(reg).to_csv(out_path.with_name("p_value_register.csv"), index=False)
    if SYNTHETIC_BANNER in provenance:
        out_path.with_name("SYNTHETIC_NOT_CLINICAL.txt").write_text(
            "The secondary summary was computed from SYNTHETIC fixtures; not clinical findings.\n")
    return out
