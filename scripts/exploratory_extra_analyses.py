"""Additional analyses fixed in the 2026-10-09 pre-outcome deviation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from retina_fragility.calibration import load_calibration
from retina_fragility.config import load_config
from retina_fragility.experiments.common import load_analysis_table
from retina_fragility.manifest import load_manifest
from retina_fragility.provenance import sha256_file, write_json
from retina_fragility.stats.primary import complete_cases, nested_lrt


def convention_agreement(table, primary, variants):
    rows = []
    groups = [('pooled', table)] + [(f'FOV={v}', x) for v, x in table.groupby('fov_mm')]
    for scope, frame in groups:
        for variant in variants:
            pair = frame[[primary, variant]].replace([np.inf, -np.inf], np.nan).dropna()
            valid = len(pair) > 1 and pair[primary].nunique() > 1 and pair[variant].nunique() > 1
            rho = float(spearmanr(pair[primary], pair[variant]).statistic) if valid else None
            disagreement = (float((np.sign(pair[primary]) != np.sign(pair[variant])).mean())
                            if len(pair) and variant.startswith('FI_') else None)
            rows.append(dict(scope=scope, variant=variant, n=len(pair), spearman_rho=rho,
                             fi_sign_disagreement_fraction=disagreement,
                             status='ok' if valid else 'correlation_not_estimable'))
    return pd.DataFrame(rows)


def fragmentation_models(table, cfg):
    st = cfg['stats']
    fi = st['fi_column']
    s0 = 'S0_' + cfg['stress']['primary_order']
    baseline = list(st['baseline_primary'])
    common, audit = complete_cases(table, baseline + [s0, fi], 'y')
    original = nested_lrt(common, 'y', baseline, fi, st,
                          'original baseline on S0-complete subset', 'exploratory')
    adjusted = nested_lrt(common, 'y', baseline + [s0], fi, st,
                          'additional S0-adjusted nested model', 'exploratory')
    assert original['n'] == adjusted['n'] == len(common)
    return dict(original_same_subset=original, s0_adjusted=adjusted, complete_case_audit=audit,
                interpretation='Exploratory, uncorrected; S0 and FI are derived from the same graph. '
                               'Adjustment assesses incremental association beyond observed fragmentation, not causality.')


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / 'outputs/exploratory_20261009'
    out = run / 'extra_analyses'
    if out.exists():
        raise SystemExit('Refusing to overwrite extra analyses')
    cfg = load_config(root / 'config.yaml')
    manifest_path = root / 'data/manifests/octa500_manifest.csv'
    cal_path = root / 'outputs/calibration/calibration.json'
    manifest, cal = load_manifest(manifest_path, cfg), load_calibration(cal_path)
    table, info = load_analysis_table(run / 'pipeline', manifest, cal, cfg['data']['primary_dataset'])
    variants = [f'{v}_{cfg["stress"]["primary_order"]}' for v in ('FI_right', 'FI_trap', 'FI_trap_len', 'qc_len')]
    agreement = convention_agreement(table, cfg['stats']['fi_column'], variants)
    models = fragmentation_models(table, cfg)
    out.mkdir()
    agreement.insert(0, 'analysis_status', 'exploratory_reference_label_analysis')
    agreement.to_csv(out / 'convention_agreement.csv', index=False)
    files = [root / 'reports/exploratory_amendment_2026-10-09.json', manifest_path, cal_path,
             run / 'pipeline/features.csv', run / 'pipeline/run.json', root / 'config.yaml', Path(__file__)]
    write_json(out / 'results.json', dict(analysis_status='exploratory_reference_label_analysis',
        human_qc_satisfied=False, data_provenance=info['data_provenance'],
        source_sha256={str(p.relative_to(root)): sha256_file(p) for p in files}, **models))
    print(json.dumps({'n': len(table), 'adjusted_model_status': models['s0_adjusted']['status']}))


if __name__ == '__main__':
    main()
