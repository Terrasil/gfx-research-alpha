#!/usr/bin/env python3
"""Analyze the paired two-layer opacity ablation without inventing missing results."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

METHODS = (
    'deferred-sparse-layer-2-edge-aware',
    'deferred-sparse-layer-2-edge-decoupled',
    'deferred-sparse-layer-2-occupancy-raw',
    'deferred-sparse-layer-2-occupancy-conditional',
)

def load(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def run(directory):
    meta = dict(line.split('=',1) for line in (directory/'manifest.txt').read_text().splitlines() if '=' in line)
    if meta.get('completed') != '1': raise ValueError('Incomplete benchmark')
    methods_expected = METHODS + (('deferred-sparse-layer-2-exposure-ordinal',) if meta.get('storage_study','').startswith('hybrid-') else ())
    data = load(directory/'attribute_quality.csv')
    if not data:
        raise ValueError('No attribute-quality data (was benchmark completed?)')
    comparisons = defaultdict(dict)
    for row in data:
        if row['method'] not in methods_expected:
            continue
        key = (row['pair_id'], row['layer'])
        if row['method'] in comparisons[key]:
            raise ValueError(f'Duplicate quality arm for {key}: {row["method"]}')
        comparisons[key][row['method']] = row
    if not comparisons:
        raise ValueError('No new opacity methods in attribute_quality.csv')
    output = []
    for (pair_id, layer), methods in sorted(comparisons.items()):
        if set(methods)!=set(methods_expected):
            raise ValueError(f'Incomplete matched arms at {pair_id}/{layer}: {sorted(methods)}')
        for name in methods_expected:
            row = methods[name]
            # Keep denominators and visibility in the output; error on matched IDs
            # is conditional on successful reconstruction, not full coverage.
            output.append({
                'pair_id':pair_id, 'layer':layer, 'method':name,
                'reference_samples':row['reference_samples'],
                'matched_samples':row['matched_samples'],
                'match_recall':row['match_recall'],
                'alpha_rmse_matched':row['alpha_rmse'],
                'alpha_signed_mean_matched':row['alpha_signed_mean'],
            })
    with (directory/'opacity_ablation.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    print('Validated', len(comparisons), 'paired geometry/layer configurations')
    for layer in sorted({row['layer'] for row in output}):
        for method in methods_expected:
            entries=[row for row in output if row['layer']==layer and row['method']==method]
            values=[float(row['alpha_rmse_matched']) for row in entries if row['alpha_rmse_matched']]
            values=[v for v in values if math.isfinite(v)]
            recall=[float(row['match_recall']) for row in entries if row['match_recall']]
            print(f'layer={layer:>2} {method:>49}  finite_rmse={len(values):3}  '
                  f'equal-case_mean_rmse={(sum(values)/len(values)) if values else float("nan"):.6f}  '
                  f'mean_recall={(sum(recall)/len(recall)) if recall else float("nan"):.5f}')
    print('Warning: RMSE uses the matched surface-ID mask; compare recall and conditional errors together.')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results',type=Path)
    run(ap.parse_args().results)
