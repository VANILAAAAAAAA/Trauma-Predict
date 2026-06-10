"""Build EHRPredict tokenizer config from field_config.json — fixed categorical tokens."""
from __future__ import annotations

import json, argparse
from pathlib import Path
from collections import defaultdict
from config import (
    CodeTCE, CategoricalTCE, NumericalRangeTCE,
    STRUCTURAL_TOKENS, TIME_BLOCK_TOKENS, G2_STAR_TOKEN,
)

BUCKET_DEFS = {
    'age': [0, 18, 30, 45, 60, 75, 90],
    'initial_ed_sbp': [0, 90, 111, 140, 180, 300],
    'rsi': [0, 1.0, 1.1, 1.8, 3.0, 20.0],
    'hr': [0, 50, 60, 80, 100, 120, 200],
    'sbp': [0, 80, 90, 110, 140, 180, 300],
    'dbp': [0, 50, 60, 80, 100, 200],
    'map': [0, 60, 70, 90, 110, 200],
    'rr': [0, 8, 12, 20, 28, 60],
    'temp': [30, 35, 36, 37.5, 38.5, 42],
    'fio2': [0, 0.21, 0.4, 0.6, 1.0],
    'base_def_48': [0, 3, 6, 10, 30],
    'lactate_48': [0, 3.0, 5.0, 10.0, 30.0],
    'rbc_48': [0, 500, 2000, 5000, 10000],
    'crys_48': [0, 2000, 5000, 10000, 30000],
    'bolus_sum_until_h': [0, 2000, 5000, 10000, 30000],
    'rbc_sum_until_h': [0, 500, 2000, 5000, 10000],
    'vent_day_sum_until_h': [0, 1, 3, 7, 14],
    'bicarb': [0, 15, 22, 26, 32, 50],
    'strong_ion': [0, 20, 30, 40, 50, 80],
    'bun': [0, 10, 20, 40, 80, 200],
    'creatinine': [0, 0.5, 1.0, 1.5, 3.0, 15.0],
    'wbc': [0, 4, 10, 15, 30, 100],
    'lymphocytes': [0, 0.5, 1.5, 5.0, 100],
    'neutrophils': [0, 1.5, 7.0, 15.0, 60],
    'uop': [0, 30, 50, 100, 200, 500],
}

def load_field_config(path): return json.load(open(path))

def build_token_config(field_config):
    tokens = []
    # Structural
    for t in STRUCTURAL_TOKENS:
        tokens.append(CodeTCE(code=t.strip('[]'), description='structural'))
    for t in TIME_BLOCK_TOKENS:
        tokens.append(CodeTCE(code=t.strip('[]'), description='time_block'))
    tokens.append(CodeTCE(code=G2_STAR_TOKEN.strip('[]'), description='first48'))
    # Fields
    for gname, group in field_config['groups'].items():
        for fdef in group['fields']:
            fn = fdef['field']; ft = fdef.get('type', 'float'); desc = fdef.get('desc', '')
            # Field identity
            tokens.append(CodeTCE(code=fn, description=desc))
            if ft == 'cat' and 'values' in fdef:
                # One token per categorical value
                for v in fdef['values']:
                    tokens.append(CategoricalTCE(code=fn, tokenization={'category': v, 'values': fdef['values']}, description=f'{desc}={v}'))
            elif ft in ('float', 'int'):
                buckets = BUCKET_DEFS.get(fn)
                if buckets:
                    unit = fdef.get('unit', '')
                    for i in range(len(buckets)-1):
                        tokens.append(NumericalRangeTCE(
                            code=fn,
                            tokenization={'unit': unit, 'range_start': buckets[i], 'range_end': buckets[i+1]},
                            description=f'{desc} [{buckets[i]}-{buckets[i+1]}]',
                        ))
    return tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--field-config', default='data dicision/field adapter/field_config.json')
    ap.add_argument('--out', default='tokenizer/tokenizer_config.json')
    args = ap.parse_args()
    fc = load_field_config(args.field_config)
    tokens = build_token_config(fc)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'tokens': [t.to_dict() for t in tokens], 'n_tokens': len(tokens)}, open(out, 'w'), indent=2)
    by_type = defaultdict(int)
    for t in tokens: by_type[t.type] += 1
    print(f'Wrote {len(tokens)} tokens to {out}')
    for k,v in sorted(by_type.items()): print(f'  {k}: {v}')

if __name__ == '__main__': main()
