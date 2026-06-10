"""EHRPredict tokenizer — extends HuggingFace PreTrainedTokenizer.

Method 1 (token-slot): produces structured token sequences with special tokens.
Method 2 (structured text): deferred to structured text compressor.
"""
from __future__ import annotations

import json
from typing import List, Dict, Optional
from transformers import PreTrainedTokenizer

from config import (
    load_tokenizer_config,
    CodeTCE, CategoricalTCE, NumericalRangeTCE,
    STRUCTURAL_TOKENS, TIME_BLOCK_TOKENS, G2_STAR_TOKEN,
    TokenConfigEntry,
)


class EHRPredictTokenizer(PreTrainedTokenizer):
    """Custom tokenizer for EHR-Predict structured patient sequences."""

    def __init__(self, path_to_tokenizer_config: str, **kwargs):
        # Build vocab before super().__init__ to satisfy get_vocab()
        self.path_to_tokenizer_config = path_to_tokenizer_config
        self.token_config: List[TokenConfigEntry] = load_tokenizer_config(path_to_tokenizer_config)

        all_tokens = STRUCTURAL_TOKENS + TIME_BLOCK_TOKENS + [G2_STAR_TOKEN]
        for entry in self.token_config:
            all_tokens.append(entry.to_token())
        all_tokens = sorted(set(all_tokens))

        # Pre-populate vocab so parent can call get_vocab() during init
        self._vocab = {t: i for i, t in enumerate(all_tokens)}

        super().__init__(
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="[MASK]",
            pad_token="[PAD]",
            **kwargs,
        )

        # Add tokens properly through parent
        self.add_tokens(all_tokens)

        # Field → token lookup
        self.field_token = {}
        self.cat_tokens = {}
        self.bucket_tokens = {}
        for entry in self.token_config:
            if isinstance(entry, CodeTCE):
                self.field_token[entry.code] = entry.to_token()
            elif isinstance(entry, CategoricalTCE):
                code = entry.code
                cat = entry.tokenization['category']
                self.cat_tokens[(code, cat)] = entry.to_token()
            elif isinstance(entry, NumericalRangeTCE):
                code = entry.code
                rs = entry.tokenization['range_start']
                re = entry.tokenization['range_end']
                self.bucket_tokens.setdefault(code, []).append((rs, re, entry.to_token()))

    def get_vocab(self) -> dict:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        return ()


# ── Block tokenizer helpers ──

CAT_MAP = {
    'male': {'1': 'M', '0': 'F'},
    'mechanism_cat': {'1': 'B', '2': 'P', '3': 'O'},
    'transfer': {'1': 'T', '0': 'D'},
    'head_injury': {'1': 'Y', '0': 'N'},
}


def tokenize_static(fields: dict, tokenizer: EHRPredictTokenizer) -> List[str]:
    """Tokenize STATIC block."""
    tokens = ["[STATIC]"]
    for fname in ['age', 'male', 'mechanism_cat', 'transfer', 'initial_ed_sbp', 'rsi', 'head_injury']:
        val = fields.get(fname)
        if val is None or val == '':
            continue
        ft = tokenizer.field_token.get(fname)
        if ft:
            tokens.append(ft)
        if fname in CAT_MAP:
            cat_val = CAT_MAP[fname].get(str(val), str(val))
            ct = tokenizer.cat_tokens.get((fname, cat_val))
            if ct:
                tokens.append(ct)
        else:
            tokens.append(str(val))
    tokens.append("[SEP]")
    return tokens


def tokenize_daily(day_idx: int, daily_row: dict, tokenizer: EHRPredictTokenizer) -> List[str]:
    """Tokenize one daily summary block."""
    tokens = [f"[DAY_{day_idx}]"]
    for key, val in daily_row.items():
        if val is None or val == '':
            continue
        base = key.split('_')[0]
        ft = tokenizer.field_token.get(key) or tokenizer.field_token.get(base)
        if ft:
            tokens.append(ft)
        tokens.append(str(val))
    tokens.append("[SEP]")
    return tokens


def tokenize_hourly(hour_idx: int, hourly_row: dict, tokenizer: EHRPredictTokenizer) -> List[str]:
    """Tokenize one hourly block."""
    tokens = [f"[HOUR_{hour_idx}]"]
    g1_order = ['hr', 'sbp', 'dbp', 'map', 'rr', 'temp', 'fio2']
    g3_order = ['bolus_sum_until_h', 'rbc_sum_until_h', 'vent_h', 'vent_day_sum_until_h']
    g4_order = ['bicarb', 'strong_ion', 'bun', 'creatinine', 'wbc', 'lymphocytes', 'neutrophils', 'uop']
    for fname in g1_order + g3_order + g4_order:
        val = hourly_row.get(fname)
        if val is None or val == '':
            continue
        ft = tokenizer.field_token.get(fname)
        if ft:
            tokens.append(ft)
        tokens.append(str(val))
    tokens.append("[SEP]")
    return tokens
