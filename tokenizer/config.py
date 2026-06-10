"""Simplified token types for EHR-Predict, adapted from hf_ehr/config.py.

Kept minimal: no stats tracking, no OMOP-specific logic.
All field codes are known upfront from field_config.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal


@dataclass
class TokenConfigEntry:
    """One token in the tokenizer vocab."""
    code: str
    type: Literal['code', 'categorical', 'numerical_range']
    description: Optional[str] = None
    tokenization: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_token(self) -> str:
        raise NotImplementedError


@dataclass
class CodeTCE(TokenConfigEntry):
    """A field-identity token. E.g. [AGE], [HR], [SBP]"""
    type: str = 'code'

    def to_token(self) -> str:
        return f"[{self.code}]"


@dataclass
class CategoricalTCE(TokenConfigEntry):
    """A single categorical value token. E.g. [SEX_M], [MECH_BLUNT]"""
    type: str = 'categorical'
    tokenization: Dict[str, Any] = field(default_factory=lambda: {
        'category': '',
        'values': [],  # all possible values for reference
    })

    def to_token(self) -> str:
        cat = self.tokenization['category']
        return f"[{self.code}_{cat}]"


@dataclass
class NumericalRangeTCE(TokenConfigEntry):
    """A numerical bucket token. E.g. [SBP_BIN_110_129]"""
    type: str = 'numerical_range'
    tokenization: Dict[str, Any] = field(default_factory=lambda: {
        'unit': None,
        'range_start': None,
        'range_end': None,
    })

    def to_token(self) -> str:
        return f"[{self.code}_BIN_{self.tokenization['range_start']}_{self.tokenization['range_end']}]"


# ── Control tokens ──
STRUCTURAL_TOKENS = [
    "[STATIC]",
    "[SEP]",
    "[MASK]",
    "[PAD]",
    "[CLS]",
]

TIME_BLOCK_TOKENS = [
    f"[DAY_{d}]" for d in range(1, 33)  # up to 32 days (12 needed + 20 buffer)
] + [
    f"[HOUR_{h}]" for h in range(0, 25)  # up to 24 hours in current day
]

G2_STAR_TOKEN = "[FIRST48]"


def load_tokenizer_config(path: str) -> List[TokenConfigEntry]:
    """Load tokenizer config from JSON."""
    raw = json.load(open(path))
    entries = []
    for e in raw['tokens']:
        t = e['type']
        if t == 'code':
            entries.append(CodeTCE(**e))
        elif t == 'categorical':
            entries.append(CategoricalTCE(**e))
        elif t == 'numerical_range':
            entries.append(NumericalRangeTCE(**e))
    return entries
