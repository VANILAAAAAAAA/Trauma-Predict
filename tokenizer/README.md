# EHRPredict Tokenizer

Adapted from hf_ehr tokenizer pipeline. Method 1: token-slot with special tokens.

## Files

```
tokenizer/
├── config.py                  ← TokenConfigEntry types (Code/Categorical/NumericalRange)
├── build_tokenizer.py         ← Generate tokenizer_config.json from field_config.json
├── ehrpredict_tokenizer.py    ← EHRPredictTokenizer (PreTrainedTokenizer subclass)
├── tokenizer_config.json      ← Generated token config (489 tokens)
├── static_token.md            ← Human-readable token list for inspection
└── README.md
```

## Token Categories

| Type | Count | Example |
|---|---|---|
| Structural | 5 | `[STATIC] [SEP] [CLS] [MASK] [PAD]` |
| Time blocks | 326 | `[DAY_1]...[DAY_13] [HOUR_0]...[HOUR_312]` |
| G2* marker | 1 | `[FIRST48]` |
| Field identity | 30 | `[age] [hr] [sbp] ...` |
| Categorical | 9 | `[male_M] [male_F] [mech_B] [mech_P] [mech_O]` |
| Numerical buckets | 118 | `[age_BIN_18_30] [sbp_BIN_110_140]` |
| **Total** | **489** | |

## Usage

```python
from ehrpredict_tokenizer import EHRPredictTokenizer, tokenize_static, tokenize_hourly

tok = EHRPredictTokenizer('tokenizer/tokenizer_config.json')
# → vocab_size = 489

static_tokens = tokenize_static({'age': 67, 'male': 1, ...}, tok)
# → ['[STATIC]', '[age]', '67.0', '[male]', '[male_M]', ...]

hourly_tokens = tokenize_hourly(0, hourly_row, tok)
# → ['[HOUR_0]', '[hr]', '73.6', '[sbp]', '104.25', ...]
```

## Update Process

1. Edit `field_config.json`
2. Run `python3 tokenizer/build_tokenizer.py`
3. Verify `tokenizer/static_token.md`
4. Sample auto-syncs via `build_current_field_sample.py`

## Method 2 (structured text)

The structured text format (e.g., `age-50 male-M mech-B`) is preserved as a second path for direct LLM input. Tokenizer only covers Method 1 (token-slot embedding).

## MLP Projector

Numerical values (e.g., `128.0` for SBP) go through MLP projection separately from token embeddings:

```text
[field_token] → embedding → e_field
value → MLP(value) → e_value
→ combined = e_field + e_value → transformer
```

Bucket tokens are optional auxiliary signals. See model design for details.
