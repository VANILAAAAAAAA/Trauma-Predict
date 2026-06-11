# Token Vocabulary Design Review

## Current status

- Token config source: `tokenizer/tokenizer_config.json`
- Generator: `tokenizer/build_tokenizer.py`
- Field registry source: `data dicision/field adapter/field_config.json`
- Current vocabulary size: 220 entries
- Time tokens: `[DAY_1]`–`[DAY_32]`, `[HOUR_0]`–`[HOUR_24]`

## What is solid

1. Custom vocabulary is the right direction for Method 1. It prevents a general BPE tokenizer from guessing clinical field identity from fragments.
2. STATIC / DAY / HOUR block organization is consistent with the agreed 13-day window: completed daily summaries + current-day detailed hours.
3. Continuous values should be preserved as numeric tensors and projected by an MLP; bucket tokens should be auxiliary, not the only numeric representation.
4. Cat thresholds from UW are useful for bins/evaluation/report labels, but raw continuous values remain primary input.

## Problems to fix before vocabulary freeze

1. **Semantic token names.** Current tokens expose implementation field names (`male`, `mechanism_cat`, `rsi`). For final training, prefer semantic tokens such as `sex`, `injury_mechanism`, `reverse_shock_index`.
2. **Categorical compression is no longer necessary.** `[mechanism_cat_B]` works, but `[injury_mechanism_blunt]` is clearer and costs the same as one special token.
3. **Numeric strings should not be converted to tokenizer IDs.** `tokenize_static()` currently emits strings like `67.0`; formal Method 1 should emit structured records: `(field_token_id, value_float, observed_flag, recency, bucket_token_id)`.
4. **Missingness needs explicit tokens/flags.** ED-missing fields require `ed_linkage` or `observed_flag`, not silent omission.
5. **Tokenizer class is a prototype.** The current subclass supports vocabulary inspection and sequence sketching; model training should use a dataset collator that creates token IDs plus value tensors.

## Recommended next implementation step

Create `tokenizer/records.py` with a dataclass:

```python
@dataclass
class TokenRecord:
    block: str
    field: str
    field_token: str
    value: float | None
    value_mask: int
    bucket_token: str | None
    categorical_token: str | None
    recency_hours: float | None
    source: str | None
```

Then tokenizer output becomes auditable records, and the model collator turns them into:

```text
field_token_ids
bucket_token_ids
categorical_token_ids
value_tensor
value_mask
recency_tensor
segment_ids
time_ids
```
