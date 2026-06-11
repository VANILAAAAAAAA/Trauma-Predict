# STATIC Token Vocabulary

Scope: only tokens used by the STATIC block. Other token families are in separate files.

## STATIC Sequence

Current model-side sequence template:

```text
[STATIC] [age] <age_value> [age_BIN_*] [male_M/F] [mechanism_cat_B/P/O] [transfer_D/T] [initial_ed_sbp] <sbp_value> [initial_ed_sbp_BIN_*] [rsi] <rsi_value> [rsi_BIN_*] [head_injury_Y/N] [SEP]
```

> Formal Method 1 note: raw numeric values should become value tensors for an MLP projector, not ordinary tokenizer IDs.

## Structural Tokens Used by STATIC

- `[STATIC]`
- `[SEP]`

## Field Identity Tokens

- `[age]` — Age at admission
- `[male]` — Sex; M=male F=female
- `[mechanism_cat]` — Injury mechanism
- `[transfer]` — Transfer context
- `[initial_ed_sbp]` — Initial ED SBP
- `[rsi]` — Reverse shock index = SBP/HR
- `[head_injury]` — Head injury from ICD

## Categorical Value Tokens

- `[male_F]` — Sex; M=male F=female=F
- `[male_M]` — Sex; M=male F=female=M
- `[mechanism_cat_B]` — Injury mechanism=B
- `[mechanism_cat_O]` — Injury mechanism=O
- `[mechanism_cat_P]` — Injury mechanism=P
- `[transfer_D]` — Transfer context=D
- `[transfer_T]` — Transfer context=T
- `[head_injury_N]` — Head injury from ICD=N
- `[head_injury_Y]` — Head injury from ICD=Y

## Numerical Bucket Tokens

- `[age_BIN_0_18]` — years
- `[age_BIN_18_30]` — years
- `[age_BIN_30_45]` — years
- `[age_BIN_45_60]` — years
- `[age_BIN_60_75]` — years
- `[age_BIN_75_90]` — years
- `[initial_ed_sbp_BIN_0_90]` — mmHg
- `[initial_ed_sbp_BIN_90_111]` — mmHg
- `[initial_ed_sbp_BIN_111_140]` — mmHg
- `[initial_ed_sbp_BIN_140_180]` — mmHg
- `[initial_ed_sbp_BIN_180_300]` — mmHg
- `[rsi_BIN_0_1.0]` — ratio
- `[rsi_BIN_1.0_1.1]` — ratio
- `[rsi_BIN_1.1_1.8]` — ratio
- `[rsi_BIN_1.8_3.0]` — ratio
- `[rsi_BIN_3.0_20.0]` — ratio

## Current Caveats Before Freezing

1. Token names are still field-code oriented (`male`, `mechanism_cat`, `rsi`). Before final vocab freeze, consider semantic aliases: `sex`, `injury_mechanism`, `reverse_shock_index`.
2. Single-letter category values are compact, but custom vocab no longer needs this compression. Consider replacing `[mechanism_cat_B]` with `[injury_mechanism_blunt]` for interpretability.
3. `rsi` means reverse shock index here; avoid confusion with rapid sequence intubation by renaming the token before training.
4. Missingness should be represented explicitly in model records (`observed_flag`, `ed_linkage`) rather than only omitting values.