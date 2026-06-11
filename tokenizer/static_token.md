# STATIC Token Vocabulary

## Sequence

```text
[STATIC]
[age] value
[sex_M|sex_F]
[injury_mechanism_blunt|injury_mechanism_penetrating|injury_mechanism_other]
[transfer_direct|transfer_transfer]
[ed_linkage_yes|ed_linkage_no]
[initial_ed_sbp] value [initial_ed_sbp_bin_*]
[reverse_shock_index] value [reverse_shock_index_bin_*]
[head_injury_yes|head_injury_no]
[SEP]
```

## Field Tokens

| Source field | Token | Value path | Bucket |
|---|---|---|---|
| `age` | `[age]` | value tensor | none frozen |
| `male` | `[sex]` | categorical token | no |
| `mechanism_cat` | `[injury_mechanism]` | categorical token | no |
| `transfer` | `[transfer]` | categorical token | no |
| ED linkage | `[ed_linkage]` | categorical token | no |
| `initial_ed_sbp` | `[initial_ed_sbp]` | value tensor | UW Cat |
| `rsi` | `[reverse_shock_index]` | value tensor | UW Cat |
| `head_injury` | `[head_injury]` | categorical token | no |

## Categorical Tokens

| Field | Tokens |
|---|---|
| sex | `[sex_M]`, `[sex_F]` |
| injury mechanism | `[injury_mechanism_blunt]`, `[injury_mechanism_penetrating]`, `[injury_mechanism_other]` |
| transfer | `[transfer_direct]`, `[transfer_transfer]` |
| ED linkage | `[ed_linkage_yes]`, `[ed_linkage_no]` |
| head injury | `[head_injury_yes]`, `[head_injury_no]` |

## Evidence-backed Buckets

| Field | Tokens | Evidence |
|---|---|---|
| initial ED SBP | `[initial_ed_sbp_bin_severely_low]`, `[initial_ed_sbp_bin_mildly_low]`, `[initial_ed_sbp_bin_normal]` | `Initial.ED.SBPCat`: ≤89 / 90–110 / ≥111 |
| reverse shock index | `[reverse_shock_index_bin_high_risk]`, `[reverse_shock_index_bin_moderate_risk]`, `[reverse_shock_index_bin_low_risk]` | `rSICat`: ≤1.0 / 1.1–1.7 / ≥1.8 |

## Do Not Freeze Yet

| Item | Reason |
|---|---|
| age bucket | no accepted threshold evidence yet |
| extra SBP hypertension bins | not supported by UW Cat evidence |
| abbreviation tokens (`rsi`, `mech`, `B/P/O`) | custom vocab should use semantic names |
