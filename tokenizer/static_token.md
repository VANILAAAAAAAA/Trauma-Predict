# STATIC Token Vocabulary

> Updated 2026-06-16. STATIC is bucket/categorical only. No numeric channels.
> Numeric MLP projection is restricted to the 7 HOUR vital signs.

## Sequence

```text
[STATIC]
[age_bin_*]
[sex_M|sex_F]
[injury_mechanism_blunt|injury_mechanism_penetrating|injury_mechanism_other]
[transfer_direct|transfer_transfer]
[ed_linkage_yes|ed_linkage_no]
[initial_ed_sbp_bin_*]
[reverse_shock_index_bin_*]
[head_injury_yes|head_injury_no]
[SEP]
```

All STATIC slots are pure vocabulary tokens (embedding lookup). No `<..._numeric>` channels.

## Field Tokens

| Source field | Token | Encoding path | Bucket |
|---|---|---|---|
| `age` | `[age_bin_*]` | bucket token | age bins |
| `male` | `[sex_M]` / `[sex_F]` | categorical token | no |
| `mechanism_cat` | `[injury_mechanism_blunt]` / `[injury_mechanism_penetrating]` / `[injury_mechanism_other]` | categorical token | no |
| `transfer` | `[transfer_direct]` / `[transfer_transfer]` | categorical token | no |
| ED linkage | `[ed_linkage_yes]` / `[ed_linkage_no]` | categorical token | no |
| `initial_ed_sbp` | `[initial_ed_sbp_bin_*]` | bucket token | UW Cat |
| `rsi` | `[reverse_shock_index_bin_*]` | bucket token | UW Cat |
| `head_injury` | `[head_injury_yes]` / `[head_injury_no]` | categorical token | no |

## Categorical Tokens

| Field | Tokens |
|---|---|
| sex | `[sex_M]`, `[sex_F]` |
| injury mechanism | `[injury_mechanism_blunt]`, `[injury_mechanism_penetrating]`, `[injury_mechanism_other]` |
| transfer | `[transfer_direct]`, `[transfer_transfer]` |
| ED linkage | `[ed_linkage_yes]`, `[ed_linkage_no]` |
| head injury | `[head_injury_yes]`, `[head_injury_no]` |

## Buckets

| Field | Tokens | Evidence |
|---|---|---|
| age | `[age_bin_18_39]`, `[age_bin_40_54]`, `[age_bin_55_64]`, `[age_bin_65_74]`, `[age_bin_75_84]`, `[age_bin_85_89]` | user-specified design strata |
| initial ED SBP | `[initial_ed_sbp_bin_hypotension]`, `[initial_ed_sbp_bin_borderline_low]`, `[initial_ed_sbp_bin_not_low]` | `Initial.ED.SBPCat`: ≤89 / 90–110 / ≥111; meaning is interpreted, not official UW codebook |
| reverse shock index | `[reverse_shock_index_bin_high_risk]`, `[reverse_shock_index_bin_intermediate]`, `[reverse_shock_index_bin_low_risk]` | `rSICat`: ≤1.0 / 1.1–1.7 / ≥1.8; meaning inferred from rSI=SBP/HR and SI literature |

## Do Not Freeze Yet

| Item | Reason |
|---|---|
| extra SBP hypertension bins | not supported by UW Cat evidence |
| abbreviation tokens (`rsi`, `mech`, `B/P/O`) | custom vocab should use semantic names |
| numeric channels in STATIC | numeric MLP projection is restricted to 7 HOUR vitals only |
