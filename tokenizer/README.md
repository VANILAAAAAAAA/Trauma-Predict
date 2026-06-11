# EHRPredict Tokenizer

Adapted from hf_ehr tokenizer ideas, but narrowed to EHR-Predict Method 1 token-slot inputs.

## Files

```text
tokenizer/
├── config.py                     # TokenConfigEntry types
├── build_tokenizer.py            # Generate tokenizer_config.json from field_config.json
├── ehrpredict_tokenizer.py       # Prototype PreTrainedTokenizer subclass for vocab inspection
├── tokenizer_config.json         # Generated token config (220 entries)
├── static_token.md               # STATIC-only vocabulary discussion
├── time_token.md                 # Structural/time-block tokens
├── field_token.md                # Non-STATIC field identity tokens
├── numerical_bucket_token.md     # Non-STATIC bucket tokens
├── token_design_review.md        # Current design risks and next changes
└── README.md
```

## Current count

| Family | Count |
|---|---:|
| Code / structural / field / time | 93 |
| Categorical | 9 |
| Numerical range | 118 |
| **Total** | **220** |

## Design boundary

Method 1 does not rely on natural-language tokenization for clinical semantics.

```text
field/category/bin tokens -> embedding table
continuous values         -> normalized value tensors -> MLP projector
```

Bucket tokens are auxiliary clinical range signals. They do not replace continuous numeric prediction.

Method 2 structured text remains a separate path for direct LLM input.

## Update process

1. Edit `data dicision/field adapter/field_config.json`.
2. Run `python3 tokenizer/build_tokenizer.py`.
3. Regenerate/split token docs if needed.
4. Validate sample token records before model training.
