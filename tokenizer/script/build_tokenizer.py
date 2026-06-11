"""Build EHRPredict tokenizer config from field_config.json.

Rule: token names are semantic display/model names, while source fields remain in
field_config.json. Numerical buckets must be evidence-backed.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from config import (
    CodeTCE,
    CategoricalTCE,
    NumericalRangeTCE,
    STRUCTURAL_TOKENS,
    TIME_BLOCK_TOKENS,
    G2_STAR_TOKEN,
)

FIELD_ALIASES = {
    # STATIC
    "male": "sex",
    "mechanism_cat": "injury_mechanism",
    "rsi": "reverse_shock_index",
    # HOUR
    "hr": "heart_rate",
    "sbp": "systolic_bp",
    "dbp": "diastolic_bp",
    "map": "mean_arterial_pressure",
    "rr": "respiratory_rate",
    "temp": "temperature",
    "bolus_sum_until_h": "crystalloid_cumulative",
    "rbc_sum_until_h": "rbc_cumulative",
    "vent_h": "ventilation_status",
    "vent_day_sum_until_h": "ventilation_days_cumulative",
    "bicarb": "bicarbonate",
    "strong_ion": "strong_ion_difference",
    "uop": "urine_output",
    # FIRST48 / DAY
    "base_def_48": "base_deficit_48h",
    "lactate_48": "lactate_48h",
    "rbc_48": "rbc_48h",
    "crys_48": "crystalloid_48h",
}

CATEGORY_ALIASES = {
    "male": {"M": "M", "F": "F"},
    "mechanism_cat": {"B": "blunt", "P": "penetrating", "O": "other"},
    "transfer": {"D": "direct", "T": "transfer"},
    "head_injury": {"Y": "yes", "N": "no"},
}

# Evidence-backed buckets only.
# Source: tokenizer/reference/uw_cat_thresholds.md
BUCKET_DEFS = {
    "initial_ed_sbp": {
        "source": "UW Initial.ED.SBPCat: <=89 / 90-110 / >=111",
        "bins": [
            (None, 90, "hypotension"),
            (90, 111, "borderline_low"),
            (111, None, "not_low"),
        ],
    },
    "rsi": {
        "source": "UW rSICat: <=1.0 / 1.1-1.7 / >=1.8; implemented as <1.1 / 1.1-1.8 / >=1.8",
        "bins": [
            (None, 1.1, "high_risk"),
            (1.1, 1.8, "intermediate"),
            (1.8, None, "low_risk"),
        ],
    },
    "base_def_48": {
        "source": "UW baseDef48Cat: 0-2.9 / 3-5.9 / 6-9.9 / >=10",
        "bins": [
            (0, 3, "normal"),
            (3, 6, "mild"),
            (6, 10, "moderate"),
            (10, None, "severe"),
        ],
    },
    "lactate_48": {
        "source": "UW lactate48Cat: <=2.9 / 3.0-5.0 / >=5.1",
        "bins": [
            (None, 3, "normal"),
            (3, 5.1, "mild"),
            (5.1, None, "severe"),
        ],
    },
    "crys_48": {
        "source": "UW crys48Cat: 0-1999 / 2000-4992 / 5000-9984 / >=10000 mL",
        "bins": [
            (0, 2000, "low_volume"),
            (2000, 5000, "moderate_volume"),
            (5000, 10000, "high_volume"),
            (10000, None, "very_high_volume"),
        ],
    },
}


def token_code(source_field: str) -> str:
    return FIELD_ALIASES.get(source_field, source_field)


def load_field_config(path: str | Path) -> dict:
    return json.load(open(path))


def build_token_config(field_config: dict):
    tokens = []

    for t in STRUCTURAL_TOKENS:
        tokens.append(CodeTCE(code=t.strip("[]"), description="structural"))
    for t in TIME_BLOCK_TOKENS:
        tokens.append(CodeTCE(code=t.strip("[]"), description="time_block"))
    tokens.append(CodeTCE(code=G2_STAR_TOKEN.strip("[]"), description="first48"))

    for _group_name, group in field_config["groups"].items():
        for fdef in group["fields"]:
            source = fdef["field"]
            code = token_code(source)
            ft = fdef.get("type", "float")
            desc = fdef.get("desc", "")
            tokens.append(CodeTCE(code=code, description=f"{desc}; source={source}"))

            if ft == "cat" and "values" in fdef:
                for raw_value in fdef["values"]:
                    value = CATEGORY_ALIASES.get(source, {}).get(raw_value, raw_value)
                    tokens.append(
                        CategoricalTCE(
                            code=code,
                            tokenization={"category": value, "source_value": raw_value, "source_field": source},
                            description=f"{desc}={value}; source={source}:{raw_value}",
                        )
                    )
            elif ft in ("float", "int") and source in BUCKET_DEFS:
                unit = fdef.get("unit", "")
                for start, end, label in BUCKET_DEFS[source]["bins"]:
                    tokens.append(
                        NumericalRangeTCE(
                            code=code,
                            tokenization={
                                "unit": unit,
                                "range_start": start,
                                "range_end": end,
                                "label": label,
                                "source": BUCKET_DEFS[source]["source"],
                                "source_field": source,
                            },
                            description=f"{desc}: {label}; source={source}",
                        )
                    )
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-config", default="data dicision/field adapter/field_config.json")
    ap.add_argument("--out", default="tokenizer/dictionary/tokenizer_config.json")
    args = ap.parse_args()

    field_config = load_field_config(args.field_config)
    tokens = build_token_config(field_config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tokens": [t.to_dict() for t in tokens], "n_tokens": len(tokens)}, open(out, "w"), indent=2)

    by_type = defaultdict(int)
    for t in tokens:
        by_type[t.type] += 1
    print(f"Wrote {len(tokens)} tokens to {out}")
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
