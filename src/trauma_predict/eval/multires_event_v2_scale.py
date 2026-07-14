from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from trauma_predict.data.multires_event_v2 import MultiresEventV2Contract
from trauma_predict.eval.multires_event_v2_projections import (
    STANDARDIZED_PRIMITIVE_COORDINATE_CONTRACT,
    STANDARDIZED_PRIMITIVE_SCALE_SCHEMA,
    STANDARDIZED_PRIMITIVE_SCALE_VERSION,
    build_standardized_primitive_schema,
    primitive_coordinate_schema_sha256,
    required_standardized_scale_keys,
)


PHYSICAL_WINDOW_KEY = (
    "subject_id",
    "stay_id",
    "absolute_start_hour",
    "absolute_end_hour",
    "field",
)
FIT_DENSE_COMPONENTS = ("last", "min", "max", "mean")


def fit_standardized_primitive_scale_artifact(
    *,
    target_root: str | Path,
    lab_scale_path: str | Path,
    expected_lab_scale_sha256: str,
    output_path: str | Path,
    expected_train_samples: int = 37_734,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit only the 38 empirically scaled phi coordinates on unique train windows.

    The fit population is the persisted train target sidecar. Duplicate anchors
    that expose the same physical M4 field window must carry byte-equivalent
    canonical process truth and contribute exactly once. No validation/test row,
    epsilon floor, MAD fallback, or clipping is permitted.
    """

    root = Path(target_root).resolve()
    destination = Path(output_path).resolve()
    if expected_train_samples < 1:
        raise ValueError("expected_train_samples must be positive")
    if not _is_sha256(expected_lab_scale_sha256):
        raise ValueError("expected_lab_scale_sha256 must be lowercase SHA-256")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)

    contract = MultiresEventV2Contract.from_dataset_root(root)
    schema = build_standardized_primitive_schema(contract)
    fitted_keys = required_standardized_scale_keys(schema)
    if len(fitted_keys) != 38:
        raise AssertionError(f"phi fitter requires 38 fitted keys, got {len(fitted_keys)}")
    lab = _load_bound_lab_scale(
        Path(lab_scale_path).resolve(),
        expected_sha256=expected_lab_scale_sha256,
        contract=contract,
    )

    fit_values: dict[str, list[float]] = {key: [] for key in fitted_keys}
    dense = tuple(contract.dense_fields)
    fitted_fields = dense + (contract.ned_field, contract.uop_field)
    seen_windows: dict[tuple[str, str, int, int, str], str] = {}
    collapsed_duplicates = 0
    train_samples = 0
    subjects: set[str] = set()
    ledger = hashlib.sha256()

    for shard_path, expected_sha256 in _train_target_shards(contract):
        if _sha256_file(shard_path) != expected_sha256:
            raise ValueError(f"train target shard hash mismatch: {shard_path}")
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError(f"{shard_path}:{line_number} must contain an object")
                if record.get("split") != "train":
                    raise ValueError(f"non-train row entered phi fit: {shard_path}:{line_number}")
                contract.validate_target_record(record, verify_content_hash=True)
                train_samples += 1
                subject_id = str(record["subject_id"])
                stay_id = str(record["stay_id"])
                prediction_hour = int(record["prediction_hour"])
                subjects.add(subject_id)
                blocks = record.get("blocks")
                if not isinstance(blocks, Sequence) or len(blocks) != 6:
                    raise ValueError("phi fit target must contain six M4 blocks")
                for block in blocks:
                    if not isinstance(block, Mapping):
                        raise ValueError("phi fit block must be an object")
                    start = prediction_hour + int(block["relative_start_hour"])
                    end = prediction_hour + int(block["relative_end_hour"])
                    processes = block.get("processes")
                    if not isinstance(processes, Mapping):
                        raise ValueError("phi fit block lacks processes")
                    for field in fitted_fields:
                        process = processes.get(field)
                        if not isinstance(process, Mapping):
                            raise ValueError(f"phi fit block lacks process {field}")
                        key = (subject_id, stay_id, start, end, field)
                        canonical_truth = _canonical_json(process)
                        previous = seen_windows.get(key)
                        if previous is not None:
                            collapsed_duplicates += 1
                            if previous != canonical_truth:
                                raise ValueError(
                                    "duplicate physical M4 field window has conflicting truth: "
                                    f"{key}"
                                )
                            continue
                        seen_windows[key] = canonical_truth
                        ledger.update(
                            (
                                _canonical_json(
                                    {"physical_window_key": key, "process": process}
                                )
                                + "\n"
                            ).encode("utf-8")
                        )
                        _collect_fit_values(
                            fit_values,
                            field=field,
                            process=process,
                            contract=contract,
                        )

    if train_samples != expected_train_samples:
        raise ValueError(
            f"phi scale fit expected {expected_train_samples} train samples, got {train_samples}"
        )
    scale_rows: dict[str, dict[str, Any]] = {}
    zero_iqr: list[str] = []
    for key in fitted_keys:
        values = sorted(fit_values[key])
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"phi fitted key {key} has empty/non-finite support")
        q25 = _linear_percentile(values, 0.25)
        center = _linear_percentile(values, 0.5)
        q75 = _linear_percentile(values, 0.75)
        scale = q75 - q25
        if not scale > 0:
            zero_iqr.append(key)
        scale_rows[key] = {
            "center": center,
            "scale": scale,
            "q25": q25,
            "q75": q75,
            "fit_count": len(values),
            "fit_kind": (
                "train_unique_physical_window_median_iqr_log_positive"
                if key.endswith("log_positive_max")
                or key.endswith("log_positive_sum")
                else "train_unique_physical_window_median_iqr_raw"
            ),
        }
    if zero_iqr:
        raise ValueError(
            "phi fitted coordinates contain zero IQR; no fallback is allowed: "
            f"{zero_iqr}"
        )

    manifest_sha256 = _sha256_file(root / "dataset_manifest.json")
    payload: dict[str, Any] = {
        "schema": STANDARDIZED_PRIMITIVE_SCALE_SCHEMA,
        "version": STANDARDIZED_PRIMITIVE_SCALE_VERSION,
        "status": "frozen_train_only_fit",
        "fit_split": "train",
        "coordinate_contract": STANDARDIZED_PRIMITIVE_COORDINATE_CONTRACT,
        "coordinate_schema_sha256": primitive_coordinate_schema_sha256(schema),
        "transform": {
            "fitted_continuous": "asinh((x-center)/scale)",
            "fitted_positive": "gate=1[x>0]; asinh((log(x)-center)/scale)",
            "lab_reuse": "asinh((x-lab_center)/lab_scale)",
            "bounded_duration": "x/4",
            "nonnegative_integer": "asinh(x)",
            "conditional_ratio": "numerator/positive_denominator_else_0",
            "clipping": "forbidden",
            "pooling": "shared_across_six_M4_blocks",
        },
        "source": {
            "sidecar_dataset_id": str(contract.manifest["dataset_id"]),
            "sidecar_dataset_manifest_sha256": manifest_sha256,
            "sidecar_sample_manifest_sha256": str(
                contract.manifest["files"]["sample_manifest"]["sha256"]
            ),
            "contract_bundle_hash": contract.contract_bundle_hash,
            "process_contract_sha256": contract.contract_hashes["process"],
            "emission_contract_sha256": contract.contract_hashes["emission"],
            "projection_contract_sha256": contract.contract_hashes["projection"],
            "lab_scale_artifact_sha256": expected_lab_scale_sha256,
        },
        "fit_population": {
            "authority": "persisted_full_sidecar_train_target_shards",
            "physical_window_key": list(PHYSICAL_WINDOW_KEY),
            "fitted_fields": list(fitted_fields),
            "duplicate_truth_policy": (
                "require_exact_canonical_json_then_count_once"
            ),
            "train_samples": train_samples,
            "train_subjects": len(subjects),
            "unique_fitted_physical_field_windows": len(seen_windows),
            "collapsed_duplicate_field_windows": collapsed_duplicates,
            "train_subject_ids_sha256": hashlib.sha256(
                ("\n".join(sorted(subjects)) + "\n").encode("utf-8")
            ).hexdigest(),
            "window_truth_ledger_sha256": ledger.hexdigest(),
        },
        "fit_audit": {
            "fitted_key_count": len(fitted_keys),
            "zero_iqr_keys": zero_iqr,
            "scale_fallback": "forbidden",
            "fitted_keys": list(fitted_keys),
        },
        "scales": scale_rows,
        "lab_scales": {
            field: {
                "center": float(lab["fields"][field]["center"]),
                "scale": float(lab["fields"][field]["scale"]),
            }
            for field in contract.lab_fields
        },
    }
    content_sha256 = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    payload["content_sha256"] = content_sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return payload


def _collect_fit_values(
    values: dict[str, list[float]],
    *,
    field: str,
    process: Mapping[str, Any],
    contract: MultiresEventV2Contract,
) -> None:
    if field in contract.dense_fields:
        state = process.get("value_state")
        if state is None:
            return
        if not isinstance(state, Mapping):
            raise ValueError(f"dense fit state for {field} is invalid")
        for component in FIT_DENSE_COMPONENTS:
            key = f"{field}|dense_joint_value_state|{component}"
            values[key].append(float(state[component]))
        return
    if field == contract.ned_field:
        state = process.get("value_state")
        if not isinstance(state, Mapping):
            raise ValueError("NED fit state is invalid")
        maximum = float(state["max"])
        if maximum > 0:
            key = f"{field}|ned_joint_value_state|log_positive_max"
            values[key].append(math.log(maximum))
        return
    if field == contract.uop_field:
        count = int(process["observation_count"])
        raw_total = process.get("sum")
        if count == 0:
            if raw_total is not None:
                raise ValueError("UOP zero-count fit state must have null SUM")
            return
        if raw_total is None:
            raise ValueError("UOP positive-count fit state must have numeric SUM")
        total = float(raw_total)
        if total > 0:
            key = f"{field}|uop_sum_given_count|log_positive_sum"
            values[key].append(math.log(total))
        return
    raise AssertionError(f"unexpected phi fitted field {field}")


def _load_bound_lab_scale(
    path: Path,
    *,
    expected_sha256: str,
    contract: MultiresEventV2Contract,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("lab scale artifact must be an object")
    canonical = _canonical_json(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    observed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if payload.get("content_sha256") != observed or observed != expected_sha256:
        raise ValueError("lab scale artifact content hash mismatch")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("lab scale artifact lacks source")
    expected_source = {
        "sidecar_dataset_id": str(contract.manifest["dataset_id"]),
        "sidecar_dataset_manifest_sha256": _sha256_file(
            contract.dataset_root / "dataset_manifest.json"
        ),
        "sidecar_sample_manifest_sha256": str(
            contract.manifest["files"]["sample_manifest"]["sha256"]
        ),
        "sidecar_contract_bundle_hash": contract.contract_bundle_hash,
        "sidecar_process_contract_sha256": contract.contract_hashes["process"],
        "sidecar_emission_contract_sha256": contract.contract_hashes["emission"],
    }
    for key, expected in expected_source.items():
        if str(source.get(key)) != expected:
            raise ValueError(f"lab scale source.{key} differs from attached sidecar")
    fields = payload.get("fields")
    if not isinstance(fields, Mapping) or set(fields) != set(contract.lab_fields):
        raise ValueError("lab scale field support differs from the V2 contract")
    compact: dict[str, dict[str, float]] = {}
    for field in contract.lab_fields:
        row = fields[field]
        if not isinstance(row, Mapping):
            raise ValueError(f"lab scale row {field} is invalid")
        center = float(row["center"])
        scale = float(row["scale"])
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"lab scale row {field} is non-finite")
        compact[field] = {"center": center, "scale": scale}
    return {"content_sha256": observed, "fields": compact}


def _train_target_shards(
    contract: MultiresEventV2Contract,
) -> tuple[tuple[Path, str], ...]:
    files = contract.manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("sidecar manifest lacks files")
    shards = files.get("target_shards")
    if not isinstance(shards, Mapping):
        raise ValueError("sidecar manifest lacks target_shards")
    result: list[tuple[Path, str]] = []
    for shard_key, row in sorted(shards.items()):
        if not str(shard_key).startswith("train-"):
            continue
        if not isinstance(row, Mapping):
            raise ValueError(f"target shard row {shard_key} is invalid")
        path = contract.dataset_root / str(row["path"])
        digest = str(row["sha256"])
        if not _is_sha256(digest):
            raise ValueError(f"target shard {shard_key} has invalid SHA-256")
        result.append((path, digest))
    if not result:
        raise ValueError("sidecar manifest contains no train target shards")
    return tuple(result)


def _linear_percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit the 38-key train-only multires_event_v2 phi scale artifact"
    )
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--lab-scale", required=True)
    parser.add_argument("--expected-lab-scale-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-train-samples", type=int, default=37_734)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    payload = fit_standardized_primitive_scale_artifact(
        target_root=args.target_root,
        lab_scale_path=args.lab_scale,
        expected_lab_scale_sha256=args.expected_lab_scale_sha256,
        output_path=args.output,
        expected_train_samples=args.expected_train_samples,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "content_sha256": payload["content_sha256"],
                "fitted_key_count": payload["fit_audit"]["fitted_key_count"],
                "zero_iqr_keys": payload["fit_audit"]["zero_iqr_keys"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["fit_standardized_primitive_scale_artifact", "main"]
