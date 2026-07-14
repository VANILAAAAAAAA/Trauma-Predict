from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from trauma_predict.data.multires_event_v2 import CoreRelationEdge, MultiresEventV2Contract
from trauma_predict.eval.multires_event_v2_promotion_contract import (
    marginal_encoding_partitions,
)
from trauma_predict.modeling.multires_event_v2.emissions import dense_abnormal_class_masks
from trauma_predict.training.multires_event_v2_loss import (
    V2_PRIMITIVE_FEEDBACK_DIMS,
    expand_enabled_core_primitives,
)


STANDARDIZED_PRIMITIVE_SCALE_SCHEMA = (
    "multires_event_v2_standardized_primitive_scale_v2"
)
STANDARDIZED_PRIMITIVE_SCALE_VERSION = (
    "2026-07-13-train-target-natural-conditional-v2"
)
STANDARDIZED_PRIMITIVE_COORDINATE_CONTRACT = (
    "registered_primitive_injective_phi_v2_natural_conditional"
)
CONTRACT_ARITHMETIC_ATOL = 1e-12


@dataclass(frozen=True)
class PhysicalProjectionSpec:
    """One registered physical view repeated over the six M4 blocks."""

    projection_id: str
    field: str
    field_id: int
    field_index: int
    operator: str
    condition: str
    likelihood_id: str
    component_index: int
    gate: str
    value_kind: str
    unit: str | None
    ordered_min: int | None = None
    ordered_max: int | None = None
    one_hot_category: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrimitiveVectorCoordinate:
    """One non-duplicated coordinate in the injective trajectory map phi(T)."""

    coordinate_id: str
    within_block_id: str
    primitive_id: str
    likelihood_id: str
    block_index: int
    field: str
    field_index: int
    component_index: int
    encoding: str
    output_index: int | None
    scale_key: str | None
    minimum: int | None
    maximum: int | None
    denominator_component_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_standardized_primitive_schema(
    contract: MultiresEventV2Contract,
) -> tuple[PrimitiveVectorCoordinate, ...]:
    """Compile the injective, non-projection trajectory coordinate map phi(T).

    Parent gates/counts are coordinates in their own registered primitives.
    Conditional child channels are zeroed when inactive. Nominal states use
    one-hot coordinates, bounded durations use their four-hour natural scale,
    counts/onsets use ``asinh(count)``, labs reuse the separately frozen lab
    affine artifact, and only 36 dense components plus conditional log-positive
    NED MAX/UOP SUM consume train-fitted median/IQR rows.
    """

    rows: list[PrimitiveVectorCoordinate] = []

    def add(
        spec: Any,
        component_name: str,
        component_index: int,
        encoding: str,
        *,
        output_index: int | None = None,
        scale_key: str | None = None,
        minimum: int | None = None,
        maximum: int | None = None,
        denominator_component_index: int | None = None,
    ) -> None:
        suffix = component_name
        if output_index is not None:
            suffix = f"{suffix}.one_hot_{output_index}"
        within = f"{spec.field}.{spec.likelihood_id}.{suffix}"
        rows.append(
            PrimitiveVectorCoordinate(
                coordinate_id=f"{spec.primitive_id}.{suffix}",
                within_block_id=within,
                primitive_id=spec.primitive_id,
                likelihood_id=spec.likelihood_id,
                block_index=spec.block_index,
                field=spec.field,
                field_index=spec.field_index,
                component_index=component_index,
                encoding=encoding,
                output_index=output_index,
                scale_key=scale_key,
                minimum=minimum,
                maximum=maximum,
                denominator_component_index=denominator_component_index,
            )
        )

    def fitted_dense(spec: Any, names: Sequence[str]) -> None:
        for component_index, name in enumerate(names):
            add(
                spec,
                str(name),
                component_index,
                "robust_affine_asinh",
                scale_key=f"{spec.field}|{spec.likelihood_id}|{name}",
            )

    for spec in expand_enabled_core_primitives(contract.process_registry):
        likelihood = spec.likelihood_id
        if likelihood == "categorical_hours_0_4":
            add(spec, "observed_hours", 0, "ordered_unit", minimum=0, maximum=4)
        elif likelihood == "dense_joint_value_state":
            fitted_dense(spec, ("last", "min", "max", "mean"))
        elif likelihood == "dense_abnormal_duration_vector":
            conditions = tuple(contract.dense_abnormal_conditions.get(spec.field, ()))
            for component_index, condition in enumerate(conditions):
                add(
                    spec,
                    f"duration_{condition}",
                    component_index,
                    "ordered_unit",
                    minimum=0,
                    maximum=4,
                )
        elif likelihood == "gcs_ordinal_triple":
            maximum = int(contract.ordinal_max[spec.field])
            for component_index, name in enumerate(("last", "min", "max")):
                add(
                    spec,
                    name,
                    component_index,
                    "ordered_unit",
                    minimum=1,
                    maximum=maximum,
                )
        elif likelihood == "gcs_verbal_ungradable_hours_given_observed":
            add(spec, "ungradable_hours", 0, "ordered_unit", minimum=0, maximum=4)
        elif likelihood == "gcs_verbal_latest_status":
            for category in (1, 2):
                add(
                    spec,
                    "latest_status",
                    0,
                    "one_hot",
                    output_index=category,
                    minimum=1,
                    maximum=2,
                )
        elif likelihood == "gcs_verbal_gradable_ordinal_triple":
            for component_index, name in enumerate(("last", "min", "max")):
                add(
                    spec,
                    name,
                    component_index,
                    "ordered_unit",
                    minimum=1,
                    maximum=5,
                )
        elif likelihood == "hurdle_negative_binomial_count":
            add(
                spec,
                "count",
                0,
                "natural_asinh_nonnegative_integer",
                minimum=0,
            )
        elif likelihood == "lab_joint_value_state":
            for component_index, name in enumerate(("last", "min", "max")):
                add(
                    spec,
                    name,
                    component_index,
                    "lab_shared_affine_asinh",
                    scale_key=spec.field,
                )
        elif likelihood == "respiratory_block_evidence":
            add(spec, "block_evidence", 0, "binary", minimum=0, maximum=1)
        elif likelihood == "respiratory_edge_evidence_given_block":
            add(spec, "edge_evidence", 0, "binary", minimum=0, maximum=1)
        elif likelihood == "respiratory_occupancy_vector":
            # Uncovered duration is exactly 4 minus the four modality durations;
            # omitting it removes a deterministic duplicate while preserving
            # injectivity of the registered primitive state.
            for component_index, name in enumerate(contract.respiratory_modalities):
                add(
                    spec,
                    str(name),
                    component_index,
                    "bounded_unit",
                    minimum=0,
                    maximum=4,
                )
        elif likelihood == "respiratory_edge_state":
            for category, name in enumerate(contract.respiratory_modalities, start=1):
                add(
                    spec,
                    f"edge_{name}",
                    0,
                    "one_hot",
                    output_index=category,
                    minimum=1,
                    maximum=4,
                )
        elif likelihood == "respiratory_onset_vector":
            for component_index, name in enumerate(contract.respiratory_modalities):
                add(
                    spec,
                    str(name),
                    component_index,
                    "natural_asinh_nonnegative_integer",
                    minimum=0,
                )
        elif likelihood == "vasopressor_duration_vector":
            for component_index, name in enumerate(contract.vasopressor_agents):
                add(
                    spec,
                    str(name),
                    component_index,
                    "bounded_unit",
                    minimum=0,
                    maximum=4,
                )
        elif likelihood == "vasopressor_edge_state_vector":
            for component_index, name in enumerate(contract.vasopressor_agents):
                add(
                    spec,
                    name,
                    component_index,
                    "binary",
                    minimum=0,
                    maximum=1,
                )
        elif likelihood == "vasopressor_onset_vector":
            for component_index, name in enumerate(contract.vasopressor_agents):
                add(
                    spec,
                    str(name),
                    component_index,
                    "natural_asinh_nonnegative_integer",
                    minimum=0,
                )
        elif likelihood == "ned_joint_value_state":
            add(spec, "positive_max_gate", 1, "positive_gate")
            add(
                spec,
                "log_positive_max",
                1,
                "positive_log_robust_affine",
                scale_key=(
                    f"{spec.field}|{spec.likelihood_id}|log_positive_max"
                ),
            )
            add(
                spec,
                "last_over_max",
                0,
                "bounded_ratio",
                minimum=0,
                maximum=1,
                denominator_component_index=1,
            )
            add(
                spec,
                "mean_over_max",
                2,
                "bounded_ratio",
                minimum=0,
                maximum=1,
                denominator_component_index=1,
            )
        elif likelihood == "uop_sum_given_count":
            add(spec, "positive_sum_gate", 0, "positive_gate")
            add(
                spec,
                "log_positive_sum",
                0,
                "positive_log_robust_affine",
                scale_key=f"{spec.field}|{spec.likelihood_id}|log_positive_sum",
            )
        else:  # pragma: no cover - fail closed on a new registry likelihood
            raise ValueError(f"standardized primitive schema lacks {likelihood!r}")

    if len({row.coordinate_id for row in rows}) != len(rows):
        raise ValueError("standardized primitive coordinate ids are not unique")
    by_block: dict[int, tuple[str, ...]] = {}
    for block_index in range(6):
        by_block[block_index] = tuple(
            row.within_block_id for row in rows if row.block_index == block_index
        )
    if any(by_block[index] != by_block[0] for index in range(1, 6)):
        raise ValueError("standardized primitive coordinates differ across M4 blocks")
    if not rows or any(
        row.encoding in {"robust_affine_asinh", "positive_log_robust_affine"}
        and not row.scale_key
        for row in rows
    ):
        raise AssertionError("standardized primitive schema lacks a scale binding")
    return tuple(rows)


def required_standardized_scale_keys(
    schema: Sequence[PrimitiveVectorCoordinate],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(row.scale_key)
                for row in schema
                if row.encoding
                in {"robust_affine_asinh", "positive_log_robust_affine"}
                and row.scale_key is not None
            }
        )
    )


def primitive_coordinate_schema_sha256(
    schema: Sequence[PrimitiveVectorCoordinate],
) -> str:
    payload = [row.as_dict() for row in schema]
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_standardized_primitive_scale_artifact(
    path: str | Path,
    *,
    expected_content_sha256: str,
    contract: MultiresEventV2Contract,
    expected_lab_scale_artifact_hash: str,
) -> dict[str, Any]:
    """Bind the train-only phi(T) scale to all attached sidecar contracts."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing standardized primitive scale artifact: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("standardized primitive scale artifact must be a JSON object")
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "content_sha256"},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    observed = hashlib.sha256(canonical).hexdigest()
    if payload.get("content_sha256") != observed or observed != expected_content_sha256:
        raise ValueError("standardized primitive scale content hash mismatch")
    required = {
        "schema": STANDARDIZED_PRIMITIVE_SCALE_SCHEMA,
        "version": STANDARDIZED_PRIMITIVE_SCALE_VERSION,
        "status": "frozen_train_only_fit",
        "fit_split": "train",
        "coordinate_contract": STANDARDIZED_PRIMITIVE_COORDINATE_CONTRACT,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"standardized primitive scale {key} must equal {expected!r}")
    transform = payload.get("transform")
    expected_transform = {
        "fitted_continuous": "asinh((x-center)/scale)",
        "fitted_positive": "gate=1[x>0]; asinh((log(x)-center)/scale)",
        "lab_reuse": "asinh((x-lab_center)/lab_scale)",
        "bounded_duration": "x/4",
        "nonnegative_integer": "asinh(x)",
        "conditional_ratio": "numerator/positive_denominator_else_0",
        "clipping": "forbidden",
        "pooling": "shared_across_six_M4_blocks",
    }
    if not isinstance(transform, Mapping) or dict(transform) != expected_transform:
        raise ValueError("standardized primitive scale transform contract changed")
    attached = payload.get("source")
    if not isinstance(attached, Mapping):
        raise ValueError("standardized primitive scale lacks source identity")
    expected_source = {
        "sidecar_dataset_id": str(contract.manifest["dataset_id"]),
        "sidecar_dataset_manifest_sha256": hashlib.sha256(
            (contract.dataset_root / "dataset_manifest.json").read_bytes()
        ).hexdigest(),
        "sidecar_sample_manifest_sha256": str(
            contract.manifest["files"]["sample_manifest"]["sha256"]
        ),
        "contract_bundle_hash": contract.contract_bundle_hash,
        "process_contract_sha256": contract.contract_hashes["process"],
        "emission_contract_sha256": contract.contract_hashes["emission"],
        "projection_contract_sha256": contract.contract_hashes["projection"],
        "lab_scale_artifact_sha256": expected_lab_scale_artifact_hash,
    }
    for key, expected in expected_source.items():
        if str(attached.get(key)) != expected:
            raise ValueError(f"standardized primitive scale source.{key} mismatch")
    schema = build_standardized_primitive_schema(contract)
    if payload.get("coordinate_schema_sha256") != primitive_coordinate_schema_sha256(
        schema
    ):
        raise ValueError("standardized primitive coordinate schema hash mismatch")
    expected_keys = set(required_standardized_scale_keys(schema))
    if len(expected_keys) != 38:
        raise AssertionError(f"V2 phi scale contract expanded to {len(expected_keys)} fitted keys")
    scales = payload.get("scales")
    if not isinstance(scales, Mapping) or set(scales) != expected_keys:
        raise ValueError("standardized primitive scale rows do not exactly cover phi(T)")
    compact: dict[str, dict[str, float]] = {}
    for key in sorted(expected_keys):
        row = scales[key]
        required_row = {"center", "scale", "q25", "q75", "fit_count", "fit_kind"}
        if not isinstance(row, Mapping) or set(row) != required_row:
            raise ValueError(f"standardized primitive scale row {key!r} is invalid")
        center = float(row["center"])
        scale = float(row["scale"])
        q25 = float(row["q25"])
        q75 = float(row["q75"])
        fit_count = int(row["fit_count"])
        expected_fit_kind = (
            "train_unique_physical_window_median_iqr_log_positive"
            if key.endswith("log_positive_max") or key.endswith("log_positive_sum")
            else "train_unique_physical_window_median_iqr_raw"
        )
        if (
            not all(math.isfinite(value) for value in (center, scale, q25, q75))
            or scale <= 0
            or fit_count < 1
            or not math.isclose(q75 - q25, scale, rel_tol=1e-12, abs_tol=1e-12)
            or row["fit_kind"] != expected_fit_kind
        ):
            raise ValueError(f"standardized primitive scale row {key!r} is non-finite")
        compact[key] = {"center": center, "scale": scale}
    fit_audit = payload.get("fit_audit")
    if not isinstance(fit_audit, Mapping) or fit_audit.get("zero_iqr_keys") != []:
        raise ValueError("standardized primitive fit audit must prove zero IQR count is zero")
    if int(fit_audit.get("fitted_key_count", -1)) != 38:
        raise ValueError("standardized primitive fit audit must cover 38 fitted keys")
    if fit_audit.get("scale_fallback") != "forbidden" or tuple(
        str(value) for value in fit_audit.get("fitted_keys", ())
    ) != tuple(sorted(expected_keys)):
        raise ValueError("standardized primitive fit audit key/fallback contract changed")
    population = payload.get("fit_population")
    if not isinstance(population, Mapping):
        raise ValueError("standardized primitive scale lacks fit_population")
    population_expected = {
        "authority": "persisted_full_sidecar_train_target_shards",
        "physical_window_key": [
            "subject_id",
            "stay_id",
            "absolute_start_hour",
            "absolute_end_hour",
            "field",
        ],
        "fitted_fields": list(contract.dense_fields)
        + [contract.ned_field, contract.uop_field],
        "duplicate_truth_policy": "require_exact_canonical_json_then_count_once",
    }
    for key, expected in population_expected.items():
        if population.get(key) != expected:
            raise ValueError(f"standardized primitive fit_population.{key} changed")
    for key in (
        "train_samples",
        "train_subjects",
        "unique_fitted_physical_field_windows",
    ):
        if int(population.get(key, 0)) < 1:
            raise ValueError(f"standardized primitive fit_population.{key} is invalid")
    if int(population.get("collapsed_duplicate_field_windows", -1)) < 0:
        raise ValueError("standardized primitive duplicate-window count is invalid")
    for key in ("train_subject_ids_sha256", "window_truth_ledger_sha256"):
        value = str(population.get(key) or "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"standardized primitive fit_population.{key} is invalid")
    lab_scales = payload.get("lab_scales")
    if not isinstance(lab_scales, Mapping) or set(lab_scales) != set(contract.lab_fields):
        raise ValueError("standardized primitive artifact must reuse every bound lab scale")
    compact_labs: dict[str, dict[str, float]] = {}
    for field in contract.lab_fields:
        row = lab_scales[field]
        if not isinstance(row, Mapping) or set(row) != {"center", "scale"}:
            raise ValueError(f"standardized primitive lab scale {field!r} is invalid")
        center = float(row["center"])
        scale = float(row["scale"])
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"standardized primitive lab scale {field!r} is non-finite")
        compact_labs[field] = {"center": center, "scale": scale}
    return {
        "schema": STANDARDIZED_PRIMITIVE_SCALE_SCHEMA,
        "version": STANDARDIZED_PRIMITIVE_SCALE_VERSION,
        "coordinate_contract": STANDARDIZED_PRIMITIVE_COORDINATE_CONTRACT,
        "content_sha256": observed,
        "scales": compact,
        "lab_scales": compact_labs,
    }


def standardize_primitive_trajectory(
    primitives: Mapping[str, torch.Tensor],
    primitive_masks: Mapping[str, torch.Tensor],
    schema: Sequence[PrimitiveVectorCoordinate],
    scale_artifact: Mapping[str, Any],
) -> torch.Tensor:
    """Map registered trajectory state injectively to ``[B,6,D]`` phi(T)."""

    rows = tuple(schema)
    if not rows:
        raise ValueError("standardized primitive schema is empty")
    within_order = tuple(row.within_block_id for row in rows if row.block_index == 0)
    dimension = len(within_order)
    if dimension < 1:
        raise ValueError("standardized primitive schema has no per-block coordinates")
    scales = scale_artifact.get("scales")
    if not isinstance(scales, Mapping):
        raise ValueError("standardized primitive scale artifact lacks scales")
    lab_scales = scale_artifact.get("lab_scales")
    if not isinstance(lab_scales, Mapping):
        raise ValueError("standardized primitive scale artifact lacks reused lab scales")
    reference = _primitive_bank(primitives, rows[0].likelihood_id)
    result = torch.zeros(
        (reference.shape[0], 6, dimension),
        dtype=torch.float64,
        device=reference.device,
    )
    position = {name: index for index, name in enumerate(within_order)}
    cuda_coordinate_checks: list[torch.Tensor] = []

    def require_valid(
        invalid: torch.Tensor,
        message: str,
        error_type: type[Exception] = ValueError,
    ) -> None:
        if invalid.device.type == "cuda":
            cuda_coordinate_checks.append((~invalid).reshape(-1))
        elif bool(invalid.any().item()):
            raise error_type(message)

    for row in rows:
        raw = _component(
            primitives,
            row.likelihood_id,
            row.field_index,
            row.component_index,
        )[:, row.block_index].double()
        value_bank = _primitive_bank(primitives, row.likelihood_id)
        mask_bank = _primitive_bank(primitive_masks, row.likelihood_id).bool()
        if mask_bank.shape[:-1] != value_bank.shape[:-1]:
            raise ValueError(
                f"phi primitive/mask leading shape differs for {row.likelihood_id}"
            )
        if mask_bank.shape[-1] == 1 and value_bank.shape[-1] > 1:
            # Collated teacher truth stores one process-activation mask per
            # [anchor,block,field]. Generated rollouts already expose the full
            # component mask. Both are the same contract after this expansion.
            mask_bank = mask_bank.expand_as(value_bank)
        elif mask_bank.shape[-1] != value_bank.shape[-1]:
            raise ValueError(
                f"phi primitive/mask width differs for {row.likelihood_id}"
            )
        active = mask_bank[:, row.block_index, row.field_index, row.component_index]
        if row.encoding == "binary":
            require_valid(
                active & ~(raw.eq(0) | raw.eq(1)),
                f"binary phi coordinate {row.coordinate_id} is invalid",
            )
            encoded = raw
        elif row.encoding == "one_hot":
            if (
                row.output_index is None
                or row.minimum is None
                or row.maximum is None
                or row.maximum < row.minimum
            ):
                raise AssertionError("one-hot phi coordinate lacks categorical support")
            require_valid(
                active
                & (
                    ~torch.isfinite(raw)
                    | raw.ne(raw.round())
                    | raw.lt(row.minimum)
                    | raw.gt(row.maximum)
                ),
                f"one-hot phi coordinate {row.coordinate_id} is invalid",
            )
            encoded = raw.round().long().eq(row.output_index).float()
        elif row.encoding == "ordered_unit":
            if row.minimum is None or row.maximum is None or row.maximum <= row.minimum:
                raise AssertionError("ordered phi coordinate lacks fixed support")
            require_valid(
                active
                & (
                    ~torch.isfinite(raw)
                    | raw.ne(raw.round())
                    | raw.lt(row.minimum)
                    | raw.gt(row.maximum)
                ),
                f"ordered phi coordinate {row.coordinate_id} is invalid",
            )
            encoded = (raw - row.minimum) / (row.maximum - row.minimum)
        elif row.encoding == "bounded_unit":
            if row.minimum is None or row.maximum is None or row.maximum <= row.minimum:
                raise AssertionError("bounded phi coordinate lacks fixed support")
            require_valid(
                active
                & (
                    ~torch.isfinite(raw)
                    | raw.lt(row.minimum)
                    | raw.gt(row.maximum)
                ),
                f"bounded phi coordinate {row.coordinate_id} is invalid",
            )
            encoded = (raw - row.minimum) / (row.maximum - row.minimum)
        elif row.encoding == "natural_asinh_nonnegative_integer":
            require_valid(
                active
                & (
                    ~torch.isfinite(raw)
                    | raw.lt(0)
                    | raw.ne(raw.round())
                ),
                f"count phi coordinate {row.coordinate_id} is invalid",
            )
            encoded = torch.asinh(raw)
        elif row.encoding == "robust_affine_asinh":
            scale_row = scales.get(row.scale_key)
            if not isinstance(scale_row, Mapping):
                raise ValueError(f"phi scale is missing for {row.scale_key!r}")
            center = float(scale_row["center"])
            scale = float(scale_row["scale"])
            if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"phi scale is invalid for {row.scale_key!r}")
            encoded = torch.asinh((raw - center) / scale)
        elif row.encoding == "lab_shared_affine_asinh":
            scale_row = lab_scales.get(row.scale_key)
            if not isinstance(scale_row, Mapping):
                raise ValueError(f"bound lab phi scale is missing for {row.scale_key!r}")
            center = float(scale_row["center"])
            scale = float(scale_row["scale"])
            if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"bound lab phi scale is invalid for {row.scale_key!r}")
            encoded = torch.asinh((raw - center) / scale)
        elif row.encoding == "positive_gate":
            require_valid(
                active & (~torch.isfinite(raw) | raw.lt(0)),
                f"positive-gate phi coordinate {row.coordinate_id} is invalid",
            )
            encoded = raw.gt(0).double()
        elif row.encoding == "positive_log_robust_affine":
            require_valid(
                active & (~torch.isfinite(raw) | raw.lt(0)),
                f"positive-log phi coordinate {row.coordinate_id} is invalid",
            )
            scale_row = scales.get(row.scale_key)
            if not isinstance(scale_row, Mapping):
                raise ValueError(f"phi scale is missing for {row.scale_key!r}")
            center = float(scale_row["center"])
            scale = float(scale_row["scale"])
            positive = active & raw.gt(0)
            safe_log = torch.where(positive, raw, torch.ones_like(raw)).log()
            encoded = torch.where(
                positive,
                torch.asinh((safe_log - center) / scale),
                torch.zeros_like(raw),
            )
        elif row.encoding == "bounded_ratio":
            denominator_index = row.denominator_component_index
            if denominator_index is None:
                raise AssertionError("ratio phi coordinate lacks denominator")
            denominator = value_bank[
                :, row.block_index, row.field_index, denominator_index
            ].double()
            positive = active & denominator.gt(0)
            invalid = active & (
                ~torch.isfinite(raw)
                | ~torch.isfinite(denominator)
                | denominator.lt(0)
                | raw.lt(0)
                | raw.gt(denominator)
                | (denominator.eq(0) & raw.ne(0))
            )
            require_valid(
                invalid,
                f"ratio phi coordinate {row.coordinate_id} is invalid",
            )
            encoded = torch.where(
                positive,
                raw / torch.where(positive, denominator, torch.ones_like(denominator)),
                torch.zeros_like(raw),
            )
        else:  # pragma: no cover
            raise ValueError(f"unknown phi encoding {row.encoding!r}")
        require_valid(
            active & ~torch.isfinite(encoded),
            f"non-finite phi coordinate {row.coordinate_id}",
            FloatingPointError,
        )
        result[:, row.block_index, position[row.within_block_id]] = torch.where(
            active, encoded, torch.zeros_like(encoded)
        )
    if cuda_coordinate_checks:
        torch._assert_async(
            torch.cat(cuda_coordinate_checks).all(),
            "invalid or non-finite active standardized primitive coordinate",
        )
    return result


def score_standardized_primitive_ensemble(
    sample_phi: torch.Tensor,
    truth_phi: torch.Tensor,
    schema: Sequence[PrimitiveVectorCoordinate],
    relation_edges: Sequence[CoreRelationEdge],
    promotion_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Score joint, temporal, relational, and marginal properties of phi(T).

    Promotion endpoints are field balanced.  The 21 cross-field relation rows
    remain canonical registry rows: undirected edges are not expanded twice.
    """

    if sample_phi.ndim != 3 or sample_phi.shape[1] != 6:
        raise ValueError("sample phi(T) must be [M,6,D]")
    if truth_phi.shape != sample_phi.shape[1:]:
        raise ValueError("truth phi(T) must be [6,D]")
    if sample_phi.shape[0] != int(
        promotion_contract["ancestral_ensemble"]["trajectories_per_anchor"]
    ):
        raise ValueError("promotion phi score requires exactly 100 trajectories")
    block_rows = tuple(row for row in schema if row.block_index == 0)
    if len(block_rows) != sample_phi.shape[2]:
        raise ValueError("promotion phi schema dimension does not match sampled vectors")
    expected_dimension = int(
        promotion_contract["standardized_primitive_vector"]["coordinates_per_block"]
    )
    if len(block_rows) != expected_dimension:
        raise ValueError("promotion phi schema must contain exactly 160 coordinates per block")
    field_indices: dict[str, list[int]] = {}
    for coordinate_index, row in enumerate(block_rows):
        field_indices.setdefault(row.field, []).append(coordinate_index)
    if len(field_indices) != int(
        promotion_contract["standardized_primitive_vector"]["fields"]
    ):
        raise ValueError("promotion phi schema must cover exactly 29 fields")

    energy = empirical_energy_score(sample_phi.flatten(1), truth_phi.flatten())
    order = float(
        promotion_contract["standardized_primitive_vector"]["variogram_order"]
    )
    observed = (truth_phi[1:] - truth_phi[:-1]).abs().pow(order)
    forecast = (sample_phi[:, 1:] - sample_phi[:, :-1]).abs().pow(order).mean(dim=0)
    temporal_errors = (observed - forecast).square()
    variogram = temporal_errors.mean()
    field_temporal = {
        field: float(temporal_errors[:, indices].mean().item())
        for field, indices in field_indices.items()
    }
    field_macro_temporal = sum(field_temporal.values()) / len(field_temporal)

    structural_edges = tuple(
        edge
        for edge in relation_edges
        if edge.lag_blocks == 0 and edge.source_field != edge.target_field
    )
    expected_edges = int(promotion_contract["relation_edge_cover"]["expected_edges"])
    if len(structural_edges) != expected_edges or len(
        {edge.edge_id for edge in structural_edges}
    ) != expected_edges:
        raise ValueError("promotion relation score requires 21 canonical lag-0 core edges")
    edge_scores: dict[str, float] = {}
    edge_types: dict[str, list[float]] = {}
    for edge in structural_edges:
        source_indices = field_indices.get(edge.source_field)
        target_indices = field_indices.get(edge.target_field)
        if not source_indices or not target_indices:
            raise ValueError(f"promotion relation edge lacks phi coordinates: {edge.edge_id}")
        truth_source = truth_phi[:, source_indices]
        truth_target = truth_phi[:, target_indices]
        sample_source = sample_phi[:, :, source_indices]
        sample_target = sample_phi[:, :, target_indices]
        observed_relation = (
            truth_source.unsqueeze(-1) - truth_target.unsqueeze(-2)
        ).abs().pow(order)
        forecast_relation = (
            sample_source.unsqueeze(-1) - sample_target.unsqueeze(-2)
        ).abs().pow(order).mean(dim=0)
        score = float((observed_relation - forecast_relation).square().mean().item())
        edge_scores[edge.edge_id] = score
        edge_types.setdefault(edge.relation_type, []).append(score)
    relation_macro = sum(edge_scores.values()) / len(edge_scores)
    relation_type_scores = {
        relation_type: sum(values) / len(values)
        for relation_type, values in sorted(edge_types.items())
    }

    coordinate_crps = empirical_coordinate_crps(sample_phi, truth_phi)
    partitions = marginal_encoding_partitions(promotion_contract)
    marginal_scores: dict[str, float] = {}
    marginal_field_scores: dict[str, dict[str, float]] = {}
    for partition_name, encodings in partitions.items():
        partition_indices = [
            index for index, row in enumerate(block_rows) if row.encoding in encodings
        ]
        expected_coordinates = int(
            promotion_contract["marginal_partitions"][partition_name][
                "expected_coordinates_per_block"
            ]
        )
        if len(partition_indices) != expected_coordinates:
            raise ValueError(
                f"promotion marginal partition {partition_name!r} has "
                f"{len(partition_indices)} coordinates, expected {expected_coordinates}"
            )
        by_field: dict[str, float] = {}
        partition_set = set(partition_indices)
        for field, indices in field_indices.items():
            selected = [index for index in indices if index in partition_set]
            if selected:
                by_field[field] = float(coordinate_crps[:, selected].mean().item())
        if not by_field:
            raise ValueError(f"promotion marginal partition {partition_name!r} is empty")
        marginal_field_scores[partition_name] = by_field
        marginal_scores[partition_name] = sum(by_field.values()) / len(by_field)
    return {
        "energy_score": float(energy),
        "lag1_variogram_score_p0_5": float(variogram.item()),
        "field_macro_lag1_variogram_score_p0_5": field_macro_temporal,
        "relation_edge_macro_variogram_score_p0_5": relation_macro,
        "marginal_value_crps": marginal_scores["value"],
        "marginal_state_crps": marginal_scores["state"],
        "field_temporal_variogram": dict(sorted(field_temporal.items())),
        "relation_variogram_by_edge": dict(sorted(edge_scores.items())),
        "relation_variogram_by_type": relation_type_scores,
        "marginal_crps_by_field": {
            name: dict(sorted(values.items()))
            for name, values in sorted(marginal_field_scores.items())
        },
    }


def empirical_coordinate_crps(
    samples: torch.Tensor,
    truth: torch.Tensor,
) -> torch.Tensor:
    """Empirical CRPS for every [block,coordinate] without an M-squared tensor."""

    ensemble = samples.detach().double()
    observation = truth.detach().double()
    if ensemble.ndim != 3 or observation.shape != ensemble.shape[1:]:
        raise ValueError("coordinate CRPS requires samples [M,6,D] and truth [6,D]")
    if ensemble.shape[0] < 1 or not bool(
        torch.isfinite(ensemble).all().item()
        and torch.isfinite(observation).all().item()
    ):
        raise ValueError("coordinate CRPS inputs must be nonempty and finite")
    count = ensemble.shape[0]
    first = (ensemble - observation).abs().mean(dim=0)
    ordered = ensemble.sort(dim=0).values
    ranks = torch.arange(
        1,
        count + 1,
        dtype=ordered.dtype,
        device=ordered.device,
    ).reshape(count, 1, 1)
    half_pairwise = (
        ordered * (2.0 * ranks - count - 1.0)
    ).sum(dim=0) / float(count * count)
    result = first - half_pairwise
    if bool((result < -1e-12).any().item()):
        raise FloatingPointError("coordinate CRPS became negative beyond arithmetic tolerance")
    return result.clamp_min(0.0)


def build_physical_projection_schema(
    contract: MultiresEventV2Contract,
) -> tuple[PhysicalProjectionSpec, ...]:
    """Compile the exact 155 deterministic core projections per M4 block.

    Care/procedure projections are deliberately absent because their joint
    objective is off. The compiled key set is checked against the attached,
    hashed projection registry instead of being accepted as an evaluator-local
    convention.
    """

    field_index = {field: index for index, field in enumerate(contract.core_fields)}
    field_id = dict(
        zip(contract.core_fields, contract.registered_core_field_ids, strict=True)
    )
    supports = contract.emission_registry.get("field_supports")
    if not isinstance(supports, Mapping):
        raise ValueError("V2 emission registry lacks field_supports")
    dense_supports = supports.get("dense_continuous")
    lab_supports = supports.get("intermittent_labs")
    if not isinstance(dense_supports, Mapping) or not isinstance(lab_supports, Mapping):
        raise ValueError("V2 emission registry lacks dense/lab physical units")

    rows: list[PhysicalProjectionSpec] = []

    def add(
        field: str,
        operator: str,
        condition: str,
        likelihood_id: str,
        component_index: int,
        gate: str,
        value_kind: str,
        unit: str | None,
        *,
        ordered_min: int | None = None,
        ordered_max: int | None = None,
        one_hot_category: int | None = None,
    ) -> None:
        if field not in field_index:
            raise ValueError(f"projection references non-core field {field!r}")
        projection_id = f"{field}.{operator}.{condition}"
        rows.append(
            PhysicalProjectionSpec(
                projection_id=projection_id,
                field=field,
                field_id=field_id[field],
                field_index=field_index[field],
                operator=operator,
                condition=condition,
                likelihood_id=likelihood_id,
                component_index=component_index,
                gate=gate,
                value_kind=value_kind,
                unit=unit,
                ordered_min=ordered_min,
                ordered_max=ordered_max,
                one_hot_category=one_hot_category,
            )
        )

    for field in contract.dense_fields:
        unit = str(dense_supports[field]["unit"])
        add(
            field,
            "DURATION",
            "OBSERVED",
            "categorical_hours_0_4",
            0,
            "always",
            "ordered",
            "hours",
            ordered_min=0,
            ordered_max=4,
        )
        for component_index, operator in enumerate(("LAST", "MIN", "MAX", "MEAN")):
            add(
                field,
                operator,
                "NONE",
                "dense_joint_value_state",
                component_index,
                "observed_hours_positive",
                "continuous",
                unit,
            )
        for component_index, condition in enumerate(
            contract.dense_abnormal_conditions.get(field, ())
        ):
            add(
                field,
                "DURATION",
                condition,
                "dense_abnormal_duration_vector",
                component_index,
                "observed_hours_positive",
                "ordered",
                "hours",
                ordered_min=0,
                ordered_max=4,
            )

    for field in contract.ordinal_fields:
        maximum = int(contract.ordinal_max[field])
        add(
            field,
            "DURATION",
            "OBSERVED",
            "categorical_hours_0_4",
            0,
            "always",
            "ordered",
            "hours",
            ordered_min=0,
            ordered_max=4,
        )
        for component_index, operator in enumerate(("LAST", "MIN", "MAX")):
            add(
                field,
                operator,
                "NONE",
                "gcs_ordinal_triple",
                component_index,
                "observed_hours_positive",
                "ordered",
                "score",
                ordered_min=1,
                ordered_max=maximum,
            )

    verbal = contract.verbal_field
    add(
        verbal,
        "DURATION",
        "OBSERVED",
        "categorical_hours_0_4",
        0,
        "always",
        "ordered",
        "hours",
        ordered_min=0,
        ordered_max=4,
    )
    add(
        verbal,
        "DURATION",
        "UNGRADABLE_RECORDED",
        "gcs_verbal_ungradable_hours_given_observed",
        0,
        "always",
        "ordered",
        "hours",
        ordered_min=0,
        ordered_max=4,
    )
    add(
        verbal,
        "STATE",
        "LAST_STATUS",
        "gcs_verbal_latest_status",
        0,
        "observed_hours_positive",
        "categorical",
        None,
        ordered_min=1,
        ordered_max=2,
    )
    for component_index, operator in enumerate(("LAST", "MIN", "MAX")):
        add(
            verbal,
            operator,
            "GRADABLE",
            "gcs_verbal_gradable_ordinal_triple",
            component_index,
            "gradable_hours_positive",
            "ordered",
            "score",
            ordered_min=1,
            ordered_max=5,
        )

    for field in contract.lab_fields:
        unit = str(lab_supports[field]["unit"])
        add(
            field,
            "COUNT",
            "OBSERVED",
            "hurdle_negative_binomial_count",
            0,
            "always",
            "count",
            "count",
        )
        for component_index, operator in enumerate(("LAST", "MIN", "MAX")):
            add(
                field,
                operator,
                "NONE",
                "lab_joint_value_state",
                component_index,
                "observation_count_positive",
                "continuous",
                unit,
            )

    respiratory = contract.respiratory_field
    for component_index, condition in enumerate(contract.respiratory_modalities):
        add(
            respiratory,
            "DURATION",
            condition,
            "respiratory_occupancy_vector",
            component_index,
            "respiratory_block_evidence",
            "continuous",
            "hours",
        )
        add(
            respiratory,
            "STATE",
            condition,
            "respiratory_edge_state",
            0,
            "respiratory_edge_evidence",
            "binary",
            None,
            ordered_min=0,
            ordered_max=1,
            one_hot_category=component_index + 1,
        )
        add(
            respiratory,
            "START",
            condition,
            "respiratory_onset_vector",
            component_index,
            "respiratory_block_evidence",
            "count",
            "count",
        )

    vasopressor = contract.vasopressor_field
    for component_index, condition in enumerate(contract.vasopressor_agents):
        add(
            vasopressor,
            "DURATION",
            condition,
            "vasopressor_duration_vector",
            component_index,
            "always",
            "continuous",
            "hours",
        )
        add(
            vasopressor,
            "STATE",
            condition,
            "vasopressor_edge_state_vector",
            component_index,
            "always",
            "binary",
            None,
            ordered_min=0,
            ordered_max=1,
        )
        add(
            vasopressor,
            "START",
            condition,
            "vasopressor_onset_vector",
            component_index,
            "always",
            "count",
            "count",
        )

    ned = contract.ned_field
    for component_index, operator in enumerate(("LAST", "MAX", "MEAN")):
        add(
            ned,
            operator,
            "NONE",
            "ned_joint_value_state",
            component_index,
            "always",
            "continuous",
            None,
        )

    uop = contract.uop_field
    add(
        uop,
        "SUM",
        "NONE",
        "uop_sum_given_count",
        0,
        "observation_count_positive",
        "continuous",
        None,
    )
    add(
        uop,
        "COUNT",
        "OBSERVED",
        "hurdle_negative_binomial_count",
        0,
        "always",
        "count",
        "count",
    )

    if len(rows) != 155 or len({row.projection_id for row in rows}) != 155:
        raise AssertionError(f"compiled {len(rows)} V2 core projections instead of 155")
    attached = _attached_core_projection_keys(contract)
    compiled = {(row.field, row.operator, row.condition) for row in rows}
    if attached != compiled:
        missing = attached.difference(compiled)
        extra = compiled.difference(attached)
        raise ValueError(
            "compiled physical projections differ from the attached projection registry: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return tuple(rows)


def project_physical_primitives(
    primitives: Mapping[str, torch.Tensor],
    contract: MultiresEventV2Contract,
    schema: Sequence[PhysicalProjectionSpec] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project primitive banks into ``[batch,6,155]`` raw physical views."""

    specs = tuple(schema or build_physical_projection_schema(contract))
    reference = _primitive_bank(primitives, specs[0].likelihood_id)
    batch_size = reference.shape[0]
    values = torch.zeros(
        (batch_size, 6, len(specs)),
        dtype=torch.float64,
        device=reference.device,
    )
    masks = torch.zeros(
        (batch_size, 6, len(specs)), dtype=torch.bool, device=reference.device
    )
    for projection_index, spec in enumerate(specs):
        source = _component(
            primitives,
            spec.likelihood_id,
            spec.field_index,
            spec.component_index,
        ).double()
        gate = _projection_gate(primitives, spec)
        if spec.one_hot_category is not None:
            source = source.round().long().eq(spec.one_hot_category).float()
        if source.shape != (batch_size, 6) or gate.shape != (batch_size, 6):
            raise AssertionError("physical projection source/gate shape drift")
        values[..., projection_index] = torch.where(gate, source, torch.zeros_like(source))
        masks[..., projection_index] = gate
    finite = torch.isfinite(values).all()
    if values.device.type == "cuda":
        # Keep the physical projection boundary fail-closed without forcing one
        # CUDA-to-host synchronization for every registered projection.
        torch._assert_async(finite, "non-finite active physical projection")
    elif not bool(finite.item()):
        invalid_projection = int(
            (~torch.isfinite(values)).any(dim=(0, 1)).nonzero()[0, 0].item()
        )
        raise FloatingPointError(
            f"non-finite physical projection {specs[invalid_projection].projection_id}"
        )
    return values, masks


def generated_coherence_report(
    primitives: Mapping[str, torch.Tensor],
    primitive_masks: Mapping[str, torch.Tensor],
    contract: MultiresEventV2Contract,
) -> list[dict[str, Any]]:
    """Return one structural/support coherence report per sampled trajectory."""

    reference = _primitive_bank(primitives, "categorical_hours_0_4")
    batch_size = reference.shape[0]
    violations: list[set[str]] = [set() for _ in range(batch_size)]
    violation_codes: list[str] = []
    violation_columns: list[torch.Tensor] = []

    if set(primitives) != set(V2_PRIMITIVE_FEEDBACK_DIMS) or set(
        primitive_masks
    ) != set(V2_PRIMITIVE_FEEDBACK_DIMS):
        raise ValueError("generated primitive banks do not exactly cover the V2 contract")

    def flag(code: str, invalid: torch.Tensor) -> None:
        reduced = invalid.bool()
        while reduced.ndim > 1:
            reduced = reduced.any(dim=-1)
        if reduced.shape != (batch_size,):
            raise ValueError(f"coherence flag {code!r} did not reduce to one value per trajectory")
        violation_codes.append(code)
        violation_columns.append(reduced)

    for likelihood_id, value in primitives.items():
        bank = _primitive_bank(primitives, likelihood_id)
        mask = _primitive_bank(primitive_masks, likelihood_id).bool()
        if bank.shape != mask.shape:
            flag(
                f"{likelihood_id}:mask_shape",
                torch.ones(batch_size, dtype=torch.bool, device=reference.device),
            )
            continue
        flag(f"{likelihood_id}:nonfinite", mask & ~torch.isfinite(bank))
        flag(f"{likelihood_id}:inactive_nonzero", ~mask & bank.ne(0))

    expected_masks = {
        likelihood_id: torch.zeros_like(_primitive_bank(primitive_masks, likelihood_id)).bool()
        for likelihood_id in primitive_masks
    }

    def activate(
        likelihood_id: str,
        field_index: int,
        active: torch.Tensor,
        width: int,
    ) -> None:
        expected_masks[likelihood_id][:, :, field_index, :width] = active.unsqueeze(-1)

    categorical = _primitive_bank(primitives, "categorical_hours_0_4")[..., 0]
    categorical_owners = (
        tuple(contract.dense_fields)
        + tuple(contract.ordinal_fields)
        + (contract.verbal_field,)
    )
    field_index = {field: index for index, field in enumerate(contract.core_fields)}
    for field in categorical_owners:
        index = field_index[field]
        hours = categorical[:, :, index]
        activate("categorical_hours_0_4", index, torch.ones_like(hours, dtype=torch.bool), 1)
        flag(f"{field}:observed_hours_support", hours.ne(hours.round()) | hours.lt(0) | hours.gt(4))

    dense_values = _primitive_bank(primitives, "dense_joint_value_state")
    abnormal = _primitive_bank(primitives, "dense_abnormal_duration_vector")
    dense_supports = contract.emission_registry["field_supports"]["dense_continuous"]
    for field in contract.dense_fields:
        index = field_index[field]
        hours = categorical[:, :, index]
        active = hours.gt(0)
        activate("dense_joint_value_state", index, active, 4)
        state = dense_values[:, :, index]
        last, minimum, maximum, mean = state.unbind(dim=-1)
        lower = float(dense_supports[field]["lower"])
        upper = float(dense_supports[field]["upper"])
        flag(
            f"{field}:dense_order",
            active
            & (
                minimum.gt(last + CONTRACT_ARITHMETIC_ATOL)
                | last.gt(maximum + CONTRACT_ARITHMETIC_ATOL)
                | minimum.gt(mean + CONTRACT_ARITHMETIC_ATOL)
                | mean.gt(maximum + CONTRACT_ARITHMETIC_ATOL)
            ),
        )
        safe_hours = hours.clamp_min(1).float()
        lower_mean = (last + (safe_hours - 1.0) * minimum) / safe_hours
        upper_mean = (last + (safe_hours - 1.0) * maximum) / safe_hours
        flag(
            f"{field}:dense_exact_mean_support",
            active
            & (
                mean.lt(lower_mean - CONTRACT_ARITHMETIC_ATOL)
                | mean.gt(upper_mean + CONTRACT_ARITHMETIC_ATOL)
                | (
                    hours.eq(1)
                    & mean.sub(last).abs().gt(CONTRACT_ARITHMETIC_ATOL)
                )
                | (maximum.eq(minimum) & (last.ne(minimum) | mean.ne(minimum)))
            ),
        )
        flag(
            f"{field}:dense_support",
            active
            & (
                state.lt(lower).any(dim=-1)
                | state.gt(upper).any(dim=-1)
            ),
        )
        condition_count = len(contract.dense_abnormal_conditions.get(field, ()))
        if condition_count:
            activate("dense_abnormal_duration_vector", index, active, condition_count)
            durations = abnormal[:, :, index, :condition_count]
            invalid = (
                durations.ne(durations.round()).any(dim=-1)
                | durations.lt(0).any(dim=-1)
                | durations.gt(4).any(dim=-1)
            )
            first_mask, second_mask = dense_abnormal_class_masks(
                field=field,
                condition_keys=contract.dense_abnormal_conditions[field],
                observed_hours=hours,
                minimum=minimum,
                maximum=maximum,
                first_duration=durations[..., 0],
            )
            first_index = durations[..., 0].round().long().clamp(0, 4)
            invalid |= ~first_mask.gather(-1, first_index.unsqueeze(-1)).squeeze(-1)
            if condition_count == 2:
                if second_mask is None:
                    raise AssertionError("two-condition abnormal contract lacks second mask")
                second_index = durations[..., 1].round().long().clamp(0, 4)
                invalid |= ~second_mask.gather(-1, second_index.unsqueeze(-1)).squeeze(-1)
            flag(f"{field}:abnormal_occupancy", active & invalid)

    ordinal = _primitive_bank(primitives, "gcs_ordinal_triple")
    for field in contract.ordinal_fields:
        index = field_index[field]
        hours = categorical[:, :, index]
        active = hours.gt(0)
        activate("gcs_ordinal_triple", index, active, 3)
        state = ordinal[:, :, index]
        last, minimum, maximum = state.unbind(dim=-1)
        invalid = (
            state.ne(state.round()).any(dim=-1)
            | state.lt(1).any(dim=-1)
            | state.gt(int(contract.ordinal_max[field])).any(dim=-1)
            | minimum.gt(last)
            | last.gt(maximum)
        )
        flag(f"{field}:ordinal_state", active & invalid)

    verbal_index = field_index[contract.verbal_field]
    h_obs = categorical[:, :, verbal_index]
    h_u = _component(
        primitives,
        "gcs_verbal_ungradable_hours_given_observed",
        verbal_index,
        0,
    )
    activate(
        "gcs_verbal_ungradable_hours_given_observed",
        verbal_index,
        torch.ones_like(h_obs, dtype=torch.bool),
        1,
    )
    flag(
        "gcs_verbal:ungradable_hours",
        h_u.ne(h_u.round()) | h_u.lt(0) | h_u.gt(h_obs),
    )
    h_gradable = h_obs - h_u
    status_active = h_obs.gt(0)
    activate("gcs_verbal_latest_status", verbal_index, status_active, 1)
    status = _component(primitives, "gcs_verbal_latest_status", verbal_index, 0)
    status_invalid = ~(
        (status.eq(1) & h_gradable.gt(0)) | (status.eq(2) & h_u.gt(0))
    )
    flag("gcs_verbal:latest_status", status_active & status_invalid)
    gradable_active = h_gradable.gt(0)
    activate(
        "gcs_verbal_gradable_ordinal_triple", verbal_index, gradable_active, 3
    )
    verbal_state = _primitive_bank(
        primitives, "gcs_verbal_gradable_ordinal_triple"
    )[:, :, verbal_index]
    last, minimum, maximum = verbal_state.unbind(dim=-1)
    flag(
        "gcs_verbal:gradable_state",
        gradable_active
        & (
            verbal_state.ne(verbal_state.round()).any(dim=-1)
            | verbal_state.lt(1).any(dim=-1)
            | verbal_state.gt(5).any(dim=-1)
            | minimum.gt(last)
            | last.gt(maximum)
            | (h_gradable.eq(1) & ~(minimum.eq(last) & last.eq(maximum)))
            | (h_gradable.eq(2) & ~(last.eq(minimum) | last.eq(maximum)))
        ),
    )

    counts = _primitive_bank(primitives, "hurdle_negative_binomial_count")[..., 0]
    lab_values = _primitive_bank(primitives, "lab_joint_value_state")
    for field in contract.lab_fields:
        index = field_index[field]
        count = counts[:, :, index]
        activate(
            "hurdle_negative_binomial_count",
            index,
            torch.ones_like(count, dtype=torch.bool),
            1,
        )
        flag(f"{field}:observation_count", count.ne(count.round()) | count.lt(0))
        active = count.gt(0)
        activate("lab_joint_value_state", index, active, 3)
        state = lab_values[:, :, index]
        last, minimum, maximum = state.unbind(dim=-1)
        flag(
            f"{field}:lab_order",
            active
            & (
                minimum.gt(last + CONTRACT_ARITHMETIC_ATOL)
                | last.gt(maximum + CONTRACT_ARITHMETIC_ATOL)
                | (
                    count.eq(1)
                    & (
                        minimum.sub(last).abs().gt(CONTRACT_ARITHMETIC_ATOL)
                        | last.sub(maximum).abs().gt(CONTRACT_ARITHMETIC_ATOL)
                    )
                )
                | (
                    count.eq(2)
                    & torch.minimum(
                        last.sub(minimum).abs(), last.sub(maximum).abs()
                    ).gt(CONTRACT_ARITHMETIC_ATOL)
                )
            ),
        )

    respiratory_index = field_index[contract.respiratory_field]
    resp_block = _component(
        primitives, "respiratory_block_evidence", respiratory_index, 0
    )
    resp_edge = _component(
        primitives, "respiratory_edge_evidence_given_block", respiratory_index, 0
    )
    always = torch.ones_like(resp_block, dtype=torch.bool)
    activate("respiratory_block_evidence", respiratory_index, always, 1)
    activate("respiratory_edge_evidence_given_block", respiratory_index, always, 1)
    flag("respiratory:block_evidence", ~(resp_block.eq(0) | resp_block.eq(1)))
    flag(
        "respiratory:edge_evidence",
        ~(resp_edge.eq(0) | resp_edge.eq(1)) | resp_edge.gt(resp_block),
    )
    block_active = resp_block.bool()
    edge_active = resp_edge.bool()
    activate("respiratory_occupancy_vector", respiratory_index, block_active, 5)
    activate("respiratory_onset_vector", respiratory_index, block_active, 4)
    activate("respiratory_edge_state", respiratory_index, edge_active, 1)
    occupancy = _primitive_bank(primitives, "respiratory_occupancy_vector")[:, :, respiratory_index]
    flag(
        "respiratory:occupancy",
        block_active
        & (
            occupancy.lt(0).any(dim=-1)
            | occupancy.sum(dim=-1).sub(4.0).abs().gt(CONTRACT_ARITHMETIC_ATOL)
        ),
    )
    onset = _primitive_bank(primitives, "respiratory_onset_vector")[:, :, respiratory_index]
    documented = occupancy[..., :4]
    edge_state = _component(primitives, "respiratory_edge_state", respiratory_index, 0)
    edge_one_hot = torch.nn.functional.one_hot(
        edge_state.round().long().sub(1).clamp(0, 3), num_classes=4
    ).bool()
    required_onset = edge_active.unsqueeze(-1) & edge_one_hot & documented.eq(0.0)
    require_any_onset = block_active & ~documented.gt(0.0).any(dim=-1)
    flag(
        "respiratory:onset",
        block_active
        & (
            onset.lt(0).any(dim=-1)
            | onset.ne(onset.round()).any(dim=-1)
            | (required_onset & onset.le(0)).any(dim=-1)
            | (require_any_onset & ~onset.gt(0).any(dim=-1))
        ),
    )
    flag(
        "respiratory:edge_state",
        edge_active & (edge_state.ne(edge_state.round()) | edge_state.lt(1) | edge_state.gt(4)),
    )

    vasopressor_index = field_index[contract.vasopressor_field]
    vaso_banks: dict[str, torch.Tensor] = {}
    for likelihood_id, kind in (
        ("vasopressor_duration_vector", "duration"),
        ("vasopressor_edge_state_vector", "binary"),
        ("vasopressor_onset_vector", "count"),
    ):
        activate(likelihood_id, vasopressor_index, always, 6)
        state = _primitive_bank(primitives, likelihood_id)[:, :, vasopressor_index]
        vaso_banks[likelihood_id] = state
        if kind == "duration":
            invalid = state.lt(0).any(dim=-1) | state.gt(4).any(dim=-1)
        elif kind == "binary":
            invalid = ~(state.eq(0) | state.eq(1)).all(dim=-1)
        else:
            invalid = state.lt(0).any(dim=-1) | state.ne(state.round()).any(dim=-1)
        flag(f"vasopressor:{kind}", invalid)
    vaso_required_onset = (
        vaso_banks["vasopressor_edge_state_vector"].bool()
        & vaso_banks["vasopressor_duration_vector"].eq(0.0)
    )
    flag(
        "vasopressor:onset_edge_consistency",
        vaso_required_onset & vaso_banks["vasopressor_onset_vector"].le(0),
    )

    ned_index = field_index[contract.ned_field]
    activate("ned_joint_value_state", ned_index, always, 3)
    ned = _primitive_bank(primitives, "ned_joint_value_state")[:, :, ned_index]
    ned_last, ned_maximum, ned_mean = ned.unbind(dim=-1)
    flag(
        "ned:value_state",
        ned.lt(0).any(dim=-1)
        | ned_last.gt(ned_maximum + CONTRACT_ARITHMETIC_ATOL)
        | ned_mean.gt(ned_maximum + CONTRACT_ARITHMETIC_ATOL),
    )
    compatible_duration = vaso_banks["vasopressor_duration_vector"][..., :5].gt(0).any(
        dim=-1
    )
    compatible_edge = vaso_banks["vasopressor_edge_state_vector"][..., :5].bool().any(
        dim=-1
    )
    flag(
        "ned:activation",
        (~compatible_duration & ned_maximum.ne(0))
        | (ned_maximum.eq(0) & (ned_last.ne(0) | ned_mean.ne(0)))
        | (ned_maximum.gt(0) & ned_mean.le(0))
        | (~compatible_edge & ned_last.gt(0)),
    )

    uop_index = field_index[contract.uop_field]
    uop_count = counts[:, :, uop_index]
    activate(
        "hurdle_negative_binomial_count",
        uop_index,
        torch.ones_like(uop_count, dtype=torch.bool),
        1,
    )
    flag("urine_output:observation_count", uop_count.lt(0) | uop_count.ne(uop_count.round()))
    uop_active = uop_count.gt(0)
    activate("uop_sum_given_count", uop_index, uop_active, 1)
    uop_sum = _component(primitives, "uop_sum_given_count", uop_index, 0)
    flag("urine_output:sum", uop_active & uop_sum.lt(0))

    for likelihood_id, expected in expected_masks.items():
        observed = _primitive_bank(primitive_masks, likelihood_id).bool()
        flag(f"{likelihood_id}:activation_mask", observed.ne(expected))

    if violation_columns:
        # One device transfer per generated ensemble, rather than one transfer
        # for every structural rule.
        violation_matrix = (
            torch.stack(violation_columns, dim=1).detach().cpu().tolist()
        )
        for trajectory_index, flags in enumerate(violation_matrix):
            violations[trajectory_index].update(
                code
                for code, invalid in zip(violation_codes, flags, strict=True)
                if invalid
            )

    return [
        {
            "coherent": not row,
            "violation_count": len(row),
            "violations": sorted(row),
        }
        for row in violations
    ]


def empirical_brier(probability: float, outcome: int | bool) -> float:
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Brier probability must be in [0,1]")
    return (probability - float(bool(outcome))) ** 2


def empirical_rps(samples: torch.Tensor, truth: float, *, minimum: int, maximum: int) -> float:
    """Normalized ranked probability score for an ordered finite support."""

    if maximum <= minimum:
        raise ValueError("RPS support must contain at least two categories")
    values = samples.detach().float().flatten()
    if values.numel() < 1 or not bool(torch.isfinite(values).all().item()):
        raise ValueError("RPS samples must be nonempty and finite")
    if values.ne(values.round()).any() or values.lt(minimum).any() or values.gt(maximum).any():
        raise ValueError("RPS samples are outside the ordered support")
    if truth != round(truth) or not minimum <= truth <= maximum:
        raise ValueError("RPS truth is outside the ordered support")
    thresholds = torch.arange(minimum, maximum, dtype=torch.float32, device=values.device)
    forecast_cdf = values.unsqueeze(-1).le(thresholds).float().mean(dim=0)
    truth_cdf = torch.tensor(float(truth), device=values.device).le(thresholds).float()
    return float((forecast_cdf - truth_cdf).square().mean().item())


def empirical_crps(samples: torch.Tensor, truth: float) -> float:
    values = samples.detach().double().flatten()
    if values.numel() < 1 or not bool(torch.isfinite(values).all().item()):
        raise ValueError("CRPS samples must be nonempty and finite")
    observation = torch.tensor(float(truth), dtype=torch.float64, device=values.device)
    first = (values - observation).abs().mean()
    second = (values[:, None] - values[None, :]).abs().mean() * 0.5
    return float((first - second).item())


def empirical_energy_score(samples: torch.Tensor, truth: torch.Tensor) -> float:
    ensemble = samples.detach().double()
    observation = truth.detach().double()
    if ensemble.ndim != 2 or observation.shape != ensemble.shape[1:]:
        raise ValueError("energy score requires samples [M,D] and truth [D]")
    if ensemble.shape[0] < 1 or not bool(
        torch.isfinite(ensemble).all().item() and torch.isfinite(observation).all().item()
    ):
        raise ValueError("energy score inputs must be nonempty and finite")
    first = torch.linalg.vector_norm(ensemble - observation, dim=-1).mean()
    second = torch.cdist(ensemble, ensemble, p=2).mean() * 0.5
    return float((first - second).item())


def lag1_variogram_score(
    samples: torch.Tensor,
    truth: torch.Tensor,
    *,
    order: float = 0.5,
) -> float:
    """Uniform-weight lag-1 variogram score for one homogeneous 6-block view."""

    ensemble = samples.detach().double()
    observation = truth.detach().double()
    if ensemble.ndim != 2 or observation.shape != ensemble.shape[1:]:
        raise ValueError("variogram score requires samples [M,D] and truth [D]")
    if ensemble.shape[1] != 6:
        raise ValueError("V2 lag-1 variogram is frozen to six blocks")
    if not 0.0 < order < 2.0:
        raise ValueError("variogram order must lie in (0,2)")
    if not bool(
        torch.isfinite(ensemble).all().item() and torch.isfinite(observation).all().item()
    ):
        raise ValueError("variogram score inputs must be finite")
    observed_increment = (observation[1:] - observation[:-1]).abs().pow(order)
    forecast_increment = (
        (ensemble[:, 1:] - ensemble[:, :-1]).abs().pow(order).mean(dim=0)
    )
    return float((observed_increment - forecast_increment).square().mean().item())


def score_physical_ensemble(
    sample_values: torch.Tensor,
    sample_masks: torch.Tensor,
    truth_values: torch.Tensor,
    truth_masks: torch.Tensor,
    schema: Sequence[PhysicalProjectionSpec],
) -> dict[str, Any]:
    """Score the 155 physical views without cross-unit aggregation.

    Conditional CRPS/median-MAE/RPS use the fixed subset where truth is active.
    Forecast trajectories are then conditioned on their generated parent gate;
    a zero-sized conditional ensemble is reported as a coverage failure, never
    imputed. Global energy/variogram scores belong to injective phi(T), not this
    deterministically redundant projection table.
    """

    specs = tuple(schema)
    if sample_values.ndim != 3 or sample_values.shape[1:] != (6, len(specs)):
        raise ValueError("sample physical projections must be [M,6,155]")
    if sample_masks.shape != sample_values.shape:
        raise ValueError("sample projection masks must align with values")
    if truth_values.shape != (6, len(specs)) or truth_masks.shape != truth_values.shape:
        raise ValueError("truth physical projections must be [6,155]")
    trajectories = sample_values.shape[0]
    if trajectories < 1:
        raise ValueError("physical scoring requires at least one trajectory")

    branch_rows: list[dict[str, Any]] = []
    seen_gate: set[tuple[str, str]] = set()
    for projection_index, spec in enumerate(specs):
        if spec.gate != "always":
            gate_key = (spec.field, spec.gate)
            if gate_key not in seen_gate:
                seen_gate.add(gate_key)
                probabilities = sample_masks[:, :, projection_index].float().mean(dim=0)
                outcomes = truth_masks[:, projection_index]
                for block_index in range(6):
                    probability = float(probabilities[block_index].item())
                    outcome = int(outcomes[block_index].item())
                    branch_rows.append(
                        {
                            "branch_id": f"{spec.field}.{spec.gate}",
                            "family": spec.gate,
                            "block_index": block_index,
                            "probability": probability,
                            "outcome": outcome,
                            "brier": empirical_brier(probability, outcome),
                        }
                    )
    rps_by_projection: dict[str, float] = {}
    crps_by_projection: dict[str, float] = {}
    mae_by_projection: dict[str, float] = {}
    brier_by_projection: dict[str, float] = {}
    coverage_by_projection: dict[str, dict[str, Any]] = {}
    coverage_failures: list[dict[str, Any]] = []
    for projection_index, spec in enumerate(specs):
        truth_active = truth_masks[:, projection_index].bool()
        if spec.gate == "always" and not bool(truth_active.all().item()):
            raise ValueError(f"always-defined truth projection {spec.projection_id} is masked")
        block_crps: list[float] = []
        block_mae: list[float] = []
        block_rps: list[float] = []
        block_brier: list[float] = []
        predicted_active_counts: list[int] = []
        for block_index in truth_active.nonzero(as_tuple=False).flatten().tolist():
            generated_active = sample_masks[:, block_index, projection_index].bool()
            generated = sample_values[generated_active, block_index, projection_index]
            count = int(generated.numel())
            predicted_active_counts.append(count)
            if count == 0:
                coverage_failures.append(
                    {
                        "projection_id": spec.projection_id,
                        "block_index": int(block_index),
                        "reason": "zero_generated_conditional_samples",
                    }
                )
                continue
            truth = float(truth_values[block_index, projection_index].item())
            if (
                spec.value_kind in {"ordered", "categorical"}
                and spec.ordered_min is not None
                and spec.ordered_max is not None
            ):
                block_rps.append(
                    empirical_rps(
                        generated,
                        truth,
                        minimum=spec.ordered_min,
                        maximum=spec.ordered_max,
                    )
                )
            if spec.value_kind == "binary":
                probability = float(generated.double().mean().item())
                outcome = int(round(truth))
                brier = empirical_brier(probability, outcome)
                block_brier.append(brier)
                branch_rows.append(
                    {
                        "branch_id": spec.projection_id,
                        "family": "binary_state",
                        "block_index": int(block_index),
                        "probability": probability,
                        "outcome": outcome,
                        "brier": brier,
                    }
                )
            elif (
                spec.value_kind == "categorical"
                and spec.ordered_min is not None
                and spec.ordered_max is not None
            ):
                # LAST_STATUS has two nominal states. Category 2 is the fixed
                # positive event, so this is ordinary binary Brier without
                # imposing a clinically meaningful category distance.
                probability = float(
                    generated.round().long().eq(spec.ordered_max).double().mean().item()
                )
                outcome = int(int(round(truth)) == spec.ordered_max)
                brier = empirical_brier(probability, outcome)
                block_brier.append(brier)
                branch_rows.append(
                    {
                        "branch_id": f"{spec.projection_id}.category_{spec.ordered_max}",
                        "family": "categorical_state",
                        "block_index": int(block_index),
                        "probability": probability,
                        "outcome": outcome,
                        "brier": brier,
                    }
                )
            elif spec.value_kind == "count":
                probability = float(generated.gt(0).double().mean().item())
                outcome = int(truth > 0)
                brier = empirical_brier(probability, outcome)
                block_brier.append(brier)
                branch_rows.append(
                    {
                        "branch_id": f"{spec.projection_id}.positive",
                        "family": "count_positive_branch",
                        "block_index": int(block_index),
                        "probability": probability,
                        "outcome": outcome,
                        "brier": brier,
                    }
                )
            elif spec.field in {
                "norepinephrine_equivalent_dose",
                "urine_output",
            } and spec.operator in {"MAX", "SUM"}:
                probability = float(generated.gt(0).double().mean().item())
                outcome = int(truth > 0)
                brier = empirical_brier(probability, outcome)
                block_brier.append(brier)
                branch_rows.append(
                    {
                        "branch_id": f"{spec.projection_id}.positive",
                        "family": "positive_value_branch",
                        "block_index": int(block_index),
                        "probability": probability,
                        "outcome": outcome,
                        "brier": brier,
                    }
                )
            if spec.value_kind != "categorical":
                block_crps.append(empirical_crps(generated, truth))
                median = float(torch.quantile(generated.float(), 0.5).item())
                block_mae.append(abs(median - truth))
        active_blocks = int(truth_active.sum().item())
        scored_blocks = len(predicted_active_counts) - sum(
            count == 0 for count in predicted_active_counts
        )
        coverage_by_projection[spec.projection_id] = {
            "truth_active_blocks": active_blocks,
            "scored_blocks": scored_blocks,
            "generated_active_counts": predicted_active_counts,
            "complete": scored_blocks == active_blocks,
        }
        if block_rps:
            rps_by_projection[spec.projection_id] = sum(block_rps) / len(block_rps)
        if block_crps:
            crps_by_projection[spec.projection_id] = sum(block_crps) / len(block_crps)
            mae_by_projection[spec.projection_id] = sum(block_mae) / len(block_mae)
        if block_brier:
            brier_by_projection[spec.projection_id] = sum(block_brier) / len(block_brier)

    by_family: dict[str, list[float]] = {}
    for row in branch_rows:
        by_family.setdefault(str(row["family"]), []).append(float(row["brier"]))
    return {
        "trajectories": trajectories,
        "branch_brier": (
            sum(float(row["brier"]) for row in branch_rows) / len(branch_rows)
            if branch_rows
            else None
        ),
        "branch_brier_by_family": {
            family: sum(values) / len(values) for family, values in sorted(by_family.items())
        },
        "branch_calibration_rows": branch_rows,
        "rps_by_projection": rps_by_projection,
        "brier_by_projection": brier_by_projection,
        "crps_by_projection": crps_by_projection,
        "median_mae_by_projection": mae_by_projection,
        "coverage_by_projection": coverage_by_projection,
        "cross_unit_aggregation": "forbidden",
        "physical_metric_contract_status": (
            "incomplete_conditional_sample_coverage" if coverage_failures else "complete"
        ),
        "physical_metric_blockers": coverage_failures,
        "care_and_procedure": {
            "status": "not_applicable",
            "reason": "care_and_procedure_joint_objective_off",
        },
    }


def calibration_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    if bins < 2:
        raise ValueError("calibration requires at least two bins")
    grouped: dict[str, list[Mapping[str, Any]]] = {"overall": list(rows)}
    for row in rows:
        grouped.setdefault(str(row["family"]), []).append(row)
    result: dict[str, Any] = {}
    for family, family_rows in sorted(grouped.items()):
        counts = [0] * bins
        probability_sum = [0.0] * bins
        outcome_sum = [0.0] * bins
        brier_sum = 0.0
        for row in family_rows:
            probability = float(row["probability"])
            outcome = int(row["outcome"])
            index = min(int(probability * bins), bins - 1)
            counts[index] += 1
            probability_sum[index] += probability
            outcome_sum[index] += outcome
            brier_sum += empirical_brier(probability, outcome)
        total = len(family_rows)
        bin_rows = []
        ece = 0.0
        for index in range(bins):
            count = counts[index]
            predicted = probability_sum[index] / count if count else None
            observed = outcome_sum[index] / count if count else None
            if count:
                ece += count / total * abs(float(predicted) - float(observed))
            bin_rows.append(
                {
                    "bin": index,
                    "lower": index / bins,
                    "upper": (index + 1) / bins,
                    "count": count,
                    "mean_probability": predicted,
                    "observed_frequency": observed,
                }
            )
        result[family] = {
            "events": total,
            "brier": brier_sum / total if total else None,
            "ece": ece if total else None,
            "bins": bin_rows,
        }
    return result


def _primitive_bank(
    primitives: Mapping[str, torch.Tensor], likelihood_id: str
) -> torch.Tensor:
    value = primitives.get(likelihood_id)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"primitive bank {likelihood_id!r} is missing")
    if value.ndim == 3:
        value = value.unsqueeze(-1)
    if value.ndim != 4 or value.shape[1:3] != (6, 29):
        raise ValueError(f"primitive bank {likelihood_id!r} must be [B,6,29,W]")
    return value


def _component(
    primitives: Mapping[str, torch.Tensor],
    likelihood_id: str,
    field_index: int,
    component_index: int,
) -> torch.Tensor:
    bank = _primitive_bank(primitives, likelihood_id)
    if component_index < 0 or component_index >= bank.shape[-1]:
        raise ValueError(
            f"primitive component {likelihood_id}[{component_index}] is unavailable"
        )
    return bank[:, :, field_index, component_index]


def _projection_gate(
    primitives: Mapping[str, torch.Tensor], spec: PhysicalProjectionSpec
) -> torch.Tensor:
    source = _component(
        primitives, spec.likelihood_id, spec.field_index, spec.component_index
    )
    if spec.gate == "always":
        return torch.ones_like(source, dtype=torch.bool)
    if spec.gate == "observed_hours_positive":
        return _component(
            primitives, "categorical_hours_0_4", spec.field_index, 0
        ).gt(0)
    if spec.gate == "gradable_hours_positive":
        observed = _component(
            primitives, "categorical_hours_0_4", spec.field_index, 0
        )
        ungradable = _component(
            primitives,
            "gcs_verbal_ungradable_hours_given_observed",
            spec.field_index,
            0,
        )
        return observed.sub(ungradable).gt(0)
    if spec.gate == "observation_count_positive":
        return _component(
            primitives, "hurdle_negative_binomial_count", spec.field_index, 0
        ).gt(0)
    if spec.gate == "respiratory_block_evidence":
        return _component(
            primitives, "respiratory_block_evidence", spec.field_index, 0
        ).bool()
    if spec.gate == "respiratory_edge_evidence":
        return _component(
            primitives,
            "respiratory_edge_evidence_given_block",
            spec.field_index,
            0,
        ).bool()
    raise ValueError(f"unknown physical projection gate {spec.gate!r}")


def _attached_core_projection_keys(
    contract: MultiresEventV2Contract,
) -> set[tuple[str, str, str]]:
    registry = contract.projection_registry
    core = set(contract.core_fields)
    keys: set[tuple[str, str, str]] = set()
    rules = registry.get("rules")
    if not isinstance(rules, Sequence):
        raise ValueError("projection registry lacks rules")
    for rule in rules:
        if not isinstance(rule, Mapping) or rule.get("action") != "deterministic_projection":
            continue
        match = rule.get("match")
        if not isinstance(match, Mapping):
            raise ValueError("deterministic projection rule lacks match")
        operators = tuple(str(value) for value in match.get("operators", ()))
        pairs = match.get("field_condition_pairs")
        if pairs is not None:
            for pair in pairs:
                field, condition = str(pair[0]), str(pair[1])
                if field in core:
                    keys.update((field, operator, condition) for operator in operators)
            continue
        fields = tuple(str(value) for value in match.get("fields", ()))
        conditions = tuple(str(value) for value in match.get("conditions", ()))
        for field in fields:
            if field in core:
                keys.update(
                    (field, operator, condition)
                    for operator in operators
                    for condition in conditions
                )
    new_rows = registry.get("new_v2_projections")
    if not isinstance(new_rows, Sequence):
        raise ValueError("projection registry lacks new_v2_projections")
    for row in new_rows:
        if not isinstance(row, Mapping):
            raise ValueError("new V2 projection row must be a mapping")
        field = str(row.get("field"))
        if field in core:
            keys.add((field, str(row.get("operator")), str(row.get("condition"))))
    if len(keys) != 155:
        raise ValueError(f"attached projection registry expands to {len(keys)} core views")
    return keys


__all__ = [
    "PhysicalProjectionSpec",
    "PrimitiveVectorCoordinate",
    "STANDARDIZED_PRIMITIVE_COORDINATE_CONTRACT",
    "STANDARDIZED_PRIMITIVE_SCALE_SCHEMA",
    "STANDARDIZED_PRIMITIVE_SCALE_VERSION",
    "build_physical_projection_schema",
    "build_standardized_primitive_schema",
    "calibration_summary",
    "empirical_brier",
    "empirical_coordinate_crps",
    "empirical_crps",
    "empirical_energy_score",
    "empirical_rps",
    "generated_coherence_report",
    "lag1_variogram_score",
    "load_standardized_primitive_scale_artifact",
    "project_physical_primitives",
    "required_standardized_scale_keys",
    "score_physical_ensemble",
    "score_standardized_primitive_ensemble",
    "standardize_primitive_trajectory",
]
