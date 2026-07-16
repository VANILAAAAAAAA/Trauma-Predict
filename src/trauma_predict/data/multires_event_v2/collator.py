from __future__ import annotations

from typing import Any, Mapping, Sequence

from trauma_predict.data.multires_event.contract import EventTemplateRegistry, SupervisionContract
from trauma_predict.data.multires_event.normalization import RobustNormalizer

from .contract import (
    BLOCK_IDS,
    DETERMINISTIC_PROJECTIONS_PER_BLOCK,
    EXPECTED_ENABLED_FACTOR_COUNT,
    MultiresEventV2Contract,
)


LIKELIHOOD_SPECS: Mapping[str, tuple[str, int | None]] = {
    "categorical_hours_0_4": ("long", None),
    "dense_joint_value_state": ("float", 4),
    "dense_abnormal_duration_vector": ("long", 2),
    "gcs_ordinal_triple": ("long", 3),
    "gcs_verbal_ungradable_hours_given_observed": ("long", None),
    "gcs_verbal_latest_status": ("long", None),
    "gcs_verbal_gradable_ordinal_triple": ("long", 3),
    "hurdle_negative_binomial_count": ("long", None),
    "lab_joint_value_state": ("float", 3),
    "respiratory_block_evidence": ("long", None),
    "respiratory_edge_evidence_given_block": ("long", None),
    "respiratory_occupancy_vector": ("float", 5),
    "respiratory_edge_state": ("long", None),
    "respiratory_onset_vector": ("long", 4),
    "vasopressor_duration_vector": ("float", 6),
    "vasopressor_edge_state_vector": ("long", 6),
    "vasopressor_onset_vector": ("long", 6),
    "ned_joint_value_state": ("float", 3),
    "uop_sum_given_count": ("float", None),
}

VALUE_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    "dense_joint_value_state": ("last", "min", "max", "mean"),
    "dense_abnormal_duration_vector": ("condition_slot_0", "condition_slot_1"),
    "gcs_ordinal_triple": ("last", "min", "max"),
    "gcs_verbal_gradable_ordinal_triple": ("last", "min", "max"),
    "lab_joint_value_state": ("last", "min", "max"),
    "respiratory_occupancy_vector": (
        "RESP_INVASIVE",
        "RESP_NONINVASIVE",
        "RESP_HIGH_FLOW",
        "RESP_OTHER_OXYGEN",
        "uncovered",
    ),
    "respiratory_onset_vector": (
        "RESP_INVASIVE",
        "RESP_NONINVASIVE",
        "RESP_HIGH_FLOW",
        "RESP_OTHER_OXYGEN",
    ),
    "vasopressor_duration_vector": (
        "VASO_NOREPINEPHRINE",
        "VASO_EPINEPHRINE",
        "VASO_PHENYLEPHRINE",
        "VASO_VASOPRESSIN",
        "VASO_DOPAMINE",
        "VASO_OTHER",
    ),
    "vasopressor_edge_state_vector": (
        "VASO_NOREPINEPHRINE",
        "VASO_EPINEPHRINE",
        "VASO_PHENYLEPHRINE",
        "VASO_VASOPRESSIN",
        "VASO_DOPAMINE",
        "VASO_OTHER",
    ),
    "vasopressor_onset_vector": (
        "VASO_NOREPINEPHRINE",
        "VASO_EPINEPHRINE",
        "VASO_PHENYLEPHRINE",
        "VASO_VASOPRESSIN",
        "VASO_DOPAMINE",
        "VASO_OTHER",
    ),
    "ned_joint_value_state": ("last", "max", "mean"),
}

VERBAL_STATUS_IDS = {"UNOBSERVED": 0, "GRADABLE": 1, "UNGRADABLE": 2}


class MultiresEventV2Collator:
    """Batch unchanged V1 input tensors with V2 stochastic-process primitives.

    All target tensors share the registered ``[batch, 6 blocks, 29 fields, ...]``
    prefix. The 155 deterministic five-tuple views are intentionally absent.
    """

    def __init__(
        self,
        *,
        contract: MultiresEventV2Contract,
        supervision: SupervisionContract,
        templates: EventTemplateRegistry,
        normalization: RobustNormalizer,
    ) -> None:
        if len(contract.core_fields) != 29:
            raise ValueError("V2 collator requires exactly 29 registered core fields")
        if len(contract.respiratory_modalities) != 4:
            raise ValueError("V2 collator requires four respiratory modality channels")
        if len(contract.vasopressor_agents) != 6:
            raise ValueError("V2 collator requires six vasopressor agent channels")
        if max((len(values) for values in contract.dense_abnormal_conditions.values()), default=0) > 2:
            raise ValueError("V2 dense abnormal tensor supports at most two registered channels per field")
        self.contract = contract
        self.supervision = supervision
        self.templates = templates
        self.normalization = normalization
        self.field_index = {field: index for index, field in enumerate(contract.core_fields)}
        self.metadata = self._build_metadata()

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("multires_event_v2 batch is empty")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is a train extra
            raise RuntimeError("MultiresEventV2Collator requires torch") from exc

        for record in records:
            self._validate_joined_record(record)
        input_batch = self._collate_input(torch, records)
        primitives, primitive_masks, derived_gates = self._collate_targets(torch, records)
        primitive_gates = self._build_gate_views(primitives, derived_gates)
        return {
            "input_batch": input_batch,
            "target_primitives": primitives,
            "target_primitive_masks": primitive_masks,
            "target_primitive_gates": primitive_gates,
            "target_primitive_metadata": self.metadata,
            "sample_id": [str(record["sample_id"]) for record in records],
            "subject_id": [str(record["subject_id"]) for record in records],
            "prediction_hour": torch.tensor(
                [int(record["prediction_hour"]) for record in records], dtype=torch.long
            ),
            "contract_bundle_hash": self.contract.contract_bundle_hash,
        }

    def _validate_joined_record(self, record: Mapping[str, Any]) -> None:
        required = {
            "sample_id",
            "subject_id",
            "hadm_id",
            "stay_id",
            "prediction_hour",
            "split",
            "base_content_hash",
            "target_content_hash",
            "input_record",
            "target_record",
        }
        if set(record) != required:
            raise ValueError("V2 collator record keys do not match the joined dataset contract")
        input_record = record["input_record"]
        target_record = record["target_record"]
        if not isinstance(input_record, Mapping) or not isinstance(target_record, Mapping):
            raise ValueError("V2 joined input/target records must be mappings")
        forbidden = {"target_events", "target_mask", "target_source_count", "target_contract"}
        if forbidden.intersection(input_record):
            raise ValueError("V1 target supervision reached the V2 collator")
        self.contract.validate_target_record(target_record, verify_content_hash=True)
        for key in ("sample_id", "subject_id", "hadm_id", "stay_id", "prediction_hour", "split"):
            if str(input_record.get(key)) != str(record[key]):
                raise ValueError(f"V2 collator input identity mismatch for {key}")
            if str(target_record.get(key)) != str(record[key]):
                raise ValueError(f"V2 collator target identity mismatch for {key}")

    def _collate_input(self, torch: Any, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        prepared = [self._prepare_input(record["input_record"]) for record in records]
        max_events = max(len(item["event_field_ids"]) for item in prepared)
        max_blocks = max(len(item["block_role_ids"]) for item in prepared)
        batch: dict[str, Any] = {
            "event_field_ids": torch.tensor(
                _pad([item["event_field_ids"] for item in prepared], max_events, 0),
                dtype=torch.long,
            ),
            "event_operator_ids": torch.tensor(
                _pad([item["event_operator_ids"] for item in prepared], max_events, 0),
                dtype=torch.long,
            ),
            "event_condition_ids": torch.tensor(
                _pad([item["event_condition_ids"] for item in prepared], max_events, 0),
                dtype=torch.long,
            ),
            "event_values": torch.tensor(
                _pad([item["event_values"] for item in prepared], max_events, 0.0),
                dtype=torch.float32,
            ),
            "event_value_mask": torch.tensor(
                _pad([item["event_value_mask"] for item in prepared], max_events, False),
                dtype=torch.bool,
            ),
            "event_study_slot_ids": torch.tensor(
                _pad([item["event_study_slot_ids"] for item in prepared], max_events, 0),
                dtype=torch.long,
            ),
            "event_block_index": torch.tensor(
                _pad([item["event_block_index"] for item in prepared], max_events, 0),
                dtype=torch.long,
            ),
            "event_mask": torch.tensor(
                _pad(
                    [[True] * len(item["event_field_ids"]) for item in prepared],
                    max_events,
                    False,
                ),
                dtype=torch.bool,
            ),
            "block_role_ids": torch.tensor(
                _pad([item["block_role_ids"] for item in prepared], max_blocks, 0),
                dtype=torch.long,
            ),
            "block_resolution_ids": torch.tensor(
                _pad([item["block_resolution_ids"] for item in prepared], max_blocks, 0),
                dtype=torch.long,
            ),
            "block_relative_start": torch.tensor(
                _pad([item["block_relative_start"] for item in prepared], max_blocks, 0.0),
                dtype=torch.float32,
            ),
            "block_relative_end": torch.tensor(
                _pad([item["block_relative_end"] for item in prepared], max_blocks, 0.0),
                dtype=torch.float32,
            ),
            "block_span": torch.tensor(
                _pad([item["block_span"] for item in prepared], max_blocks, 0.0),
                dtype=torch.float32,
            ),
            "block_mask": torch.tensor(
                _pad(
                    [[True] * len(item["block_role_ids"]) for item in prepared],
                    max_blocks,
                    False,
                ),
                dtype=torch.bool,
            ),
            "latest_input_block_index": torch.tensor(
                [item["latest_input_block_index"] for item in prepared], dtype=torch.long
            ),
            "static_numeric": torch.tensor(
                [item["static_numeric"] for item in prepared], dtype=torch.float32
            ),
            "static_numeric_mask": torch.tensor(
                [item["static_numeric_mask"] for item in prepared], dtype=torch.bool
            ),
            "static_categorical": torch.tensor(
                [item["static_categorical"] for item in prepared], dtype=torch.long
            ),
            "static_categorical_mask": torch.tensor(
                [item["static_categorical_mask"] for item in prepared], dtype=torch.bool
            ),
            "prediction_hour": torch.tensor(
                [int(item["prediction_hour"]) for item in prepared], dtype=torch.long
            ),
            "sample_id": [str(item["sample_id"]) for item in prepared],
            "subject_id": [str(item["subject_id"]) for item in prepared],
        }
        batch["operator_ids"] = batch["event_operator_ids"]
        batch["condition_ids"] = batch["event_condition_ids"]
        batch["values"] = batch["event_values"]
        batch["block_index"] = batch["event_block_index"]
        batch["resolution_ids"] = batch["block_resolution_ids"]
        batch["relative_start"] = batch["block_relative_start"]
        batch["relative_end"] = batch["block_relative_end"]
        batch["span"] = batch["block_span"]
        forbidden = {
            "target_values",
            "target_raw_values",
            "target_mask",
            "f24_target_raw_values",
            "f24_target_mask",
            "query_field_ids",
        }
        if forbidden.intersection(batch):
            raise AssertionError("V1 target tensors leaked into input_batch")
        return batch

    def _prepare_input(self, record: Mapping[str, Any]) -> dict[str, Any]:
        input_blocks = sorted(record["block_table"], key=lambda item: int(item["block_id"]))
        if any(block.get("side") != "input" for block in input_blocks):
            raise ValueError("V2 input_record block table contains a target block")
        relative_ends = [float(block["relative_end_hour"]) for block in input_blocks]
        latest_candidates = [
            index for index, relative_end in enumerate(relative_ends) if relative_end == 0.0
        ]
        if len(latest_candidates) != 1:
            raise ValueError(
                "V2 input block table must contain exactly one block ending at prediction time"
            )
        latest_input_block_index = latest_candidates[0]
        if any(relative_end > 0.0 for relative_end in relative_ends):
            raise ValueError("V2 input block table contains a post-anchor block")
        if any(left > right for left, right in zip(relative_ends, relative_ends[1:])):
            raise ValueError("V2 input block ids are not in chronological order")
        if latest_input_block_index != len(input_blocks) - 1:
            raise ValueError("the block ending at prediction time must be the final input block")
        latest_block = input_blocks[latest_input_block_index]
        if (
            latest_block.get("role") != "NEAR"
            or latest_block.get("resolution") != "H1"
            or float(latest_block.get("relative_start_hour")) != -1.0
            or float(latest_block.get("span_hours")) != 1.0
        ):
            raise ValueError(
                "the final input block must be the frozen NEAR/H1 (-1h, 0h] block"
            )
        block_position = {
            int(block["block_id"]): index for index, block in enumerate(input_blocks)
        }
        blocks_by_id = {int(item["block_id"]): item for item in input_blocks}
        result: dict[str, Any] = {
            "event_field_ids": [],
            "event_operator_ids": [],
            "event_condition_ids": [],
            "event_values": [],
            "event_value_mask": [],
            "event_study_slot_ids": [],
            "event_block_index": [],
        }
        for event in record["input_events"]:
            field_id, operator_id, condition_id, raw_value, block_id = event
            if int(field_id) in self.supervision.excluded_input_field_ids:
                raise ValueError("a model-side excluded V1 input field reached the V2 collator")
            block = blocks_by_id.get(int(block_id))
            if block is None:
                raise ValueError("V1 input event references a non-input block")
            template = self.templates.get(int(field_id), int(operator_id), int(condition_id))
            study_slot_id = 0
            if template.value_type == "study_slot":
                study_slot_id = int(raw_value)
                if not 1 <= study_slot_id <= 8:
                    raise ValueError("CXR study_slot must be in 1..8")
                transformed, observed = 0.0, False
            else:
                transformed, observed = self.normalization.transform_event(
                    raw_value,
                    template=template,
                    resolution=str(block["resolution"]),
                    span_hours=float(block["span_hours"]),
                )
            result["event_field_ids"].append(int(field_id))
            result["event_operator_ids"].append(int(operator_id))
            result["event_condition_ids"].append(int(condition_id))
            result["event_values"].append(transformed)
            result["event_value_mask"].append(observed)
            result["event_study_slot_ids"].append(study_slot_id)
            result["event_block_index"].append(block_position[int(block_id)])

        static = record.get("static") or {}
        static_numeric: list[float] = []
        static_numeric_mask: list[bool] = []
        for field in self.supervision.static_numeric_fields:
            transformed, observed = self.normalization.transform_static(field, static.get(field))
            static_numeric.append(transformed)
            static_numeric_mask.append(observed)
        static_categorical: list[int] = []
        static_categorical_mask: list[bool] = []
        for field in self.supervision.static_categorical_fields:
            value = static.get(field)
            vocabulary = self.supervision.static_category_ids[field]
            category_id = vocabulary.get(str(value), self.supervision.static_unknown_id)
            if value is not None and category_id == self.supervision.static_unknown_id:
                raise ValueError(f"static field {field} has unregistered category {value}")
            static_categorical.append(category_id)
            static_categorical_mask.append(category_id != self.supervision.static_unknown_id)
        result.update(
            {
                "block_role_ids": [int(item["role_id"]) for item in input_blocks],
                "block_resolution_ids": [int(item["resolution_id"]) for item in input_blocks],
                "block_relative_start": [float(item["relative_start_hour"]) for item in input_blocks],
                "block_relative_end": [float(item["relative_end_hour"]) for item in input_blocks],
                "block_span": [float(item["span_hours"]) for item in input_blocks],
                "latest_input_block_index": latest_input_block_index,
                "static_numeric": static_numeric,
                "static_numeric_mask": static_numeric_mask,
                "static_categorical": static_categorical,
                "static_categorical_mask": static_categorical_mask,
                "sample_id": record["sample_id"],
                "subject_id": record["subject_id"],
                "prediction_hour": record["prediction_hour"],
            }
        )
        return result

    def _collate_targets(
        self,
        torch: Any,
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        batch_size = len(records)
        prefix = (batch_size, len(BLOCK_IDS), len(self.contract.core_fields))
        primitives: dict[str, Any] = {}
        masks: dict[str, Any] = {}
        for likelihood_id, (dtype_name, width) in LIKELIHOOD_SPECS.items():
            shape = prefix if width is None else prefix + (width,)
            # Physical-unit supervision must retain the serialized double
            # precision.  In particular, dense v can acquire a real 1e-5-scale
            # support error if MIN/LAST/MAX/MEAN are rounded to float32 before
            # canonicalization.  Model inputs and sampled feedback remain
            # float32; only teacher likelihood targets use this bank.
            dtype = torch.float64 if dtype_name == "float" else torch.long
            primitives[likelihood_id] = torch.zeros(shape, dtype=dtype)
            masks[likelihood_id] = torch.zeros(prefix, dtype=torch.bool)
        h_gradable = torch.zeros(prefix, dtype=torch.long)
        ned_compatible_active = torch.zeros(prefix, dtype=torch.bool)

        for batch_index, joined in enumerate(records):
            for block_index, block in enumerate(joined["target_record"]["blocks"]):
                processes = block["processes"]
                for field in self.contract.dense_fields:
                    field_index = self.field_index[field]
                    process = processes[field]
                    observed = int(process["observed_hours"])
                    _put_scalar(
                        primitives,
                        masks,
                        "categorical_hours_0_4",
                        batch_index,
                        block_index,
                        field_index,
                        observed,
                        True,
                    )
                    state = process["value_state"]
                    if state is not None:
                        _put_vector(
                            primitives,
                            masks,
                            "dense_joint_value_state",
                            batch_index,
                            block_index,
                            field_index,
                            [state[key] for key in VALUE_COMPONENTS["dense_joint_value_state"]],
                            True,
                        )
                    conditions = self.contract.dense_abnormal_conditions.get(field, ())
                    if conditions and observed > 0:
                        vector = [int(process["abnormal_occupancy"][condition]) for condition in conditions]
                        vector += [0] * (2 - len(vector))
                        _put_vector(
                            primitives,
                            masks,
                            "dense_abnormal_duration_vector",
                            batch_index,
                            block_index,
                            field_index,
                            vector,
                            True,
                        )

                for field in self.contract.ordinal_fields:
                    field_index = self.field_index[field]
                    process = processes[field]
                    observed = int(process["observed_hours"])
                    _put_scalar(
                        primitives,
                        masks,
                        "categorical_hours_0_4",
                        batch_index,
                        block_index,
                        field_index,
                        observed,
                        True,
                    )
                    state = process["ordinal_state"]
                    if state is not None:
                        _put_vector(
                            primitives,
                            masks,
                            "gcs_ordinal_triple",
                            batch_index,
                            block_index,
                            field_index,
                            [state[key] for key in VALUE_COMPONENTS["gcs_ordinal_triple"]],
                            True,
                        )

                verbal_index = self.field_index[self.contract.verbal_field]
                verbal = processes[self.contract.verbal_field]
                h_obs = int(verbal["observed_hours"])
                h_u = int(verbal["ungradable_hours"])
                h_g = int(verbal["gradable_hours"])
                _put_scalar(
                    primitives,
                    masks,
                    "categorical_hours_0_4",
                    batch_index,
                    block_index,
                    verbal_index,
                    h_obs,
                    True,
                )
                _put_scalar(
                    primitives,
                    masks,
                    "gcs_verbal_ungradable_hours_given_observed",
                    batch_index,
                    block_index,
                    verbal_index,
                    h_u,
                    True,
                )
                _put_scalar(
                    primitives,
                    masks,
                    "gcs_verbal_latest_status",
                    batch_index,
                    block_index,
                    verbal_index,
                    VERBAL_STATUS_IDS[verbal["last_observation_status"]],
                    h_obs > 0,
                )
                h_gradable[batch_index, block_index, verbal_index] = h_g
                state = verbal["gradable_state"]
                if state is not None:
                    _put_vector(
                        primitives,
                        masks,
                        "gcs_verbal_gradable_ordinal_triple",
                        batch_index,
                        block_index,
                        verbal_index,
                        [
                            state[key]
                            for key in VALUE_COMPONENTS[
                                "gcs_verbal_gradable_ordinal_triple"
                            ]
                        ],
                        True,
                    )

                for field in self.contract.lab_fields:
                    field_index = self.field_index[field]
                    process = processes[field]
                    count = int(process["observation_count"])
                    _put_scalar(
                        primitives,
                        masks,
                        "hurdle_negative_binomial_count",
                        batch_index,
                        block_index,
                        field_index,
                        count,
                        True,
                    )
                    state = process["value_state"]
                    if state is not None:
                        _put_vector(
                            primitives,
                            masks,
                            "lab_joint_value_state",
                            batch_index,
                            block_index,
                            field_index,
                            [state[key] for key in VALUE_COMPONENTS["lab_joint_value_state"]],
                            True,
                        )

                respiratory_index = self.field_index[self.contract.respiratory_field]
                respiratory = processes[self.contract.respiratory_field]
                block_evidence = bool(respiratory["block_evidence"])
                edge_evidence = bool(respiratory["edge_evidence"])
                _put_scalar(
                    primitives,
                    masks,
                    "respiratory_block_evidence",
                    batch_index,
                    block_index,
                    respiratory_index,
                    int(block_evidence),
                    True,
                )
                _put_scalar(
                    primitives,
                    masks,
                    "respiratory_edge_evidence_given_block",
                    batch_index,
                    block_index,
                    respiratory_index,
                    int(edge_evidence),
                    True,
                )
                if block_evidence:
                    durations = respiratory["documented_duration"]
                    _put_vector(
                        primitives,
                        masks,
                        "respiratory_occupancy_vector",
                        batch_index,
                        block_index,
                        respiratory_index,
                        [durations[name] for name in self.contract.respiratory_modalities]
                        + [respiratory["uncovered_duration"]],
                        True,
                    )
                    onsets = respiratory["onset_count"]
                    _put_vector(
                        primitives,
                        masks,
                        "respiratory_onset_vector",
                        batch_index,
                        block_index,
                        respiratory_index,
                        [onsets[name] for name in self.contract.respiratory_modalities],
                        True,
                    )
                if edge_evidence:
                    edge_category_id = (
                        self.contract.respiratory_modalities.index(
                            respiratory["edge_category"]
                        )
                        + 1
                    )
                    _put_scalar(
                        primitives,
                        masks,
                        "respiratory_edge_state",
                        batch_index,
                        block_index,
                        respiratory_index,
                        edge_category_id,
                        True,
                    )

                vasopressor_index = self.field_index[self.contract.vasopressor_field]
                vasopressor = processes[self.contract.vasopressor_field]
                for likelihood_id, source_key in (
                    ("vasopressor_duration_vector", "duration"),
                    ("vasopressor_edge_state_vector", "edge_state"),
                    ("vasopressor_onset_vector", "onset_count"),
                ):
                    vector = vasopressor[source_key]
                    _put_vector(
                        primitives,
                        masks,
                        likelihood_id,
                        batch_index,
                        block_index,
                        vasopressor_index,
                        [vector[name] for name in self.contract.vasopressor_agents],
                        True,
                    )

                ned_index = self.field_index[self.contract.ned_field]
                ned = processes[self.contract.ned_field]["value_state"]
                _put_vector(
                    primitives,
                    masks,
                    "ned_joint_value_state",
                    batch_index,
                    block_index,
                    ned_index,
                    [ned[key] for key in VALUE_COMPONENTS["ned_joint_value_state"]],
                    True,
                )
                compatible_agents = self.contract.vasopressor_agents[:-1]
                ned_compatible_active[batch_index, block_index, ned_index] = any(
                    float(vasopressor["duration"][agent]) > 0.0 for agent in compatible_agents
                )

                uop_index = self.field_index[self.contract.uop_field]
                uop = processes[self.contract.uop_field]
                uop_count = int(uop["observation_count"])
                _put_scalar(
                    primitives,
                    masks,
                    "hurdle_negative_binomial_count",
                    batch_index,
                    block_index,
                    uop_index,
                    uop_count,
                    True,
                )
                if uop["sum"] is not None:
                    _put_scalar(
                        primitives,
                        masks,
                        "uop_sum_given_count",
                        batch_index,
                        block_index,
                        uop_index,
                        float(uop["sum"]),
                        True,
                    )

        return primitives, masks, {
            "gcs_verbal_gradable_hours": h_gradable,
            "ned_compatible_active": ned_compatible_active,
        }

    def _build_gate_views(
        self,
        primitives: Mapping[str, Any],
        derived: Mapping[str, Any],
    ) -> dict[str, Mapping[str, Any]]:
        hours = primitives["categorical_hours_0_4"]
        counts = primitives["hurdle_negative_binomial_count"]
        verbal_ungradable = primitives["gcs_verbal_ungradable_hours_given_observed"]
        resp_block = primitives["respiratory_block_evidence"]
        resp_edge = primitives["respiratory_edge_evidence_given_block"]
        return {
            "dense_joint_value_state": {"observed_hours": hours},
            "dense_abnormal_duration_vector": {
                "observed_hours": hours,
                "upper_hours": hours,
            },
            "gcs_ordinal_triple": {"observed_hours": hours},
            "gcs_verbal_ungradable_hours_given_observed": {"observed_hours": hours},
            "gcs_verbal_latest_status": {
                "observed_hours": hours,
                "ungradable_hours": verbal_ungradable,
            },
            "gcs_verbal_gradable_ordinal_triple": {
                "observed_hours": hours,
                "ungradable_hours": verbal_ungradable,
                "gradable_hours": derived["gcs_verbal_gradable_hours"],
            },
            "lab_joint_value_state": {"observation_count": counts},
            "respiratory_edge_evidence_given_block": {"block_evidence": resp_block},
            "respiratory_occupancy_vector": {"block_evidence": resp_block},
            "respiratory_edge_state": {"edge_evidence": resp_edge},
            "respiratory_onset_vector": {"block_evidence": resp_block},
            "ned_joint_value_state": {
                "compatible_active": derived["ned_compatible_active"]
            },
            "uop_sum_given_count": {"observation_count": counts},
        }

    def _build_metadata(self) -> Mapping[str, Any]:
        field_supports = self.contract.emission_registry.get("field_supports")
        if not isinstance(field_supports, Mapping):
            raise ValueError("V2 emission registry lacks field_supports")
        dense_supports = field_supports.get("dense_continuous")
        if not isinstance(dense_supports, Mapping) or set(dense_supports) != set(
            self.contract.dense_fields
        ):
            raise ValueError(
                "V2 emission registry dense supports must exactly cover dense fields"
            )
        valid_ranges: dict[str, tuple[float, float]] = {}
        for field in self.contract.dense_fields:
            row = dense_supports[field]
            if not isinstance(row, Mapping):
                raise ValueError(f"V2 dense support for {field!r} must be a mapping")
            lower = float(row.get("lower"))
            upper = float(row.get("upper"))
            if not lower < upper:
                raise ValueError(f"V2 dense support for {field!r} must satisfy lower < upper")
            valid_ranges[field] = (lower, upper)
        likelihoods_by_field = {
            field: _likelihoods_for_field(self.contract, field)
            for field in self.contract.core_fields
        }
        factor_order = []
        for block_index, block_id in enumerate(BLOCK_IDS):
            for field_index, field in enumerate(self.contract.core_fields):
                field_id = self.contract.registered_core_field_ids[field_index]
                for likelihood_id in likelihoods_by_field[field]:
                    factor_order.append(
                        {
                            "block_index": block_index,
                            "block_id": block_id,
                            "field_index": field_index,
                            "field_id": field_id,
                            "field": field,
                            "likelihood_id": likelihood_id,
                        }
                    )
        if len(factor_order) != EXPECTED_ENABLED_FACTOR_COUNT:
            raise ValueError(
                f"V2 metadata expanded {len(factor_order)} factors, expected "
                f"{EXPECTED_ENABLED_FACTOR_COUNT}"
            )
        return {
            "axis_contract": "batch_block_registered_field_then_likelihood_components",
            "block_order": BLOCK_IDS,
            "field_order": self.contract.core_fields,
            "field_ids": self.contract.registered_core_field_ids,
            "field_index": dict(self.field_index),
            "likelihood_order": tuple(LIKELIHOOD_SPECS),
            "value_components": dict(VALUE_COMPONENTS),
            "dense_abnormal_conditions": {
                field: self.contract.dense_abnormal_conditions.get(field, ())
                for field in self.contract.dense_fields
            },
            "respiratory_category_ids": {
                name: index + 1
                for index, name in enumerate(self.contract.respiratory_modalities)
            },
            "verbal_status_ids": dict(VERBAL_STATUS_IDS),
            "ordinal_max": dict(self.contract.ordinal_max),
            "valid_ranges": valid_ranges,
            "likelihoods_by_field": likelihoods_by_field,
            "factor_order": tuple(factor_order),
            "enabled_factor_count": EXPECTED_ENABLED_FACTOR_COUNT,
            "deterministic_projection_views_per_block": DETERMINISTIC_PROJECTIONS_PER_BLOCK,
            "deterministic_projections_have_direct_loss": False,
        }


def _likelihoods_for_field(
    contract: MultiresEventV2Contract,
    field: str,
) -> tuple[str, ...]:
    if field in contract.dense_fields:
        values = ["categorical_hours_0_4", "dense_joint_value_state"]
        if field in contract.dense_abnormal_conditions:
            values.append("dense_abnormal_duration_vector")
        return tuple(values)
    if field in contract.ordinal_fields:
        return ("categorical_hours_0_4", "gcs_ordinal_triple")
    if field == contract.verbal_field:
        return (
            "categorical_hours_0_4",
            "gcs_verbal_ungradable_hours_given_observed",
            "gcs_verbal_latest_status",
            "gcs_verbal_gradable_ordinal_triple",
        )
    if field in contract.lab_fields:
        return ("hurdle_negative_binomial_count", "lab_joint_value_state")
    if field == contract.respiratory_field:
        return (
            "respiratory_block_evidence",
            "respiratory_edge_evidence_given_block",
            "respiratory_occupancy_vector",
            "respiratory_edge_state",
            "respiratory_onset_vector",
        )
    if field == contract.vasopressor_field:
        return (
            "vasopressor_duration_vector",
            "vasopressor_edge_state_vector",
            "vasopressor_onset_vector",
        )
    if field == contract.ned_field:
        return ("ned_joint_value_state",)
    if field == contract.uop_field:
        return ("hurdle_negative_binomial_count", "uop_sum_given_count")
    raise ValueError(f"field has no enabled V2 likelihoods: {field}")


def _put_scalar(
    primitives: Mapping[str, Any],
    masks: Mapping[str, Any],
    likelihood_id: str,
    batch_index: int,
    block_index: int,
    field_index: int,
    value: Any,
    active: bool,
) -> None:
    primitives[likelihood_id][batch_index, block_index, field_index] = value
    masks[likelihood_id][batch_index, block_index, field_index] = active


def _put_vector(
    primitives: Mapping[str, Any],
    masks: Mapping[str, Any],
    likelihood_id: str,
    batch_index: int,
    block_index: int,
    field_index: int,
    value: Sequence[Any],
    active: bool,
) -> None:
    primitives[likelihood_id][batch_index, block_index, field_index] = primitives[
        likelihood_id
    ].new_tensor(value)
    masks[likelihood_id][batch_index, block_index, field_index] = active


def _pad(rows: Sequence[Sequence[Any]], width: int, value: Any) -> list[list[Any]]:
    return [list(row) + [value] * (width - len(row)) for row in rows]
