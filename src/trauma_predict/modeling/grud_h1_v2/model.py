from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import nn

from trauma_predict.modeling.multires_event_v2.field_state import (
    FutureFieldStateQueries,
    PrimitiveFeedbackEncoder,
    PrimitiveParameterHeads,
)
from trauma_predict.training.multires_event_v2_loss import (
    EXPECTED_ENABLED_CORE_PRIMITIVES,
    REGISTERED_CORE_FIELD_IDS,
    V2_PRIMITIVE_FEEDBACK_DIMS,
    V2_PRIMITIVE_HEAD_DIMS,
)


PrimitiveSampler = Callable[
    [int, int, Mapping[str, torch.Tensor]],
    tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
]


@dataclass(frozen=True)
class GRUDH1JointM4Config:
    """Architecture contract for the H1 GRU-D matched baseline.

    The input and output dimensions are frozen to the audited H1 sidecar and
    r9 target contract.  Capacity parameters remain explicit so they can be
    frozen in the formal training configuration without changing the task.
    """

    input_channels: int = 118
    hidden_size: int = 192
    dropout: float = 0.1

    static_numeric_fields: int = 4
    static_categorical_fields: int = 5
    static_categorical_vocab_size: int = 32

    field_vocab_size: int = 38
    future_block_count: int = 6
    target_field_count: int = 29
    target_field_ids: tuple[int, ...] = tuple(REGISTERED_CORE_FIELD_IDS)

    def __post_init__(self) -> None:
        if self.input_channels != 118:
            raise ValueError("the frozen GRU-D H1 input contract requires 118 channels")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.static_numeric_fields != 4 or self.static_categorical_fields != 5:
            raise ValueError(
                "the matched static contract requires 4 numeric and 5 categorical fields"
            )
        if self.static_categorical_vocab_size < 2:
            raise ValueError("static_categorical_vocab_size must include padding and values")
        if self.field_vocab_size != 38:
            raise ValueError("the frozen field address space contains 37 fields plus padding")
        if self.future_block_count != 6 or self.target_field_count != 29:
            raise ValueError("the r9 target contract requires six blocks and 29 fields")
        if tuple(self.target_field_ids) != tuple(REGISTERED_CORE_FIELD_IDS):
            raise ValueError("target_field_ids must equal the frozen r9 registered order")

    @property
    def primitive_dims(self) -> dict[str, int]:
        return {str(name): int(width) for name, width in V2_PRIMITIVE_HEAD_DIMS.items()}

    @property
    def feedback_dims(self) -> dict[str, int]:
        return {
            str(name): int(width)
            for name, width in V2_PRIMITIVE_FEEDBACK_DIMS.items()
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GRUDH1JointM4Config:
        if value.get("schema_version") == "trauma_predict.grud_h1_joint_m4_model_config.v1":
            return cls._from_frozen_model_yaml(value)
        raw = value.get("architecture", value)
        if not isinstance(raw, Mapping):
            raise TypeError("GRU-D model architecture must be a mapping")
        aliases = {
            "input_size": "input_channels",
            "h1_channels": "input_channels",
        }
        accepted = {item.name for item in fields(cls)}
        normalized: dict[str, Any] = {}
        for raw_key, item in raw.items():
            key = aliases.get(str(raw_key), str(raw_key))
            if key not in accepted:
                raise ValueError(f"unknown GRU-D H1 architecture key: {raw_key!r}")
            if key == "target_field_ids":
                item = tuple(int(field_id) for field_id in item)
            normalized[key] = item
        return cls(**normalized)

    @classmethod
    def _from_frozen_model_yaml(
        cls,
        value: Mapping[str, Any],
    ) -> GRUDH1JointM4Config:
        if (
            value.get("route") != "grud_h1_to_joint_m4_v2"
            or value.get("role") != "matched_classic_baseline"
            or value.get("initialization") != "from_scratch"
        ):
            raise ValueError("GRU-D model YAML identity differs from the matched baseline")
        input_config = _mapping(value.get("input"), "model.input")
        grud = _mapping(value.get("grud"), "model.grud")
        decoder = _mapping(value.get("decoder"), "model.decoder")
        output = _mapping(value.get("output"), "model.output")
        formal = _mapping(value.get("formal_contract"), "model.formal_contract")
        excluded = _mapping(
            value.get("excluded_method_components"),
            "model.excluded_method_components",
        )
        expected_input = {
            "resolution": "H1",
            "channel_count": 118,
            "max_history_hours": 312,
            "missing_tuple": "mask_zero",
            "registered_zero": "mask_one_value_zero",
            "cxr_study_slot_policy": "count_label_occurrences_within_hour",
        }
        if {key: input_config.get(key) for key in expected_input} != expected_input:
            raise ValueError("GRU-D model YAML input contract differs from frozen H1")
        expected_grud = {
            "value_decay": "diagonal",
            "hidden_decay": "full_from_channel_deltas",
            "decay_initialization": "near_identity_positive_bias_1e-3",
            "feature_center": "normalized_zero",
            "concatenate_observation_mask": True,
        }
        if {key: grud.get(key) for key in expected_grud} != expected_grud:
            raise ValueError("GRU-D decay contract differs from the classic baseline")
        expected_decoder = {
            "type": "block_major_gru_cell",
            "future_blocks": 6,
            "target_fields": 29,
            "causal_positions": 174,
            "order": "block_then_frozen_field_order",
            "teacher_feedback": "right_shifted_registered_primitives",
            "rollout_feedback": "right_shifted_sampled_registered_primitives",
        }
        if {key: decoder.get(key) for key in expected_decoder} != expected_decoder:
            raise ValueError("GRU-D decoder contract differs from the frozen causal route")
        expected_output = {
            "primitive_parameter_heads": "registered_v2",
            "stochastic_factors": EXPECTED_ENABLED_CORE_PRIMITIVES,
            "deterministic_projection_loss": False,
            "h1_head": False,
            "f24_head": False,
        }
        if {key: output.get(key) for key in expected_output} != expected_output:
            raise ValueError("GRU-D output contract differs from the matched V2 task")
        expected_formal = {
            "model_parameter_count": 1_596_987,
            "causal_field_positions": 174,
            "stochastic_factors": EXPECTED_ENABLED_CORE_PRIMITIVES,
            "relation_parameters": 0,
        }
        if {key: formal.get(key) for key in expected_formal} != expected_formal:
            raise ValueError("GRU-D formal architecture contract changed")
        required_exclusions = {
            "multi_resolution_input",
            "transformer",
            "relation_bias",
            "relation_table",
            "temporal_fusion",
            "target_attention",
        }
        if set(excluded) != required_exclusions or any(
            excluded[key] is not True for key in required_exclusions
        ):
            raise ValueError("GRU-D model YAML must exclude every primary-method component")
        hidden_size = int(grud.get("hidden_size", -1))
        if int(decoder.get("hidden_size", -2)) != hidden_size:
            raise ValueError("GRU-D encoder and decoder hidden sizes must match")
        return cls(
            input_channels=int(input_config["channel_count"]),
            hidden_size=hidden_size,
            dropout=float(grud.get("dropout", -1.0)),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primitive_head_dims"] = self.primitive_dims
        payload["primitive_feedback_dims"] = self.feedback_dims
        return payload


class _StaticContextEncoder(nn.Module):
    """Encode the matched static fields without importing a history encoder."""

    def __init__(self, config: GRUDH1JointM4Config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.numeric_fields = config.static_numeric_fields
        self.categorical_fields = config.static_categorical_fields
        self.categorical_vocab_size = config.static_categorical_vocab_size
        self.numeric = nn.Sequential(
            nn.Linear(self.numeric_fields * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.categorical = nn.ModuleList(
            nn.Embedding(
                self.categorical_vocab_size,
                hidden_size,
                padding_idx=0,
            )
            for _ in range(self.categorical_fields)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(hidden_size),
        )

    def forward(
        self,
        static_numeric: torch.Tensor,
        static_numeric_mask: torch.Tensor,
        static_categorical: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = static_numeric.shape[0]
        if static_numeric.shape != (batch_size, self.numeric_fields):
            raise ValueError("static_numeric does not match the frozen numeric fields")
        if static_numeric_mask.shape != static_numeric.shape:
            raise ValueError("static_numeric_mask must align with static_numeric")
        if static_categorical.shape != (batch_size, self.categorical_fields):
            raise ValueError("static_categorical does not match the frozen categorical fields")
        categorical_ids = static_categorical.to(dtype=torch.long)
        if categorical_ids.numel() and (
            categorical_ids.min().item() < 0
            or categorical_ids.max().item() >= self.categorical_vocab_size
        ):
            raise ValueError("static categorical id is outside its frozen vocabulary")
        numeric_mask = static_numeric_mask.bool()
        finite_numeric = torch.isfinite(static_numeric)
        if bool((numeric_mask & ~finite_numeric).any().item()):
            raise ValueError("observed static numeric values must be finite")
        numeric_values = torch.where(
            numeric_mask,
            torch.nan_to_num(static_numeric),
            torch.zeros_like(static_numeric),
        )
        numeric = self.numeric(
            torch.cat((numeric_values, numeric_mask.to(dtype=numeric_values.dtype)), dim=-1)
        )
        categorical_parts = [
            embedding(categorical_ids[:, index])
            for index, embedding in enumerate(self.categorical)
        ]
        categorical = torch.stack(categorical_parts, dim=1).sum(dim=1)
        denominator = categorical_ids.ne(0).sum(dim=1, keepdim=True).clamp_min(1)
        categorical = categorical / denominator.to(dtype=categorical.dtype)
        return self.output(torch.cat((numeric, categorical), dim=-1))


class GRUDH1JointM4Model(nn.Module):
    """GRU-D H1 encoder with a plain causal recurrent six-M4 decoder.

    This baseline deliberately contains no relation table, attention layer,
    Transformer, or multi-resolution history route.  Task adaptation is limited
    to the frozen r9 registered queries, parameter heads, and primitive feedback.
    """

    def __init__(
        self,
        config: GRUDH1JointM4Config,
        *,
        input_means: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        means = (
            torch.zeros(config.input_channels, dtype=torch.float32)
            if input_means is None
            else torch.as_tensor(input_means, dtype=torch.float32).detach().clone()
        )
        if means.shape != (config.input_channels,) or not bool(torch.isfinite(means).all()):
            raise ValueError("input_means must contain 118 finite training-set means")
        self.register_buffer("input_means", means, persistent=True)

        self.static_encoder = _StaticContextEncoder(config)
        self.input_decay_weight = nn.Parameter(
            torch.full((config.input_channels,), 1.0e-3)
        )
        self.input_decay_bias = nn.Parameter(
            torch.full((config.input_channels,), 1.0e-3)
        )
        self.hidden_decay = nn.Linear(config.input_channels, config.hidden_size)
        nn.init.constant_(
            self.hidden_decay.weight,
            1.0e-3 / float(config.input_channels),
        )
        nn.init.constant_(self.hidden_decay.bias, 1.0e-3)
        self.history_cell = nn.GRUCell(config.input_channels * 2, config.hidden_size)
        self.history_norm = nn.LayerNorm(config.hidden_size)

        self.target_field_embedding = nn.Embedding(
            config.field_vocab_size,
            config.hidden_size,
            padding_idx=0,
        )
        self.field_queries = FutureFieldStateQueries(
            config.hidden_size,
            config.future_block_count,
            config.target_field_count,
            config.dropout,
        )
        self.register_buffer(
            "target_field_ids",
            torch.tensor(config.target_field_ids, dtype=torch.long),
            persistent=True,
        )

        self.start_feedback = nn.Parameter(torch.zeros(config.hidden_size))
        self.decoder_input = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_size),
        )
        self.decoder_cell = nn.GRUCell(config.hidden_size, config.hidden_size)
        self.decoder_output = nn.LayerNorm(config.hidden_size)
        self.primitive_heads = PrimitiveParameterHeads(
            config.hidden_size,
            config.primitive_dims,
            config.dropout,
        )
        self.feedback_encoder = PrimitiveFeedbackEncoder(
            config.hidden_size,
            config.feedback_dims,
            config.dropout,
        )

    def forward(
        self,
        *,
        h1_values: torch.Tensor,
        h1_observed_mask: torch.Tensor,
        h1_delta_hours: torch.Tensor,
        h1_sequence_mask: torch.Tensor,
        static_numeric: torch.Tensor,
        static_numeric_mask: torch.Tensor,
        static_categorical: torch.Tensor,
        target_primitives: Mapping[str, torch.Tensor] | None = None,
        target_primitive_masks: Mapping[str, torch.Tensor] | None = None,
        sampler: PrimitiveSampler | None = None,
    ) -> dict[str, Any]:
        history_state, query_tokens = self.encode_for_rollout(
            h1_values=h1_values,
            h1_observed_mask=h1_observed_mask,
            h1_delta_hours=h1_delta_hours,
            h1_sequence_mask=h1_sequence_mask,
            static_numeric=static_numeric,
            static_numeric_mask=static_numeric_mask,
            static_categorical=static_categorical,
        )
        if target_primitives is None:
            if target_primitive_masks is not None:
                raise ValueError("target_primitive_masks requires target_primitives")
            if sampler is None:
                raise ValueError("autoregressive forward requires a likelihood sampler")
            return self.rollout_from_encoded(
                history_state,
                query_tokens,
                sampler=sampler,
            )
        if target_primitive_masks is None:
            raise ValueError("teacher forcing requires target_primitive_masks")
        if sampler is not None:
            raise ValueError("teacher forcing and sampled rollout are separate routes")
        teacher_feedback, _ = self.feedback_encoder(
            target_primitives,
            target_primitive_masks,
            leading_shape=(
                h1_values.shape[0],
                self.config.future_block_count,
                self.config.target_field_count,
            ),
        )
        field_states = self._decode_teacher(
            history_state,
            query_tokens,
            teacher_feedback,
        )
        return self._outputs(field_states, history_state)

    def encode_history(
        self,
        *,
        h1_values: torch.Tensor,
        h1_observed_mask: torch.Tensor,
        h1_delta_hours: torch.Tensor,
        h1_sequence_mask: torch.Tensor,
        static_numeric: torch.Tensor,
        static_numeric_mask: torch.Tensor,
        static_categorical: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_steps = self._validate_h1_input(
            h1_values,
            h1_observed_mask,
            h1_delta_hours,
            h1_sequence_mask,
        )
        history = self.static_encoder(
            static_numeric,
            static_numeric_mask,
            static_categorical,
        )
        if history.shape != (batch_size, self.config.hidden_size):
            raise ValueError("static context batch does not align with H1 history")
        dtype = history.dtype
        values = h1_values.to(dtype=dtype)
        observed_mask = h1_observed_mask.bool()
        delta_hours = h1_delta_hours.to(dtype=dtype)
        sequence_mask = h1_sequence_mask.bool()
        input_means = self.input_means.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        last_values = input_means

        for step in range(time_steps):
            valid = sequence_mask[:, step].unsqueeze(-1)
            observed = observed_mask[:, step] & valid
            delta = torch.where(valid, delta_hours[:, step], torch.zeros_like(delta_hours[:, step]))
            input_decay = torch.exp(
                -torch.relu(
                    delta * self.input_decay_weight.to(dtype=dtype)
                    + self.input_decay_bias.to(dtype=dtype)
                )
            )
            hidden_decay = torch.exp(-torch.relu(self.hidden_decay(delta)))
            current = torch.nan_to_num(values[:, step])
            imputed = torch.where(
                observed,
                current,
                input_decay * last_values + (1.0 - input_decay) * input_means,
            )
            decayed_history = hidden_decay * history
            candidate = self.history_cell(
                torch.cat((imputed, observed.to(dtype=dtype)), dim=-1),
                decayed_history,
            )
            history = torch.where(valid, candidate, history)
            updated_last = torch.where(observed, current, last_values)
            last_values = torch.where(valid, updated_last, last_values)
        return self.history_norm(history)

    def encode_for_rollout(
        self,
        *,
        h1_values: torch.Tensor,
        h1_observed_mask: torch.Tensor,
        h1_delta_hours: torch.Tensor,
        h1_sequence_mask: torch.Tensor,
        static_numeric: torch.Tensor,
        static_numeric_mask: torch.Tensor,
        static_categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_state = self.encode_history(
            h1_values=h1_values,
            h1_observed_mask=h1_observed_mask,
            h1_delta_hours=h1_delta_hours,
            h1_sequence_mask=h1_sequence_mask,
            static_numeric=static_numeric,
            static_numeric_mask=static_numeric_mask,
            static_categorical=static_categorical,
        )
        registered_fields = self.target_field_embedding(self.target_field_ids)
        query_tokens = self.field_queries(registered_fields, history_state.shape[0])
        return history_state, query_tokens

    def rollout(
        self,
        *,
        h1_values: torch.Tensor,
        h1_observed_mask: torch.Tensor,
        h1_delta_hours: torch.Tensor,
        h1_sequence_mask: torch.Tensor,
        static_numeric: torch.Tensor,
        static_numeric_mask: torch.Tensor,
        static_categorical: torch.Tensor,
        sampler: PrimitiveSampler,
    ) -> dict[str, Any]:
        """Generate one block-major trajectory without accepting future truth."""

        history_state, query_tokens = self.encode_for_rollout(
            h1_values=h1_values,
            h1_observed_mask=h1_observed_mask,
            h1_delta_hours=h1_delta_hours,
            h1_sequence_mask=h1_sequence_mask,
            static_numeric=static_numeric,
            static_numeric_mask=static_numeric_mask,
            static_categorical=static_categorical,
        )
        return self.rollout_from_encoded(history_state, query_tokens, sampler=sampler)

    def rollout_from_encoded(
        self,
        history_state: torch.Tensor,
        query_tokens: torch.Tensor,
        *,
        sampler: PrimitiveSampler,
    ) -> dict[str, Any]:
        if self.training:
            raise RuntimeError("sampled rollout is inference-only; call model.eval() first")
        batch_size = history_state.shape[0]
        if history_state.shape != (batch_size, self.config.hidden_size):
            raise ValueError("history_state must be [batch, hidden_size]")
        expected_queries = (
            batch_size,
            self.config.future_block_count,
            self.config.target_field_count,
            self.config.hidden_size,
        )
        if query_tokens.shape != expected_queries:
            raise ValueError(
                f"query_tokens shape={tuple(query_tokens.shape)} does not match {expected_queries}"
            )
        flat_queries = query_tokens.reshape(batch_size, -1, self.config.hidden_size)
        decoder_state = history_state
        previous_feedback = self.start_feedback.view(1, -1).expand(batch_size, -1)
        states: list[torch.Tensor] = []
        parameter_rows: dict[str, list[torch.Tensor]] = {
            key: [] for key in self.config.primitive_dims
        }
        primitive_rows: dict[str, list[torch.Tensor]] = {
            key: [] for key in self.config.feedback_dims
        }
        primitive_mask_rows: dict[str, list[torch.Tensor]] = {
            key: [] for key in self.config.feedback_dims
        }
        feedback_valid_rows: list[torch.Tensor] = []
        for position in range(self.config.future_block_count * self.config.target_field_count):
            query = flat_queries[:, position]
            decoder_state, field_state = self._decoder_step(
                decoder_state,
                history_state,
                query,
                previous_feedback,
            )
            parameters = self.primitive_heads(field_state)
            block_index, field_index = divmod(position, self.config.target_field_count)
            sampled, sampled_masks = sampler(block_index, field_index, parameters)
            feedback, feedback_valid = self.feedback_encoder(
                sampled,
                sampled_masks,
                leading_shape=(batch_size,),
            )
            states.append(field_state)
            previous_feedback = feedback
            feedback_valid_rows.append(feedback_valid)
            for key, width in self.config.primitive_dims.items():
                value = parameters[key]
                if value.shape != (batch_size, width):
                    raise ValueError(f"sampler parameter {key!r} has an invalid shape")
                parameter_rows[key].append(value)
            for key, width in self.config.feedback_dims.items():
                value = sampled[key]
                mask = sampled_masks[key]
                if value.shape != (batch_size, width):
                    raise ValueError(f"sampled primitive {key!r} has an invalid shape")
                if mask.shape == (batch_size,):
                    mask = mask.unsqueeze(-1).expand(batch_size, width)
                if mask.shape != (batch_size, width):
                    raise ValueError(f"sampled primitive mask {key!r} has an invalid shape")
                primitive_rows[key].append(value)
                primitive_mask_rows[key].append(mask.bool())
        feedback_contract = torch.stack(feedback_valid_rows, dim=1).all()
        feedback_error = "sampler must return at least one valid primitive at every field"
        if history_state.device.type == "cuda":
            torch._assert_async(feedback_contract, feedback_error)
        elif not bool(feedback_contract.item()):
            raise ValueError(feedback_error)
        field_states = torch.stack(states, dim=1).reshape(*expected_queries)
        primitive_parameters = {
            key: torch.stack(values, dim=1).reshape(
                batch_size,
                self.config.future_block_count,
                self.config.target_field_count,
                self.config.primitive_dims[key],
            )
            for key, values in parameter_rows.items()
        }
        generated_primitives = {
            key: torch.stack(values, dim=1).reshape(
                batch_size,
                self.config.future_block_count,
                self.config.target_field_count,
                self.config.feedback_dims[key],
            )
            for key, values in primitive_rows.items()
        }
        generated_masks = {
            key: torch.stack(values, dim=1).reshape(
                batch_size,
                self.config.future_block_count,
                self.config.target_field_count,
                self.config.feedback_dims[key],
            )
            for key, values in primitive_mask_rows.items()
        }
        return self._outputs(
            field_states,
            history_state,
            primitive_parameters=primitive_parameters,
            generated_primitives=generated_primitives,
            generated_primitive_masks=generated_masks,
        )

    def _decode_teacher(
        self,
        history_state: torch.Tensor,
        query_tokens: torch.Tensor,
        teacher_feedback: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = history_state.shape[0]
        flat_queries = query_tokens.reshape(batch_size, -1, self.config.hidden_size)
        flat_feedback = teacher_feedback.reshape(batch_size, -1, self.config.hidden_size)
        expected_positions = self.config.future_block_count * self.config.target_field_count
        if flat_feedback.shape != (batch_size, expected_positions, self.config.hidden_size):
            raise ValueError("teacher feedback must align with the frozen block-major order")
        decoder_state = history_state
        previous_feedback = self.start_feedback.view(1, -1).expand(batch_size, -1)
        states: list[torch.Tensor] = []
        for position in range(expected_positions):
            decoder_state, field_state = self._decoder_step(
                decoder_state,
                history_state,
                flat_queries[:, position],
                previous_feedback,
            )
            states.append(field_state)
            previous_feedback = flat_feedback[:, position]
        return torch.stack(states, dim=1).reshape(
            batch_size,
            self.config.future_block_count,
            self.config.target_field_count,
            self.config.hidden_size,
        )

    def _decoder_step(
        self,
        decoder_state: torch.Tensor,
        history_state: torch.Tensor,
        query: torch.Tensor,
        previous_feedback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder_input = self.decoder_input(
            torch.cat((query, previous_feedback, history_state), dim=-1)
        )
        next_state = self.decoder_cell(decoder_input, decoder_state)
        return next_state, self.decoder_output(next_state + query)

    def _outputs(
        self,
        field_states: torch.Tensor,
        history_state: torch.Tensor,
        *,
        primitive_parameters: Mapping[str, torch.Tensor] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        parameters = (
            self.primitive_heads(field_states)
            if primitive_parameters is None
            else dict(primitive_parameters)
        )
        return {
            "field_states": field_states,
            "primitive_parameters": parameters,
            "primitive_parameter_dims": self.config.primitive_dims,
            "primitive_feedback_dims": self.config.feedback_dims,
            "history_state": history_state,
            **extra,
        }

    def _validate_h1_input(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        delta_hours: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> tuple[int, int]:
        if values.ndim != 3 or values.shape[-1] != self.config.input_channels:
            raise ValueError("h1_values must be [batch,time,118]")
        if observed_mask.shape != values.shape:
            raise ValueError("h1_observed_mask must align with h1_values")
        if delta_hours.shape != values.shape:
            raise ValueError("h1_delta_hours must align with h1_values")
        batch_size, time_steps, _ = values.shape
        if time_steps < 1 or sequence_mask.shape != (batch_size, time_steps):
            raise ValueError("h1_sequence_mask must be [batch,time] with time >= 1")
        sequence = sequence_mask.bool()
        if not bool(sequence.any(dim=1).all().item()):
            raise ValueError("every H1 sequence must contain at least one visible hour")
        if time_steps > 1 and bool((sequence[:, 1:] & ~sequence[:, :-1]).any().item()):
            raise ValueError("h1_sequence_mask must be a left-aligned valid prefix")
        observed = observed_mask.bool()
        if bool((observed & ~sequence.unsqueeze(-1)).any().item()):
            raise ValueError("padded H1 hours cannot contain observed channels")
        if bool((observed & ~torch.isfinite(values)).any().item()):
            raise ValueError("observed H1 values must be finite")
        valid_deltas = delta_hours[sequence.unsqueeze(-1).expand_as(delta_hours)]
        if not bool(torch.isfinite(valid_deltas).all().item()) or bool(
            valid_deltas.lt(0).any().item()
        ):
            raise ValueError("visible H1 delta hours must be finite and nonnegative")
        return batch_size, time_steps


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _validate_task_contract(contract: Any | None) -> None:
    if contract is None:
        return
    if not isinstance(contract, Mapping):
        core_fields = getattr(contract, "core_fields", None)
        process_registry = getattr(contract, "process_registry", None)
        emission_registry = getattr(contract, "emission_registry", None)
        registered_ids = getattr(contract, "registered_core_field_ids", None)
        if (
            not isinstance(core_fields, tuple)
            or len(core_fields) != 29
            or tuple(registered_ids or ()) != tuple(REGISTERED_CORE_FIELD_IDS)
        ):
            raise ValueError("GRU-D baseline requires the real 29-field r9 target contract")
        process = _mapping(process_registry, "contract.process_registry")
        scope = _mapping(process.get("scope"), "contract.process_registry.scope")
        if (
            len(tuple(scope.get("future_blocks") or ())) != 6
            or int(scope.get("expanded_enabled_core_primitives", -1))
            != EXPECTED_ENABLED_CORE_PRIMITIVES
        ):
            raise ValueError("GRU-D baseline contract must contain six blocks and 414 factors")
        emission = _mapping(emission_registry, "contract.emission_registry")
        head_contract = _mapping(
            emission.get("enabled_core_head_contract"),
            "contract.emission_registry.enabled_core_head_contract",
        )
        layouts = _mapping(
            head_contract.get("layouts"),
            "contract.emission_registry.enabled_core_head_contract.layouts",
        )
        observed_widths = {
            str(key): int(_mapping(row, f"contract layout {key!r}").get("width", -1))
            for key, row in layouts.items()
        }
        if observed_widths != dict(V2_PRIMITIVE_HEAD_DIMS):
            raise ValueError("GRU-D baseline emission heads differ from frozen V2")
        return
    candidates: list[Mapping[str, Any]] = [contract]
    for key in ("target", "primitive_contract", "formal_contract"):
        child = contract.get(key)
        if isinstance(child, Mapping):
            candidates.append(child)
    expected = {
        "future_block_count": 6,
        "ordered_m4_blocks": 6,
        "output_blocks": 6,
        "target_field_count": 29,
        "field_processes": 29,
        "registered_fields": 29,
        "stochastic_factors": EXPECTED_ENABLED_CORE_PRIMITIVES,
        "stochastic_factors_per_anchor": EXPECTED_ENABLED_CORE_PRIMITIVES,
    }
    for candidate in candidates:
        for key, expected_value in expected.items():
            if key in candidate and int(candidate[key]) != expected_value:
                raise ValueError(
                    f"GRU-D baseline contract {key} must equal {expected_value}"
                )


def build_grud_h1_joint_m4_model(
    config: GRUDH1JointM4Config | Mapping[str, Any],
    *,
    input_means: torch.Tensor | None = None,
    contract: Any | None = None,
) -> GRUDH1JointM4Model:
    """Build the matched baseline without accepting method-specific overrides."""

    resolved = (
        config
        if isinstance(config, GRUDH1JointM4Config)
        else GRUDH1JointM4Config.from_mapping(config)
    )
    _validate_task_contract(contract)
    return GRUDH1JointM4Model(resolved, input_means=input_means)


__all__ = [
    "GRUDH1JointM4Config",
    "GRUDH1JointM4Model",
    "PrimitiveSampler",
    "build_grud_h1_joint_m4_model",
]
