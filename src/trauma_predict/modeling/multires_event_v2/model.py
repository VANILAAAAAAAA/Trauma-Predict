from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..multires_event.embeddings import (
    BlockContextEmbedding,
    EmbeddingVocabulary,
    EventEmbedding,
    SemanticEmbeddingTables,
    StaticContextEncoder,
)
from ..multires_event.encoder import BlockLatentCompressor, TrajectoryEncoder
from .config import MultiResolutionEventV2Config, resolve_mode
from .field_state import (
    FutureFieldStateQueries,
    PrimitiveFeedbackEncoder,
    PrimitiveParameterHeads,
)
from .rollout import AutoregressiveFieldStateRollout, PrimitiveSampler
from .trajectory import FieldStateTrajectoryDecoder


class MultiResolutionEventV2Model(nn.Module):
    """V1 input hierarchy with a joint six-block, 29-field M4 target process."""

    def __init__(self, config: MultiResolutionEventV2Config) -> None:
        super().__init__()
        self.config = config
        vocab = EmbeddingVocabulary(
            fields=config.field_vocab_size,
            operators=config.operator_vocab_size,
            conditions=config.condition_vocab_size,
            roles=config.role_vocab_size,
            resolutions=config.resolution_vocab_size,
        )
        self.semantic_embeddings = SemanticEmbeddingTables(vocab, config.hidden_size)
        self.block_embedding = BlockContextEmbedding(
            config.role_vocab_size,
            config.hidden_size,
            time_scale_hours=config.time_scale_hours,
        )
        self.event_embedding = EventEmbedding(
            config.hidden_size,
            config.dropout,
            config.study_slot_vocab_size,
        )
        self.static_encoder = StaticContextEncoder(
            config.hidden_size,
            config.static_numeric_fields,
            config.static_categorical_fields,
            config.static_categorical_vocab_size,
            config.dropout,
        )
        self.block_compressor = BlockLatentCompressor(
            config.hidden_size,
            config.num_attention_heads,
            config.block_latent_count,
            config.block_compressor_layers,
            config.dropout,
        )
        self.trajectory_encoder = TrajectoryEncoder(
            config.hidden_size,
            config.num_attention_heads,
            config.trajectory_encoder_layers,
            config.dropout,
        )
        self.field_queries = FutureFieldStateQueries(
            config.hidden_size,
            config.future_block_count,
            config.target_field_count,
            config.dropout,
        )
        self.target_decoder = FieldStateTrajectoryDecoder(
            config.hidden_size,
            config.num_attention_heads,
            config.target_decoder_layers,
            config.dropout,
            config.future_block_count,
            config.target_field_count,
            config.relation_type_count,
        )
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
        self.autoregressive_rollout = AutoregressiveFieldStateRollout(
            config.future_block_count,
            config.target_field_count,
        )
        self.register_buffer(
            "target_field_ids",
            torch.tensor(config.target_field_ids, dtype=torch.long),
            persistent=True,
        )

    def forward(
        self,
        event_field_ids: torch.Tensor,
        event_operator_ids: torch.Tensor,
        event_condition_ids: torch.Tensor,
        event_values: torch.Tensor,
        event_value_mask: torch.Tensor,
        event_study_slot_ids: torch.Tensor,
        block_index: torch.Tensor,
        event_mask: torch.Tensor,
        block_role_ids: torch.Tensor,
        resolution_ids: torch.Tensor,
        relative_start: torch.Tensor,
        relative_end: torch.Tensor,
        span: torch.Tensor,
        block_mask: torch.Tensor,
        static_numeric: torch.Tensor,
        static_numeric_mask: torch.Tensor,
        static_categorical: torch.Tensor,
        target_primitives: Mapping[str, torch.Tensor] | None = None,
        target_primitive_masks: Mapping[str, torch.Tensor] | None = None,
        sampler: PrimitiveSampler | None = None,
        relation_adjacency: torch.Tensor | None = None,
        relation_type_lags: torch.Tensor | None = None,
        mode: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        resolved_mode = resolve_mode(mode, self.config.mode)
        memory, memory_mask = self._encode_input(
            event_field_ids=event_field_ids,
            event_operator_ids=event_operator_ids,
            event_condition_ids=event_condition_ids,
            event_values=event_values,
            event_value_mask=event_value_mask,
            event_study_slot_ids=event_study_slot_ids,
            block_index=block_index,
            event_mask=event_mask,
            block_role_ids=block_role_ids,
            resolution_ids=resolution_ids,
            relative_start=relative_start,
            relative_end=relative_end,
            span=span,
            block_mask=block_mask,
            static_numeric=static_numeric,
            static_numeric_mask=static_numeric_mask,
            static_categorical=static_categorical,
        )
        query_tokens = self._field_queries(event_field_ids.shape[0])
        generated: dict[str, Any] = {}
        if target_primitives is None:
            if target_primitive_masks is not None:
                raise ValueError("target_primitive_masks requires target_primitives")
            if sampler is None:
                raise ValueError("autoregressive forward requires a likelihood sampler")
            field_states, generated_primitives, generated_masks = self.autoregressive_rollout(
                query_tokens,
                memory,
                memory_mask,
                decoder=self.target_decoder,
                primitive_heads=self.primitive_heads,
                feedback_encoder=self.feedback_encoder,
                mode=resolved_mode,
                sampler=sampler,
                relation_adjacency=relation_adjacency,
                relation_type_lags=relation_type_lags,
            )
            generated = {
                "generated_primitives": generated_primitives,
                "generated_primitive_masks": generated_masks,
            }
        else:
            if target_primitive_masks is None:
                raise ValueError("teacher forcing requires target_primitive_masks")
            teacher_feedback, teacher_feedback_mask = self.encode_teacher_targets(
                target_primitives,
                target_primitive_masks,
                batch_size=event_field_ids.shape[0],
            )
            decoded = self.target_decoder(
                query_tokens,
                memory,
                memory_mask,
                mode=resolved_mode,
                context_states=teacher_feedback,
                context_mask=teacher_feedback_mask,
                relation_adjacency=relation_adjacency,
                relation_type_lags=relation_type_lags,
            )
            field_states = decoded.reshape(
                event_field_ids.shape[0],
                self.config.future_block_count,
                self.config.target_field_count,
                self.config.hidden_size,
            )
        return self._outputs(field_states, memory_mask, **generated)

    def rollout(
        self,
        event_field_ids: torch.Tensor,
        event_operator_ids: torch.Tensor,
        event_condition_ids: torch.Tensor,
        event_values: torch.Tensor,
        event_value_mask: torch.Tensor,
        event_study_slot_ids: torch.Tensor,
        block_index: torch.Tensor,
        event_mask: torch.Tensor,
        block_role_ids: torch.Tensor,
        resolution_ids: torch.Tensor,
        relative_start: torch.Tensor,
        relative_end: torch.Tensor,
        span: torch.Tensor,
        block_mask: torch.Tensor,
        static_numeric: torch.Tensor,
        static_numeric_mask: torch.Tensor,
        static_categorical: torch.Tensor,
        sampler: PrimitiveSampler,
        relation_adjacency: torch.Tensor | None = None,
        relation_type_lags: torch.Tensor | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Autoregressive inference API; future truth is intentionally not accepted."""

        memory, memory_mask, query_tokens = self.encode_for_rollout(
            event_field_ids=event_field_ids,
            event_operator_ids=event_operator_ids,
            event_condition_ids=event_condition_ids,
            event_values=event_values,
            event_value_mask=event_value_mask,
            event_study_slot_ids=event_study_slot_ids,
            block_index=block_index,
            event_mask=event_mask,
            block_role_ids=block_role_ids,
            resolution_ids=resolution_ids,
            relative_start=relative_start,
            relative_end=relative_end,
            span=span,
            block_mask=block_mask,
            static_numeric=static_numeric,
            static_numeric_mask=static_numeric_mask,
            static_categorical=static_categorical,
        )
        return self.rollout_from_encoded(
            memory,
            memory_mask,
            query_tokens,
            sampler=sampler,
            relation_adjacency=relation_adjacency,
            relation_type_lags=relation_type_lags,
            mode=mode,
        )

    def encode_for_rollout(
        self,
        event_field_ids: torch.Tensor,
        event_operator_ids: torch.Tensor,
        event_condition_ids: torch.Tensor,
        event_values: torch.Tensor,
        event_value_mask: torch.Tensor,
        event_study_slot_ids: torch.Tensor,
        block_index: torch.Tensor,
        event_mask: torch.Tensor,
        block_role_ids: torch.Tensor,
        resolution_ids: torch.Tensor,
        relative_start: torch.Tensor,
        relative_end: torch.Tensor,
        span: torch.Tensor,
        block_mask: torch.Tensor,
        static_numeric: torch.Tensor,
        static_numeric_mask: torch.Tensor,
        static_categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode an observed history once for repeated ancestral trajectories.

        The returned tensors contain input-derived state and registered query
        identities only.  No future target or target mask is accepted by this
        interface, so evaluators can safely replicate the encoded tensors over
        a Monte Carlo ensemble without re-running the input encoder.
        """

        memory, memory_mask = self._encode_input(
            event_field_ids=event_field_ids,
            event_operator_ids=event_operator_ids,
            event_condition_ids=event_condition_ids,
            event_values=event_values,
            event_value_mask=event_value_mask,
            event_study_slot_ids=event_study_slot_ids,
            block_index=block_index,
            event_mask=event_mask,
            block_role_ids=block_role_ids,
            resolution_ids=resolution_ids,
            relative_start=relative_start,
            relative_end=relative_end,
            span=span,
            block_mask=block_mask,
            static_numeric=static_numeric,
            static_numeric_mask=static_numeric_mask,
            static_categorical=static_categorical,
        )
        return memory, memory_mask, self._field_queries(event_field_ids.shape[0])

    def rollout_from_encoded(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        query_tokens: torch.Tensor,
        *,
        sampler: PrimitiveSampler,
        relation_adjacency: torch.Tensor | None = None,
        relation_type_lags: torch.Tensor | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Draw one trajectory per encoded row without re-encoding its history."""

        resolved_mode = resolve_mode(mode, self.config.mode)
        expected_queries = (
            memory.shape[0],
            self.config.future_block_count,
            self.config.target_field_count,
            self.config.hidden_size,
        )
        if query_tokens.shape != expected_queries:
            raise ValueError(
                f"query_tokens shape={tuple(query_tokens.shape)} does not match "
                f"{expected_queries}"
            )
        if memory.ndim != 3 or memory.shape[-1] != self.config.hidden_size:
            raise ValueError("memory must be [batch, memory_tokens, hidden_size]")
        if memory_mask.shape != memory.shape[:2]:
            raise ValueError("memory_mask must align with encoded memory")
        field_states, generated_primitives, generated_masks = self.autoregressive_rollout(
            query_tokens,
            memory,
            memory_mask,
            decoder=self.target_decoder,
            primitive_heads=self.primitive_heads,
            feedback_encoder=self.feedback_encoder,
            mode=resolved_mode,
            sampler=sampler,
            relation_adjacency=relation_adjacency,
            relation_type_lags=relation_type_lags,
        )
        return self._outputs(
            field_states,
            memory_mask,
            generated_primitives=generated_primitives,
            generated_primitive_masks=generated_masks,
        )

    def encode_teacher_targets(
        self,
        target_primitives: Mapping[str, torch.Tensor],
        target_primitive_masks: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode actual target primitives; opaque external target latents are forbidden."""

        return self.feedback_encoder(
            target_primitives,
            target_primitive_masks,
            leading_shape=(
                batch_size,
                self.config.future_block_count,
                self.config.target_field_count,
            ),
        )

    def _field_queries(self, batch_size: int) -> torch.Tensor:
        registered_fields = self.semantic_embeddings.field(self.target_field_ids)
        return self.field_queries(registered_fields, batch_size)

    def _outputs(
        self,
        field_states: torch.Tensor,
        memory_mask: torch.Tensor,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "field_states": field_states,
            "primitive_parameters": self.primitive_heads(field_states),
            "primitive_parameter_dims": self.config.primitive_dims,
            "primitive_feedback_dims": self.config.feedback_dims,
            "memory_mask": memory_mask,
            **extra,
        }

    def _encode_input(
        self,
        *,
        event_field_ids: torch.Tensor,
        event_operator_ids: torch.Tensor,
        event_condition_ids: torch.Tensor,
        event_values: torch.Tensor,
        event_value_mask: torch.Tensor,
        event_study_slot_ids: torch.Tensor,
        block_index: torch.Tensor,
        event_mask: torch.Tensor,
        block_role_ids: torch.Tensor,
        resolution_ids: torch.Tensor,
        relative_start: torch.Tensor,
        relative_end: torch.Tensor,
        span: torch.Tensor,
        block_mask: torch.Tensor,
        static_numeric: torch.Tensor,
        static_numeric_mask: torch.Tensor,
        static_categorical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, event_count = event_field_ids.shape
        for name, value in (
            ("event_operator_ids", event_operator_ids),
            ("event_condition_ids", event_condition_ids),
            ("event_values", event_values),
            ("event_value_mask", event_value_mask),
            ("event_study_slot_ids", event_study_slot_ids),
            ("block_index", block_index),
            ("event_mask", event_mask),
        ):
            if value.shape != (batch_size, event_count):
                raise ValueError(f"{name} must align with event_field_ids")
        block_count = block_role_ids.shape[1]
        expected_block_shape = (batch_size, block_count)
        for name, value in (
            ("resolution_ids", resolution_ids),
            ("relative_start", relative_start),
            ("relative_end", relative_end),
            ("span", span),
            ("block_mask", block_mask),
        ):
            if value.shape != expected_block_shape:
                raise ValueError(
                    f"{name} shape={tuple(value.shape)} does not match {expected_block_shape}"
                )

        block_resolution = self.semantic_embeddings.resolution(resolution_ids)
        block_context = self.block_embedding(
            block_role_ids,
            block_resolution,
            relative_start,
            relative_end,
            span,
        )
        safe_block_index = block_index.clamp(min=0, max=max(block_count - 1, 0))
        gathered_block_context = torch.gather(
            block_context,
            dim=1,
            index=safe_block_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size),
        )
        valid_events = event_mask.bool() & block_index.ge(0) & block_index.lt(block_count)
        event_semantics = self.semantic_embeddings(
            event_field_ids,
            event_operator_ids,
            event_condition_ids,
        )
        embedded_events = self.event_embedding(
            event_semantics,
            event_values,
            valid_events & event_value_mask.bool() & event_study_slot_ids.eq(0),
            event_study_slot_ids,
            gathered_block_context,
        ) * valid_events.unsqueeze(-1)
        static_token = self.static_encoder(
            static_numeric,
            static_numeric_mask,
            static_categorical,
        )
        block_latents, encoded_block_mask = self.block_compressor(
            embedded_events,
            block_index,
            valid_events,
            block_context,
            block_mask,
        )
        return self.trajectory_encoder(static_token, block_latents, encoded_block_mask)

    def config_dict(self) -> dict[str, Any]:
        return self.config.as_dict()


def build_multires_v2_model(
    config: MultiResolutionEventV2Config | dict[str, Any],
) -> MultiResolutionEventV2Model:
    resolved = (
        config
        if isinstance(config, MultiResolutionEventV2Config)
        else MultiResolutionEventV2Config.from_mapping(config)
    )
    return MultiResolutionEventV2Model(resolved)
