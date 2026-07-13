from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch
from torch import nn

from .decoder import BlockLocalFutureQueryDecoder, FutureQueryEmbedding
from .embeddings import (
    BlockContextEmbedding,
    EmbeddingVocabulary,
    EventEmbedding,
    SemanticEmbeddingTables,
    StaticContextEncoder,
)
from .encoder import BlockLatentCompressor, TrajectoryEncoder
from .heads import TypedPredictionHeads


@dataclass(frozen=True)
class MultiResolutionEventConfig:
    hidden_size: int = 256
    num_attention_heads: int = 8
    trajectory_encoder_layers: int = 4
    query_decoder_layers: int = 3
    block_compressor_layers: int = 1
    block_latent_count: int = 8
    dropout: float = 0.1
    field_vocab_size: int = 38
    operator_vocab_size: int = 11
    condition_vocab_size: int = 64
    role_vocab_size: int = 8
    resolution_vocab_size: int = 4
    static_numeric_fields: int = 4
    static_categorical_fields: int = 5
    static_categorical_vocab_size: int = 32
    study_slot_vocab_size: int = 16
    max_query_time_index: int = 8
    expected_query_count: int = 986
    expected_query_blocks: int = 7
    time_scale_hours: float = 24.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MultiResolutionEventConfig":
        aliases = {
            "d_model": "hidden_size",
            "num_heads": "num_attention_heads",
            "encoder_layers": "trajectory_encoder_layers",
            "decoder_layers": "query_decoder_layers",
            "query_layers": "query_decoder_layers",
            "block_latents": "block_latent_count",
            "latent_count": "block_latent_count",
        }
        accepted = {item.name for item in fields(cls)}
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            name = aliases.get(str(key), str(key))
            if name in accepted:
                normalized[name] = item
        config = cls(**normalized)
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if config.block_latent_count < 1:
            raise ValueError("block_latent_count must be positive")
        return config


class MultiResolutionEventModel(nn.Module):
    """Scratch hierarchical event Transformer with fixed block-local future queries."""

    def __init__(self, config: MultiResolutionEventConfig) -> None:
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
        self.query_embedding = FutureQueryEmbedding(
            config.hidden_size,
            config.max_query_time_index,
            config.dropout,
        )
        self.query_decoder = BlockLocalFutureQueryDecoder(
            config.hidden_size,
            config.num_attention_heads,
            config.query_decoder_layers,
            config.dropout,
            expected_block_count=config.expected_query_blocks,
        )
        self.prediction_heads = TypedPredictionHeads(config.hidden_size, config.dropout)

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
        query_field_ids: torch.Tensor,
        query_operator_ids: torch.Tensor,
        query_condition_ids: torch.Tensor,
        query_resolution_ids: torch.Tensor,
        query_time_index: torch.Tensor,
        query_span: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        batch_size, event_count = event_field_ids.shape
        if event_operator_ids.shape != (batch_size, event_count):
            raise ValueError("event operator tensor shape mismatch")
        if event_condition_ids.shape != (batch_size, event_count):
            raise ValueError("event condition tensor shape mismatch")
        if event_study_slot_ids.shape != (batch_size, event_count):
            raise ValueError("event study-slot tensor shape mismatch")
        if event_value_mask.shape != (batch_size, event_count):
            raise ValueError("event value-mask tensor shape mismatch")
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
                raise ValueError(f"{name} shape={tuple(value.shape)} does not match {expected_block_shape}")

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
        memory, memory_mask = self.trajectory_encoder(
            static_token,
            block_latents,
            encoded_block_mask,
        )

        query_count = query_field_ids.shape[1]
        if self.config.expected_query_count and query_count != self.config.expected_query_count:
            raise ValueError(
                f"query_count={query_count} does not match the frozen "
                f"expected_query_count={self.config.expected_query_count}"
            )
        if query_mask is None:
            query_mask = query_field_ids.ne(0)
        query_semantics = self.semantic_embeddings(
            query_field_ids,
            query_operator_ids,
            query_condition_ids,
            query_resolution_ids,
        )
        embedded_queries = self.query_embedding(
            query_semantics,
            query_time_index,
            query_span,
        )
        query_hidden = self.query_decoder(
            embedded_queries,
            query_resolution_ids,
            query_time_index,
            query_mask,
            memory,
            memory_mask,
        )
        return {
            **self.prediction_heads(query_hidden),
            "query_hidden": query_hidden,
            "query_mask": query_mask.bool(),
            "memory_mask": memory_mask,
        }

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)


def _infer_vocab_size(
    model_config: dict[str, Any],
    target_contract: Any,
    config_key: str,
    contract_key: str,
) -> None:
    if config_key in model_config:
        return
    vocab = getattr(target_contract, "vocab_sizes", None)
    if vocab is None and isinstance(target_contract, Mapping):
        vocab = target_contract.get("vocab_sizes")
    if isinstance(vocab, Mapping) and contract_key in vocab:
        model_config[config_key] = int(vocab[contract_key])


def build_multires_model(
    model_config: Mapping[str, Any],
    target_contract: Any,
) -> MultiResolutionEventModel:
    """Build the model while pinning dimensions to the target contract when available."""

    resolved: dict[str, Any] = {}
    embedding = model_config.get("embedding", {})
    pooling = model_config.get("block_pooling", {})
    encoder = model_config.get("encoder", {})
    decoder = model_config.get("decoder", {})
    if not all(isinstance(item, Mapping) for item in (embedding, pooling, encoder, decoder)):
        raise ValueError("model embedding/pooling/encoder/decoder sections must be mappings")
    resolved.update(model_config)
    nested_values = {
        "hidden_size": embedding.get("d_model"),
        "dropout": encoder.get("dropout", embedding.get("dropout")),
        "block_latent_count": pooling.get("latent_tokens_per_block"),
        "block_compressor_layers": pooling.get("cross_attention_layers"),
        "trajectory_encoder_layers": encoder.get("trajectory_layers"),
        "num_attention_heads": encoder.get("attention_heads"),
        "query_decoder_layers": decoder.get("query_layers"),
        "expected_query_count": decoder.get("primary_direct_queries"),
        "study_slot_vocab_size": (
            None
            if embedding.get("max_study_slots_per_block") is None
            else int(embedding["max_study_slots_per_block"]) + 1
        ),
    }
    if decoder.get("m4_blocks") is not None:
        nested_values["expected_query_blocks"] = 1 + int(decoder["m4_blocks"])
    for key, value in nested_values.items():
        if value is not None:
            resolved[key] = value
    decoder_heads = decoder.get("attention_heads")
    if (
        decoder_heads is not None
        and resolved.get("num_attention_heads") is not None
        and int(decoder_heads) != int(resolved["num_attention_heads"])
    ):
        raise ValueError("encoder and decoder attention head counts must match")
    for config_key, contract_key in (
        ("field_vocab_size", "field"),
        ("operator_vocab_size", "operator"),
        ("condition_vocab_size", "condition"),
        ("role_vocab_size", "role"),
        ("resolution_vocab_size", "resolution"),
        ("static_categorical_vocab_size", "static_categorical"),
        ("study_slot_vocab_size", "study_slot"),
    ):
        _infer_vocab_size(resolved, target_contract, config_key, contract_key)
    queries = getattr(target_contract, "queries", None)
    if queries is None and isinstance(target_contract, Mapping):
        queries = target_contract.get("queries")
    if "expected_query_count" not in resolved and isinstance(queries, list):
        resolved["expected_query_count"] = len(queries)
    numeric_fields = getattr(target_contract, "static_numeric_fields", None)
    categorical_fields = getattr(target_contract, "static_categorical_fields", None)
    if isinstance(target_contract, Mapping):
        static_contract = target_contract.get("static", {})
        if isinstance(static_contract, Mapping):
            numeric_fields = numeric_fields or static_contract.get("numeric_fields")
            categorical_fields = categorical_fields or static_contract.get("categorical_fields")
    if numeric_fields is not None:
        resolved["static_numeric_fields"] = len(numeric_fields)
    if categorical_fields is not None:
        resolved["static_categorical_fields"] = len(categorical_fields)
    return MultiResolutionEventModel(MultiResolutionEventConfig.from_mapping(resolved))
