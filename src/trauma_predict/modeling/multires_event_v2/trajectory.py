from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .field_state import FieldStateContextEmbedding
from .relation_bias import RegisteredRelationBias


TARGET_SCOPE_SAME_BLOCK = 0
TARGET_SCOPE_ADJACENT_BLOCKS = 1
INPUT_SCOPE_LATEST_TO_FIRST = 0
INPUT_SCOPE_ALL_TO_EACH = 1


@dataclass
class FieldStateTrajectoryLayerCache:
    """Projected inference state for one target decoder layer."""

    target_keys: list[torch.Tensor]
    target_values: list[torch.Tensor]
    memory_keys: torch.Tensor
    memory_values: torch.Tensor


@dataclass
class FieldStateTrajectoryCache:
    """Inference-only cache for exact block-major incremental decoding."""

    flat_queries: torch.Tensor
    memory_attention_bias: torch.Tensor
    layers: list[FieldStateTrajectoryLayerCache]
    target_relation_adjacency: torch.Tensor
    target_time_scope_ids: torch.Tensor
    position: int = 0
    awaiting_feedback: bool = False


def _project_query(
    attention: nn.MultiheadAttention,
    value: torch.Tensor,
) -> torch.Tensor:
    hidden_size = attention.embed_dim
    weight = attention.in_proj_weight[:hidden_size]
    bias = None if attention.in_proj_bias is None else attention.in_proj_bias[:hidden_size]
    return F.linear(value, weight, bias)


def _project_key_value(
    attention: nn.MultiheadAttention,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = attention.embed_dim
    key_weight = attention.in_proj_weight[hidden_size : hidden_size * 2]
    value_weight = attention.in_proj_weight[hidden_size * 2 :]
    key_bias = (
        None
        if attention.in_proj_bias is None
        else attention.in_proj_bias[hidden_size : hidden_size * 2]
    )
    value_bias = (
        None
        if attention.in_proj_bias is None
        else attention.in_proj_bias[hidden_size * 2 :]
    )
    key = F.linear(value, key_weight, key_bias)
    projected_value = F.linear(value, value_weight, value_bias)
    return key, projected_value


def _split_heads(
    value: torch.Tensor,
    attention: nn.MultiheadAttention,
) -> torch.Tensor:
    batch_size, token_count, hidden_size = value.shape
    head_size = hidden_size // attention.num_heads
    return value.view(batch_size, token_count, attention.num_heads, head_size).transpose(1, 2)


def _projected_attention(
    attention: nn.MultiheadAttention,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_bias: torch.Tensor | None,
) -> torch.Tensor:
    projected_query = _split_heads(_project_query(attention, query), attention)
    attended = F.scaled_dot_product_attention(
        projected_query,
        keys,
        values,
        attn_mask=attention_bias,
        dropout_p=0.0,
        is_causal=False,
    )
    batch_size, _, query_count, _ = attended.shape
    merged = attended.transpose(1, 2).contiguous().view(
        batch_size,
        query_count,
        attention.embed_dim,
    )
    return attention.out_proj(merged)


def build_joint_target_access_mask(
    block_count: int,
    field_count: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the frozen joint causal contract in block-major field order."""

    positions = torch.arange(block_count * field_count, device=device)
    query_block = positions.div(field_count, rounding_mode="floor").view(-1, 1)
    query_field = positions.remainder(field_count).view(-1, 1)
    key_block = query_block.transpose(0, 1)
    key_field = query_field.transpose(0, 1)
    return key_block.lt(query_block) | (
        key_block.eq(query_block) & key_field.lt(query_field)
    )


class FieldStateTrajectoryLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.target_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.memory_query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.memory_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        target_context: torch.Tensor,
        target_attention_bias: torch.Tensor,
        memory: torch.Tensor,
        memory_attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.target_attention(
            self.query_norm(hidden),
            self.context_norm(target_context),
            self.context_norm(target_context),
            attn_mask=target_attention_bias,
            need_weights=False,
        )
        hidden = hidden + attended
        attended_memory, _ = self.memory_attention(
            self.memory_query_norm(hidden),
            self.memory_norm(memory),
            self.memory_norm(memory),
            attn_mask=memory_attention_bias,
            need_weights=False,
        )
        hidden = hidden + attended_memory
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class FieldStateTrajectoryDecoder(nn.Module):
    """Decode one joint trajectory with both registered relation paths active."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        layers: int,
        dropout: float,
        block_count: int,
        field_count: int,
        input_field_count: int,
        target_parameter_keys: Sequence[str],
        input_parameter_keys: Sequence[str],
        target_input_field_ids: Sequence[int],
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.block_count = int(block_count)
        self.field_count = int(field_count)
        self.input_field_count = int(input_field_count)
        self.target_count = self.block_count * self.field_count
        self.context_embedding = FieldStateContextEmbedding(hidden_size, dropout)
        self.target_bos = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.target_bos, std=0.02)
        self.target_relation_bias = RegisteredRelationBias(
            target_parameter_keys,
            num_heads,
        )
        self.input_target_relation_bias = RegisteredRelationBias(
            input_parameter_keys,
            num_heads,
        )
        target_input_field_ids = tuple(int(field_id) for field_id in target_input_field_ids)
        if len(target_input_field_ids) != self.field_count:
            raise ValueError("target_input_field_ids must align with target fields")
        if min(target_input_field_ids, default=0) < 1 or max(
            target_input_field_ids, default=0
        ) > self.input_field_count:
            raise ValueError("target_input_field_ids is outside input field memory")
        required_input_mask = torch.zeros(self.input_field_count, dtype=torch.bool)
        required_input_mask[
            torch.tensor(target_input_field_ids, dtype=torch.long) - 1
        ] = True
        self.register_buffer(
            "required_input_field_mask",
            required_input_mask,
            persistent=True,
        )
        self.layers = nn.ModuleList(
            FieldStateTrajectoryLayer(hidden_size, num_heads, dropout) for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        *,
        target_relation_adjacency: torch.Tensor,
        target_time_scope_ids: torch.Tensor,
        input_target_relation_adjacency: torch.Tensor,
        input_target_time_scope_ids: torch.Tensor,
        context_states: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query_tokens.shape[0]
        expected = (batch_size, self.block_count, self.field_count, self.hidden_size)
        if query_tokens.shape != expected:
            raise ValueError(
                f"query_tokens shape={tuple(query_tokens.shape)} does not match {expected}"
            )
        self._validate_memory(memory, memory_mask, batch_size)

        flat_queries = query_tokens.reshape(batch_size, self.target_count, self.hidden_size)
        if query_positions is None:
            positions = torch.arange(self.target_count, device=query_tokens.device)
        else:
            positions = query_positions.to(device=query_tokens.device, dtype=torch.long).flatten()
            if positions.numel() == 0:
                raise ValueError("query_positions cannot be empty")
            if positions.min().item() < 0 or positions.max().item() >= self.target_count:
                raise ValueError("query_positions is outside the target sequence")
        hidden = flat_queries.index_select(1, positions)

        if context_states is None:
            flat_context_states = flat_queries.new_zeros(flat_queries.shape)
            available = torch.zeros(
                (batch_size, self.target_count),
                dtype=torch.bool,
                device=query_tokens.device,
            )
        else:
            if context_states.shape != expected:
                raise ValueError(
                    f"context_states shape={tuple(context_states.shape)} does not match {expected}"
                )
            flat_context_states = context_states.reshape(
                batch_size,
                self.target_count,
                self.hidden_size,
            )
            if context_mask is None:
                available = torch.ones(
                    (batch_size, self.target_count),
                    dtype=torch.bool,
                    device=query_tokens.device,
                )
            else:
                if context_mask.shape != expected[:3]:
                    raise ValueError("context_mask must be [batch, blocks, fields]")
                available = context_mask.reshape(batch_size, self.target_count).bool()

        embedded_context = self.context_embedding(flat_context_states) + flat_queries
        embedded_context = embedded_context * available.unsqueeze(-1)
        bos = self.target_bos.view(1, 1, -1).expand(batch_size, -1, -1)
        target_context = torch.cat((bos, embedded_context), dim=1)
        target_attention_bias = self._target_attention_bias(
            batch_size=batch_size,
            available=available,
            positions=positions,
            target_relation_adjacency=target_relation_adjacency,
            target_time_scope_ids=target_time_scope_ids,
            dtype=query_tokens.dtype,
        )
        memory_attention_bias = self._memory_attention_bias(
            batch_size=batch_size,
            memory_mask=memory_mask,
            positions=positions,
            input_target_relation_adjacency=input_target_relation_adjacency,
            input_target_time_scope_ids=input_target_time_scope_ids,
            dtype=query_tokens.dtype,
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                target_context,
                target_attention_bias,
                memory,
                memory_attention_bias,
            )
        return self.output_norm(hidden)

    def initialize_incremental_cache(
        self,
        query_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        *,
        target_relation_adjacency: torch.Tensor,
        target_time_scope_ids: torch.Tensor,
        input_target_relation_adjacency: torch.Tensor,
        input_target_time_scope_ids: torch.Tensor,
    ) -> FieldStateTrajectoryCache:
        """Project immutable target BOS and memory K/V once for free rollout."""

        if self.training:
            raise RuntimeError(
                "incremental target cache is inference-only; call decoder.eval() first"
            )
        batch_size = query_tokens.shape[0]
        expected = (batch_size, self.block_count, self.field_count, self.hidden_size)
        if query_tokens.shape != expected:
            raise ValueError(
                f"query_tokens shape={tuple(query_tokens.shape)} does not match {expected}"
            )
        self._validate_memory(memory, memory_mask, batch_size)

        flat_queries = query_tokens.reshape(batch_size, self.target_count, self.hidden_size)
        bos = self.target_bos.view(1, 1, -1).expand(batch_size, -1, -1)
        layer_caches: list[FieldStateTrajectoryLayerCache] = []
        for layer in self.layers:
            normalized_bos = layer.context_norm(bos)
            bos_key, bos_value = _project_key_value(layer.target_attention, normalized_bos)
            normalized_memory = layer.memory_norm(memory)
            memory_key, memory_value = _project_key_value(
                layer.memory_attention,
                normalized_memory,
            )
            layer_caches.append(
                FieldStateTrajectoryLayerCache(
                    target_keys=[_split_heads(bos_key, layer.target_attention)],
                    target_values=[_split_heads(bos_value, layer.target_attention)],
                    memory_keys=_split_heads(memory_key, layer.memory_attention),
                    memory_values=_split_heads(memory_value, layer.memory_attention),
                )
            )

        all_positions = torch.arange(self.target_count, device=query_tokens.device)
        memory_attention_bias = self._memory_attention_bias(
            batch_size=batch_size,
            memory_mask=memory_mask,
            positions=all_positions,
            input_target_relation_adjacency=input_target_relation_adjacency,
            input_target_time_scope_ids=input_target_time_scope_ids,
            dtype=query_tokens.dtype,
        ).reshape(
            batch_size,
            self.num_heads,
            self.target_count,
            memory.shape[1],
        )
        return FieldStateTrajectoryCache(
            flat_queries=flat_queries,
            memory_attention_bias=memory_attention_bias,
            layers=layer_caches,
            target_relation_adjacency=target_relation_adjacency,
            target_time_scope_ids=target_time_scope_ids,
        )

    def incremental_step(self, cache: FieldStateTrajectoryCache) -> torch.Tensor:
        """Decode the next registered field from cached projected context."""

        if cache.awaiting_feedback:
            raise RuntimeError("append generated feedback before decoding the next field")
        if cache.position >= self.target_count:
            raise RuntimeError("incremental target sequence is already complete")
        position = cache.position
        hidden = cache.flat_queries[:, position : position + 1]
        key_positions = torch.arange(position, dtype=torch.long, device=hidden.device)
        target_relation_bias = self._incremental_target_relation_bias(
            cache,
            position=position,
            key_positions=key_positions,
            dtype=hidden.dtype,
        )
        memory_attention_bias = cache.memory_attention_bias[:, :, position : position + 1]

        for layer, layer_cache in zip(self.layers, cache.layers, strict=True):
            target_keys = torch.cat(layer_cache.target_keys, dim=2)
            target_values = torch.cat(layer_cache.target_values, dim=2)
            attended = _projected_attention(
                layer.target_attention,
                layer.query_norm(hidden),
                target_keys,
                target_values,
                target_relation_bias,
            )
            hidden = hidden + attended
            attended_memory = _projected_attention(
                layer.memory_attention,
                layer.memory_query_norm(hidden),
                layer_cache.memory_keys,
                layer_cache.memory_values,
                memory_attention_bias,
            )
            hidden = hidden + attended_memory
            hidden = hidden + layer.feed_forward(layer.feed_forward_norm(hidden))

        cache.awaiting_feedback = True
        return self.output_norm(hidden)[:, 0]

    def append_incremental_context(
        self,
        cache: FieldStateTrajectoryCache,
        feedback: torch.Tensor,
    ) -> None:
        """Append the sampled current-field feedback to every layer's K/V cache."""

        if not cache.awaiting_feedback:
            raise RuntimeError("decode a field before appending its generated feedback")
        expected = (cache.flat_queries.shape[0], self.hidden_size)
        if feedback.shape != expected:
            raise ValueError(
                f"feedback shape={tuple(feedback.shape)} does not match {expected}"
            )
        position = cache.position
        context = (
            self.context_embedding(feedback.unsqueeze(1))
            + cache.flat_queries[:, position : position + 1]
        )
        for layer, layer_cache in zip(self.layers, cache.layers, strict=True):
            normalized_context = layer.context_norm(context)
            key, value = _project_key_value(layer.target_attention, normalized_context)
            layer_cache.target_keys.append(_split_heads(key, layer.target_attention))
            layer_cache.target_values.append(_split_heads(value, layer.target_attention))
        cache.position += 1
        cache.awaiting_feedback = False

    def _validate_memory(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        batch_size: int,
    ) -> None:
        if memory.ndim != 3 or memory.shape[0] != batch_size:
            raise ValueError("memory must be [batch, memory_tokens, hidden]")
        if memory.shape[-1] != self.hidden_size:
            raise ValueError("memory hidden width must match the decoder")
        if memory.shape[1] < self.input_field_count:
            raise ValueError("memory must end with all registered input-field tokens")
        if memory_mask.shape != memory.shape[:2]:
            raise ValueError("memory_mask must align with memory tokens")
        field_memory_mask = memory_mask[:, -self.input_field_count :].bool()
        if not field_memory_mask[:, self.required_input_field_mask].all():
            raise ValueError(
                "all 29 output-field history tokens, including unobserved, must be visible"
            )

    def _incremental_target_relation_bias(
        self,
        cache: FieldStateTrajectoryCache,
        *,
        position: int,
        key_positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_position = torch.tensor([position], device=cache.flat_queries.device)
        relation = self._target_relation_values(
            positions=query_position,
            key_positions=key_positions,
            target_relation_adjacency=cache.target_relation_adjacency,
            target_time_scope_ids=cache.target_time_scope_ids,
            dtype=dtype,
        )
        batch_size = cache.flat_queries.shape[0]
        relation = relation.expand(batch_size, -1, -1, -1)
        bos_bias = relation.new_zeros((batch_size, self.num_heads, 1, 1))
        return torch.cat((bos_bias, relation), dim=-1)

    def _target_relation_values(
        self,
        *,
        positions: torch.Tensor,
        key_positions: torch.Tensor,
        target_relation_adjacency: torch.Tensor,
        target_time_scope_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_fields = positions.remainder(self.field_count)
        query_blocks = positions.div(self.field_count, rounding_mode="floor")
        key_fields = key_positions.remainder(self.field_count)
        key_blocks = key_positions.div(self.field_count, rounding_mode="floor")
        delta = query_blocks[:, None] - key_blocks[None, :]
        scopes = target_time_scope_ids.to(device=positions.device, dtype=torch.long).flatten()
        if scopes.shape != (self.target_relation_bias.relation_count,):
            raise ValueError("target_time_scope_ids must contain one id per target edge")
        if not torch.isin(
            scopes,
            torch.tensor(
                [TARGET_SCOPE_SAME_BLOCK, TARGET_SCOPE_ADJACENT_BLOCKS],
                device=scopes.device,
            ),
        ).all():
            raise ValueError("target_time_scope_ids contains an unknown scope")
        scope_mask = torch.where(
            scopes[:, None, None].eq(TARGET_SCOPE_SAME_BLOCK),
            delta.unsqueeze(0).eq(0),
            delta.unsqueeze(0).eq(1),
        )
        return self.target_relation_bias(
            target_relation_adjacency.to(device=positions.device),
            query_fields,
            key_fields,
            relation_scope_mask=scope_mask,
        ).to(dtype=dtype)

    def _target_attention_bias(
        self,
        *,
        batch_size: int,
        available: torch.Tensor,
        positions: torch.Tensor,
        target_relation_adjacency: torch.Tensor,
        target_time_scope_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        access = build_joint_target_access_mask(
            self.block_count,
            self.field_count,
            device=available.device,
        ).index_select(0, positions)
        allowed = access.unsqueeze(0) & available.unsqueeze(1)
        bos_allowed = torch.ones(
            (batch_size, positions.numel(), 1),
            dtype=torch.bool,
            device=available.device,
        )
        allowed_with_bos = torch.cat((bos_allowed, allowed), dim=-1)
        bias = torch.zeros(
            (batch_size, self.num_heads, positions.numel(), self.target_count + 1),
            dtype=dtype,
            device=available.device,
        ).masked_fill(~allowed_with_bos.unsqueeze(1), float("-inf"))
        key_positions = torch.arange(self.target_count, device=available.device)
        relation = self._target_relation_values(
            positions=positions,
            key_positions=key_positions,
            target_relation_adjacency=target_relation_adjacency,
            target_time_scope_ids=target_time_scope_ids,
            dtype=dtype,
        ).expand(batch_size, -1, -1, -1)
        bias[..., 1:] = bias[..., 1:] + relation
        return bias.reshape(
            batch_size * self.num_heads,
            positions.numel(),
            self.target_count + 1,
        )

    def _memory_attention_bias(
        self,
        *,
        batch_size: int,
        memory_mask: torch.Tensor,
        positions: torch.Tensor,
        input_target_relation_adjacency: torch.Tensor,
        input_target_time_scope_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        memory_count = memory_mask.shape[1]
        bias = torch.zeros(
            (batch_size, self.num_heads, positions.numel(), memory_count),
            dtype=dtype,
            device=memory_mask.device,
        ).masked_fill(~memory_mask[:, None, None, :].bool(), float("-inf"))
        query_fields = positions.remainder(self.field_count)
        query_blocks = positions.div(self.field_count, rounding_mode="floor")
        input_fields = torch.arange(self.input_field_count, device=memory_mask.device)
        scopes = input_target_time_scope_ids.to(
            device=memory_mask.device,
            dtype=torch.long,
        ).flatten()
        if scopes.shape != (self.input_target_relation_bias.relation_count,):
            raise ValueError(
                "input_target_time_scope_ids must contain one id per input-target edge"
            )
        if not torch.isin(
            scopes,
            torch.tensor(
                [INPUT_SCOPE_LATEST_TO_FIRST, INPUT_SCOPE_ALL_TO_EACH],
                device=scopes.device,
            ),
        ).all():
            raise ValueError("input_target_time_scope_ids contains an unknown scope")
        bridge_scope = query_blocks.eq(0).view(1, -1, 1).expand(
            scopes.numel(),
            -1,
            self.input_field_count,
        )
        all_scope = torch.ones_like(bridge_scope)
        scope_mask = torch.where(
            scopes[:, None, None].eq(INPUT_SCOPE_LATEST_TO_FIRST),
            bridge_scope,
            all_scope,
        )
        relation = self.input_target_relation_bias(
            input_target_relation_adjacency.to(device=memory_mask.device),
            query_fields,
            input_fields,
            relation_scope_mask=scope_mask,
        ).to(dtype=dtype)
        relation = relation.expand(batch_size, -1, -1, -1)
        bias[..., -self.input_field_count :] = (
            bias[..., -self.input_field_count :] + relation
        )
        return bias.reshape(
            batch_size * self.num_heads,
            positions.numel(),
            memory_count,
        )
