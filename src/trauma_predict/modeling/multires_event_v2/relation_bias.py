from __future__ import annotations

import torch
from torch import nn


class TypedRelationBias(nn.Module):
    """Map a typed field graph to additive per-head attention bias.

    Input orientation is ``[relation_type, query_field, key_field]``.  Zero-valued
    nonedges remain an additive zero; this module never creates an attention mask.
    """

    def __init__(self, relation_type_count: int, num_attention_heads: int) -> None:
        super().__init__()
        self.relation_type_count = int(relation_type_count)
        self.num_attention_heads = int(num_attention_heads)
        self.type_head_bias = nn.Parameter(
            torch.empty(self.relation_type_count, self.num_attention_heads)
        )
        # Relational and relation-neutral runs must start from the same
        # conditional function, not merely the same parameter shapes.  A zero
        # initialization makes the typed prior an exact learned residual at
        # step zero while retaining nonzero gradients in relational mode.
        nn.init.zeros_(self.type_head_bias)

    def forward(
        self,
        relation_adjacency: torch.Tensor,
        query_field_indices: torch.Tensor,
        key_field_indices: torch.Tensor,
        *,
        relation_type_lags: torch.Tensor | None = None,
        query_block_indices: torch.Tensor | None = None,
        key_block_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        adjacency = relation_adjacency
        if adjacency.ndim == 3:
            adjacency = adjacency.unsqueeze(0)
        if adjacency.ndim != 4:
            raise ValueError(
                "relation_adjacency must be [types, fields, fields] or "
                "[batch, types, fields, fields]"
            )
        if adjacency.shape[1] != self.relation_type_count:
            raise ValueError(
                f"relation_adjacency type width={adjacency.shape[1]} does not match "
                f"configured relation_type_count={self.relation_type_count}"
            )
        if adjacency.shape[2] != adjacency.shape[3]:
            raise ValueError("relation_adjacency field axes must be square")
        field_count = adjacency.shape[2]
        if query_field_indices.numel() and (
            query_field_indices.min().item() < 0 or query_field_indices.max().item() >= field_count
        ):
            raise ValueError("query field index is outside relation_adjacency")
        if key_field_indices.numel() and (
            key_field_indices.min().item() < 0 or key_field_indices.max().item() >= field_count
        ):
            raise ValueError("key field index is outside relation_adjacency")

        adjacency = adjacency.to(dtype=self.type_head_bias.dtype)
        selected = adjacency[:, :, query_field_indices][:, :, :, key_field_indices]
        lag_arguments = (relation_type_lags, query_block_indices, key_block_indices)
        if any(item is not None for item in lag_arguments):
            if any(item is None for item in lag_arguments):
                raise ValueError(
                    "relation lag gating requires type lags and query/key block indices"
                )
            assert relation_type_lags is not None
            assert query_block_indices is not None
            assert key_block_indices is not None
            lags = relation_type_lags.to(device=adjacency.device, dtype=torch.long).flatten()
            if lags.shape != (self.relation_type_count,):
                raise ValueError("relation_type_lags must contain one lag per relation type")
            if lags.lt(0).any():
                raise ValueError("relation_type_lags cannot be negative")
            query_blocks = query_block_indices.to(device=adjacency.device, dtype=torch.long).flatten()
            key_blocks = key_block_indices.to(device=adjacency.device, dtype=torch.long).flatten()
            if query_blocks.numel() != query_field_indices.numel():
                raise ValueError("query block and field indices must align")
            if key_blocks.numel() != key_field_indices.numel():
                raise ValueError("key block and field indices must align")
            block_delta = query_blocks[:, None] - key_blocks[None, :]
            lag_match = block_delta.unsqueeze(0).eq(lags[:, None, None])
            selected = selected * lag_match.unsqueeze(0)
        field_bias = torch.einsum(
            "brqk,rh->bhqk",
            selected,
            self.type_head_bias,
        )
        return field_bias
