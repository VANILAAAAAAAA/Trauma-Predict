from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


class RegisteredRelationBias(nn.Module):
    """Map explicit registered edges to additive per-head attention bias.

    The first adjacency axis is the frozen ``parameter_key`` order, not a
    relation-type vocabulary.  Consequently two edges with the same clinical
    relation type still learn independent attention residuals.  Rectangular
    target-by-input adjacency is supported for input-target cross-attention.
    Nonedges remain finite zero bias and never become a hard mask.
    """

    def __init__(
        self,
        parameter_keys: Sequence[str],
        num_attention_heads: int = 1,
    ) -> None:
        super().__init__()
        self.parameter_keys = tuple(str(key) for key in parameter_keys)
        if not self.parameter_keys or any(not key for key in self.parameter_keys):
            raise ValueError("parameter_keys must be non-empty strings")
        if len(set(self.parameter_keys)) != len(self.parameter_keys):
            raise ValueError("parameter_keys must be unique")
        self.num_attention_heads = int(num_attention_heads)
        self.edge_head_bias = nn.Parameter(
            torch.empty(len(self.parameter_keys), self.num_attention_heads)
        )
        # A zero residual is not an off-gate: every registered edge is in the
        # forward graph and receives gradients from its first legal use.
        nn.init.zeros_(self.edge_head_bias)

    @property
    def relation_count(self) -> int:
        return len(self.parameter_keys)

    def get_extra_state(self) -> dict[str, Any]:
        return {"parameter_keys": self.parameter_keys}

    def set_extra_state(self, state: Any) -> None:
        checkpoint_keys = tuple(state.get("parameter_keys", ())) if isinstance(state, dict) else ()
        if checkpoint_keys != self.parameter_keys:
            raise RuntimeError(
                "relation parameter_key order differs from the checkpoint: "
                f"expected {self.parameter_keys}, got {checkpoint_keys}"
            )

    def forward(
        self,
        relation_adjacency: torch.Tensor,
        query_field_indices: torch.Tensor,
        key_field_indices: torch.Tensor,
        *,
        relation_scope_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        adjacency = relation_adjacency
        if adjacency.ndim != 3:
            raise ValueError(
                "relation_adjacency must be [parameter_keys, query_fields, key_fields]"
            )
        if adjacency.shape[0] != self.relation_count:
            raise ValueError(
                f"relation_adjacency width={adjacency.shape[0]} does not match "
                f"registered parameter_key count={self.relation_count}"
            )
        query_fields = query_field_indices.to(device=adjacency.device, dtype=torch.long).flatten()
        key_fields = key_field_indices.to(device=adjacency.device, dtype=torch.long).flatten()
        if query_fields.numel() and (
            query_fields.min().item() < 0
            or query_fields.max().item() >= adjacency.shape[1]
        ):
            raise ValueError("query field index is outside relation_adjacency")
        if key_fields.numel() and (
            key_fields.min().item() < 0
            or key_fields.max().item() >= adjacency.shape[2]
        ):
            raise ValueError("key field index is outside relation_adjacency")

        selected = adjacency[:, query_fields][:, :, key_fields].to(
            dtype=self.edge_head_bias.dtype
        )
        if relation_scope_mask is not None:
            scope = relation_scope_mask.to(device=adjacency.device, dtype=selected.dtype)
            expected = (self.relation_count, query_fields.numel(), key_fields.numel())
            if scope.shape != expected:
                raise ValueError(
                    f"relation_scope_mask shape={tuple(scope.shape)} does not match {expected}"
                )
            selected = selected * scope
        return torch.einsum(
            "rqk,rh->hqk",
            selected,
            self.edge_head_bias,
        ).unsqueeze(0)
