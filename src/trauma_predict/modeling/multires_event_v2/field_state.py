from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class FutureFieldStateQueries(nn.Module):
    """Create one registered-field query for every future M4 block."""

    def __init__(
        self,
        hidden_size: int,
        block_count: int,
        field_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_count = int(block_count)
        self.field_count = int(field_count)
        self.block = nn.Embedding(self.block_count, hidden_size)
        self.field_order = nn.Embedding(self.field_count, hidden_size)
        self.query_seed = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.query_seed, std=0.02)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        registered_field_embeddings: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if registered_field_embeddings.shape[:1] != (self.field_count,):
            raise ValueError("registered_field_embeddings must align with the frozen field order")
        device = registered_field_embeddings.device
        block_ids = torch.arange(self.block_count, device=device)
        field_order = torch.arange(self.field_count, device=device)
        queries = (
            registered_field_embeddings.view(1, self.field_count, -1)
            + self.block(block_ids).view(self.block_count, 1, -1)
            + self.field_order(field_order).view(1, self.field_count, -1)
            + self.query_seed.view(1, 1, -1)
        )
        queries = self.dropout(self.norm(queries))
        return queries.unsqueeze(0).expand(int(batch_size), -1, -1, -1)


class FieldStateContextEmbedding(nn.Module):
    """Adapt teacher-forced or generated field states for target-state attention."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(torch.nan_to_num(states))


class PrimitiveParameterHeads(nn.Module):
    """Contract-defined emission parameters with no loss semantics in the model."""

    def __init__(
        self,
        hidden_size: int,
        head_dims: Mapping[str, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.head_dims = {name: int(width) for name, width in head_dims.items()}
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, int(width)),
                )
                for name, width in head_dims.items()
            }
        )

    def forward(self, field_states: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(field_states) for name, head in self.heads.items()}

    def forward_selected(
        self,
        field_states: torch.Tensor,
        likelihood_ids: tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        """Evaluate registered heads while preserving the full sampler mapping.

        Unselected banks are shape-correct zeros.  They are never sampled at
        this field position, but retaining them preserves the strict
        ``RegistryPrimitiveSampler`` key order and call contract.
        """

        if self.training:
            raise RuntimeError(
                "selected primitive heads are inference-only; call primitive_heads.eval() first"
            )
        selected = frozenset(likelihood_ids)
        unknown = selected.difference(self.heads)
        if unknown:
            raise ValueError(f"unknown selected primitive heads: {sorted(unknown)}")
        leading_shape = field_states.shape[:-1]
        return {
            name: (
                head(field_states)
                if name in selected
                else field_states.new_zeros((*leading_shape, self.head_dims[name]))
            )
            for name, head in self.heads.items()
        }


class PrimitiveFeedbackEncoder(nn.Module):
    """Encode observed/generated primitives into the state used as causal feedback.

    Each key is one likelihood id.  Its value and component-validity mask have a
    registry-defined final width, so no opaque target latent enters the decoder.
    Physical-unit values use a fixed signed-log transform inside this feedback
    path.  The likelihood path still receives exact raw units; the transform is
    only a scale-stable representation of already-visible target history.
    """

    def __init__(
        self,
        hidden_size: int,
        feedback_dims: Mapping[str, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.feedback_dims = dict(feedback_dims)
        self.projections = nn.ModuleDict(
            {
                likelihood_id: nn.Sequential(
                    nn.Linear(int(width) * 2, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_size),
                )
                for likelihood_id, width in self.feedback_dims.items()
            }
        )
        self.empty_feedback = nn.Parameter(torch.zeros(hidden_size))
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def forward(
        self,
        target_primitives: Mapping[str, torch.Tensor],
        target_primitive_masks: Mapping[str, torch.Tensor],
        *,
        leading_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_keys = set(self.feedback_dims)
        if set(target_primitives) != expected_keys:
            raise ValueError(
                "target_primitives likelihood ids do not match primitive_feedback_dims"
            )
        if set(target_primitive_masks) != expected_keys:
            raise ValueError(
                "target_primitive_masks likelihood ids do not match primitive_feedback_dims"
            )

        encoded: list[torch.Tensor] = []
        present: list[torch.Tensor] = []
        for likelihood_id, width in self.feedback_dims.items():
            values = target_primitives[likelihood_id]
            masks = target_primitive_masks[likelihood_id]
            expected_shape = (*leading_shape, width)
            if values.shape != expected_shape:
                raise ValueError(
                    f"target primitive {likelihood_id!r} shape={tuple(values.shape)} "
                    f"does not match {expected_shape}"
                )
            if masks.shape == leading_shape:
                masks = masks.unsqueeze(-1).expand_as(values)
            if masks.shape != expected_shape:
                raise ValueError(
                    f"target primitive mask {likelihood_id!r} must be {expected_shape} "
                    f"or {leading_shape}"
                )
            component_mask = masks.bool()
            raw_values = torch.nan_to_num(values.float()) * component_mask
            safe_values = torch.sign(raw_values) * torch.log1p(raw_values.abs())
            features = torch.cat((safe_values, component_mask.float()), dim=-1)
            is_present = component_mask.any(dim=-1)
            encoded.append(self.projections[likelihood_id](features) * is_present.unsqueeze(-1))
            present.append(is_present)

        if not encoded:
            state = self.empty_feedback.view(*((1,) * len(leading_shape)), -1).expand(
                *leading_shape,
                -1,
            )
            valid = torch.zeros(leading_shape, dtype=torch.bool, device=state.device)
            return state, valid
        stacked = torch.stack(encoded, dim=-2)
        valid_by_likelihood = torch.stack(present, dim=-1)
        denominator = valid_by_likelihood.sum(dim=-1, keepdim=True).clamp_min(1).float()
        pooled = stacked.sum(dim=-2) / denominator
        valid = valid_by_likelihood.any(dim=-1)
        empty = self.empty_feedback.view(*((1,) * len(leading_shape)), -1)
        pooled = torch.where(valid.unsqueeze(-1), pooled, empty)
        return self.output(pooled), valid
