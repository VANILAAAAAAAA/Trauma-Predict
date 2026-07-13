from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EmbeddingVocabulary:
    """Vocabulary sizes include padding id zero."""

    fields: int
    operators: int
    conditions: int
    roles: int
    resolutions: int


class SemanticEmbeddingTables(nn.Module):
    """Shared categorical tables for input events and future queries."""

    def __init__(self, vocab: EmbeddingVocabulary, hidden_size: int) -> None:
        super().__init__()
        self.field = nn.Embedding(vocab.fields, hidden_size, padding_idx=0)
        self.operator = nn.Embedding(vocab.operators, hidden_size, padding_idx=0)
        self.condition = nn.Embedding(vocab.conditions, hidden_size, padding_idx=0)
        self.resolution = nn.Embedding(vocab.resolutions, hidden_size, padding_idx=0)

    def forward(
        self,
        field_ids: torch.Tensor,
        operator_ids: torch.Tensor,
        condition_ids: torch.Tensor,
        resolution_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = self.field(field_ids) + self.operator(operator_ids) + self.condition(condition_ids)
        if resolution_ids is not None:
            result = result + self.resolution(resolution_ids)
        return result


class RelativeTimeEmbedding(nn.Module):
    """Project prediction-relative geometry without tokenizing block names."""

    def __init__(self, hidden_size: int, time_scale_hours: float = 24.0) -> None:
        super().__init__()
        self.time_scale_hours = float(time_scale_hours)
        self.network = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        relative_start: torch.Tensor,
        relative_end: torch.Tensor,
        span: torch.Tensor,
    ) -> torch.Tensor:
        scale = max(self.time_scale_hours, 1e-6)
        start = relative_start.float() / scale
        end = relative_end.float() / scale
        block_span = span.float() / scale
        features = torch.stack(
            (
                start,
                end,
                block_span,
                torch.sign(start) * torch.log1p(start.abs()),
                torch.sign(end) * torch.log1p(end.abs()),
                torch.log1p(block_span.clamp_min(0.0)),
            ),
            dim=-1,
        )
        return self.network(features)


class BlockContextEmbedding(nn.Module):
    def __init__(
        self,
        role_vocab_size: int,
        hidden_size: int,
        time_scale_hours: float = 24.0,
    ) -> None:
        super().__init__()
        self.role = nn.Embedding(role_vocab_size, hidden_size, padding_idx=0)
        self.time = RelativeTimeEmbedding(hidden_size, time_scale_hours=time_scale_hours)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        role_ids: torch.Tensor,
        resolution_embedding: torch.Tensor,
        relative_start: torch.Tensor,
        relative_end: torch.Tensor,
        span: torch.Tensor,
    ) -> torch.Tensor:
        return self.norm(
            self.role(role_ids)
            + resolution_embedding
            + self.time(relative_start, relative_end, span)
        )


class EventEmbedding(nn.Module):
    """Combine tuple semantics, typed value content, and gathered block context.

    Numeric observations use the value projection.  CXR ``study_slot`` is a
    categorical identifier (zero means not applicable), so it has an independent
    embedding and never masquerades as a scalar measurement.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        study_slot_vocab_size: int,
    ) -> None:
        super().__init__()
        if study_slot_vocab_size < 9:
            raise ValueError("study_slot_vocab_size must cover padding plus slots 1..8")
        self.value_projection = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.study_slot = nn.Embedding(
            study_slot_vocab_size,
            hidden_size,
            padding_idx=0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        semantic_embedding: torch.Tensor,
        values: torch.Tensor,
        value_mask: torch.Tensor,
        study_slot_ids: torch.Tensor,
        block_context: torch.Tensor,
    ) -> torch.Tensor:
        if study_slot_ids.shape != values.shape:
            raise ValueError("study_slot_ids must align with event values")
        if study_slot_ids.numel() and (
            study_slot_ids.min().item() < 0
            or study_slot_ids.max().item() >= self.study_slot.num_embeddings
        ):
            raise ValueError("event_study_slot_ids is outside the configured vocabulary")
        safe_values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
        numeric = torch.stack((safe_values, value_mask.float()), dim=-1)
        result = (
            semantic_embedding
            + self.value_projection(numeric)
            + self.study_slot(study_slot_ids)
            + block_context
        )
        return self.dropout(self.norm(result))


class StaticContextEncoder(nn.Module):
    """Encode heterogeneous STATIC fields into one independent context token."""

    def __init__(
        self,
        hidden_size: int,
        numeric_fields: int,
        categorical_fields: int,
        categorical_vocab_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.numeric_fields = int(numeric_fields)
        self.categorical_fields = int(categorical_fields)
        self.numeric_projection = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.numeric_field = nn.Embedding(max(self.numeric_fields, 1), hidden_size)
        self.categorical_value = nn.Embedding(categorical_vocab_size, hidden_size, padding_idx=0)
        self.categorical_field = nn.Embedding(max(self.categorical_fields, 1), hidden_size)
        self.empty_static = nn.Parameter(torch.zeros(hidden_size))
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def forward(
        self,
        static_numeric: torch.Tensor,
        static_numeric_mask: torch.Tensor,
        static_categorical: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = static_numeric.shape[0]
        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        if static_numeric.shape[1] != self.numeric_fields:
            raise ValueError(
                f"static_numeric width={static_numeric.shape[1]} does not match "
                f"configured numeric_fields={self.numeric_fields}"
            )
        if self.numeric_fields:
            numeric_mask = static_numeric_mask.bool()
            safe = torch.nan_to_num(static_numeric.float(), nan=0.0, posinf=0.0, neginf=0.0)
            projected = self.numeric_projection(
                torch.stack((safe, numeric_mask.float()), dim=-1)
            )
            field_ids = torch.arange(self.numeric_fields, device=safe.device)
            pieces.append(projected + self.numeric_field(field_ids).unsqueeze(0))
            masks.append(numeric_mask)

        if static_categorical.shape[1] != self.categorical_fields:
            raise ValueError(
                f"static_categorical width={static_categorical.shape[1]} does not match "
                f"configured categorical_fields={self.categorical_fields}"
            )
        if self.categorical_fields:
            categorical_mask = static_categorical.ne(0)
            field_ids = torch.arange(self.categorical_fields, device=static_categorical.device)
            pieces.append(
                self.categorical_value(static_categorical)
                + self.categorical_field(field_ids).unsqueeze(0)
            )
            masks.append(categorical_mask)

        if not pieces:
            return self.empty_static.view(1, -1).expand(batch_size, -1)
        values = torch.cat(pieces, dim=1)
        valid = torch.cat(masks, dim=1)
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1).float()
        pooled = (values * valid.unsqueeze(-1)).sum(dim=1) / denominator
        no_static = ~valid.any(dim=1)
        pooled = torch.where(no_static.unsqueeze(-1), self.empty_static, pooled)
        return self.fusion(pooled)
