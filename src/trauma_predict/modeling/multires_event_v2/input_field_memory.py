from __future__ import annotations

import torch
from torch import nn


INPUT_ONLY_FIELD_IDS = (29, 30, 31, 32, 33, 34, 36, 37)
TEMPORAL_GEOMETRY_DIM = 6


class InputFieldMemoryEncoder(nn.Module):
    """Build one explicit history token for every registered input field.

    All target fields are read only from the single global final chronological
    input block and carry that block state into the first future block.  There
    is no field-specific last-nonempty fallback.  Input-only fields summarize
    all visible history because their registered edges may affect every future
    block.  Every target field emits a token: absence in the final block is a
    learned unobserved state instead of disappearing behind a padding mask.
    """

    def __init__(
        self,
        hidden_size: int,
        input_field_count: int,
        target_field_ids: tuple[int, ...],
        dropout: float,
        time_scale_hours: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.input_field_count = int(input_field_count)
        self.target_field_ids = tuple(int(field_id) for field_id in target_field_ids)
        expected_ids = set(range(1, self.input_field_count + 1))
        if not set(self.target_field_ids) < expected_ids:
            raise ValueError("target_field_ids must be a strict subset of input field ids")
        input_only_field_ids = tuple(sorted(expected_ids.difference(self.target_field_ids)))
        if self.input_field_count == 37 and input_only_field_ids != INPUT_ONLY_FIELD_IDS:
            raise ValueError(
                "strict relation V2 input-only field order must equal "
                f"{INPUT_ONLY_FIELD_IDS}"
            )
        self.time_scale_hours = float(time_scale_hours)
        if self.time_scale_hours <= 0.0:
            raise ValueError("time_scale_hours must be positive")
        self.register_buffer(
            "target_field_mask",
            torch.tensor(
                [
                    field_id in self.target_field_ids
                    for field_id in range(1, self.input_field_count + 1)
                ],
                dtype=torch.bool,
            ),
            persistent=True,
        )
        self.register_buffer(
            "input_only_field_indices",
            torch.tensor(
                [field_id - 1 for field_id in input_only_field_ids],
                dtype=torch.long,
            ),
            persistent=True,
        )
        # One six-feature block-geometry scorer per input-only field.  Zero
        # initialization is an exact continuity point: masked softmax becomes
        # uniform over that field's existing five-tuples, reproducing the
        # previous all-history arithmetic mean before learning any recency.
        self.input_only_temporal_weight = nn.Parameter(
            torch.zeros(len(input_only_field_ids), TEMPORAL_GEOMETRY_DIM)
        )
        self.observation_state = nn.Embedding(2, self.hidden_size)
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        event_embeddings: torch.Tensor,
        event_field_ids: torch.Tensor,
        block_index: torch.Tensor,
        latest_input_block_index: torch.Tensor,
        event_mask: torch.Tensor,
        event_relative_start: torch.Tensor,
        event_relative_end: torch.Tensor,
        event_span: torch.Tensor,
        registered_field_embeddings: torch.Tensor,
        latest_block_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[batch, 37, hidden]`` tokens and their observed indicators."""

        if event_embeddings.ndim != 3:
            raise ValueError("event_embeddings must be [batch, events, hidden]")
        batch_size, event_count, hidden_size = event_embeddings.shape
        if hidden_size != self.hidden_size:
            raise ValueError("event embedding width does not match field-memory encoder")
        for name, value in (
            ("event_field_ids", event_field_ids),
            ("block_index", block_index),
            ("event_mask", event_mask),
            ("event_relative_start", event_relative_start),
            ("event_relative_end", event_relative_end),
            ("event_span", event_span),
        ):
            if value.shape != (batch_size, event_count):
                raise ValueError(f"{name} must align with event_embeddings")
        if latest_input_block_index.shape != (batch_size,):
            raise ValueError("latest_input_block_index must contain one block per sample")
        if latest_input_block_index.lt(0).any():
            raise ValueError("latest_input_block_index cannot be negative")
        expected_fields = (self.input_field_count, self.hidden_size)
        if registered_field_embeddings.shape != expected_fields:
            raise ValueError(
                "registered_field_embeddings shape="
                f"{tuple(registered_field_embeddings.shape)} does not match {expected_fields}"
            )
        if latest_block_context.shape != (batch_size, self.hidden_size):
            raise ValueError("latest_block_context must be [batch, hidden]")

        field_ids = torch.arange(
            1,
            self.input_field_count + 1,
            dtype=event_field_ids.dtype,
            device=event_field_ids.device,
        )
        valid = event_mask.bool() & block_index.ge(0)
        belongs_to_field = (
            event_field_ids[:, None, :].eq(field_ids[None, :, None])
            & valid[:, None, :]
        )

        # Every output bridge is anchored to the same globally final chronological
        # input block.  It must never fall back to an older nonempty field block.
        # Input-only fields retain every visible tuple across history.
        event_blocks = block_index[:, None, :].expand(-1, self.input_field_count, -1)
        latest_block_events = belongs_to_field & event_blocks.eq(
            latest_input_block_index[:, None, None]
        )
        target_field_mask = self.target_field_mask.to(device=event_embeddings.device)
        selected = torch.where(
            target_field_mask.view(1, -1, 1),
            latest_block_events,
            belongs_to_field,
        )

        counts = selected.sum(dim=-1)
        pooled = torch.einsum(
            "bfe,beh->bfh",
            selected.to(dtype=event_embeddings.dtype),
            event_embeddings,
        ) / counts.clamp_min(1).unsqueeze(-1).to(dtype=event_embeddings.dtype)

        # Input-only fields keep the same source five-tuples and the same one
        # token per field.  Their pooling weights may now learn from the exact
        # boundaries of each tuple's existing H1/M4/F24 block.  These are block
        # bounds, not invented event timestamps or LAST_OBS tuples.
        scale = self.time_scale_hours
        age_lower = (-event_relative_end.float()).clamp_min(0.0) / scale
        age_upper = (-event_relative_start.float()).clamp_min(0.0) / scale
        block_span = event_span.float().clamp_min(0.0) / scale
        geometry = torch.stack(
            (
                age_lower,
                age_upper,
                block_span,
                torch.log1p(age_lower),
                torch.log1p(age_upper),
                torch.log1p(block_span),
            ),
            dim=-1,
        )
        input_only_indices = self.input_only_field_indices.to(
            device=event_embeddings.device
        )
        input_only_events = belongs_to_field.index_select(1, input_only_indices)
        temporal_logits = torch.einsum(
            "bek,fk->bfe",
            geometry,
            self.input_only_temporal_weight.float(),
        )
        has_event = input_only_events.any(dim=-1, keepdim=True)
        masked_logits = temporal_logits.masked_fill(~input_only_events, float("-inf"))
        safe_logits = torch.where(has_event, masked_logits, torch.zeros_like(masked_logits))
        temporal_weights = torch.softmax(safe_logits, dim=-1)
        temporal_weights = temporal_weights * input_only_events.to(
            dtype=temporal_weights.dtype
        )
        temporal_weights = temporal_weights / temporal_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(temporal_weights.dtype).tiny)
        temporal_pooled = torch.einsum(
            "bfe,beh->bfh",
            temporal_weights.to(dtype=event_embeddings.dtype),
            event_embeddings,
        )
        pooled = pooled.index_copy(
            1,
            input_only_indices,
            temporal_pooled.to(dtype=pooled.dtype),
        )
        observed = counts.gt(0)
        observation_state = self.observation_state(observed.long())
        identity = registered_field_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        final_geometry = latest_block_context[:, None, :] * target_field_mask.view(1, -1, 1)
        fused_input = pooled + identity + observation_state + final_geometry
        tokens = self.output_norm(fused_input + self.fusion(fused_input))
        return tokens, observed
