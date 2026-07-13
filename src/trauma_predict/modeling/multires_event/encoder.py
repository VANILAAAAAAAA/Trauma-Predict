from __future__ import annotations

import torch
from torch import nn


class CrossAttentionCompressorLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.event_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
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
        latents: torch.Tensor,
        events: torch.Tensor,
        event_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            self.query_norm(latents),
            self.event_norm(events),
            self.event_norm(events),
            key_padding_mask=event_padding_mask,
            need_weights=False,
        )
        latents = latents + attended
        return latents + self.feed_forward(self.feed_forward_norm(latents))


def pack_events_by_block(
    event_embeddings: torch.Tensor,
    block_index: torch.Tensor,
    event_mask: torch.Tensor,
    block_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ragged event groups into a dense (batch*block, max_events, hidden) tensor."""

    batch_size, _, hidden_size = event_embeddings.shape
    group_count = batch_size * block_count
    valid = event_mask.bool() & block_index.ge(0) & block_index.lt(block_count)
    event_count = event_embeddings.shape[1]
    flat_valid = valid.reshape(-1)
    flat_block = block_index.reshape(-1)[flat_valid]
    flat_batch = (
        torch.arange(batch_size, device=event_embeddings.device)
        .view(-1, 1)
        .expand(-1, event_count)
        .reshape(-1)[flat_valid]
    )
    group_ids = flat_batch * block_count + flat_block
    order = torch.argsort(group_ids, stable=True)
    sorted_groups = group_ids.index_select(0, order)
    sorted_events = event_embeddings.reshape(-1, hidden_size)[flat_valid].index_select(0, order)
    counts = torch.bincount(sorted_groups, minlength=group_count)
    max_events = max(int(counts.max().item()) if counts.numel() else 0, 1)
    packed = event_embeddings.new_zeros((group_count, max_events, hidden_size))
    padding_mask = torch.ones(
        (group_count, max_events),
        dtype=torch.bool,
        device=event_embeddings.device,
    )
    if sorted_groups.numel():
        starts = torch.cumsum(counts, dim=0) - counts
        offsets = torch.arange(
            sorted_groups.numel(),
            device=event_embeddings.device,
        ) - torch.repeat_interleave(starts, counts)
        packed[sorted_groups, offsets] = sorted_events
        padding_mask[sorted_groups, offsets] = False

    # MultiheadAttention returns NaN when every key is masked. Empty/padded blocks use a
    # zero-valued sentinel and are removed again with group_has_events.
    empty = counts.eq(0)
    padding_mask[empty, 0] = False
    return packed, padding_mask, ~empty


class BlockLatentCompressor(nn.Module):
    """Compress each variable-size event block into a fixed learned latent set."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        latent_count: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_count = int(latent_count)
        self.latents = nn.Parameter(torch.empty(self.latent_count, hidden_size))
        nn.init.normal_(self.latents, std=0.02)
        self.layers = nn.ModuleList(
            CrossAttentionCompressorLayer(hidden_size, num_heads, dropout)
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        event_embeddings: torch.Tensor,
        block_index: torch.Tensor,
        event_mask: torch.Tensor,
        block_context: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, block_count, hidden_size = block_context.shape
        packed, event_padding_mask, _group_has_events = pack_events_by_block(
            event_embeddings,
            block_index,
            event_mask,
            block_count,
        )
        context = block_context.reshape(batch_size * block_count, 1, hidden_size)
        latents = self.latents.view(1, self.latent_count, hidden_size).expand(
            batch_size * block_count,
            -1,
            -1,
        )
        latents = latents + context
        for layer in self.layers:
            latents = layer(latents, packed, event_padding_mask)
        # An observed time block with no emitted tuples is still part of the trajectory:
        # the learned latents carry its temporal context and explicit event absence.
        valid_group = block_mask.reshape(-1).bool()
        latents = self.output_norm(latents) * valid_group.view(-1, 1, 1)
        return (
            latents.reshape(batch_size, block_count, self.latent_count, hidden_size),
            valid_group.reshape(batch_size, block_count),
        )


class TrajectoryEncoder(nn.Module):
    """Encode STATIC plus ordered block latents into one shared trajectory memory."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        static_token: torch.Tensor,
        block_latents: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, block_count, latent_count, hidden_size = block_latents.shape
        flattened = block_latents.reshape(batch_size, block_count * latent_count, hidden_size)
        latent_valid = block_mask.bool().repeat_interleave(latent_count, dim=1)
        memory = torch.cat((static_token.unsqueeze(1), flattened), dim=1)
        valid = torch.cat(
            (
                torch.ones((batch_size, 1), dtype=torch.bool, device=memory.device),
                latent_valid,
            ),
            dim=1,
        )
        encoded = self.encoder(memory, src_key_padding_mask=~valid)
        return self.output_norm(encoded), valid
