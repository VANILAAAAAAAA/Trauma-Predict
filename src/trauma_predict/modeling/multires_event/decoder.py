from __future__ import annotations

import torch
from torch import nn


class FutureQueryEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_time_index: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_index = nn.Embedding(max_time_index, hidden_size)
        self.span_projection = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        semantic_embedding: torch.Tensor,
        time_index: torch.Tensor,
        span: torch.Tensor,
    ) -> torch.Tensor:
        if time_index.min().item() < 0 or time_index.max().item() >= self.time_index.num_embeddings:
            raise ValueError("query_time_index is outside the configured embedding range")
        span_value = span.float()
        span_features = torch.stack(
            (span_value / 24.0, torch.log1p(span_value.clamp_min(0.0))),
            dim=-1,
        )
        result = semantic_embedding + self.time_index(time_index) + self.span_projection(span_features)
        return self.dropout(self.norm(result))


class BlockLocalFutureQueryDecoder(nn.Module):
    """Decode each of H1/M4_01..06 locally while sharing trajectory memory.

    Query self-attention is deliberately restricted to one future time block. For the
    frozen 986-query contract this avoids a dense 986x986 attention matrix while still
    allowing every query to cross-attend to the complete encoded history.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        layers: int,
        dropout: float,
        expected_block_count: int = 7,
    ) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.expected_block_count = int(expected_block_count)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        query_resolution_ids: torch.Tensor,
        query_time_index: torch.Tensor,
        query_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query_resolution_ids.shape != query_time_index.shape:
            raise ValueError("query resolution and time-index tensors must have the same shape")
        if not torch.equal(query_resolution_ids, query_resolution_ids[:1].expand_as(query_resolution_ids)):
            raise ValueError("fixed query resolution order must be identical across the batch")
        if not torch.equal(query_time_index, query_time_index[:1].expand_as(query_time_index)):
            raise ValueError("fixed query time order must be identical across the batch")

        metadata = torch.stack((query_resolution_ids[0], query_time_index[0]), dim=-1)
        groups = torch.unique(metadata, dim=0, sorted=True)
        if self.expected_block_count and groups.shape[0] != self.expected_block_count:
            raise ValueError(
                f"fixed query contract has {groups.shape[0]} time blocks; "
                f"expected {self.expected_block_count}"
            )

        decoded = torch.zeros_like(query_embeddings)
        for resolution_id, time_index in groups.tolist():
            positions = (
                metadata[:, 0].eq(int(resolution_id))
                & metadata[:, 1].eq(int(time_index))
            ).nonzero(as_tuple=False).flatten()
            local_query = query_embeddings.index_select(1, positions)
            local_mask = query_mask.index_select(1, positions).bool()
            local_decoded = self.decoder(
                tgt=local_query,
                memory=memory,
                tgt_key_padding_mask=~local_mask,
                memory_key_padding_mask=~memory_mask.bool(),
            )
            local_decoded = self.output_norm(local_decoded) * local_mask.unsqueeze(-1)
            decoded.index_copy_(1, positions, local_decoded)
        return decoded
