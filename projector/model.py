from __future__ import annotations

import torch
from torch import nn


class VitalValueProjector(nn.Module):
    """Project fixed [B,T,7] vital values + masks into hour embeddings."""

    def __init__(self, n_vitals: int = 7, history_len: int = 24, d_model: int = 64):
        super().__init__()
        self.n_vitals = n_vitals
        self.history_len = history_len
        self.d_model = d_model
        self.field_emb = nn.Embedding(n_vitals, d_model)
        self.time_emb = nn.Embedding(history_len, d_model)
        self.mask_emb = nn.Embedding(2, d_model)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.slot_norm = nn.LayerNorm(d_model)
        self.hour_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, values_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return hour embeddings [B,T,D].

        values_norm: [B,T,7], standardized; arbitrary placeholder allowed where mask=0.
        mask: [B,T,7], 1 observed / 0 missing.
        """
        b, t, f = values_norm.shape
        assert f == self.n_vitals
        device = values_norm.device
        field_ids = torch.arange(f, device=device).view(1, 1, f)
        time_ids = torch.arange(t, device=device).view(1, t, 1)
        mlong = mask.long().clamp(0, 1)
        base = self.field_emb(field_ids) + self.time_emb(time_ids) + self.mask_emb(mlong)
        value_e = self.value_mlp(values_norm.unsqueeze(-1)) * mask.unsqueeze(-1)
        slot_e = self.slot_norm(base + value_e)  # [B,T,F,D]
        denom = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        hour_e = (slot_e * mask.unsqueeze(-1)).sum(dim=2) / denom
        return self.hour_pool(hour_e)


class VitalNextHourModel(nn.Module):
    """Small smoke-test model for next-hour 7-vital regression."""

    def __init__(self, n_vitals: int = 7, history_len: int = 24, d_model: int = 64, n_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.projector = VitalValueProjector(n_vitals=n_vitals, history_len=history_len, d_model=d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True, dropout=0.1, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_vitals),
        )

    def forward(self, values_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.projector(values_norm, mask)
        z = self.encoder(h)
        return self.head(z[:, -1])


def masked_huber_loss(pred: torch.Tensor, target_norm: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.huber_loss(pred, target_norm, reduction='none')
    denom = target_mask.sum().clamp_min(1.0)
    return (loss * target_mask).sum() / denom
