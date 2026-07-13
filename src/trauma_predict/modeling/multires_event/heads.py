from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TypedPredictionHeads(nn.Module):
    """Emit parameters for every registered loss family at every future query.

    The loss registry selects the applicable family. Producing compact parameter banks
    keeps query ordering independent from Python-side head dispatch and makes the output
    straightforward to gather under DDP.
    """

    def __init__(self, hidden_size: int, dropout: float, max_ordinal_thresholds: int = 5) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.continuous = nn.Linear(hidden_size, 2)
        self.ordinal = nn.Linear(hidden_size, max_ordinal_thresholds)
        self.count = nn.Linear(hidden_size, 3)
        self.point_duration = nn.Linear(hidden_size, 1)
        self.interval_duration = nn.Linear(hidden_size, 5)
        self.binary = nn.Linear(hidden_size, 1)
        self.structured_score = nn.Linear(hidden_size, 1)
        self.nonnegative = nn.Linear(hidden_size, 3)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.trunk(hidden)
        continuous = self.continuous(features)
        count = self.count(features)
        interval = self.interval_duration(features)
        nonnegative = self.nonnegative(features)
        return {
            "continuous_loc": continuous[..., 0],
            "continuous_scale": F.softplus(continuous[..., 1]) + 1e-4,
            "ordinal_logits": self.ordinal(features),
            "count_gate_logits": count[..., 0],
            "count_total_count": F.softplus(count[..., 1]) + 1e-4,
            "count_nb_logits": count[..., 2],
            "point_duration_logits": self.point_duration(features).squeeze(-1),
            "interval_mixture_logits": interval[..., :3],
            "interval_alpha": F.softplus(interval[..., 3]) + 1e-4,
            "interval_beta": F.softplus(interval[..., 4]) + 1e-4,
            "binary_logits": self.binary(features).squeeze(-1),
            "structured_scores": self.structured_score(features).squeeze(-1),
            "nonnegative_gate_logits": nonnegative[..., 0],
            "nonnegative_loc": nonnegative[..., 1],
            "nonnegative_scale": F.softplus(nonnegative[..., 2]) + 1e-4,
        }
