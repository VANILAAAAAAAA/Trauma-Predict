from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trauma_predict.data.main_route_contract import (
    BINARY_NEXT24_FIELD_SPECS,
    HOUR_SPECIAL_TOKENS,
    HOUR_VALUE_ORDER,
    MULTICLASS_NEXT24_FIELD_SPECS,
    TARGET_DOMAINS,
)


def _nn_module():  # type: ignore[no-untyped-def]
    import torch.nn as nn

    return nn.Module


class HourStateAdapter(_nn_module()):
    def __init__(
        self,
        hidden_size: int,
        adapter_hidden_size: int,
        dropout: float,
        field_hidden_size: int = 64,
    ) -> None:
        super().__init__()
        import torch.nn as nn

        self.field_hidden_size = field_hidden_size
        self.field_embedding = nn.Embedding(len(HOUR_VALUE_ORDER) + 1, field_hidden_size)
        self.vital_value_projections = nn.ModuleList([
            nn.Linear(1, field_hidden_size)
            for _ in HOUR_VALUE_ORDER
        ])
        self.vital_mask_embedding = nn.Embedding(2, field_hidden_size)
        self.vent_state_embedding = nn.Embedding(2, field_hidden_size)
        self.field_norm = nn.LayerNorm(field_hidden_size)
        self.hour_network = nn.Sequential(
            nn.LayerNorm(field_hidden_size * (len(HOUR_VALUE_ORDER) + 1)),
            nn.Linear(field_hidden_size * (len(HOUR_VALUE_ORDER) + 1), adapter_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, hour_values, hour_mask, hour_vent):  # type: ignore[no-untyped-def]
        import torch

        values = torch.nan_to_num(hour_values, nan=0.0, posinf=0.0, neginf=0.0)
        masks = hour_mask.float().clamp(0.0, 1.0)
        field_ids = torch.arange(len(HOUR_VALUE_ORDER) + 1, device=values.device)
        field_embeddings = self.field_embedding(field_ids)

        field_features = []
        for index, value_projection in enumerate(self.vital_value_projections):
            value = values[..., index:index + 1] * masks[..., index:index + 1]
            value_feature = value_projection(value)
            mask_feature = self.vital_mask_embedding(masks[..., index].long())
            field_feature = field_embeddings[index].view(1, 1, -1)
            field_features.append(self.field_norm(value_feature + mask_feature + field_feature))

        vent_state = hour_vent.float().clamp(0.0, 1.0).squeeze(-1).long()
        vent_feature = self.vent_state_embedding(vent_state) + field_embeddings[-1].view(1, 1, -1)
        field_features.append(self.field_norm(vent_feature))

        return self.hour_network(torch.cat(field_features, dim=-1))


class MainRouteModel(_nn_module()):
    def __init__(
        self,
        base_model: str,
        tokenizer_length: int,
        adapter_hidden_size: int = 256,
        hour_field_hidden_size: int = 64,
        dropout: float = 0.1,
        loss_weights: dict[str, float] | None = None,
        active_losses: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        import torch
        import torch.nn as nn
        from transformers import AutoModel

        self.base_model = base_model
        self.encoder = AutoModel.from_pretrained(base_model)
        self.encoder.resize_token_embeddings(tokenizer_length)
        hidden_size = int(getattr(self.encoder.config, "hidden_size"))
        self.hidden_size = hidden_size
        self.hour_adapter = HourStateAdapter(
            hidden_size=hidden_size,
            adapter_hidden_size=adapter_hidden_size,
            dropout=dropout,
            field_hidden_size=hour_field_hidden_size,
        )
        self.hour_time_embedding = nn.Embedding(len(HOUR_SPECIAL_TOKENS), hidden_size)
        self.hour_segment_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.next_hour_values_head = nn.Linear(hidden_size, len(HOUR_VALUE_ORDER))
        self.next_hour_vent_head = nn.Linear(hidden_size, 1)
        self.next24_domain_head = nn.Linear(hidden_size, len(TARGET_DOMAINS))
        self.next24_binary_heads = nn.ModuleDict({
            spec.safe_key: nn.Linear(hidden_size, 1)
            for spec in BINARY_NEXT24_FIELD_SPECS
        })
        self.next24_multiclass_heads = nn.ModuleDict({
            spec.safe_key: nn.Linear(hidden_size, len(spec.values) + 1)
            for spec in MULTICLASS_NEXT24_FIELD_SPECS
        })
        self.loss_weights = {
            "next_hour_values": 1.0,
            "next_hour_vent": 0.25,
            "next24_domain": 0.25,
            "next24_binary": 0.5,
            "next24_multiclass": 0.5,
        }
        if loss_weights:
            self.loss_weights.update({key: float(value) for key, value in loss_weights.items()})
        self.active_losses = {
            "next_hour_values": True,
            "next_hour_vent": True,
            "next24_domain": True,
            "next24_binary": True,
            "next24_multiclass": True,
        }
        if active_losses:
            self.active_losses.update({key: bool(value) for key, value in active_losses.items()})
        self.supports_global_attention = str(getattr(self.encoder.config, "model_type", "")).lower() in {
            "longformer",
        }

    def forward(
        self,
        input_ids=None,  # type: ignore[no-untyped-def]
        attention_mask=None,  # type: ignore[no-untyped-def]
        hour_values=None,  # type: ignore[no-untyped-def]
        hour_mask=None,  # type: ignore[no-untyped-def]
        hour_vent=None,  # type: ignore[no-untyped-def]
        hour_positions=None,  # type: ignore[no-untyped-def]
        hour_position_mask=None,  # type: ignore[no-untyped-def]
        hour_time_indices=None,  # type: ignore[no-untyped-def]
        state_position=None,  # type: ignore[no-untyped-def]
        next_hour_values=None,  # type: ignore[no-untyped-def]
        next_hour_mask=None,  # type: ignore[no-untyped-def]
        next_hour_vent=None,  # type: ignore[no-untyped-def]
        next24_domain_labels=None,  # type: ignore[no-untyped-def]
        next24_binary_labels=None,  # type: ignore[no-untyped-def]
        next24_multiclass_labels=None,  # type: ignore[no-untyped-def]
        **_: Any,
    ) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required")
        if hour_values is None or hour_mask is None or hour_vent is None:
            raise ValueError("HOUR side tensors are required")
        if hour_positions is None or hour_position_mask is None or hour_time_indices is None:
            raise ValueError("HOUR token positions and time indices are required")
        if state_position is None:
            raise ValueError("state_position is required")

        token_embeddings = self.encoder.get_input_embeddings()(input_ids)
        hour_embeddings = self.hour_adapter(hour_values, hour_mask, hour_vent)
        hour_embeddings = (
            hour_embeddings
            + self.hour_time_embedding(hour_time_indices)
            + self.hour_segment_embedding.view(1, 1, -1)
        )
        inputs_embeds = token_embeddings.clone()
        valid = hour_position_mask.bool()
        if bool(valid.any()):
            batch_index = torch.arange(input_ids.shape[0], device=input_ids.device).unsqueeze(1).expand_as(hour_positions)
            inputs_embeds[batch_index[valid], hour_positions[valid]] = (
                inputs_embeds[batch_index[valid], hour_positions[valid]] + hour_embeddings[valid]
            )

        encoder_kwargs: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        if self.supports_global_attention:
            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[torch.arange(input_ids.shape[0], device=input_ids.device), state_position] = 1
            encoder_kwargs["global_attention_mask"] = global_attention_mask
        outputs = self.encoder(**encoder_kwargs)
        hidden = outputs.last_hidden_state
        state_vector = hidden[torch.arange(hidden.shape[0], device=hidden.device), state_position]
        state_vector = self.dropout(state_vector)

        next_hour_value_logits = self.next_hour_values_head(state_vector)
        next_hour_vent_logits = self.next_hour_vent_head(state_vector)
        next24_domain_logits = self.next24_domain_head(state_vector)
        next24_binary_logits = torch.cat(
            [self.next24_binary_heads[spec.safe_key](state_vector) for spec in BINARY_NEXT24_FIELD_SPECS],
            dim=-1,
        ) if BINARY_NEXT24_FIELD_SPECS else state_vector.new_zeros((state_vector.shape[0], 0))
        next24_multiclass_logits = [
            self.next24_multiclass_heads[spec.safe_key](state_vector)
            for spec in MULTICLASS_NEXT24_FIELD_SPECS
        ]

        loss = state_vector.new_tensor(0.0)
        loss_parts: dict[str, Any] = {}
        if self._loss_enabled("next_hour_values") and next_hour_values is not None and next_hour_mask is not None:
            observed = next_hour_mask.float()
            observed_count = observed.sum().clamp_min(1.0)
            hour_value_loss = F.smooth_l1_loss(
                next_hour_value_logits * observed,
                next_hour_values.float() * observed,
                reduction="sum",
            ) / observed_count
            loss = loss + self.loss_weights["next_hour_values"] * hour_value_loss
            loss_parts["next_hour_values"] = hour_value_loss.detach()
        if self._loss_enabled("next_hour_vent") and next_hour_vent is not None:
            hour_vent_loss = F.binary_cross_entropy_with_logits(
                next_hour_vent_logits,
                next_hour_vent.float(),
            )
            loss = loss + self.loss_weights["next_hour_vent"] * hour_vent_loss
            loss_parts["next_hour_vent"] = hour_vent_loss.detach()
        if self._loss_enabled("next24_domain") and next24_domain_labels is not None:
            domain_loss = F.binary_cross_entropy_with_logits(
                next24_domain_logits,
                next24_domain_labels.float(),
            )
            loss = loss + self.loss_weights["next24_domain"] * domain_loss
            loss_parts["next24_domain"] = domain_loss.detach()
        if self._loss_enabled("next24_binary") and next24_binary_labels is not None and next24_binary_logits.shape[-1]:
            binary_loss = F.binary_cross_entropy_with_logits(
                next24_binary_logits,
                next24_binary_labels.float(),
            )
            loss = loss + self.loss_weights["next24_binary"] * binary_loss
            loss_parts["next24_binary"] = binary_loss.detach()
        if self._loss_enabled("next24_multiclass") and next24_multiclass_labels is not None and next24_multiclass_logits:
            multiclass_losses = []
            for index, logits in enumerate(next24_multiclass_logits):
                multiclass_losses.append(F.cross_entropy(logits, next24_multiclass_labels[:, index]))
            multiclass_loss = torch.stack(multiclass_losses).mean()
            loss = loss + self.loss_weights["next24_multiclass"] * multiclass_loss
            loss_parts["next24_multiclass"] = multiclass_loss.detach()

        return {
            "loss": loss,
            "logits": next_hour_value_logits,
            "next_hour_value_logits": next_hour_value_logits,
            "next_hour_vent_logits": next_hour_vent_logits,
            "next24_domain_logits": next24_domain_logits,
            "next24_binary_logits": next24_binary_logits,
            "next24_multiclass_logits": tuple(next24_multiclass_logits),
            "loss_parts": loss_parts,
        }

    def _loss_enabled(self, key: str) -> bool:
        return bool(self.active_losses.get(key, False)) and float(self.loss_weights.get(key, 0.0)) != 0.0

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False

    def save_main_route(self, output_dir: Path) -> None:
        import torch

        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "main_route_model.pt")
        payload = {
            "base_model": self.base_model,
            "hidden_size": self.hidden_size,
            "hour_adapter": {
                "type": "field_aware",
                "field_hidden_size": self.hour_adapter.field_hidden_size,
            },
            "hour_value_order": list(HOUR_VALUE_ORDER),
            "target_domains": list(TARGET_DOMAINS),
            "binary_next24_fields": [spec.key for spec in BINARY_NEXT24_FIELD_SPECS],
            "multiclass_next24_fields": [spec.key for spec in MULTICLASS_NEXT24_FIELD_SPECS],
            "active_losses": self.active_losses,
            "loss_weights": self.loss_weights,
        }
        (output_dir / "main_route_model_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
