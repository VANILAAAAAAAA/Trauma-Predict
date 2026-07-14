from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn

from .config import TrajectoryMode
from .field_state import PrimitiveFeedbackEncoder, PrimitiveParameterHeads
from .trajectory import FieldStateTrajectoryDecoder


PrimitiveSampler = Callable[
    [int, int, Mapping[str, torch.Tensor]],
    tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
]


class AutoregressiveFieldStateRollout(nn.Module):
    """Generate block-major field states without accepting future ground truth."""

    def __init__(
        self,
        block_count: int,
        field_count: int,
    ) -> None:
        super().__init__()
        self.block_count = int(block_count)
        self.field_count = int(field_count)
        self.target_count = self.block_count * self.field_count

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        *,
        decoder: FieldStateTrajectoryDecoder,
        primitive_heads: PrimitiveParameterHeads,
        feedback_encoder: PrimitiveFeedbackEncoder,
        mode: TrajectoryMode,
        sampler: PrimitiveSampler,
        relation_adjacency: torch.Tensor | None = None,
        relation_type_lags: torch.Tensor | None = None,
        use_cache: bool | None = None,
        use_selected_heads: bool | None = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        batch_size = query_tokens.shape[0]
        hidden_size = query_tokens.shape[-1]
        generated_feedback: list[torch.Tensor] = []
        generated_feedback_valid: list[torch.Tensor] = []
        generated_states: list[torch.Tensor] = []
        generated_primitives: dict[str, list[torch.Tensor]] = {
            key: [] for key in feedback_encoder.feedback_dims
        }
        generated_masks: dict[str, list[torch.Tensor]] = {
            key: [] for key in feedback_encoder.feedback_dims
        }
        cache_enabled = not decoder.training if use_cache is None else use_cache
        selected_heads_enabled = (
            not primitive_heads.training
            if use_selected_heads is None
            else use_selected_heads
        )
        head_selector = getattr(sampler, "required_likelihood_ids", None)
        incremental_cache = (
            decoder.initialize_incremental_cache(
                query_tokens,
                memory,
                memory_mask,
                mode=mode,
                relation_adjacency=relation_adjacency,
                relation_type_lags=relation_type_lags,
            )
            if cache_enabled
            else None
        )
        for position in range(self.target_count):
            if incremental_cache is not None:
                current = decoder.incremental_step(incremental_cache)
            else:
                if generated_feedback:
                    prefix = torch.stack(generated_feedback, dim=1)
                else:
                    prefix = query_tokens.new_zeros((batch_size, 0, hidden_size))
                padding = query_tokens.new_zeros(
                    (batch_size, self.target_count - position, hidden_size)
                )
                context = torch.cat((prefix, padding), dim=1).reshape(
                    batch_size,
                    self.block_count,
                    self.field_count,
                    hidden_size,
                )
                available = (
                    torch.arange(self.target_count, device=query_tokens.device)
                    .lt(position)
                    .view(1, self.block_count, self.field_count)
                    .expand(batch_size, -1, -1)
                )
                current = decoder(
                    query_tokens,
                    memory,
                    memory_mask,
                    mode=mode,
                    context_states=context,
                    context_mask=available,
                    relation_adjacency=relation_adjacency,
                    relation_type_lags=relation_type_lags,
                    query_positions=torch.tensor([position], device=query_tokens.device),
                )[:, 0]
            block_id, field_order = divmod(position, self.field_count)
            if selected_heads_enabled and callable(head_selector):
                required_likelihoods = tuple(head_selector(block_id, field_order))
                current_parameters = primitive_heads.forward_selected(
                    current,
                    required_likelihoods,
                )
            else:
                current_parameters = primitive_heads(current)
            sampled_primitives, sampled_masks = sampler(
                block_id,
                field_order,
                current_parameters,
            )
            feedback, feedback_valid = feedback_encoder(
                sampled_primitives,
                sampled_masks,
                leading_shape=(batch_size,),
            )
            generated_feedback_valid.append(feedback_valid)
            if incremental_cache is not None:
                decoder.append_incremental_context(incremental_cache, feedback)
            generated_feedback.append(feedback)
            generated_states.append(current)
            for key in generated_primitives:
                value = sampled_primitives[key]
                mask = sampled_masks[key]
                width = feedback_encoder.feedback_dims[key]
                if mask.shape == (batch_size,):
                    mask = mask.unsqueeze(-1).expand(batch_size, width)
                generated_primitives[key].append(value)
                generated_masks[key].append(mask)
        feedback_contract = torch.stack(generated_feedback_valid, dim=1).all()
        feedback_error = (
            "sampler must return at least one valid primitive for every generated field"
        )
        if query_tokens.device.type == "cuda":
            # Validate the whole 6x29 trajectory once without serializing every
            # generated position through a CUDA-to-host boolean conversion.
            torch._assert_async(feedback_contract, feedback_error)
        elif not bool(feedback_contract.item()):
            raise ValueError(feedback_error)
        field_states = torch.stack(generated_states, dim=1).reshape(
            batch_size,
            self.block_count,
            self.field_count,
            hidden_size,
        )
        primitive_outputs = {
            key: torch.stack(values, dim=1).reshape(
                batch_size,
                self.block_count,
                self.field_count,
                feedback_encoder.feedback_dims[key],
            )
            for key, values in generated_primitives.items()
        }
        primitive_masks = {
            key: torch.stack(values, dim=1).reshape(
                batch_size,
                self.block_count,
                self.field_count,
                feedback_encoder.feedback_dims[key],
            )
            for key, values in generated_masks.items()
        }
        return field_states, primitive_outputs, primitive_masks
