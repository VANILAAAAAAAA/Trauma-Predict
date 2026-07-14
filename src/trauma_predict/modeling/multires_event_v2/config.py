from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, Mapping


TrajectoryMode = Literal["block", "trajectory", "relational"]
VALID_TRAJECTORY_MODES = frozenset({"block", "trajectory", "relational"})


@dataclass(frozen=True)
class MultiResolutionEventV2Config:
    """Architecture contract for the six-block M4 field-state model.

    ``primitive_head_dims`` and ``primitive_feedback_dims`` are deliberately
    contract-driven.  Their identical keys are the enabled likelihood ids; their
    widths respectively describe emitted distribution parameters and raw primitive
    feedback.  Distribution semantics and losses remain outside this structure layer.
    """

    hidden_size: int = 256
    num_attention_heads: int = 8
    trajectory_encoder_layers: int = 4
    target_decoder_layers: int = 3
    block_compressor_layers: int = 1
    block_latent_count: int = 8
    dropout: float = 0.1

    field_vocab_size: int = 38
    operator_vocab_size: int = 11
    condition_vocab_size: int = 64
    role_vocab_size: int = 8
    resolution_vocab_size: int = 4
    static_numeric_fields: int = 4
    static_categorical_fields: int = 5
    static_categorical_vocab_size: int = 32
    study_slot_vocab_size: int = 16
    time_scale_hours: float = 24.0

    future_block_count: int = 6
    target_field_count: int = 29
    # Core registry order: ids 1..28 plus urine_output=35.  Id 29 is the
    # crystalloid auxiliary field and must not silently enter the core target.
    target_field_ids: tuple[int, ...] = (
        1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 7,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 35,
    )
    relation_type_count: int = 14
    mode: TrajectoryMode = "trajectory"
    primitive_head_dims: Mapping[str, int] | tuple[tuple[str, int], ...] = ()
    primitive_feedback_dims: Mapping[str, int] | tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.trajectory_encoder_layers < 1 or self.target_decoder_layers < 1:
            raise ValueError("encoder and target decoder must each have at least one layer")
        if self.block_latent_count < 1 or self.block_compressor_layers < 1:
            raise ValueError("block compressor dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.future_block_count != 6:
            raise ValueError("the V2 target contract is frozen to six M4 blocks")
        if self.target_field_count != 29:
            raise ValueError("the V2 target contract is frozen to 29 registered fields")
        if len(self.target_field_ids) != self.target_field_count:
            raise ValueError("target_field_ids must contain one id per registered target field")
        if len(set(self.target_field_ids)) != len(self.target_field_ids):
            raise ValueError("target_field_ids must be unique and ordered")
        if min(self.target_field_ids, default=0) < 1:
            raise ValueError("target field id zero is reserved for padding")
        if max(self.target_field_ids, default=0) >= self.field_vocab_size:
            raise ValueError("target_field_ids exceeds field_vocab_size")
        if self.relation_type_count < 1:
            raise ValueError("relation_type_count must be positive")
        if self.mode not in VALID_TRAJECTORY_MODES:
            raise ValueError(f"unsupported trajectory mode: {self.mode!r}")
        _validate_primitive_heads(self.primitive_head_dims)
        _validate_primitive_heads(self.primitive_feedback_dims)
        if set(self.primitive_dims) != set(self.feedback_dims):
            raise ValueError(
                "primitive_head_dims and primitive_feedback_dims must use identical "
                "likelihood_id keys"
            )

    @property
    def primitive_dims(self) -> dict[str, int]:
        if isinstance(self.primitive_head_dims, Mapping):
            return {str(name): int(width) for name, width in self.primitive_head_dims.items()}
        return {str(name): int(width) for name, width in self.primitive_head_dims}

    @property
    def feedback_dims(self) -> dict[str, int]:
        if isinstance(self.primitive_feedback_dims, Mapping):
            return {str(name): int(width) for name, width in self.primitive_feedback_dims.items()}
        return {str(name): int(width) for name, width in self.primitive_feedback_dims}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MultiResolutionEventV2Config":
        aliases = {
            "d_model": "hidden_size",
            "num_heads": "num_attention_heads",
            "encoder_layers": "trajectory_encoder_layers",
            "decoder_layers": "target_decoder_layers",
            "query_layers": "target_decoder_layers",
            "block_latents": "block_latent_count",
            "latent_count": "block_latent_count",
            "m4_blocks": "future_block_count",
            "field_count": "target_field_count",
            "relation_types": "relation_type_count",
        }
        accepted = {item.name for item in fields(cls)}
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            name = aliases.get(str(key), str(key))
            if name not in accepted:
                continue
            if name == "target_field_ids":
                item = tuple(int(field_id) for field_id in item)
            normalized[name] = item
        return cls(**normalized)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_mode(mode: str | None, default: TrajectoryMode) -> TrajectoryMode:
    resolved = default if mode is None else str(mode)
    if resolved not in VALID_TRAJECTORY_MODES:
        raise ValueError(f"unsupported trajectory mode: {resolved!r}")
    return resolved  # type: ignore[return-value]


def _validate_primitive_heads(
    value: Mapping[str, int] | tuple[tuple[str, int], ...],
) -> None:
    items = value.items() if isinstance(value, Mapping) else value
    names: set[str] = set()
    for raw_name, raw_width in items:
        name = str(raw_name)
        if not name or "." in name:
            raise ValueError("primitive head names must be non-empty and cannot contain '.'")
        if name in names:
            raise ValueError(f"duplicate primitive head name: {name}")
        if int(raw_width) < 1:
            raise ValueError(f"primitive head {name!r} must have a positive width")
        names.add(name)
