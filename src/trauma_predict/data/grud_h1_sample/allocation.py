from __future__ import annotations

from dataclasses import asdict, dataclass


MAX_HISTORY_HOURS = 312


@dataclass(frozen=True)
class TimeBlock:
    time_block: str
    side: str
    role: str
    resolution: str
    start_hour: int
    end_hour: int
    relative_start_hour: int
    relative_end_hour: int

    @property
    def span_hours(self) -> int:
        return self.end_hour - self.start_hour

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["span_hours"] = self.span_hours
        return payload


@dataclass(frozen=True)
class H1Allocation:
    prediction_hour: int
    history_start_hour: int
    history_hours: int
    max_history_hours: int
    blocks: tuple[TimeBlock, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "resolution": "H1",
            "prediction_hour": self.prediction_hour,
            "history_start_hour": self.history_start_hour,
            "history_end_hour": self.prediction_hour,
            "history_hours": self.history_hours,
            "max_history_hours": self.max_history_hours,
            "block_count": len(self.blocks),
            "block_id_semantics": "zero_based_chronological_H1_from_history_start",
        }


def format_time_block_id(
    side: str,
    resolution: str,
    relative_start_hour: int,
    relative_end_hour: int,
) -> str:
    if side != "input":
        raise ValueError(f"GRU-D H1 sidecar only supports input blocks, found {side!r}")
    return f"IN_{resolution}_R{relative_start_hour:+04d}_R{relative_end_hour:+04d}"


def allocate_h1_input_blocks(
    prediction_hour: int,
    *,
    max_history_hours: int = MAX_HISTORY_HOURS,
) -> H1Allocation:
    prediction_hour = int(prediction_hour)
    max_history_hours = int(max_history_hours)
    if prediction_hour < 1:
        raise ValueError("prediction_hour must be positive")
    if max_history_hours < 1:
        raise ValueError("max_history_hours must be positive")
    history_start = max(0, prediction_hour - max_history_hours)
    blocks = tuple(
        TimeBlock(
            time_block=format_time_block_id(
                "input",
                "H1",
                block_start - prediction_hour,
                block_start + 1 - prediction_hour,
            ),
            side="input",
            role="HISTORY_H1",
            resolution="H1",
            start_hour=block_start,
            end_hour=block_start + 1,
            relative_start_hour=block_start - prediction_hour,
            relative_end_hour=block_start + 1 - prediction_hour,
        )
        for block_start in range(history_start, prediction_hour)
    )
    allocation = H1Allocation(
        prediction_hour=prediction_hour,
        history_start_hour=history_start,
        history_hours=prediction_hour - history_start,
        max_history_hours=max_history_hours,
        blocks=blocks,
    )
    validate_h1_partition(allocation)
    return allocation


def validate_h1_partition(allocation: H1Allocation) -> None:
    blocks = allocation.blocks
    if not blocks:
        raise ValueError("H1 allocation has no blocks")
    if blocks[0].start_hour != allocation.history_start_hour:
        raise ValueError("H1 allocation starts at the wrong history hour")
    if blocks[-1].end_hour != allocation.prediction_hour:
        raise ValueError("H1 allocation does not end at the prediction anchor")
    if len(blocks) != allocation.history_hours or len(blocks) > allocation.max_history_hours:
        raise ValueError("H1 allocation length differs from the frozen history contract")
    for expected_id, block in enumerate(blocks):
        if block.span_hours != 1 or block.resolution != "H1" or block.side != "input":
            raise ValueError(f"Invalid H1 block at position {expected_id}: {block}")
        if expected_id and blocks[expected_id - 1].end_hour != block.start_hour:
            raise ValueError("H1 input blocks contain a gap or overlap")


__all__ = [
    "H1Allocation",
    "MAX_HISTORY_HOURS",
    "TimeBlock",
    "allocate_h1_input_blocks",
    "format_time_block_id",
    "validate_h1_partition",
]
