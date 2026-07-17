from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

from .allocation import TimeBlock
from .io import CxrEvent, IntervalEvent, PointEvent, StayData
from .registry import Template


@dataclass(frozen=True)
class AggregateResult:
    value: float | None
    valid: bool
    observed: bool
    source_count: int
    coverage_hours: float | None = None
    note: str | None = None


def aggregate_template(stay: StayData, block: TimeBlock, template: Template) -> AggregateResult:
    name = template.aggregation
    if name in {"last_point", "min_point", "max_point", "mean_point", "count_point"}:
        return _aggregate_points(stay, block, template)
    if name == "hourly_condition_duration":
        return _hourly_condition_duration(stay, block, template)
    if name == "hourly_observed_duration":
        return _hourly_observed_duration(stay, block, template)
    if name == "interval_amount_sum":
        return _interval_amount_sum(stay, block, template)
    if name == "interval_start_count":
        return _interval_start_count(stay, block, template)
    if name == "interval_overlap_duration":
        return _interval_overlap_duration(stay, block, template)
    if name == "interval_end_state":
        return _interval_end_state(stay, block, template)
    if name in {"interval_rate_last", "interval_rate_max", "interval_rate_mean"}:
        return _interval_rate(stay, block, template)
    if name in {"uop_sum", "uop_count"}:
        return _aggregate_uop(stay, block, template)
    if name in {"cxr_study_slots", "cxr_study_count"}:
        raise ValueError("CXR is multi-instance; call aggregate_cxr_template")
    raise ValueError(f"Unsupported aggregation: {name}")


def aggregate_cxr_template(stay: StayData, block: TimeBlock, template: Template) -> list[AggregateResult]:
    studies = _cxr_in_block(stay.cxr_events, block)
    if template.aggregation == "cxr_study_count":
        return [AggregateResult(float(len(studies)), True, bool(studies), len(studies))]
    if template.aggregation != "cxr_study_slots":
        raise ValueError(f"Unsupported CXR aggregation: {template.aggregation}")
    return [
        AggregateResult(float(study_slot), True, True, 1)
        for study_slot, study in enumerate(studies, start=1)
        if template.condition in study.labels
    ]


def should_emit_input(template: Template, result: AggregateResult) -> bool:
    if not result.valid:
        return False
    if template.input_emit == "always":
        return True
    if template.input_emit == "positive_only":
        return result.value is not None and result.value > 0
    if template.input_emit in {"observed_only", "observed_required"}:
        return result.observed
    if template.input_emit == "known_state":
        return result.value in {0, 1, 0.0, 1.0}
    raise ValueError(f"Unknown input_emit: {template.input_emit}")


def _aggregate_points(stay: StayData, block: TimeBlock, template: Template) -> AggregateResult:
    raw_points = _points_in_block(stay.points.get(template.field, []), block)
    points = _valid_points(raw_points, template)
    excluded = len(raw_points) - len(points)
    note = f"excluded_out_of_range={excluded}" if excluded else None
    if template.aggregation == "count_point":
        return AggregateResult(float(len(points)), True, bool(points), len(points), note=note)
    summary_points = points
    if template.domain in {"vitals", "neurologic", "respiratory"} and template.aggregation == "mean_point":
        summary_points = _latest_point_per_hour(points)
    values = [event.value for event in summary_points]
    if not summary_points:
        return AggregateResult(None, False, False, 0, note=note)
    if template.aggregation == "last_point":
        value = summary_points[-1].value
    elif template.aggregation == "min_point":
        value = min(values)
    elif template.aggregation == "max_point":
        value = max(values)
    elif template.aggregation == "mean_point":
        value = fmean(values)
    else:
        raise AssertionError(template.aggregation)
    return AggregateResult(float(value), True, True, len(summary_points), note=note)


def _hourly_condition_duration(stay: StayData, block: TimeBlock, template: Template) -> AggregateResult:
    condition = template.condition_spec or {}
    if condition.get("kind") not in {"numeric_predicate", "numeric_interval"}:
        raise ValueError(f"Condition is not a numeric predicate: {template.condition}")
    raw_points = _points_in_block(stay.points.get(template.field, []), block)
    points = _valid_points(raw_points, template)
    excluded = len(raw_points) - len(points)
    groups = _points_by_hour(points)
    if not groups:
        note = f"excluded_out_of_range={excluded}" if excluded else None
        return AggregateResult(None, False, False, 0, coverage_hours=0.0, note=note)
    reducer = condition.get("within_hour_reducer")
    if reducer not in {"min", "max"}:
        raise ValueError(f"Missing within-hour reducer for {template.condition}")
    representatives = [
        min(event.value for event in values) if reducer == "min" else max(event.value for event in values)
        for values in groups.values()
    ]
    duration = float(sum(_matches_registered_condition(value, condition) for value in representatives))
    note = f"excluded_out_of_range={excluded}" if excluded else None
    return AggregateResult(duration, True, True, len(points), float(len(groups)), note)


def _hourly_observed_duration(stay: StayData, block: TimeBlock, template: Template) -> AggregateResult:
    raw_points = _points_in_block(stay.points.get(template.field, []), block)
    points = _valid_points(raw_points, template)
    excluded = len(raw_points) - len(points)
    observed_hours = len(_points_by_hour(points))
    note = f"excluded_out_of_range={excluded}" if excluded else None
    return AggregateResult(
        float(observed_hours),
        True,
        observed_hours > 0,
        len(points),
        coverage_hours=float(observed_hours),
        note=note,
    )


def _interval_amount_sum(stay: StayData, block: TimeBlock, template: Template) -> AggregateResult:
    total = 0.0
    count = 0
    for event in _intervals_for_template(stay, block, template):
        overlap = _overlap(event.start_hour, event.end_hour, block.start_hour, block.end_hour)
        if overlap <= 0 and not (event.end_hour == event.start_hour and _point_in_block(event.start_hour, block)):
            continue
        if event.value is None:
            continue
        duration = event.end_hour - event.start_hour
        total += event.value * (overlap / duration) if duration > 0 else event.value
        count += 1
    return AggregateResult(float(total), True, count > 0, count)


def _interval_start_count(stay: StayData, block: TimeBlock, template: Template) -> AggregateResult:
    if template.field == "respiratory_support" and not _respiratory_block_evidence(stay, block):
        return AggregateResult(None, False, False, 0, coverage_hours=0.0)
    if template.field in {"respiratory_support", "vasopressor_support"}:
        anchor = _prediction_hour(block)
        starts = [
            event
            for event in _state_episode_starts(stay, template.field, template.condition)
            if _point_in_block(event.start_hour, block)
            and (block.side != "input" or event.available_hour <= anchor + 1e-9)
        ]
    else:
        starts = [
            event
            for event in _intervals_for_template(stay, block, template)
            if _point_in_block(event.start_hour, block)
        ]
    return AggregateResult(float(len(starts)), True, bool(starts), len(starts))


def _interval_overlap_duration(stay: StayData, block: TimeBlock, template: Template) -> AggregateResult:
    if template.field == "respiratory_support" and not _respiratory_block_evidence(stay, block):
        return AggregateResult(None, False, False, 0, coverage_hours=0.0)
    overlaps: list[tuple[float, float]] = []
    for event in _intervals_for_template(stay, block, template):
        start = max(event.start_hour, block.start_hour)
        end = min(event.end_hour, block.end_hour)
        if end > start:
            overlaps.append((start, end))
    duration = _union_duration(overlaps)
    return AggregateResult(float(duration), True, duration > 0, len(overlaps), float(block.span_hours))


def _interval_end_state(stay: StayData, block: TimeBlock, template: Template) -> AggregateResult:
    if template.field == "respiratory_support":
        edge_evidence = _respiratory_edge_evidence(stay, block)
        if not edge_evidence:
            return AggregateResult(None, False, False, 0, coverage_hours=0.0)
        active = [event for event in edge_evidence if event.condition == template.condition]
        return AggregateResult(float(int(bool(active))), True, True, len(active), 1.0)
    active = any(
        event.start_hour <= block.end_hour < event.end_hour
        for event in _intervals_for_template(stay, block, template)
    )
    return AggregateResult(float(int(active)), True, True, 1 if active else 0, 1.0)


def _interval_rate(stay: StayData, block: TimeBlock, template: Template) -> AggregateResult:
    events = [
        event for event in _intervals_for_template(stay, block, template)
        if event.value is not None and event.value >= 0 and event.end_hour > block.start_hour and event.start_hour < block.end_hour
    ]
    if template.aggregation == "interval_rate_last":
        value = sum(
            float(event.value)
            for event in events
            if event.start_hour <= block.end_hour < event.end_hour
        )
        return AggregateResult(float(value), True, bool(events), len(events), float(block.span_hours))
    boundaries = sorted(
        {float(block.start_hour), float(block.end_hour)}
        | {max(float(block.start_hour), event.start_hour) for event in events}
        | {min(float(block.end_hour), event.end_hour) for event in events}
    )
    segments: list[tuple[float, float]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        rate = sum(float(event.value) for event in events if event.start_hour <= midpoint < event.end_hour)
        segments.append((end - start, rate))
    if template.aggregation == "interval_rate_max":
        value = max((rate for _, rate in segments), default=0.0)
    elif template.aggregation == "interval_rate_mean":
        value = sum(duration * rate for duration, rate in segments) / float(block.span_hours)
    else:
        raise AssertionError(template.aggregation)
    return AggregateResult(float(value), True, bool(events), len(events), float(block.span_hours))


def _aggregate_uop(stay: StayData, block: TimeBlock, template: Template) -> AggregateResult:
    points = _points_in_block(stay.points.get("urine_output", []), block)
    observed_hours = len({_event_hour_index(event.clinical_hour) for event in points})
    if template.aggregation == "uop_sum":
        if not points:
            return AggregateResult(None, False, False, 0, float(observed_hours))
        total = float(sum(event.value for event in points))
        return AggregateResult(total, True, True, len(points), float(observed_hours))
    if template.aggregation == "uop_count":
        return AggregateResult(float(len(points)), True, bool(points), len(points), float(observed_hours))
    raise AssertionError(template.aggregation)


def _points_in_block(points: list[PointEvent], block: TimeBlock) -> list[PointEvent]:
    anchor = _prediction_hour(block)
    return [
        event for event in points
        if _point_in_block(event.clinical_hour, block)
        and (block.side != "input" or event.available_hour <= anchor + 1e-9)
    ]


def _intervals_for_template(stay: StayData, block: TimeBlock, template: Template) -> list[IntervalEvent]:
    anchor = _prediction_hour(block)
    condition_kind = (template.condition_spec or {}).get("kind")
    return [
        event for event in stay.intervals.get(template.field, [])
        if (block.side != "input" or event.available_hour <= anchor + 1e-9)
        and (condition_kind not in {"event_subtype", "antibiotic_source"} or event.condition == template.condition)
    ]


def _intervals_for_field(stay: StayData, block: TimeBlock, field: str) -> list[IntervalEvent]:
    anchor = _prediction_hour(block)
    return [
        event for event in stay.intervals.get(field, [])
        if block.side != "input" or event.available_hour <= anchor + 1e-9
    ]


def _respiratory_block_evidence(stay: StayData, block: TimeBlock) -> list[IntervalEvent]:
    return [
        event for event in _intervals_for_field(stay, block, "respiratory_support")
        if _overlap(event.start_hour, event.end_hour, block.start_hour, block.end_hour) > 0
        or _point_in_block(event.start_hour, block)
    ]


def _respiratory_edge_evidence(stay: StayData, block: TimeBlock) -> list[IntervalEvent]:
    edge = float(block.end_hour)
    return [
        event for event in _intervals_for_field(stay, block, "respiratory_support")
        if event.start_hour <= edge < event.end_hour
        or (event.start_hour == edge and event.end_hour == edge)
    ]


def _state_episode_starts(stay: StayData, field: str, condition: str) -> tuple[IntervalEvent, ...]:
    """Return one onset row per continuous/overlapping state episode.

    Availability segments remain intact for duration and end-state aggregation.  The onset row
    retains the availability time of the earliest clinical segment, so a later-visible
    continuation cannot be mislabeled as a newly started episode in an input sample.
    """
    key = (field, condition)
    cached = stay.episode_start_cache.get(key)
    if cached is not None:
        return cached
    events = sorted(
        (event for event in stay.intervals.get(field, []) if event.condition == condition),
        key=lambda event: (event.start_hour, event.end_hour, event.available_hour, event.source_id),
    )
    if not events:
        stay.episode_start_cache[key] = ()
        return ()
    starts: list[IntervalEvent] = []
    onset = events[0]
    episode_end = events[0].end_hour
    for event in events[1:]:
        if event.start_hour <= episode_end + 1e-9:
            episode_end = max(episode_end, event.end_hour)
            if abs(event.start_hour - onset.start_hour) <= 1e-9 and (
                event.available_hour,
                event.end_hour,
                event.source_id,
            ) < (
                onset.available_hour,
                onset.end_hour,
                onset.source_id,
            ):
                onset = event
            continue
        starts.append(onset)
        onset = event
        episode_end = event.end_hour
    starts.append(onset)
    result = tuple(starts)
    stay.episode_start_cache[key] = result
    return result


def _cxr_in_block(events: list[CxrEvent], block: TimeBlock) -> list[CxrEvent]:
    anchor = _prediction_hour(block)
    return [
        event for event in events
        if _point_in_block(event.clinical_hour, block)
        and block.side == "input"
        and event.available_hour <= anchor + 1e-9
    ]


def _valid_points(points: list[PointEvent], template: Template) -> list[PointEvent]:
    if template.valid_min is None or template.valid_max is None:
        return points
    return [event for event in points if template.valid_min <= event.value <= template.valid_max]


def _points_by_hour(points: list[PointEvent]) -> dict[int, list[PointEvent]]:
    grouped: dict[int, list[PointEvent]] = {}
    for event in points:
        grouped.setdefault(_event_hour_index(event.clinical_hour), []).append(event)
    return grouped


def _latest_point_per_hour(points: list[PointEvent]) -> list[PointEvent]:
    latest: dict[int, PointEvent] = {}
    for event in points:
        hour_index = _event_hour_index(event.clinical_hour)
        current = latest.get(hour_index)
        if current is None or (event.clinical_hour, event.available_hour, event.source_id) >= (
            current.clinical_hour, current.available_hour, current.source_id
        ):
            latest[hour_index] = event
    return [latest[index] for index in sorted(latest)]


def _matches_registered_condition(value: float, condition: dict[str, object]) -> bool:
    if condition["kind"] == "numeric_interval":
        lower = float(condition["lower"])
        upper = float(condition["upper"])
        lower_ok = value >= lower if condition.get("lower_inclusive") else value > lower
        upper_ok = value <= upper if condition.get("upper_inclusive") else value < upper
        return lower_ok and upper_ok
    threshold = float(condition["threshold"])
    comparator = condition["comparator"]
    return {
        "<": value < threshold,
        "<=": value <= threshold,
        ">": value > threshold,
        ">=": value >= threshold,
    }[str(comparator)]


def _point_in_block(hour: float, block: TimeBlock) -> bool:
    if block.side == "input" and block.start_hour == 0 and hour == 0:
        return True
    return block.start_hour < hour <= block.end_hour


def _event_hour_index(hour: float) -> int:
    return 0 if hour <= 0 else int(math.ceil(hour) - 1)


def _prediction_hour(block: TimeBlock) -> float:
    return float(block.start_hour - block.relative_start_hour)


def _overlap(start: float, end: float, block_start: float, block_end: float) -> float:
    return max(0.0, min(end, block_end) - max(start, block_start))


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start
