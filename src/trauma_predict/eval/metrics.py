from __future__ import annotations

from collections.abc import Iterable


def mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    if count == 0:
        raise ValueError("cannot compute mean of an empty iterable")
    return total / count
