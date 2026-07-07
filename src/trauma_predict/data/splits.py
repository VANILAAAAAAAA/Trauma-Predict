from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping


def assert_patient_level_split(rows: Iterable[Mapping[str, object]]) -> None:
    subject_to_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subject_id = str(row.get("subject_id") or "")
        split = str(row.get("split") or "")
        if not subject_id:
            raise ValueError("sample row missing subject_id")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split for subject_id={subject_id}: {split}")
        subject_to_split[subject_id].add(split)

    leaked = {subject_id: splits for subject_id, splits in subject_to_split.items() if len(splits) > 1}
    if leaked:
        preview = sorted(leaked.items())[:5]
        raise ValueError(f"subject_id appears in multiple splits: {preview}")
