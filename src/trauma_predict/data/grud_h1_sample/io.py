from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, TextIO


@dataclass(frozen=True)
class PointEvent:
    field: str
    clinical_hour: float
    available_hour: float
    value: float
    unit: str
    source_table: str
    source_id: str
    source_condition: str

    @property
    def hour(self) -> float:
        """Compatibility alias for audit code; block geometry uses clinical_hour."""
        return self.clinical_hour


@dataclass(frozen=True)
class IntervalEvent:
    field: str
    start_hour: float
    end_hour: float
    available_hour: float
    value: float | None
    unit: str
    condition: str
    source_table: str
    source_id: str

    @property
    def amount(self) -> float | None:
        return self.value


@dataclass(frozen=True)
class CxrEvent:
    study_id: str
    clinical_hour: float
    available_hour: float
    labels: tuple[str, ...]


@dataclass
class StayData:
    source_dir: Path
    subject_id: str
    hadm_id: str
    stay_id: str
    sample_key: str
    icu_intime: str
    icu_outtime: str
    available_until_hour: float
    static: dict[str, Any]
    points: dict[str, list[PointEvent]]
    intervals: dict[str, list[IntervalEvent]]
    cxr_events: list[CxrEvent]
    source_counts: dict[str, int]
    episode_start_cache: dict[tuple[str, str], tuple[IntervalEvent, ...]] = dataclass_field(
        default_factory=dict, repr=False
    )


POINT_COLUMNS = (
    "field", "clinical_hour", "available_hour", "value", "unit", "source_table", "source_id", "source_condition"
)
INTERVAL_COLUMNS = (
    "field", "start_hour", "end_hour", "available_hour", "value", "unit", "condition", "source_table", "source_id"
)
CXR_COLUMNS = ("study_id", "clinical_hour", "available_hour", "label")
MANIFEST_SCHEMA = "multires_field_ready_manifest_v1"
STATIC_SCHEMA = "multires_static_context_v1"
FIELD_READY_CONTRACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "contracts"
    / "grud_h1_baseline"
    / "field_ready_contract_v1.json"
)


def load_stay(path: Path) -> StayData:
    paths = {
        "static": _required_path(path, "static_context.json"),
        "points": _required_path(path, "point_events.csv.gz"),
        "intervals": _required_path(path, "interval_events.csv.gz"),
        "cxr": _required_path(path, "cxr_events.csv.gz"),
        "manifest": _required_path(path, "manifest.json"),
    }
    manifest = read_json(paths["manifest"])
    static_payload = read_json(paths["static"])
    if not isinstance(manifest, dict) or not isinstance(static_payload, dict):
        raise ValueError("field-ready manifest/static_context must be JSON objects")
    _validate_manifest(path, manifest, paths)
    keys = manifest["source_keys"]
    if static_payload.get("schema") != STATIC_SCHEMA:
        raise ValueError(f"static_context schema mismatch: {static_payload.get('schema')!r}")
    if static_payload.get("source_keys") != keys:
        raise ValueError("static_context source_keys do not match manifest")
    points: dict[str, list[PointEvent]] = {}
    intervals: dict[str, list[IntervalEvent]] = {}
    source_counts: dict[str, int] = {}

    with _open_csv(paths["points"]) as handle:
        reader = csv.DictReader(handle)
        _require_headers(paths["points"], reader.fieldnames, POINT_COLUMNS)
        point_count = 0
        for row_number, row in enumerate(reader, start=2):
            field = str(row["field"]).strip()
            if not field:
                raise ValueError(f"empty point field in {paths['points']}:{row_number}")
            clinical_hour = _required_finite(row["clinical_hour"], "clinical_hour", paths["points"], row_number)
            available_hour = _required_finite(row["available_hour"], "available_hour", paths["points"], row_number)
            value = _required_finite(row["value"], "value", paths["points"], row_number)
            if available_hour < clinical_hour:
                raise ValueError(
                    f"point available_hour precedes clinical_hour in {paths['points']}:{row_number}"
                )
            event = PointEvent(
                field=field,
                clinical_hour=clinical_hour,
                available_hour=available_hour,
                value=value,
                unit=str(row.get("unit") or ""),
                source_table=str(row.get("source_table") or ""),
                source_id=str(row.get("source_id") or ""),
                source_condition=str(row.get("source_condition") or "NONE"),
            )
            points.setdefault(field, []).append(event)
            source_counts[field] = source_counts.get(field, 0) + 1
            point_count += 1

    with _open_csv(paths["intervals"]) as handle:
        reader = csv.DictReader(handle)
        _require_headers(paths["intervals"], reader.fieldnames, INTERVAL_COLUMNS)
        interval_count = 0
        for row_number, row in enumerate(reader, start=2):
            field = str(row["field"]).strip()
            if not field:
                raise ValueError(f"empty interval field in {paths['intervals']}:{row_number}")
            start_hour = _required_finite(row["start_hour"], "start_hour", paths["intervals"], row_number)
            end_hour = _required_finite(row["end_hour"], "end_hour", paths["intervals"], row_number)
            if end_hour < start_hour:
                raise ValueError(f"interval end precedes start in {paths['intervals']}:{row_number}")
            available_hour = _required_finite(row["available_hour"], "available_hour", paths["intervals"], row_number)
            value = _required_finite(row["value"], "value", paths["intervals"], row_number)
            condition = str(row["condition"]).strip()
            if not condition:
                raise ValueError(f"empty interval condition in {paths['intervals']}:{row_number}")
            event = IntervalEvent(
                field=field,
                start_hour=start_hour,
                end_hour=end_hour,
                available_hour=available_hour,
                value=value,
                unit=str(row.get("unit") or ""),
                condition=condition,
                source_table=str(row.get("source_table") or ""),
                source_id=str(row.get("source_id") or ""),
            )
            intervals.setdefault(field, []).append(event)
            source_counts[field] = source_counts.get(field, 0) + 1
            interval_count += 1

    cxr_rows: dict[str, dict[str, Any]] = {}
    with _open_csv(paths["cxr"]) as handle:
        reader = csv.DictReader(handle)
        _require_headers(paths["cxr"], reader.fieldnames, CXR_COLUMNS)
        cxr_count = 0
        for row_number, row in enumerate(reader, start=2):
            study_id = str(row["study_id"]).strip()
            label = str(row["label"]).strip()
            if not study_id or not label:
                raise ValueError(f"empty CXR study_id/label in {paths['cxr']}:{row_number}")
            clinical_hour = _required_finite(row["clinical_hour"], "clinical_hour", paths["cxr"], row_number)
            available_hour = _required_finite(row["available_hour"], "available_hour", paths["cxr"], row_number)
            if available_hour < clinical_hour:
                raise ValueError(f"CXR available_hour precedes clinical_hour in {paths['cxr']}:{row_number}")
            current = cxr_rows.setdefault(
                study_id,
                {"clinical_hour": clinical_hour, "available_hour": available_hour, "labels": set()},
            )
            if abs(float(current["clinical_hour"]) - clinical_hour) > 1e-9:
                raise ValueError(f"inconsistent CXR clinical_hour for study {study_id}")
            current["available_hour"] = max(float(current["available_hour"]), available_hour)
            current["labels"].add(label)
            cxr_count += 1
    cxr_events = [
        CxrEvent(
            study_id=study_id,
            clinical_hour=float(payload["clinical_hour"]),
            available_hour=float(payload["available_hour"]),
            labels=tuple(sorted(payload["labels"])),
        )
        for study_id, payload in cxr_rows.items()
    ]
    if cxr_events:
        source_counts["cxr"] = cxr_count

    for values in points.values():
        values.sort(key=lambda event: (event.clinical_hour, event.available_hour, event.source_id))
    for values in intervals.values():
        values.sort(key=lambda event: (event.start_hour, event.end_hour, event.condition, event.source_id))
    cxr_events.sort(key=lambda event: (event.clinical_hour, event.study_id))

    expected_counts = {"point_events": point_count, "interval_events": interval_count, "cxr_events": cxr_count}
    if manifest.get("counts") != expected_counts:
        raise ValueError(f"manifest count mismatch: {manifest.get('counts')} != {expected_counts}")
    intime = str(manifest["icu_intime"])
    outtime = str(manifest["icu_outtime"])
    available_until = _required_finite(manifest["available_until_hour"], "available_until_hour", paths["manifest"], 1)
    static = static_payload.get("static")
    if not isinstance(static, dict):
        raise ValueError("static_context.static must be an object")
    return StayData(
        source_dir=path,
        subject_id=str(keys["subject_id"]),
        hadm_id=str(keys["hadm_id"]),
        stay_id=str(keys["stay_id"]),
        sample_key=str(keys["sample_key"]),
        icu_intime=intime,
        icu_outtime=outtime,
        available_until_hour=float(available_until),
        static=dict(static or {}),
        points=points,
        intervals=intervals,
        cxr_events=cxr_events,
        source_counts=source_counts,
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_datetime(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported datetime: {value}")


def _validate_manifest(root: Path, manifest: dict[str, Any], paths: dict[str, Path]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"field-ready manifest schema mismatch under {root}: {manifest.get('schema')!r}")
    if manifest.get("sample_unit") != "icu_stay" or manifest.get("cxr_input_only") is not True:
        raise ValueError("field-ready manifest sample_unit/CXR contract mismatch")
    keys = manifest.get("source_keys")
    required_keys = {"subject_id", "hadm_id", "stay_id", "sample_key"}
    if not isinstance(keys, dict) or set(keys) != required_keys or any(not str(keys[key]).strip() for key in required_keys):
        raise ValueError(f"field-ready manifest source_keys mismatch: {keys!r}")
    for name in ("icu_intime", "icu_outtime"):
        if not str(manifest.get(name) or "").strip():
            raise ValueError(f"field-ready manifest lacks {name}")
        parse_datetime(str(manifest[name]))
    _required_finite(manifest.get("available_until_hour"), "available_until_hour", paths["manifest"], 1)
    history_hours = _required_finite(manifest.get("history_hours"), "history_hours", paths["manifest"], 1)
    if history_hours <= 0:
        raise ValueError("field-ready manifest history_hours must be positive")

    contract_version = str(manifest.get("contract_version") or "")
    contract_hash = str(manifest.get("contract_hash") or "")
    if not contract_version or len(contract_hash) != 64 or any(char not in "0123456789abcdef" for char in contract_hash):
        raise ValueError("field-ready manifest lacks a valid contract_version/contract_hash")
    if FIELD_READY_CONTRACT_PATH.is_file():
        expected_contract = read_json(FIELD_READY_CONTRACT_PATH)
        expected_hash = _sha256_file(FIELD_READY_CONTRACT_PATH)
        if contract_version != str(expected_contract.get("version") or "") or contract_hash != expected_hash:
            raise ValueError(
                f"field-ready contract drift: manifest={contract_version}/{contract_hash} "
                f"expected={expected_contract.get('version')}/{expected_hash}"
            )

    expected_files = {
        paths["static"].name: paths["static"],
        paths["points"].name: paths["points"],
        paths["intervals"].name: paths["intervals"],
        paths["cxr"].name: paths["cxr"],
    }
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(expected_files):
        raise ValueError(f"field-ready manifest file set mismatch: {sorted(files) if isinstance(files, dict) else files!r}")
    for name, file_path in expected_files.items():
        declared = files.get(name)
        if not isinstance(declared, dict) or declared.get("sha256") != _sha256_file(file_path):
            raise ValueError(f"field-ready file hash mismatch: {name}")


def _require_headers(path: Path, actual: list[str] | None, expected: tuple[str, ...]) -> None:
    if tuple(actual or ()) != expected:
        raise ValueError(f"field-ready headers mismatch in {path}: {tuple(actual or ())} != {expected}")


def _required_finite(value: Any, name: str, path: Path, row_number: int) -> float:
    numeric = parse_float(value)
    if numeric is None or not math.isfinite(numeric):
        raise ValueError(f"missing/non-finite {name} in {path}:{row_number}")
    return numeric


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_path(root: Path, *names: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing field-ready file under {root}; expected one of {list(names)}")


@contextmanager
def _open_csv(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield handle
