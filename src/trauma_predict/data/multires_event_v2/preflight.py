from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import BLOCK_IDS, MultiresEventV2Contract, sha256_file
from .dataset import MultiresEventV2Dataset


@dataclass(frozen=True)
class MultiresEventV2PreflightResult:
    dataset_id: str
    dataset_manifest_sha256: str
    contract_bundle_hash: str
    base_dataset_id: str
    base_dataset_fingerprint: str
    split: str
    sample_count: int
    subject_count: int
    shard_count: int
    block_count: int
    core_field_count: int
    enabled_factor_count: int
    relation_total_edges: int
    relation_active_core_edges: int
    relation_deferred_edges: int
    validated_record_count: int
    verified_target_shard_hashes: bool


def preflight_multires_event_v2(
    base_dataset: Any,
    target_root: str | Path,
    *,
    verify_target_shard_hashes: bool = False,
    verify_all_records: bool = False,
    verify_one_record_per_shard: bool = False,
) -> MultiresEventV2PreflightResult:
    """Validate the relocated V1/V2 authority boundary before runtime.

    Contract files and all manifest/authority hashes are always verified.
    Target shard byte hashes and every record are optional because the formal C4
    artifact is large. Even in the fast path, strict joins are executed for the
    first and last sample of the selected split.
    """

    root = Path(target_root).resolve()
    contract = MultiresEventV2Contract.from_dataset_root(
        root,
        verify_contract_hashes=True,
    )
    dataset = MultiresEventV2Dataset(
        base_dataset,
        root,
        contract=contract,
        strict=True,
        verify_shard_hashes=verify_target_shard_hashes,
    )
    if verify_all_records:
        indices = list(range(len(dataset)))
    elif verify_one_record_per_shard:
        first_by_shard: dict[str, int] = {}
        for index, shard_key in enumerate(dataset.shard_keys):
            first_by_shard.setdefault(shard_key, index)
        indices = sorted(set(first_by_shard.values()) | {0, len(dataset) - 1})
    else:
        indices = sorted({0, len(dataset) - 1})
    for index in indices:
        joined = dataset[index]
        if len(joined["target_record"]["blocks"]) != len(BLOCK_IDS):
            raise AssertionError("strict V2 join returned a non-six-block target")
        if any(
            len(block["processes"]) != len(contract.core_fields)
            for block in joined["target_record"]["blocks"]
        ):
            raise AssertionError("strict V2 join returned a non-29-field target block")

    base = contract.manifest["base_dataset"]
    return MultiresEventV2PreflightResult(
        dataset_id=str(contract.manifest["dataset_id"]),
        dataset_manifest_sha256=sha256_file(root / "dataset_manifest.json"),
        contract_bundle_hash=contract.contract_bundle_hash,
        base_dataset_id=str(base["dataset_id"]),
        base_dataset_fingerprint=str(base["fingerprint"]),
        split=dataset.split,
        sample_count=len(dataset),
        subject_count=len(set(dataset.subject_ids)),
        shard_count=len(set(dataset.shard_keys)),
        block_count=len(BLOCK_IDS),
        core_field_count=len(contract.core_fields),
        enabled_factor_count=int(contract.process_registry["scope"]["expanded_enabled_core_primitives"]),
        relation_total_edges=contract.relation_total_edges,
        relation_active_core_edges=contract.relation_active_core_edges,
        relation_deferred_edges=contract.relation_deferred_edges,
        validated_record_count=len(indices),
        verified_target_shard_hashes=bool(verify_target_shard_hashes),
    )
