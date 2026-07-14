"""Strict V1-input/V2-target data boundary for M4 trajectory modeling."""

from .collator import LIKELIHOOD_SPECS, MultiresEventV2Collator
from .contract import (
    BLOCK_IDS,
    DETERMINISTIC_PROJECTIONS_PER_BLOCK,
    EXPECTED_CORE_FIELD_COUNT,
    EXPECTED_ENABLED_FACTOR_COUNT,
    CoreRelationEdge,
    MultiresEventV2Contract,
)
from .dataset import MultiresEventV2Dataset, TargetManifestEntry, TargetShardSpec
from .preflight import MultiresEventV2PreflightResult, preflight_multires_event_v2

__all__ = [
    "BLOCK_IDS",
    "DETERMINISTIC_PROJECTIONS_PER_BLOCK",
    "EXPECTED_CORE_FIELD_COUNT",
    "EXPECTED_ENABLED_FACTOR_COUNT",
    "CoreRelationEdge",
    "LIKELIHOOD_SPECS",
    "MultiresEventV2Collator",
    "MultiresEventV2Contract",
    "MultiresEventV2Dataset",
    "MultiresEventV2PreflightResult",
    "TargetManifestEntry",
    "TargetShardSpec",
    "preflight_multires_event_v2",
]
