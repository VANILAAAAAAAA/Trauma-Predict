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
from .relation_contract import (
    INPUT_TARGET_RELATION_EDGE_COUNT,
    RELATION_CONTRACT_VERSION,
    RELATION_FIELD_COUNT,
    TARGET_RELATION_EDGE_COUNT,
    MultiresEventV2RelationContract,
    RegisteredRelationEdge,
    RelationEvidence,
    RelationField,
)

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
    "MultiresEventV2RelationContract",
    "RegisteredRelationEdge",
    "RelationEvidence",
    "RelationField",
    "RELATION_CONTRACT_VERSION",
    "RELATION_FIELD_COUNT",
    "TARGET_RELATION_EDGE_COUNT",
    "INPUT_TARGET_RELATION_EDGE_COUNT",
    "TargetManifestEntry",
    "TargetShardSpec",
    "preflight_multires_event_v2",
]
