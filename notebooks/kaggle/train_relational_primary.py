from __future__ import annotations


HISTORICAL_ROUTE_ID = "multires_event_v2_m4_relational_primary_v8"
RELATION_V2_ROUTE_ID = "multires_event_v2_m4_relation_v2"
HOSTED_ROUTE_STATUS = "pending"
DISABLED_MESSAGE = (
    "HISTORICAL_RELATIONAL_PRIMARY_DISABLED: the v8 Kaggle entrypoint does not "
    "implement Relation Contract V2 (52 target-target edges, 39 input-target "
    "edges, and 91 edge-specific parameters). The Relation V2 hosted route is "
    "pending a separately frozen notebook and bundle; no hosted training is "
    "authorized from this file."
)


def main() -> None:
    """Fail closed before importing training code or resolving any config."""

    raise RuntimeError(DISABLED_MESSAGE)


if __name__ == "__main__":
    main()
