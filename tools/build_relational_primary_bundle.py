from __future__ import annotations


HISTORICAL_BUNDLE_SCHEMA = "trauma_predict.multires_event_v2_relational_primary_bundle.v2"
RELATION_V2_ROUTE_ID = "multires_event_v2_m4_relation_v2"
HOSTED_ROUTE_STATUS = "pending"
DISABLED_MESSAGE = (
    "HISTORICAL_RELATIONAL_PRIMARY_BUNDLE_BUILD_DISABLED: the v8 bundle builder "
    "cannot package Relation Contract V2. Do not publish or resume a hosted run "
    "until a separate Relation V2 hosted contract, bundle, and notebook are "
    "frozen."
)


def main() -> int:
    """Fail closed without creating a bundle or touching patient artifacts."""

    raise RuntimeError(DISABLED_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(main())
