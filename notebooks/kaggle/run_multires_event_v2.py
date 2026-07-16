from __future__ import annotations


HISTORICAL_ROUTE_ID = "multires_event_v2_m4_v8_modes"
RELATION_V2_ROUTE_ID = "multires_event_v2_m4_relation_v2"
HOSTED_ROUTE_STATUS = "pending"
HISTORICAL_DISABLED_MESSAGE = (
    "HISTORICAL_MULTIRES_EVENT_V2_KAGGLE_DISABLED: the v8 mode/promotion launcher "
    "does not implement Relation Contract V2. No Relation V2 hosted route is "
    "frozen or authorized."
)


def main() -> None:
    """Fail closed without exposing a mode, promotion, config, or training API."""

    raise RuntimeError(HISTORICAL_DISABLED_MESSAGE)


if __name__ == "__main__":
    main()
