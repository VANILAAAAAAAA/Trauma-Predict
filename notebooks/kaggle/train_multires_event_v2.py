from __future__ import annotations


HISTORICAL_ROUTE_ID = "multires_event_v2_m4_v8_modes"
RELATION_V2_ROUTE_ID = "multires_event_v2_m4_relation_v2"
HOSTED_ROUTE_STATUS = "pending"
DISABLED_MESSAGE = (
    "HISTORICAL_MULTIRES_EVENT_V2_ENTRYPOINT_DISABLED: this v8 mode/capacity "
    "entrypoint cannot run Relation Contract V2. No Relation V2 hosted route is "
    "frozen or authorized."
)


def main() -> None:
    """Fail closed without resolving a deleted v8 config or importing training."""

    raise RuntimeError(DISABLED_MESSAGE)


if __name__ == "__main__":
    main()
