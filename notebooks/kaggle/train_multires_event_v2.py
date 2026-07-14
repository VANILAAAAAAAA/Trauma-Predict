from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch.distributed.elastic.multiprocessing.errors import record  # noqa: E402
from trauma_predict.training.multires_event_v2 import (  # noqa: E402
    build_multires_event_v2_model,
    build_multires_event_v2_runtime,
    load_multires_event_v2_configs,
    require_multires_event_v2_training_authorization,
    resolve_repo_path,
    run_multires_event_v2_capacity_gated_training,
    run_multires_event_v2_rank_artifact_preflight_only,
    run_multires_event_v2_training,
    run_multires_event_v2_verification_probe,
)
from trauma_predict.training.config import load_yaml_config  # noqa: E402
from trauma_predict.training.observability import (  # noqa: E402
    atomic_write_json,
    utc_now,
)


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch/DDP entry point for the six-block M4 relational primary."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--capacity-probe-output", type=Path)
    parser.add_argument("--elapsed-before-capacity-seconds", type=float)
    parser.add_argument("--verification-only", action="store_true")
    parser.add_argument("--rank-artifact-preflight-output", type=Path)
    parser.add_argument(
        "--rank-artifact-preflight-mode",
        choices=("block", "trajectory", "relational"),
    )
    args = parser.parse_args()
    early_preflight = args.rank_artifact_preflight_output is not None or (
        args.rank_artifact_preflight_mode is not None
    )
    if early_preflight:
        if args.rank_artifact_preflight_output is None or (
            args.rank_artifact_preflight_mode is None
        ):
            parser.error(
                "--rank-artifact-preflight-output and "
                "--rank-artifact-preflight-mode must be paired"
            )
        if (
            args.config is not None
            or args.dry_run
            or args.capacity_probe_output is not None
            or args.elapsed_before_capacity_seconds is not None
            or args.verification_only
        ):
            parser.error(
                "rank artifact preflight is a config-free pre-Dataset action"
            )
        return args
    if args.config is None:
        parser.error("--config is required outside rank artifact preflight")
    if args.dry_run and args.capacity_probe_output is not None:
        parser.error("--dry-run and --capacity-probe-output are mutually exclusive")
    if (args.capacity_probe_output is None) != (
        args.elapsed_before_capacity_seconds is None
    ):
        parser.error(
            "--capacity-probe-output and --elapsed-before-capacity-seconds must be paired"
        )
    if args.verification_only and args.capacity_probe_output is None:
        parser.error("--verification-only requires the paired capacity-probe arguments")
    return args


@record
def main() -> None:
    args = parse_args()
    if args.rank_artifact_preflight_output is not None:
        run_multires_event_v2_rank_artifact_preflight_only(
            output_dir=args.rank_artifact_preflight_output,
            mode=str(args.rank_artifact_preflight_mode),
        )
        return
    assert args.config is not None
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if args.dry_run:
        run_dry_preflight(config_path)
        return
    train = load_yaml_config(config_path.resolve())
    if args.verification_only:
        run_multires_event_v2_verification_probe(
            config_path.resolve(),
            repo_root=REPO_ROOT,
            output_dir=args.capacity_probe_output,
            elapsed_before_capacity_seconds=float(args.elapsed_before_capacity_seconds),
        )
        return
    require_multires_event_v2_training_authorization(train)
    if args.capacity_probe_output is not None:
        run_multires_event_v2_capacity_gated_training(
            config_path.resolve(),
            repo_root=REPO_ROOT,
            capacity_output_dir=args.capacity_probe_output,
            elapsed_before_capacity_seconds=float(args.elapsed_before_capacity_seconds),
        )
        return
    run_multires_event_v2_training(config_path.resolve(), repo_root=REPO_ROOT)


def run_dry_preflight(config_path: Path) -> dict[str, Any]:
    """Materialize the exact joined runtime without initializing CUDA/DDP.

    This is intentionally a data-path preflight, not a light YAML check: it
    verifies both frozen artifacts, their content-hash join, the train-only
    normalization authority, the 414-factor batch contract, and the model
    parameterization before a hosted optimizer process is launched.
    """

    train, dataset, model_config, dataset_path, model_path = load_multires_event_v2_configs(
        config_path.resolve(), repo_root=REPO_ROOT
    )
    lab_scale = verify_repo_lab_scale_artifact(train)
    output_dir = resolve_repo_path(str(train["outputs"]["output_dir"]), REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = build_multires_event_v2_runtime(
        train,
        dataset,
        repo_root=REPO_ROOT,
        rank=0,
        world_size=1,
        phase="interval",
    )
    first_batch = next(iter(runtime.train_loader))
    metadata = _mapping(first_batch.get("target_primitive_metadata"), "batch metadata")
    factor_order = metadata.get("factor_order")
    if not isinstance(factor_order, (list, tuple)) or len(factor_order) != 414:
        raise ValueError("V2 dry-run batch must expose the frozen 414-factor order")
    if tuple(metadata.get("block_order") or ()) != tuple(f"M4_{index:02d}" for index in range(1, 7)):
        raise ValueError("V2 dry-run batch does not contain exactly six future M4 blocks")
    if len(tuple(metadata.get("field_order") or ())) != 29:
        raise ValueError("V2 dry-run batch does not contain the registered 29-field order")

    model = build_multires_event_v2_model(model_config, mode=str(train["mode"]))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    snapshot = {
        "schema_version": "trauma_predict.multires_event_v2_dry_run.v1",
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path.resolve()),
        "dataset_config_path": str(dataset_path),
        "model_config_path": str(model_path),
        "mode": str(train["mode"]),
        "run_name": str(train["run_name"]),
        "parameter_count": int(parameter_count),
        "runtime_identity": dict(runtime.identity),
        "contract_bundle_hash": runtime.contract.contract_bundle_hash,
        "process_contract_sha256": runtime.contract.contract_hashes["process"],
        "emission_contract_sha256": runtime.contract.contract_hashes["emission"],
        "projection_contract_sha256": runtime.contract.contract_hashes["projection"],
        "relation_contract_sha256": runtime.contract.contract_hashes["relation"],
        "sidecar_schema_sha256": runtime.contract.contract_hashes["sidecar_schema"],
        "lab_scale_artifact": lab_scale,
        "standardized_primitive_scale_artifact": {
            "path": runtime.identity["standardized_primitive_scale_artifact"],
            "sha256": runtime.identity["standardized_primitive_scale_sha256"],
            "fit_split": "train",
        },
        "batch_contract": {
            "future_blocks": 6,
            "registered_fields": 29,
            "stochastic_primitive_factors": len(factor_order),
            "relation_types": len(tuple(metadata.get("relation_types") or ())),
            "active_relation_edges": int(
                _mapping(metadata.get("relation_edge_counts"), "relation edge counts")[
                    "active_core"
                ]
            ),
        },
        "train": train,
        "dataset": dataset,
        "model": model_config,
    }
    atomic_write_json(output_dir / "dry_run_preflight.json", snapshot)
    print(
        "MULTIRES_EVENT_V2_PREFLIGHT_OK",
        json.dumps(
            {
                "run_name": train["run_name"],
                "mode": train["mode"],
                "base_dataset_id": runtime.identity["base_dataset_id"],
                "target_dataset_id": runtime.identity["target_dataset_id"],
                "counts": runtime.identity["counts"],
                "contract_bundle_hash": runtime.contract.contract_bundle_hash,
                "relation_contract_sha256": runtime.contract.contract_hashes[
                    "relation"
                ],
                "sidecar_schema_sha256": runtime.contract.contract_hashes[
                    "sidecar_schema"
                ],
                "lab_scale_artifact_sha256": lab_scale["sha256"],
                "standardized_primitive_scale_sha256": runtime.identity[
                    "standardized_primitive_scale_sha256"
                ],
                "parameter_count": parameter_count,
                "stochastic_primitive_factors": len(factor_order),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return snapshot


def verify_repo_lab_scale_artifact(train: Mapping[str, Any]) -> dict[str, Any]:
    value = train.get("lab_scale_artifact")
    expected = str(train.get("lab_scale_artifact_hash") or "")
    if not value:
        raise ValueError("V2 train config must freeze lab_scale_artifact")
    if not SHA256_PATTERN.fullmatch(expected):
        raise ValueError("V2 train config must freeze a lowercase lab_scale_artifact_hash")
    path = resolve_repo_path(str(value), REPO_ROOT)
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("V2 lab scale artifact must be a repository file") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "content_sha256"},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    observed = hashlib.sha256(canonical).hexdigest()
    if observed != expected:
        raise ValueError(f"V2 lab scale artifact hash mismatch: {observed} != {expected}")
    if payload.get("content_sha256") != observed:
        raise ValueError("V2 lab scale artifact self hash mismatch")
    if payload.get("schema") != "multires_event_v2_lab_affine_scale_v1":
        raise ValueError("V2 lab scale artifact schema mismatch")
    if payload.get("fit_split") != "train" or payload.get("status") != "frozen_train_only_fit":
        raise ValueError("V2 lab scale artifact must be fit on train subjects only")
    return {
        "path": str(path),
        "sha256": observed,
        "schema": payload["schema"],
        "fit_split": payload["fit_split"],
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


if __name__ == "__main__":
    main()
