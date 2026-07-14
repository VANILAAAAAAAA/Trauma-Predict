from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
KAGGLE_SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, KAGGLE_SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_multires_event_v1 as v1_route  # noqa: E402
from trauma_predict.eval.multires_event_v2_free_running import (  # noqa: E402
    evaluate_multires_event_v2_promotion,
)
from trauma_predict.eval.multires_event_v2_promotion_contract import (  # noqa: E402
    load_promotion_metric_contract,
)
from trauma_predict.training.config import load_yaml_config  # noqa: E402
from trauma_predict.training.multires_event_v2 import (  # noqa: E402
    AUTHORIZED_TRAINING_RUN_NAMES,
    CAPACITY_PROBE_OPTIMIZER_STEPS,
    CAPACITY_PROBE_SCHEMA,
    CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR,
    CAPACITY_PROBE_VALIDATION_ANCHORS,
    CAPACITY_SESSION_BUDGET_SECONDS,
    CAPACITY_SESSION_RESERVE_SECONDS,
    CAPACITY_STRUCTURAL_METRICS,
    EXPECTED_OPTIMIZER_CONTRACT,
    OPTIMIZER_CONTRACT_VERSION,
    RAW_JOINT_NLL_REDUCTION,
    TRAINING_AUTHORIZATION_REASON as CORE_TRAINING_AUTHORIZATION_REASON,
    TRAINING_AUTHORIZED as CORE_TRAINING_AUTHORIZED,
    validate_multires_event_v2_configs,
    validate_optimizer_health_summary,
)
from trauma_predict.training.observability import (  # noqa: E402
    atomic_write_json,
    heartbeat,
    next_attempt_dir,
    sha256_file,
    sha256_payload,
    utc_now,
)


# Single audit surface for the frozen r8 data and contract identity.
BASE_AUTHORITY = {
    "dataset_id": "multires_event_v1_c4_full_20260712",
    "fingerprint": "d58d003b6a9b2dd7c1f8d269a1867b534ea475a91118d7d4d44804bee69f9e47",
    "manifest_sha256": "4e7742900907e0e2f774099ba1dd485468210ff3da9ddaef3ec3bf67957000c3",
    "sample_manifest_sha256": "b3d4305353997320fe310c4df6e15619026db6f229a124b0c9a5e1d89898f05e",
    "subject_split_sha256": "89deb50c2c6415dff5ce00338a980e25531433e8dee835b004a27d561e7adb6d",
}
TARGET_AUTHORITY = {
    "dataset_id": "multires_event_m4_target_v2_c4_full_20260713_r8",
    "manifest_sha256": "fb8748a5d396c5342be143032096acef03af2345bdd80e53dc82f69a7875b8b6",
    "sample_manifest_sha256": "96ce73f2cfb4a2a8af0bd21cbbab9634bd02268d03e7cda68ac4f21229596a4e",
    "contract_bundle_hash": "10e9ed6c2fb94610fa61edc5061b8465e967ef6c222f22455877da583420cd10",
    "process_contract_sha256": "3f90bec35d6473a0e9dc69f3654d1b55eaf1c9d3f9850078df1361e84b2cd7db",
    "emission_contract_sha256": "e926e1a3e6e3e71039a26548ca8d3f35bf2eee5725be3195992d4d47f715e96c",
    "projection_contract_sha256": "7efdf7d3c0415e6aa26d99411f5df66907b5ff74b30f6880e72de72fe4c3d34b",
    "relation_contract_sha256": "65286cd9fb7e1038270de39ea17daafffb160cf9c5ab7bb3beb2556a9aa8eea0",
    "sidecar_schema_sha256": "a2e4018d9dac3c4245ad13852036e6cb3ff9014eea9dc996fa9b0b6235251e8f",
    "process_contract_version": "2026-07-13-r8",
    "emission_contract_version": "2026-07-13-r8",
    "projection_contract_version": "2026-07-13-r8",
}
EXPECTED_LAB_SCALE_ARTIFACT_SHA256 = "dbd5b14254338ff8c42fbfbaf02ca024050b83860f21ffb3b58a27899469cd12"
EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_SHA256 = (
    "0f13f933ae008613ca0665a2de21674de571cbd1102eeca21376e00e582b49e7"
)
EXPECTED_PROMOTION_METRIC_CONTRACT = (
    "configs/evaluation/multires_event_v2_promotion_v2.json"
)
EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256 = (
    "7b5b85d5d3b3604308e1fe8b1471bc6c5c0c20bb16e3b9aaffd0c5e3afb53f3f"
)
TRAINING_AUTHORIZED = CORE_TRAINING_AUTHORIZED
TRAINING_AUTHORIZATION_REASON = CORE_TRAINING_AUTHORIZATION_REASON

BASE_DATASET_REF = "vanilaaaa/trauma-predict-multires-event-v1-c4-20260712"
TARGET_DATASET_REF = "vanilaaaa/trauma-predict-multires-event-v2-c4-r8-20260713"
EXPECTED_COUNTS = {"samples": 50350, "train": 37734, "val": 6309, "test": 6307, "shards": 52}
EXPECTED_SHARD_COUNTS = {"train": 38, "val": 7, "test": 7}
TARGET_CONTRACT_FILES = (
    "target_process_registry_v2.json",
    "target_emission_registry_v2.json",
    "target_projection_registry_v2.json",
    "field_category_matrix_v1.csv",
    "field_relation_edges_v1.csv",
    "event_element_extension_v2.json",
    "target_sidecar_schema_v2.json",
)

STAGE_CONFIGS = {
    "smoke": "configs/train/t4x2_multires_event_v2_smoke.yaml",
    "block": "configs/train/t4x2_multires_event_v2_block.yaml",
    "trajectory": "configs/train/t4x2_multires_event_v2_trajectory.yaml",
    "relational": "configs/train/t4x2_multires_event_v2_relational.yaml",
}
V2_ACTIONS = (*tuple(STAGE_CONFIGS), "promotion")
PROMOTION_RUN_ROOT_ENV = {
    "block": "TRAUMA_PREDICT_V2_BLOCK_RUN_ROOT",
    "trajectory": "TRAUMA_PREDICT_V2_TRAJECTORY_RUN_ROOT",
    "relational": "TRAUMA_PREDICT_V2_RELATIONAL_RUN_ROOT",
}
PROMOTION_MODES = tuple(PROMOTION_RUN_ROOT_ENV)
CONTRACT_IDENTITY_KEYS = (
    "base_dataset_id",
    "base_fingerprint",
    "base_dataset_manifest_sha256",
    "target_dataset_id",
    "dataset_id",
    "target_dataset_manifest_sha256",
    "contract_bundle_hash",
    "process_contract_sha256",
    "emission_contract_sha256",
    "projection_contract_sha256",
    "relation_contract_sha256",
    "sidecar_schema_sha256",
    "lab_scale_artifact_sha256",
    "lab_scale_artifact_file_sha256",
    "standardized_primitive_scale_sha256",
    "standardized_primitive_scale_artifact_file_sha256",
    "promotion_metric_contract_sha256",
    "promotion_metric_contract_file_sha256",
    "semantic_runtime_identity_sha256",
)
PORTABLE_RUN_ARTIFACTS = {
    "input_normalization": "artifacts/input_normalization.json",
    "lab_affine_scale": "artifacts/lab_affine_scale.json",
    "standardized_primitive_scale": "artifacts/standardized_primitive_scale.json",
    "promotion_metric_contract": "artifacts/promotion_metric_contract.json",
    "runtime_environment": "artifacts/runtime_environment.json",
    "train_config": "artifacts/config/train.yaml",
    "dataset_config": "artifacts/config/dataset.yaml",
    "model_config": "artifacts/config/model.yaml",
}
TRAIN_ENTRYPOINT = "notebooks/kaggle/train_multires_event_v2.py"
KAGGLE_WORKING = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
OUTPUT_ROOT = Path(
    os.environ.get("TRAUMA_PREDICT_OUTPUT_ROOT", KAGGLE_WORKING / "trauma-predict-runs")
)
PREPARED_BASE_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_PREPARED_DATA_ROOT",
        KAGGLE_WORKING / "trauma-predict-multires-event-v1-c4-20260712",
    )
)
PREPARED_TARGET_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_PREPARED_V2_TARGET_ROOT",
        KAGGLE_WORKING / TARGET_AUTHORITY["dataset_id"],
    )
)
BASE_DOWNLOAD_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_V1_DOWNLOAD_ROOT",
        KAGGLE_WORKING / "kaggle-dataset-multires-event-v1-c4-20260712",
    )
)
TARGET_DOWNLOAD_ROOT = Path(
    os.environ.get(
        "TRAUMA_PREDICT_V2_DOWNLOAD_ROOT",
        KAGGLE_WORKING / "kaggle-dataset-multires-event-v2-target",
    )
)
FAILURE_TAIL_LINES = int(os.environ.get("TRAUMA_PREDICT_FAILURE_TAIL_LINES", "80"))
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
STREAM_PREFIXES = (
    "V2_TRAIN_NLL ",
    "V2_EVAL_NLL ",
    "MULTIRES_EVENT_V2_PREFLIGHT_OK",
    "MULTIRES_EVENT_V2_TRAINING_COMPLETE",
    "RESUME_CHECKPOINT ",
)


def main() -> None:
    session_started = time.monotonic()
    print("repo_root", REPO_ROOT, flush=True)
    print("output_root", OUTPUT_ROOT, flush=True)
    require_frozen_authority_constants()
    action = selected_action()
    print("v2_action", action, flush=True)
    source = verify_source_identity()
    if action == "promotion":
        attempt_dir = next_attempt_dir(OUTPUT_ROOT / "t4x2_multires_event_v2_promotion")
        suite = verify_matched_suite_and_lab_scale()
        validated = validate_promotion_run_roots(
            require_all=True,
            require_attached=is_kaggle_runtime(),
        )
        promotion = run_promotion(validated, attempt_dir)
        atomic_write_json(
            attempt_dir / "promotion_only_complete.json",
            {
                "schema_version": (
                    "trauma_predict.multires_event_v2_promotion_only_complete.v1"
                ),
                "completed_at": utc_now(),
                "source": source,
                "matched_factor_signature": suite["matched_factor_signature"],
                "validated_runs": validated,
                "promotion_path": str(attempt_dir / "promotion.json"),
                "promotion_sha256": sha256_file(attempt_dir / "promotion.json"),
                "promoted": bool(promotion["promoted"]),
            },
        )
        print("MULTIRES_EVENT_V2_PROMOTION_ONLY_FINISHED", flush=True)
        return
    require_t4x2_runtime()
    stage = action
    config = STAGE_CONFIGS[stage]
    attempt_dir = next_attempt_dir(OUTPUT_ROOT / f"t4x2_multires_event_v2_{stage}")
    print("attempt_log_dir", attempt_dir, flush=True)
    print("selected_stage", stage, flush=True)

    base_source = explicit_or_attached_base_root(attempt_dir)
    target_source = explicit_or_attached_target_root(attempt_dir)
    print("base_dataset_source", base_source, flush=True)
    print("target_dataset_source", target_source, flush=True)
    base_root = v1_route.prepare_dataset_root(base_source, PREPARED_BASE_ROOT, attempt_dir)
    target_root = prepare_target_root(target_source, PREPARED_TARGET_ROOT, attempt_dir)
    print("prepared_base_root", base_root, flush=True)
    print("prepared_target_root", target_root, flush=True)

    install_requirements(attempt_dir)
    runtime_guard()
    suite = verify_matched_suite_and_lab_scale()
    atomic_write_json(
        attempt_dir / "attempt_manifest.json",
        {
            "schema_version": "trauma_predict.multires_event_v2_kaggle_attempt.v1",
            "started_at": utc_now(),
            "source": source,
            "base_dataset_ref": BASE_DATASET_REF,
            "target_dataset_ref": TARGET_DATASET_REF,
            "base_dataset_source": str(base_source),
            "target_dataset_source": str(target_source),
            "prepared_base_root": str(base_root),
            "prepared_target_root": str(target_root),
            "action": action,
            "stage": stage,
            "stage_config": config,
            "base_authority": dict(BASE_AUTHORITY),
            "target_authority": dict(TARGET_AUTHORITY),
            "lab_scale_artifact": suite["lab_scale_artifact"],
            "standardized_primitive_scale_artifact": suite[
                "standardized_primitive_scale_artifact"
            ],
            "launcher_matched_factor_signature": suite["matched_factor_signature"],
            "training_authorized": TRAINING_AUTHORIZED,
            "training_authorization_reason": TRAINING_AUTHORIZATION_REASON,
        },
    )

    env = repo_env(base_root, target_root, source["git_ref"])
    run_to_log(
        [sys.executable, TRAIN_ENTRYPOINT, "--config", config, "--dry-run"],
        attempt_dir / f"dry-run-{stage}.log",
        env=env,
        label=f"DRY_RUN_{stage.upper()}",
    )
    if os.environ.get("TRAUMA_PREDICT_DRY_RUN_ONLY") == "1":
        atomic_write_json(
            attempt_dir / "attempt_dry_run_complete.json",
            {
                "schema_version": "trauma_predict.multires_event_v2_kaggle_dry_run_complete.v1",
                "completed_at": utc_now(),
                "action": action,
                "stage": stage,
                "training_authorized": TRAINING_AUTHORIZED,
                "training_authorization_reason": TRAINING_AUTHORIZATION_REASON,
            },
        )
        print("MULTIRES_EVENT_V2_DRY_RUN_ONLY_FINISHED", flush=True)
        return

    require_training_authorization(stage)
    train_config = load_yaml_config(REPO_ROOT / config)
    run_dir = resolve_output_dir(train_config)
    if stage == "smoke":
        archive_previous_smoke_output()
    print_run_contract(stage, config, attempt_dir / f"{stage}.log")
    validation = (
        reusable_completed_run(
            run_dir,
            expected_mode=str(train_config["mode"]),
            require_free_running=True,
        )
        if stage != "smoke"
        else None
    )
    capacity: dict[str, Any] | None = None
    if validation is None:
        capacity_root = attempt_dir / "capacity-probe" if stage != "smoke" else None
        run_torchrun(
            config,
            attempt_dir / f"{stage}.log",
            env=env,
            label=stage.upper(),
            capacity_output_dir=capacity_root,
            elapsed_before_capacity_seconds=(
                time.monotonic() - session_started if capacity_root is not None else None
            ),
        )
        if capacity_root is not None:
            capacity = validate_capacity_probe_report(
                capacity_root,
                expected_mode=str(train_config["mode"]),
            )
        validation = validate_completed_run(
            run_dir,
            expected_mode=str(train_config["mode"]),
            require_free_running=stage != "smoke",
        )
    completed_row = {
        "stage": stage,
        "mode": str(train_config["mode"]),
        "run_dir": str(run_dir),
        "run_manifest": str(run_dir / "run_manifest.json"),
        "capacity_probe": capacity,
        **validation,
    }
    print(f"MULTIRES_EVENT_V2_STAGE_OK stage={stage} run_dir={run_dir}", flush=True)

    atomic_write_json(
        attempt_dir / "attempt_complete.json",
        {
            "schema_version": "trauma_predict.multires_event_v2_kaggle_attempt_complete.v1",
            "completed_at": utc_now(),
            "action": action,
            "stage": stage,
            "completed": completed_row,
            "promotion": None,
        },
    )
    print("MULTIRES_EVENT_V2_KAGGLE_RUN_FINISHED", flush=True)


def require_frozen_authority_constants() -> None:
    values = {
        "target manifest": TARGET_AUTHORITY["manifest_sha256"],
        "target sample manifest": TARGET_AUTHORITY["sample_manifest_sha256"],
        "contract bundle": TARGET_AUTHORITY["contract_bundle_hash"],
        "process contract": TARGET_AUTHORITY["process_contract_sha256"],
        "emission contract": TARGET_AUTHORITY["emission_contract_sha256"],
        "projection contract": TARGET_AUTHORITY["projection_contract_sha256"],
        "relation contract": TARGET_AUTHORITY["relation_contract_sha256"],
        "sidecar schema": TARGET_AUTHORITY["sidecar_schema_sha256"],
        "lab scale artifact": EXPECTED_LAB_SCALE_ARTIFACT_SHA256,
        "standardized primitive scale artifact": (
            EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_SHA256
        ),
    }
    pending = [label for label, value in values.items() if not SHA256_PATTERN.fullmatch(str(value))]
    if pending:
        raise RuntimeError(
            "V2 Kaggle route is intentionally blocked until final authority hashes are frozen: "
            + ", ".join(pending)
        )


def require_training_authorization(stage: str) -> None:
    if not TRAINING_AUTHORIZED:
        raise RuntimeError(
            "V2 hosted training is not authorized; run with TRAUMA_PREDICT_DRY_RUN_ONLY=1 "
            f"for preflight only. Reason: {TRAINING_AUTHORIZATION_REASON}."
        )
    config = STAGE_CONFIGS.get(stage)
    if config is None:
        raise RuntimeError(f"V2 training authorization requires one training action: {stage!r}")
    run_name = str(load_yaml_config(REPO_ROOT / config).get("run_name") or "")
    if run_name not in AUTHORIZED_TRAINING_RUN_NAMES:
        raise RuntimeError(
            f"V2 hosted training is not authorized for action={stage!r}, "
            f"run_name={run_name!r}; authorized={AUTHORIZED_TRAINING_RUN_NAMES!r}. "
            f"Reason: {TRAINING_AUTHORIZATION_REASON}."
        )


def selected_action(value: str | None = None) -> str:
    legacy = os.environ.get("TRAUMA_PREDICT_V2_STAGES", "").strip()
    if legacy:
        raise ValueError(
            "TRAUMA_PREDICT_V2_STAGES is forbidden; select one TRAUMA_PREDICT_V2_ACTION"
        )
    raw = value if value is not None else os.environ.get("TRAUMA_PREDICT_V2_ACTION", "smoke")
    action = str(raw).strip().lower()
    if not action:
        action = "smoke"
    if "," in action or any(character.isspace() for character in action):
        raise ValueError("TRAUMA_PREDICT_V2_ACTION must contain exactly one action")
    if action not in V2_ACTIONS:
        raise ValueError(f"unknown V2 action {action!r}; allowed={V2_ACTIONS}")
    return action


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT.is_dir()


def verify_source_identity() -> dict[str, str]:
    required = os.environ.get("REQUIRED_GIT_REF", "").strip()
    if not required:
        raise RuntimeError("REQUIRED_GIT_REF must name a published immutable tag or exact commit")
    if any(character.isspace() for character in required):
        raise ValueError("REQUIRED_GIT_REF cannot contain whitespace")
    if COMMIT_PATTERN.fullmatch(required):
        kind = "commit"
        resolved = _git_text("rev-parse", f"{required}^{{commit}}")
    else:
        tag_ref = f"refs/tags/{required}"
        if _git_result("show-ref", "--verify", "--quiet", tag_ref).returncode != 0:
            raise RuntimeError(
                "REQUIRED_GIT_REF must be an exact 40-character commit or an existing tag; "
                f"not an immutable tag: {required}"
            )
        kind = "tag"
        resolved = _git_text("rev-parse", f"{tag_ref}^{{commit}}")
    head = _git_text("rev-parse", "HEAD")
    if head != resolved:
        raise RuntimeError(f"HEAD {head} does not match REQUIRED_GIT_REF {required} ({resolved})")
    status = _git_text("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "immutable V2 source checkout is dirty; refusing hosted training: "
            + status[:2000]
        )
    tree = _git_text("rev-parse", "HEAD^{tree}")
    payload = {"git_ref": required, "kind": kind, "commit": head, "tree": tree}
    print("source_identity", json.dumps(payload, sort_keys=True), flush=True)
    return payload


def find_exact_base_attached_dataset(input_root: Path) -> Path:
    return _find_one_exact_dataset(input_root, _matches_base_authority, "immutable V1 base")


def find_exact_target_attached_dataset(input_root: Path) -> Path:
    return _find_one_exact_dataset(input_root, _matches_target_authority, "V2 target sidecar")


def _find_one_exact_dataset(
    input_root: Path,
    matcher: Callable[[Path, Mapping[str, Any]], bool],
    label: str,
) -> Path:
    if not input_root.is_dir():
        raise FileNotFoundError(f"dataset search root is absent: {input_root}")
    exact: list[Path] = []
    inspected: list[dict[str, Any]] = []
    for manifest_path in sorted(input_root.rglob("dataset_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        inspected.append(
            {
                "root": str(manifest_path.parent),
                "dataset_id": manifest.get("dataset_id"),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
        if matcher(manifest_path.parent, manifest):
            exact.append(manifest_path.parent.resolve())
    unique = sorted(set(exact))
    if len(unique) > 1:
        raise RuntimeError(f"multiple exact {label} datasets are attached; retain one: {unique}")
    if not unique:
        raise FileNotFoundError(f"no exact {label} dataset found; inspected={inspected}")
    return unique[0]


def _matches_base_authority(root: Path, manifest: Mapping[str, Any]) -> bool:
    counts = _base_counts(manifest)
    return (
        manifest.get("dataset_id") == BASE_AUTHORITY["dataset_id"]
        and manifest.get("fingerprint") == BASE_AUTHORITY["fingerprint"]
        and counts == EXPECTED_COUNTS
        and sha256_file(root / "dataset_manifest.json") == BASE_AUTHORITY["manifest_sha256"]
        and _file_hash_or_empty(root / "sample_manifest.csv")
        == BASE_AUTHORITY["sample_manifest_sha256"]
        and _file_hash_or_empty(root / "subject_split.csv")
        == BASE_AUTHORITY["subject_split_sha256"]
    )


def _matches_target_authority(root: Path, manifest: Mapping[str, Any]) -> bool:
    hashes = manifest.get("contract_hashes") or {}
    return (
        manifest.get("dataset_id") == TARGET_AUTHORITY["dataset_id"]
        and _target_counts(manifest) == EXPECTED_COUNTS
        and sha256_file(root / "dataset_manifest.json") == TARGET_AUTHORITY["manifest_sha256"]
        and _file_hash_or_empty(root / "sample_manifest.csv")
        == TARGET_AUTHORITY["sample_manifest_sha256"]
        and manifest.get("contract_bundle_hash") == TARGET_AUTHORITY["contract_bundle_hash"]
        and hashes.get("process") == TARGET_AUTHORITY["process_contract_sha256"]
        and hashes.get("emission") == TARGET_AUTHORITY["emission_contract_sha256"]
        and hashes.get("projection") == TARGET_AUTHORITY["projection_contract_sha256"]
        and hashes.get("relation") == TARGET_AUTHORITY["relation_contract_sha256"]
        and hashes.get("sidecar_schema") == TARGET_AUTHORITY["sidecar_schema_sha256"]
    )


def explicit_or_attached_base_root(log_dir: Path) -> Path:
    explicit = os.environ.get("TRAUMA_PREDICT_DATA_ROOT")
    if explicit:
        root = Path(explicit).resolve()
        if not root.is_dir() or not _matches_base_authority(
            root, _read_json(root / "dataset_manifest.json")
        ):
            raise ValueError(f"TRAUMA_PREDICT_DATA_ROOT is not the frozen V1 base: {root}")
        return root
    try:
        return find_exact_base_attached_dataset(KAGGLE_INPUT)
    except FileNotFoundError:
        return download_exact_dataset(
            dataset_ref=BASE_DATASET_REF,
            download_root=BASE_DOWNLOAD_ROOT,
            finder=find_exact_base_attached_dataset,
            usable=v1_route.has_usable_shard_payload,
            log_path=log_dir / "base_dataset_download.log",
            label="BASE_DATASET_DOWNLOAD",
        )


def explicit_or_attached_target_root(log_dir: Path) -> Path:
    dataset_ref = resolved_target_dataset_ref()
    explicit = os.environ.get("TRAUMA_PREDICT_V2_TARGET_ROOT")
    if explicit:
        root = Path(explicit).resolve()
        if not root.is_dir() or not _matches_target_authority(
            root, _read_json(root / "dataset_manifest.json")
        ):
            raise ValueError(
                f"TRAUMA_PREDICT_V2_TARGET_ROOT is not the frozen target sidecar: {root}"
            )
        return root
    try:
        return find_exact_target_attached_dataset(KAGGLE_INPUT)
    except FileNotFoundError:
        return download_exact_dataset(
            dataset_ref=dataset_ref,
            download_root=TARGET_DOWNLOAD_ROOT,
            finder=find_exact_target_attached_dataset,
            usable=has_usable_target_payload,
            log_path=log_dir / "target_dataset_download.log",
            label="TARGET_DATASET_DOWNLOAD",
        )


def resolved_target_dataset_ref() -> str:
    override = os.environ.get("TRAUMA_PREDICT_V2_DATASET_REF")
    if override is not None and override != TARGET_DATASET_REF:
        raise ValueError(
            "TRAUMA_PREDICT_V2_DATASET_REF must exactly equal the frozen source ref "
            f"{TARGET_DATASET_REF!r}; got {override!r}"
        )
    return TARGET_DATASET_REF


def download_exact_dataset(
    *,
    dataset_ref: str,
    download_root: Path,
    finder: Callable[[Path], Path],
    usable: Callable[[Path], bool],
    log_path: Path,
    label: str,
) -> Path:
    if download_root.is_dir():
        try:
            existing = finder(download_root)
            if usable(existing):
                print("using_existing_download", existing, flush=True)
                return existing
        except FileNotFoundError:
            pass
        archive = download_root.with_name(f"{download_root.name}.invalid-{os.getpid()}")
        download_root.rename(archive)
        print("archived_invalid_download", archive, flush=True)
    download_root.mkdir(parents=True, exist_ok=True)
    v1_route.configure_kaggle_credentials()
    run_to_log(
        ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", download_root],
        log_path,
        env=os.environ.copy(),
        label=label,
    )
    archives = sorted(download_root.glob("*.zip"))
    if len(archives) != 1:
        raise RuntimeError(
            f"controlled {label} must produce exactly one outer ZIP; found={archives}"
        )
    package_root = download_root / "dataset-package"
    v1_route.safe_extract_dataset_package(archives[0], package_root)
    discovered = finder(download_root)
    if not usable(discovered):
        raise FileNotFoundError(f"downloaded exact dataset has no usable payload: {discovered}")
    return discovered


def prepare_target_root(source_root: Path, destination: Path, log_dir: Path) -> Path:
    if is_prepared_target(destination):
        print("using_existing_prepared_target", destination, flush=True)
        return destination.resolve()
    if is_prepared_target(source_root):
        print("using_unpacked_target_source", source_root, flush=True)
        return source_root.resolve()
    if not _matches_target_authority(source_root, _read_json(source_root / "dataset_manifest.json")):
        raise ValueError("target preparation source is not the frozen V2 authority")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.prepare-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name in ("dataset_manifest.json", "sample_manifest.csv", "subject_split.csv", "SUCCEEDED"):
        source = source_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, temporary / name)
    contract_layout = materialize_target_contracts(source_root, temporary / "contracts")
    archive = _one_optional_archive(source_root, ("target_shards.zip", "shards.zip"))
    if archive is not None:
        shard_layout = archive.name
        shard_count = safe_extract_target_shards(archive, temporary)
    else:
        shard_layout = "kaggle_hosted_extracted_target_tree"
        shard_count = copy_extracted_target_shards(source_root, temporary)
    if shard_count != EXPECTED_COUNTS["shards"]:
        raise RuntimeError(f"V2 target payload materialized {shard_count} shards, expected 52")
    if not is_prepared_target(temporary):
        raise ValueError("prepared V2 target failed exact post-materialization identity checks")
    if destination.exists():
        archived = destination.with_name(f"{destination.name}.invalid-{os.getpid()}")
        destination.rename(archived)
        print("archived_invalid_prepared_target", archived, flush=True)
    temporary.replace(destination)
    atomic_write_json(
        log_dir / "target_dataset_prepare.json",
        {
            "schema_version": "trauma_predict.multires_event_v2_target_prepare.v1",
            "created_at": utc_now(),
            "source_root": str(source_root),
            "destination": str(destination),
            "contract_layout": contract_layout,
            "target_shard_layout": shard_layout,
            "materialized_target_shards": shard_count,
        },
    )
    return destination.resolve()


def materialize_target_contracts(source_root: Path, destination: Path) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    direct: dict[str, Path] = {}
    for name in TARGET_CONTRACT_FILES:
        candidates = sorted(path for path in source_root.rglob(name) if path.is_file())
        if len(candidates) == 1:
            direct[name] = candidates[0]
    if len(direct) == len(TARGET_CONTRACT_FILES):
        for name, source in direct.items():
            shutil.copy2(source, destination / name)
        _verify_target_contract_files(destination, source_root / "dataset_manifest.json")
        return "extracted_contract_tree"

    archive = _one_optional_archive(source_root, ("contracts.zip",))
    if archive is None:
        missing = sorted(set(TARGET_CONTRACT_FILES) - set(direct))
        raise FileNotFoundError(f"V2 target contracts are incomplete: {missing}")
    with zipfile.ZipFile(archive) as handle:
        members: dict[str, zipfile.ZipInfo] = {}
        for info in handle.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe contracts.zip member: {info.filename}")
            if member.name in TARGET_CONTRACT_FILES:
                if member.name in members:
                    raise RuntimeError(f"duplicate contract in contracts.zip: {member.name}")
                members[member.name] = info
        if set(members) != set(TARGET_CONTRACT_FILES):
            raise FileNotFoundError("contracts.zip does not contain the exact training contract set")
        for name, info in members.items():
            with handle.open(info) as source, (destination / name).open("wb") as output:
                shutil.copyfileobj(source, output)
    _verify_target_contract_files(destination, source_root / "dataset_manifest.json")
    return "contracts_zip"


def safe_extract_target_shards(archive_path: Path, destination: Path) -> int:
    count = 0
    seen: set[tuple[str, str]] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe target shard member: {info.filename}")
            parts = list(member.parts)
            for marker in ("target_shards", "shards"):
                if marker in parts:
                    parts = parts[parts.index(marker) + 1 :]
                    break
            if len(parts) != 2 or parts[0] not in EXPECTED_SHARD_COUNTS:
                raise ValueError(f"target shard member lacks split/name: {info.filename}")
            relative = Path(*parts)
            if relative.suffixes[-2:] != [".jsonl", ".gz"]:
                raise ValueError(f"unexpected non-gzip target shard: {info.filename}")
            key = (parts[0], relative.name)
            if key in seen:
                raise RuntimeError(f"duplicate target shard: {key}")
            seen.add(key)
            target = destination / "target_shards" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    return count


def discover_extracted_target_shards(source_root: Path) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {split: [] for split in EXPECTED_SHARD_COUNTS}
    candidates = set(source_root.rglob("*.jsonl.gz"))
    candidates.update(source_root.rglob("*.jsonl"))
    for source in sorted(candidates):
        parts = source.relative_to(source_root).parts
        if any(part in {"validation", "manifests", "audit"} for part in parts):
            continue
        logical = tuple(part.removesuffix(".zip") for part in parts)
        split: str | None = None
        for marker in ("target_shards", "shards"):
            if marker in logical:
                index = logical.index(marker)
                split = logical[index + 1] if len(logical) > index + 1 else None
                break
        if split is None:
            candidates_split = [part for part in logical[:-1] if part in discovered]
            split = candidates_split[0] if len(candidates_split) == 1 else None
        if split in discovered and source.name.startswith(f"{split}-"):
            discovered[split].append(source)
    return discovered


def copy_extracted_target_shards(source_root: Path, destination: Path) -> int:
    discovered = discover_extracted_target_shards(source_root)
    observed = {split: len(paths) for split, paths in discovered.items()}
    if observed != EXPECTED_SHARD_COUNTS:
        raise FileNotFoundError(
            f"target shard archive is absent and extracted counts are {observed}; "
            f"expected {EXPECTED_SHARD_COUNTS}"
        )
    for split, paths in discovered.items():
        for source in paths:
            target_name = source.name if source.name.endswith(".jsonl.gz") else f"{source.name}.gz"
            target = destination / "target_shards" / split / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".jsonl":
                recompress_target_shard_like_builder(source, target)
            else:
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
    return sum(observed.values())


def recompress_target_shard_like_builder(source: Path, target: Path) -> None:
    """Restore the exact deterministic gzip bytes emitted by the r8 builder.

    Kaggle can expose uploaded ``*.jsonl.gz`` files as plain ``*.jsonl``.  The
    r8 manifest binds compressed bytes, so recompression must reproduce the
    builder's line-wise TextIOWrapper buffering and default gzip level rather
    than merely produce an equivalent decompressed stream.
    """

    raw_output = target.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0)
    output = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    try:
        with source.open("r", encoding="utf-8", newline="") as input_handle:
            for line_number, line in enumerate(input_handle, start=1):
                if not line.endswith("\n"):
                    raise ValueError(
                        f"plain hosted target shard lacks LF at line {line_number}: {source}"
                    )
                output.write(line)
    finally:
        output.flush()
        output.close()
        raw_output.close()


def has_usable_target_payload(root: Path) -> bool:
    if is_prepared_target(root):
        return True
    if _one_optional_archive(root, ("target_shards.zip", "shards.zip")) is not None:
        return True
    observed = {
        split: len(paths) for split, paths in discover_extracted_target_shards(root).items()
    }
    return observed == EXPECTED_SHARD_COUNTS


def is_prepared_target(root: Path) -> bool:
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    if not _matches_target_authority(root, manifest):
        return False
    try:
        _verify_target_contract_files(root / "contracts", manifest_path)
    except (FileNotFoundError, ValueError):
        return False
    observed = {
        split: len(list((root / "target_shards" / split).glob("*.jsonl.gz")))
        for split in EXPECTED_SHARD_COUNTS
    }
    if observed != EXPECTED_SHARD_COUNTS:
        return False
    try:
        _verify_target_shard_files(root, manifest)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _verify_target_contract_files(contract_root: Path, manifest_path: Path) -> None:
    manifest = _read_json(manifest_path)
    declared = manifest.get("contract_hashes") or {}
    key_by_file = {
        "target_process_registry_v2.json": "process",
        "target_emission_registry_v2.json": "emission",
        "target_projection_registry_v2.json": "projection",
        "field_category_matrix_v1.csv": "category",
        "field_relation_edges_v1.csv": "relation",
        "event_element_extension_v2.json": "element_extension",
        "target_sidecar_schema_v2.json": "sidecar_schema",
    }
    for filename, key in key_by_file.items():
        path = contract_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != declared.get(key):
            raise ValueError(f"target contract hash mismatch: {filename}")
    process = _read_json(contract_root / "target_process_registry_v2.json")
    emission = _read_json(contract_root / "target_emission_registry_v2.json")
    projection = _read_json(contract_root / "target_projection_registry_v2.json")
    if process.get("version") != TARGET_AUTHORITY["process_contract_version"]:
        raise ValueError("target process registry is not the frozen r8 source identity")
    if emission.get("version") != TARGET_AUTHORITY["emission_contract_version"]:
        raise ValueError("target emission registry is not the frozen r8 source identity")
    if projection.get("version") != TARGET_AUTHORITY["projection_contract_version"]:
        raise ValueError("target projection registry is not the frozen r8 source identity")


def _verify_target_shard_files(root: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files") or {}
    declared = files.get("target_shards") or {}
    if not isinstance(declared, Mapping) or len(declared) != EXPECTED_COUNTS["shards"]:
        raise ValueError("target manifest must declare exactly 52 target shard hashes")
    split_counts = {split: 0 for split in EXPECTED_SHARD_COUNTS}
    split_samples = {split: 0 for split in EXPECTED_SHARD_COUNTS}
    seen_paths: set[str] = set()
    for key, raw_metadata in declared.items():
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(f"target shard metadata must be a mapping: {key}")
        relative = Path(str(raw_metadata.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe declared target shard path: {relative}")
        if len(relative.parts) != 3 or relative.parts[0] != "target_shards":
            raise ValueError(f"declared target shard path violates layout: {relative}")
        split = relative.parts[1]
        if split not in split_counts or not relative.name.startswith(f"{split}-"):
            raise ValueError(f"declared target shard split/name mismatch: {relative}")
        relative_text = relative.as_posix()
        if relative_text in seen_paths:
            raise ValueError(f"duplicate declared target shard path: {relative}")
        seen_paths.add(relative_text)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = str(raw_metadata.get("sha256") or "")
        if not SHA256_PATTERN.fullmatch(expected_hash) or sha256_file(path) != expected_hash:
            raise ValueError(f"target shard byte hash mismatch: {relative}")
        samples = int(raw_metadata.get("samples", -1))
        if samples < 1:
            raise ValueError(f"target shard sample count must be positive: {relative}")
        split_counts[split] += 1
        split_samples[split] += samples
    if split_counts != EXPECTED_SHARD_COUNTS:
        raise ValueError(f"target shard split counts mismatch: {split_counts}")
    if split_samples != {key: EXPECTED_COUNTS[key] for key in split_samples}:
        raise ValueError(f"target shard sample totals mismatch: {split_samples}")


def verify_matched_suite_and_lab_scale() -> dict[str, Any]:
    configs = {
        stage: load_yaml_config(REPO_ROOT / path) for stage, path in STAGE_CONFIGS.items()
    }
    signatures = {
        stage: _matched_signature(configs[stage]) for stage in ("block", "trajectory", "relational")
    }
    if len(set(signatures.values())) != 1:
        raise ValueError(f"block/trajectory/relational configs are not mode-only matched: {signatures}")
    expected_modes = {"block": "block", "trajectory": "trajectory", "relational": "relational"}
    for stage, mode in expected_modes.items():
        if configs[stage].get("mode") != mode:
            raise ValueError(f"{stage} config must declare mode={mode}")
    scale_paths = {str(config.get("lab_scale_artifact") or "") for config in configs.values()}
    scale_hashes = {str(config.get("lab_scale_artifact_hash") or "") for config in configs.values()}
    if len(scale_paths) != 1 or "" in scale_paths:
        raise ValueError("all V2 configs must reference one repo lab_scale_artifact")
    if scale_hashes != {EXPECTED_LAB_SCALE_ARTIFACT_SHA256}:
        raise ValueError("all V2 configs must freeze the final lab scale artifact hash")
    path = (REPO_ROOT / next(iter(scale_paths))).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("lab_scale_artifact must remain inside the source repository") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _read_json(path)
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "content_sha256"},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    observed_scale_hash = hashlib.sha256(canonical).hexdigest()
    if observed_scale_hash != EXPECTED_LAB_SCALE_ARTIFACT_SHA256:
        raise ValueError("repo lab scale artifact differs from the frozen content hash")
    if payload.get("content_sha256") != observed_scale_hash:
        raise ValueError("repo lab scale artifact self hash mismatch")
    if payload.get("schema") != "multires_event_v2_lab_affine_scale_v1":
        raise ValueError("repo lab scale artifact schema mismatch")
    if payload.get("fit_split") != "train" or payload.get("status") != "frozen_train_only_fit":
        raise ValueError("repo lab scale artifact is not train-only")
    phi_paths = {
        str(config.get("standardized_primitive_scale_artifact") or "")
        for config in configs.values()
    }
    phi_hashes = {
        str(config.get("standardized_primitive_scale_artifact_hash") or "")
        for config in configs.values()
    }
    if len(phi_paths) != 1 or "" in phi_paths:
        raise ValueError("all V2 configs must reference one repo phi scale artifact")
    if phi_hashes != {EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_SHA256}:
        raise ValueError("all V2 configs must freeze the final phi scale artifact hash")
    phi_path = (REPO_ROOT / next(iter(phi_paths))).resolve()
    try:
        phi_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("standardized primitive scale must remain inside the repository") from exc
    if not phi_path.is_file():
        raise FileNotFoundError(phi_path)
    phi_payload = _read_json(phi_path)
    phi_canonical = json.dumps(
        {key: value for key, value in phi_payload.items() if key != "content_sha256"},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    observed_phi_hash = hashlib.sha256(phi_canonical).hexdigest()
    if observed_phi_hash != EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_SHA256:
        raise ValueError("repo phi scale artifact differs from the frozen content hash")
    if phi_payload.get("content_sha256") != observed_phi_hash:
        raise ValueError("repo phi scale artifact self hash mismatch")
    if phi_payload.get("schema") != "multires_event_v2_standardized_primitive_scale_v2":
        raise ValueError("repo phi scale artifact schema mismatch")
    if (
        phi_payload.get("fit_split") != "train"
        or phi_payload.get("status") != "frozen_train_only_fit"
        or int((phi_payload.get("fit_audit") or {}).get("fitted_key_count", -1)) != 38
        or (phi_payload.get("fit_audit") or {}).get("zero_iqr_keys") != []
    ):
        raise ValueError("repo phi scale artifact lacks the frozen train-only fit proof")
    promotion_paths = {
        str(config.get("promotion_metric_contract") or "") for config in configs.values()
    }
    promotion_hashes = {
        str(config.get("promotion_metric_contract_hash") or "")
        for config in configs.values()
    }
    if promotion_paths != {EXPECTED_PROMOTION_METRIC_CONTRACT} or promotion_hashes != {
        EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256
    }:
        raise ValueError("all V2 configs must freeze one promotion metric contract")
    promotion_path = (REPO_ROOT / EXPECTED_PROMOTION_METRIC_CONTRACT).resolve()
    load_promotion_metric_contract(
        promotion_path,
        expected_sha256=EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256,
    )
    return {
        "matched_factor_signature": next(iter(signatures.values())),
        "lab_scale_artifact": {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": observed_scale_hash,
            "fit_split": payload["fit_split"],
        },
        "standardized_primitive_scale_artifact": {
            "path": str(phi_path.relative_to(REPO_ROOT)),
            "sha256": observed_phi_hash,
            "fit_split": phi_payload["fit_split"],
            "fitted_key_count": int(phi_payload["fit_audit"]["fitted_key_count"]),
        },
        "promotion_metric_contract": {
            "path": EXPECTED_PROMOTION_METRIC_CONTRACT,
            "sha256": EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256,
        },
    }


def _matched_signature(config: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(config))
    payload.pop("mode", None)
    payload.pop("run_name", None)
    payload.pop("outputs", None)
    return sha256_payload(payload)


def repo_env(base_root: Path, target_root: Path, git_ref: str) -> dict[str, str]:
    env = os.environ.copy()
    env["TRAUMA_PREDICT_DATA_ROOT"] = str(base_root)
    env["TRAUMA_PREDICT_V2_TARGET_ROOT"] = str(target_root)
    env["TRAUMA_PREDICT_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    env["REQUIRED_GIT_REF"] = git_ref
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_torchrun(
    config: str,
    log_path: Path,
    *,
    env: dict[str, str],
    label: str,
    capacity_output_dir: Path | None = None,
    elapsed_before_capacity_seconds: float | None = None,
) -> None:
    if (capacity_output_dir is None) != (elapsed_before_capacity_seconds is None):
        raise ValueError("capacity output and elapsed-session inputs must be paired")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        TRAIN_ENTRYPOINT,
        "--config",
        config,
    ]
    if capacity_output_dir is not None:
        command.extend(
            (
                "--capacity-probe-output",
                str(capacity_output_dir.resolve()),
                "--elapsed-before-capacity-seconds",
                f"{float(elapsed_before_capacity_seconds):.6f}",
            )
        )
    run_to_log(
        command,
        log_path,
        env=env,
        label=label,
    )


def validate_capacity_probe_report(root: Path, *, expected_mode: str) -> dict[str, Any]:
    report_path = root.resolve() / "capacity_probe.json"
    if not report_path.is_file():
        raise FileNotFoundError("formal action lacks the attempt-local capacity report")
    report = _read_json(report_path)
    if (
        report.get("schema_version") != CAPACITY_PROBE_SCHEMA
        or report.get("status") != "PASSED"
        or report.get("mode") != expected_mode
        or Path(str(report.get("report_path") or "")).resolve() != report_path
        or report.get("failures") != []
    ):
        raise ValueError("capacity report schema/status/mode gate failed")
    contract = _mapping(report.get("contract"), "capacity report contract")
    expected_contract = {
        "optimizer_steps": CAPACITY_PROBE_OPTIMIZER_STEPS,
        "per_device_train_batch_size": 32,
        "world_size": 2,
        "precision": "fp16",
        "validation_selection": "persisted_val_manifest_prefix",
        "validation_anchors": CAPACITY_PROBE_VALIDATION_ANCHORS,
        "trajectories_per_anchor": CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR,
        "formal_validation_anchors": EXPECTED_COUNTS["val"],
        "formal_trajectories_per_anchor": 100,
    }
    if dict(contract) != expected_contract:
        raise ValueError("capacity report changed the frozen 100-anchor or formal 6309 contract")
    hardware = report.get("hardware")
    if not isinstance(hardware, list) or len(hardware) != 2 or any(
        "T4" not in str(row.get("device_name", "")).upper()
        or int(row.get("rank", -1)) not in {0, 1}
        or int(row.get("local_rank", -1)) not in {0, 1}
        or not isinstance(row.get("compute_capability"), list)
        or len(row.get("compute_capability")) != 2
        or int(row.get("total_memory_bytes", 0)) <= 0
        or int(row.get("peak_allocated_bytes", 0)) <= 0
        or int(row.get("peak_reserved_bytes", 0))
        < int(row.get("peak_allocated_bytes", 0))
        or int(row.get("peak_reserved_bytes", 0)) > int(row.get("total_memory_bytes", -1))
        for row in hardware
        if isinstance(row, Mapping)
    ) or any(not isinstance(row, Mapping) for row in hardware) or {
        int(row["rank"]) for row in hardware
    } != {0, 1}:
        raise ValueError("capacity report does not prove two valid T4 devices")
    optimizer = _mapping(report.get("optimizer"), "capacity optimizer")
    steps = optimizer.get("steps")
    if (
        optimizer.get("optimizer_contract_version") != OPTIMIZER_CONTRACT_VERSION
        or optimizer.get("loss_reduction") != RAW_JOINT_NLL_REDUCTION
        or optimizer.get("gradient_clipping") != "disabled"
        or optimizer.get("configured_contract") != EXPECTED_OPTIMIZER_CONTRACT
        or optimizer.get("scaler_skipped_steps") != 0
        or not isinstance(steps, list)
        or len(steps) != CAPACITY_PROBE_OPTIMIZER_STEPS
        or any(
            row.get("optimizer_updated") is not True
            or int(row.get("global_anchors", -1)) != 64
            or int(row.get("step", -1)) not in {1, 2}
            or not math.isfinite(float(row.get("wall_seconds", math.nan)))
            or float(row.get("wall_seconds", math.nan)) <= 0.0
            or not math.isfinite(float(row.get("joint_nll_anchor_mean", math.nan)))
            or not _capacity_optimizer_step_health_valid(row)
            or float(row.get("scaler_scale_before", math.nan)) != 32.0
            or float(row.get("scaler_scale_after", math.nan)) != 32.0
            for row in steps
            if isinstance(row, Mapping)
        )
        or any(not isinstance(row, Mapping) for row in steps)
        or {int(row["step"]) for row in steps} != {1, 2}
    ):
        raise ValueError("capacity report lacks two successful exact-B64 optimizer steps")
    teacher = _mapping(report.get("teacher_probe"), "capacity teacher probe")
    if (
        int(teacher.get("anchors", -1)) != CAPACITY_PROBE_VALIDATION_ANCHORS
        or int(teacher.get("subjects", -1)) < 1
        or not math.isfinite(float(teacher.get("wall_seconds", math.nan)))
        or float(teacher.get("wall_seconds", math.nan)) <= 0.0
        or not math.isfinite(float(teacher.get("joint_nll_subject_macro", math.nan)))
    ):
        raise ValueError("capacity report teacher probe is incomplete")
    free = _mapping(report.get("free_running_probe"), "capacity free-running")
    structural = _mapping(
        free.get("structural_subject_macro"), "capacity structural metrics"
    )
    if (
        set(structural) != set(CAPACITY_STRUCTURAL_METRICS)
        or any(not math.isfinite(float(structural[key])) for key in CAPACITY_STRUCTURAL_METRICS)
        or int(free.get("anchors", -1)) != CAPACITY_PROBE_VALIDATION_ANCHORS
        or int(free.get("trajectories_per_anchor", -1))
        != CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR
        or float(free.get("coherence_rate", -1.0)) != 1.0
        or int(free.get("coherent_trajectories", -1))
        != CAPACITY_PROBE_VALIDATION_ANCHORS
        * CAPACITY_PROBE_TRAJECTORIES_PER_ANCHOR
        or free.get("selection_verified") is not True
    ):
        raise ValueError("capacity report structural/coherence gate failed")
    identity = _mapping(report.get("identity"), "capacity identity")
    expected_set_sha = str(identity.get("first_100_sample_id_set_sha256") or "")
    if (
        identity.get("dataset_id") != TARGET_AUTHORITY["dataset_id"]
        or identity.get("contract_bundle_hash")
        != TARGET_AUTHORITY["contract_bundle_hash"]
        or identity.get("relation_contract_sha256")
        != TARGET_AUTHORITY["relation_contract_sha256"]
        or identity.get("sidecar_schema_sha256")
        != TARGET_AUTHORITY["sidecar_schema_sha256"]
        or not SHA256_PATTERN.fullmatch(
            str(identity.get("contract_bundle_hash") or "")
        )
        or not SHA256_PATTERN.fullmatch(
            str(identity.get("input_normalization_sha256") or "")
        )
        or identity.get("promotion_metric_contract_sha256")
        != EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256
        or not SHA256_PATTERN.fullmatch(
            str(identity.get("first_100_sample_ids_sha256") or "")
        )
        or not SHA256_PATTERN.fullmatch(expected_set_sha)
        or str(free.get("observed_sample_ids_sha256") or "") != expected_set_sha
    ):
        raise ValueError("capacity report does not bind the first persisted 100 anchors")
    projection = _mapping(report.get("projection"), "capacity projection")
    components = _mapping(
        projection.get("components_seconds"), "capacity projection components"
    )
    expected_projection_counts = {
        "formal_max_steps": 4000,
        "formal_eval_steps": 250,
        "interval_teacher_passes": 16,
        "final_teacher_passes": 1,
        "total_teacher_passes": 17,
    }
    if any(
        int(projection.get(key, -1)) != expected
        for key, expected in expected_projection_counts.items()
    ) or set(components) != {"optimizer", "teacher_forced", "free_running"}:
        raise ValueError("capacity report changed the frozen formal projection contract")
    projection_scalars = (
        float(projection.get("optimizer_seconds_per_step", math.nan)),
        float(projection.get("teacher_seconds_per_anchor", math.nan)),
        float(projection.get("free_running_seconds_per_anchor", math.nan)),
        *(float(components[key]) for key in sorted(components)),
        float(projection.get("projected_formal_runtime_seconds", math.nan)),
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in projection_scalars):
        raise ValueError("capacity report formal projection is non-finite")
    projected_seconds = float(projection["projected_formal_runtime_seconds"])
    if not math.isclose(
        projected_seconds,
        sum(float(value) for value in components.values()),
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise ValueError("capacity report projection components do not close")
    budget = _mapping(report.get("budget"), "capacity budget")
    elapsed_before = float(budget.get("elapsed_before_capacity_seconds", math.nan))
    probe_elapsed = float(budget.get("capacity_probe_elapsed_seconds", math.nan))
    required_seconds = float(budget.get("required_session_seconds", math.nan))
    remaining_seconds = float(budget.get("remaining_headroom_seconds", math.nan))
    recomputed_required = (
        elapsed_before
        + probe_elapsed
        + projected_seconds
        + CAPACITY_SESSION_RESERVE_SECONDS
    )
    if (
        int(budget.get("session_budget_seconds", -1)) != CAPACITY_SESSION_BUDGET_SECONDS
        or int(budget.get("reserved_finalization_seconds", -1))
        != CAPACITY_SESSION_RESERVE_SECONDS
        or not math.isfinite(elapsed_before)
        or elapsed_before < 0.0
        or not math.isfinite(probe_elapsed)
        or probe_elapsed <= 0.0
        or not math.isfinite(required_seconds)
        or required_seconds > CAPACITY_SESSION_BUDGET_SECONDS
        or not math.isclose(required_seconds, recomputed_required, abs_tol=1e-6)
        or not math.isclose(
            remaining_seconds,
            CAPACITY_SESSION_BUDGET_SECONDS - required_seconds,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("capacity report exceeds the frozen session budget")
    if list(root.rglob("SUCCESS")):
        raise ValueError("capacity output contains a forbidden formal SUCCESS marker")
    return {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "status": "PASSED",
        "mode": expected_mode,
        "projected_formal_runtime_seconds": float(
            projection["projected_formal_runtime_seconds"]
        ),
        "remaining_headroom_seconds": float(budget["remaining_headroom_seconds"]),
    }


def _capacity_optimizer_step_health_valid(row: Mapping[str, Any]) -> bool:
    gradient = row.get("gradient_health")
    state = row.get("optimizer_state_health")
    configuration = state.get("optimizer_configuration") if isinstance(state, Mapping) else None
    if (
        not isinstance(gradient, Mapping)
        or not isinstance(state, Mapping)
        or not isinstance(configuration, Mapping)
    ):
        return False
    try:
        trainable = int(gradient.get("trainable_parameter_tensors", -1))
        gradient_l2 = float(gradient.get("global_l2_norm", math.nan))
        probe_gradient = float(gradient.get("probe_gradient_abs", math.nan))
        state_minimum = float(state.get("exp_avg_sq_minimum", math.nan))
        probe_before = float(state.get("probe_value_before", math.nan))
        probe_after = float(state.get("probe_value_after", math.nan))
        probe_changed = state.get("probe_parameter_changed")
        step = int(row.get("step", -1))
        expected_learning_rate = (
            float(EXPECTED_OPTIMIZER_CONTRACT["learning_rate"]) * step / 400.0
        )
        learning_rate_used = float(row.get("learning_rate_used", math.nan))
        gradient_audit_seconds = float(gradient.get("audit_wall_seconds", math.nan))
        state_audit_seconds = float(state.get("audit_wall_seconds", math.nan))
        audit_wall_seconds = float(row.get("optimizer_audit_wall_seconds", math.nan))
        return (
            row.get("event") == "v2_optimizer_health"
            and step in {1, 2}
            and int(row.get("local_anchors", -1)) == 32
            and int(row.get("world_size", -1)) == 2
            and int(row.get("global_anchors", -1)) == 64
            and row.get("optimizer_contract_version") == OPTIMIZER_CONTRACT_VERSION
            and row.get("loss_reduction") == RAW_JOINT_NLL_REDUCTION
            and "grad_norm_after_unscale_before_clip" not in row
            and "max_grad_norm" not in row
            and math.isfinite(learning_rate_used)
            and learning_rate_used == expected_learning_rate
            and float(row.get("expected_learning_rate_used", math.nan))
            == expected_learning_rate
            and int(row.get("expected_optimizer_step", -1)) == step
            and float(row.get("observed_optimizer_step_min", math.nan)) == float(step)
            and float(row.get("observed_optimizer_step_max", math.nan)) == float(step)
            and math.isfinite(gradient_audit_seconds)
            and gradient_audit_seconds > 0.0
            and math.isfinite(state_audit_seconds)
            and state_audit_seconds > 0.0
            and math.isfinite(audit_wall_seconds)
            and audit_wall_seconds > 0.0
            and audit_wall_seconds <= float(row.get("wall_seconds", math.nan))
            and math.isclose(
                audit_wall_seconds,
                gradient_audit_seconds + state_audit_seconds,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and gradient.get("optimizer_contract_version") == OPTIMIZER_CONTRACT_VERSION
            and trainable > 0
            and int(gradient.get("gradient_tensors", -1)) == trainable
            and int(gradient.get("missing_gradient_tensors", -1)) == 0
            and gradient.get("all_gradients_finite") is True
            and math.isfinite(gradient_l2)
            and gradient_l2 > 0.0
            and gradient.get("global_l2_positive") is True
            and gradient.get("gradient_clipping") == "disabled"
            and gradient.get("gradient_modified_after_unscale") is False
            and math.isfinite(probe_gradient)
            and probe_gradient > 0.0
            and isinstance(gradient.get("probe_parameter"), str)
            and bool(gradient.get("probe_parameter"))
            and int(gradient.get("probe_flat_index", -1)) >= 0
            and state.get("optimizer_contract_version") == OPTIMIZER_CONTRACT_VERSION
            and int(state.get("trainable_parameter_tensors", -1)) == trainable
            and int(state.get("optimizer_state_entries", -1)) == trainable
            and state.get("state_complete") is True
            and int(state.get("expected_optimizer_step", -1)) == step
            and float(state.get("observed_optimizer_step_min", math.nan)) == float(step)
            and float(state.get("observed_optimizer_step_max", math.nan)) == float(step)
            and state.get("state_steps_complete_equal_expected") is True
            and state.get("parameters_finite") is True
            and state.get("exp_avg_finite") is True
            and state.get("exp_avg_sq_finite") is True
            and state.get("exp_avg_sq_nonnegative") is True
            and math.isfinite(state_minimum)
            and state_minimum >= 0.0
            and state.get("probe_parameter") == gradient.get("probe_parameter")
            and int(state.get("probe_flat_index", -1))
            == int(gradient.get("probe_flat_index", -2))
            and math.isfinite(probe_before)
            and math.isfinite(probe_after)
            and isinstance(probe_changed, bool)
            and probe_changed is (probe_before != probe_after)
            and state.get("optimizer_updated") is True
            and configuration.get("optimizer") == "AdamW"
            and int(configuration.get("parameter_group_count", -1)) == 1
            and float(configuration.get("base_learning_rate", math.nan))
            == float(EXPECTED_OPTIMIZER_CONTRACT["learning_rate"])
            and float(configuration.get("current_learning_rate", math.nan))
            == learning_rate_used
            and float(configuration.get("weight_decay", math.nan))
            == float(EXPECTED_OPTIMIZER_CONTRACT["weight_decay"])
            and list(configuration.get("adamw_betas", ()))
            == EXPECTED_OPTIMIZER_CONTRACT["adamw_betas"]
            and float(configuration.get("adamw_eps", math.nan))
            == float(EXPECTED_OPTIMIZER_CONTRACT["adamw_eps"])
            and configuration.get("adamw_amsgrad")
            is EXPECTED_OPTIMIZER_CONTRACT["adamw_amsgrad"]
            and configuration.get("adamw_maximize")
            is EXPECTED_OPTIMIZER_CONTRACT["adamw_maximize"]
            and configuration.get("adamw_foreach")
            is EXPECTED_OPTIMIZER_CONTRACT["adamw_foreach"]
            and configuration.get("adamw_fused")
            is EXPECTED_OPTIMIZER_CONTRACT["adamw_fused"]
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_semantic_runtime_identity(identity: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "python",
        "torch",
        "cuda_runtime",
        "cudnn",
        "devices",
        "world_size",
        "precision",
        "requirements_sha256",
        "lock_sha256",
        "dependency_versions",
    }
    if set(identity) != required:
        raise ValueError("semantic runtime identity fields are incomplete")
    python = _mapping(identity.get("python"), "semantic runtime python")
    devices = identity.get("devices")
    dependencies = identity.get("dependency_versions")
    lock_sha = identity.get("lock_sha256")
    if (
        identity.get("schema_version")
        != "trauma_predict.multires_event_v2_semantic_runtime.v1"
        or set(python) != {"implementation", "version"}
        or not str(python.get("implementation") or "")
        or not str(python.get("version") or "")
        or not str(identity.get("torch") or "")
        or not str(identity.get("cuda_runtime") or "")
        or int(identity.get("cudnn", 0)) <= 0
        or int(identity.get("world_size", -1)) != 2
        or identity.get("precision") != "fp16"
        or not SHA256_PATTERN.fullmatch(str(identity.get("requirements_sha256") or ""))
        or (lock_sha is not None and not SHA256_PATTERN.fullmatch(str(lock_sha)))
        or not isinstance(dependencies, Mapping)
        or set(dependencies) != {"numpy", "PyYAML", "safetensors"}
        or any(not str(value or "") for value in dependencies.values())
        or not isinstance(devices, list)
        or len(devices) != 2
    ):
        raise ValueError("semantic runtime identity violates the hosted contract")
    for device in devices:
        if (
            not isinstance(device, Mapping)
            or set(device) != {"name", "compute_capability"}
            or "T4" not in str(device.get("name") or "").upper()
            or not isinstance(device.get("compute_capability"), list)
            or len(device["compute_capability"]) != 2
        ):
            raise ValueError("semantic runtime identity does not describe two T4 devices")


def print_run_contract(stage: str, config_path: str, log_path: Path) -> None:
    config = load_yaml_config(REPO_ROOT / config_path)
    training = config["training"]
    print(
        "MULTIRES_EVENT_V2_RUN_CONTRACT",
        json.dumps(
            {
                "stage": stage,
                "run_name": config["run_name"],
                "mode": config["mode"],
                "route": config["route"],
                "max_steps": int(training["max_steps"]),
                "logging_steps": int(training["logging_steps"]),
                "eval_steps": int(training["eval_steps"]),
                "save_steps": int(training["save_steps"]),
                "output_dir": str(resolve_output_dir(config)),
                "full_log": str(log_path),
                "stochastic_primitive_factors": int(
                    config["objective"]["stochastic_primitive_factors"]
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_to_log(
    command: list[Any],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    label: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [str(part) for part in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(command), ">>", log_path, flush=True)
    with log_path.open("a", encoding="utf-8") as log, heartbeat(
        label, log_path, seconds=300
    ):
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            stripped = line.rstrip()
            if stripped.startswith(STREAM_PREFIXES):
                print(stripped, flush=True)
        returncode = process.wait()
    if returncode != 0:
        if check:
            print_failure_tail(log_path)
            raise subprocess.CalledProcessError(returncode, command)
        print(f"{label}_NONZERO returncode={returncode} log={log_path}", flush=True)
    else:
        print(f"{label}_OK log={log_path}", flush=True)
    return subprocess.CompletedProcess(command, returncode)


def install_requirements(log_dir: Path) -> None:
    if os.environ.get("TRAUMA_PREDICT_SKIP_INSTALL") == "1":
        print("SKIP_MULTIRES_V2_PIP_INSTALL", flush=True)
        return
    run_to_log(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-multires-kaggle.txt"],
        log_dir / "pip_install.log",
        env=os.environ.copy(),
        label="PIP_INSTALL",
    )
    run_to_log(
        [sys.executable, "-m", "pip", "check"],
        log_dir / "pip_check.log",
        env=os.environ.copy(),
        label="PIP_CHECK",
        check=False,
    )


def require_t4x2_runtime() -> None:
    result = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout.strip(), flush=True)
    count = sum(1 for line in result.stdout.splitlines() if line.startswith("GPU "))
    if count < 2:
        raise RuntimeError(f"select Kaggle T4 x2; detected {count} GPU(s)")


def runtime_guard() -> None:
    import torch

    payload = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    print("runtime", json.dumps(payload, sort_keys=True), flush=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("multires_event_v2 matched experiments require two visible CUDA devices")
    for index in range(2):
        if float(torch.ones(1, device=f"cuda:{index}").item()) != 1.0:
            raise RuntimeError(f"CUDA tensor smoke failed on device {index}")
    print("MULTIRES_EVENT_V2_RUNTIME_GUARD_OK", flush=True)


def archive_previous_smoke_output() -> None:
    config = load_yaml_config(REPO_ROOT / STAGE_CONFIGS["smoke"])
    smoke_dir = resolve_output_dir(config)
    if not smoke_dir.exists():
        return
    archive_root = OUTPUT_ROOT / "smoke-history"
    archive_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (archive_root / f"{smoke_dir.name}-attempt-{index:04d}").exists():
        index += 1
    destination = archive_root / f"{smoke_dir.name}-attempt-{index:04d}"
    smoke_dir.rename(destination)
    print("archived_previous_smoke", destination, flush=True)


def reusable_completed_run(
    run_dir: Path,
    *,
    expected_mode: str,
    require_free_running: bool,
) -> dict[str, Any] | None:
    """Reuse an immutable successful run, while failing closed on a corrupt success."""

    if not (run_dir / "SUCCESS").is_file():
        return None
    validation = validate_completed_run(
        run_dir,
        expected_mode=expected_mode,
        require_free_running=require_free_running,
    )
    print(
        f"MULTIRES_EVENT_V2_REUSE_SUCCESS mode={expected_mode} run_dir={run_dir}",
        flush=True,
    )
    return validation


def validate_completed_run(
    run_dir: Path,
    *,
    expected_mode: str,
    require_free_running: bool,
) -> dict[str, Any]:
    required = [
        "resolved_config.json",
        "artifacts/manifest.json",
        *PORTABLE_RUN_ARTIFACTS.values(),
        "dataset_identity.json",
        "objective_contract.json",
        "model_identity.json",
        "normalization_identity.json",
        "source_identity.json",
        "identity_hashes.json",
        "metrics.jsonl",
        "optimizer_health_summary.json",
        "best_checkpoint.json",
        "best_checkpoint/model.pt",
        "best_checkpoint/identity_hashes.json",
        "final_model/model.pt",
        "final_model/model_manifest.json",
        "val_per_anchor_joint_nll.jsonl",
        "evaluation.json",
        "run_manifest.json",
        "SUCCESS",
    ]
    if require_free_running:
        required.extend(
            (
                "free_running/evaluation.json",
                "free_running/manifest.json",
                "free_running/sample_schema.json",
            )
        )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"completed V2 run lacks required outputs: {missing}")
    manifest_path = run_dir / "run_manifest.json"
    success = _read_json(run_dir / "SUCCESS")
    manifest = _read_json(manifest_path)
    optimizer_health_pointer = manifest.get("optimizer_health_summary")
    if not isinstance(optimizer_health_pointer, Mapping):
        raise ValueError("completed run manifest lacks optimizer health summary identity")
    summary_path = run_dir / "optimizer_health_summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    summary_sha256 = sha256_file(summary_path)
    metrics_sha256 = sha256_file(metrics_path)
    if (
        success.get("schema_version") != "trauma_predict.multires_event_v2_success.v1"
        or success.get("run_manifest_sha256") != sha256_file(manifest_path)
        or success.get("optimizer_health_summary_sha256") != summary_sha256
        or success.get("metrics_jsonl_sha256") != metrics_sha256
        or optimizer_health_pointer.get("path") != "optimizer_health_summary.json"
        or optimizer_health_pointer.get("sha256") != summary_sha256
        or optimizer_health_pointer.get("metrics_path") != "metrics.jsonl"
        or optimizer_health_pointer.get("metrics_sha256") != metrics_sha256
    ):
        raise ValueError("completed V2 SUCCESS/run manifest optimizer hash chain is invalid")
    if (
        manifest.get("schema_version")
        != "trauma_predict.multires_event_v2_run_manifest.v1"
        or manifest.get("status") != "SUCCEEDED"
        or manifest.get("mode") != expected_mode
    ):
        raise ValueError(f"V2 run manifest status/mode mismatch: {manifest}")
    identity = _read_json(run_dir / "dataset_identity.json")
    expected_contract_identity = {
        "base_dataset_id": BASE_AUTHORITY["dataset_id"],
        "base_fingerprint": BASE_AUTHORITY["fingerprint"],
        "base_dataset_manifest_sha256": BASE_AUTHORITY["manifest_sha256"],
        "target_dataset_id": TARGET_AUTHORITY["dataset_id"],
        "dataset_id": TARGET_AUTHORITY["dataset_id"],
        "target_dataset_manifest_sha256": TARGET_AUTHORITY["manifest_sha256"],
        "contract_bundle_hash": TARGET_AUTHORITY["contract_bundle_hash"],
        "process_contract_sha256": TARGET_AUTHORITY["process_contract_sha256"],
        "emission_contract_sha256": TARGET_AUTHORITY["emission_contract_sha256"],
        "projection_contract_sha256": TARGET_AUTHORITY["projection_contract_sha256"],
        "relation_contract_sha256": TARGET_AUTHORITY["relation_contract_sha256"],
        "sidecar_schema_sha256": TARGET_AUTHORITY["sidecar_schema_sha256"],
        "lab_scale_artifact_sha256": EXPECTED_LAB_SCALE_ARTIFACT_SHA256,
        "standardized_primitive_scale_sha256": (
            EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_SHA256
        ),
        "promotion_metric_contract_sha256": (
            EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256
        ),
    }
    for key, expected in expected_contract_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"completed run contract identity.{key} mismatch")
    objective_contract = _read_json(run_dir / "objective_contract.json")
    if objective_contract.get("contract_bundle_hash") != TARGET_AUTHORITY[
        "contract_bundle_hash"
    ]:
        raise ValueError("completed objective contract bundle differs from launcher authority")
    objective_contract_hashes = objective_contract.get("contract_hashes")
    expected_objective_hashes = {
        "process": TARGET_AUTHORITY["process_contract_sha256"],
        "emission": TARGET_AUTHORITY["emission_contract_sha256"],
        "projection": TARGET_AUTHORITY["projection_contract_sha256"],
        "relation": TARGET_AUTHORITY["relation_contract_sha256"],
        "sidecar_schema": TARGET_AUTHORITY["sidecar_schema_sha256"],
    }
    if not isinstance(objective_contract_hashes, Mapping) or any(
        objective_contract_hashes.get(key) != expected
        for key, expected in expected_objective_hashes.items()
    ):
        raise ValueError("completed objective contract hashes differ from launcher authority")
    artifact_manifest = _read_json(run_dir / "artifacts/manifest.json")
    artifact_entries = artifact_manifest.get("artifacts")
    if (
        artifact_manifest.get("schema_version")
        != "trauma_predict.multires_event_v2_run_artifacts.v1"
        or not isinstance(artifact_entries, Mapping)
        or set(artifact_entries) != set(PORTABLE_RUN_ARTIFACTS)
    ):
        raise ValueError("completed run portable artifact manifest is incomplete")
    artifact_manifest_pointer = manifest.get("artifact_manifest")
    if (
        not isinstance(artifact_manifest_pointer, Mapping)
        or artifact_manifest_pointer.get("path") != "artifacts/manifest.json"
        or artifact_manifest_pointer.get("sha256")
        != sha256_file(run_dir / "artifacts/manifest.json")
    ):
        raise ValueError("completed run manifest does not bind portable artifacts")
    artifact_file_hashes: dict[str, str] = {}
    for name, relative in PORTABLE_RUN_ARTIFACTS.items():
        entry = artifact_entries.get(name)
        if not isinstance(entry, Mapping) or entry.get("path") != relative:
            raise ValueError(f"completed run portable artifact pointer mismatch: {name}")
        file_sha256 = str(entry.get("file_sha256") or "")
        if (
            not SHA256_PATTERN.fullmatch(file_sha256)
            or sha256_file(run_dir / relative) != file_sha256
        ):
            raise ValueError(f"completed run portable artifact hash mismatch: {name}")
        artifact_file_hashes[name] = file_sha256
    if artifact_entries["lab_affine_scale"].get(
        "semantic_sha256"
    ) != EXPECTED_LAB_SCALE_ARTIFACT_SHA256:
        raise ValueError("completed run portable lab scale semantic identity mismatch")
    if artifact_entries["standardized_primitive_scale"].get(
        "semantic_sha256"
    ) != EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_SHA256:
        raise ValueError("completed run portable phi scale semantic identity mismatch")
    if artifact_entries["promotion_metric_contract"].get(
        "semantic_sha256"
    ) != EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256:
        raise ValueError("completed run portable promotion contract identity mismatch")
    runtime_environment = _read_json(
        run_dir / PORTABLE_RUN_ARTIFACTS["runtime_environment"]
    )
    semantic_runtime_identity = runtime_environment.get("semantic_runtime_identity")
    semantic_runtime_sha256 = str(
        runtime_environment.get("semantic_runtime_identity_sha256") or ""
    )
    if (
        runtime_environment.get("schema_version")
        != "trauma_predict.multires_event_v2_runtime_environment.v1"
        or not isinstance(semantic_runtime_identity, Mapping)
        or sha256_payload(semantic_runtime_identity) != semantic_runtime_sha256
        or artifact_entries["runtime_environment"].get("semantic_sha256")
        != semantic_runtime_sha256
    ):
        raise ValueError("completed run runtime environment semantic identity is invalid")
    _validate_semantic_runtime_identity(semantic_runtime_identity)
    expected_portable_identity = {
        "normalization_artifact": PORTABLE_RUN_ARTIFACTS["input_normalization"],
        "normalization_artifact_file_sha256": artifact_file_hashes[
            "input_normalization"
        ],
        "lab_scale_artifact": PORTABLE_RUN_ARTIFACTS["lab_affine_scale"],
        "lab_scale_artifact_file_sha256": artifact_file_hashes["lab_affine_scale"],
        "standardized_primitive_scale_artifact": PORTABLE_RUN_ARTIFACTS[
            "standardized_primitive_scale"
        ],
        "standardized_primitive_scale_artifact_file_sha256": artifact_file_hashes[
            "standardized_primitive_scale"
        ],
        "promotion_metric_contract": PORTABLE_RUN_ARTIFACTS[
            "promotion_metric_contract"
        ],
        "promotion_metric_contract_file_sha256": artifact_file_hashes[
            "promotion_metric_contract"
        ],
        "runtime_environment_artifact": PORTABLE_RUN_ARTIFACTS[
            "runtime_environment"
        ],
        "runtime_environment_artifact_file_sha256": artifact_file_hashes[
            "runtime_environment"
        ],
        "semantic_runtime_identity_sha256": semantic_runtime_sha256,
    }
    for key, expected in expected_portable_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"completed run portable dataset identity.{key} mismatch")
    contract_identity = {key: identity[key] for key in CONTRACT_IDENTITY_KEYS}
    contract_identity["objective_contract_sha256"] = sha256_file(
        run_dir / "objective_contract.json"
    )
    normalization_sha256 = str(identity.get("input_normalization_sha256") or "")
    if not SHA256_PATTERN.fullmatch(normalization_sha256):
        raise ValueError("completed run lacks the full input normalization file hash")
    normalization_identity = _read_json(run_dir / "normalization_identity.json")
    if (
        normalization_identity.get("artifact_path")
        != PORTABLE_RUN_ARTIFACTS["input_normalization"]
        or normalization_identity.get("artifact_sha256") != normalization_sha256
        or normalization_identity.get("artifact_file_sha256")
        != artifact_file_hashes["input_normalization"]
        or artifact_file_hashes["input_normalization"] != normalization_sha256
    ):
        raise ValueError(
            "completed portable normalization identity disagrees with dataset identity"
        )
    source_identity = _read_json(run_dir / "source_identity.json")
    git_commit = str(source_identity.get("git_commit") or "")
    git_head_tree = str(source_identity.get("git_head_tree") or "")
    source_tree_sha256 = str(source_identity.get("source_tree_sha256") or "")
    if (
        not COMMIT_PATTERN.fullmatch(git_commit)
        or not COMMIT_PATTERN.fullmatch(git_head_tree)
        or not SHA256_PATTERN.fullmatch(source_tree_sha256)
        or source_identity.get("git_clean") is not True
    ):
        raise ValueError("completed run lacks a clean source tree and Git identity")
    if (
        git_commit != _git_text("rev-parse", "HEAD")
        or git_head_tree != _git_text("rev-parse", "HEAD^{tree}")
    ):
        raise ValueError("completed run source commit/tree differs from current immutable source")
    evaluation = _read_json(run_dir / "evaluation.json")
    if manifest.get("evaluation") != evaluation:
        raise ValueError("completed run manifest does not bind the teacher evaluation")
    expected_evaluation = {
        "phase": "final",
        "mode": expected_mode,
        "samples": EXPECTED_COUNTS["val"],
        "subjects": 505,
        "primitive_factors_per_anchor": 414,
        "active_target_denominator": False,
        "deterministic_projection_loss": False,
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"completed V2 evaluation.{key} must equal {expected!r}")
    per_anchor_path = run_dir / "val_per_anchor_joint_nll.jsonl"
    if evaluation.get("per_anchor_output_sha256") != sha256_file(per_anchor_path):
        raise ValueError("completed V2 per-anchor paired-evaluation artifact hash mismatch")
    teacher_identity = evaluation.get("identity")
    if not isinstance(teacher_identity, Mapping):
        raise ValueError("completed teacher evaluation lacks row-level contract identity")
    evaluation_contract_identity = {
        "dataset_id": TARGET_AUTHORITY["dataset_id"],
        "contract_bundle_hash": TARGET_AUTHORITY["contract_bundle_hash"],
        "process_contract_sha256": TARGET_AUTHORITY["process_contract_sha256"],
        "emission_contract_sha256": TARGET_AUTHORITY["emission_contract_sha256"],
        "projection_contract_sha256": TARGET_AUTHORITY["projection_contract_sha256"],
        "relation_contract_sha256": TARGET_AUTHORITY["relation_contract_sha256"],
        "sidecar_schema_sha256": TARGET_AUTHORITY["sidecar_schema_sha256"],
        "lab_scale_artifact_sha256": EXPECTED_LAB_SCALE_ARTIFACT_SHA256,
        "standardized_primitive_scale_sha256": (
            EXPECTED_STANDARDIZED_PRIMITIVE_SCALE_SHA256
        ),
        "input_normalization_sha256": normalization_sha256,
        "promotion_metric_contract_sha256": (
            EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256
        ),
        "semantic_runtime_identity_sha256": semantic_runtime_sha256,
    }
    for key, expected in evaluation_contract_identity.items():
        if teacher_identity.get(key) != expected:
            raise ValueError(f"completed teacher evaluation identity.{key} mismatch")
    free_pointer = manifest.get("free_running_evaluation")
    free_evaluation: dict[str, Any] | None = None
    if require_free_running:
        if not isinstance(free_pointer, Mapping):
            raise ValueError("completed matched run lacks free-running manifest pointer")
        free_path = run_dir / "free_running/evaluation.json"
        free_manifest_path = run_dir / "free_running/manifest.json"
        if (
            free_pointer.get("path") != "free_running/evaluation.json"
            or free_pointer.get("manifest_path") != "free_running/manifest.json"
        ):
            raise ValueError("completed free-running pointers are not portable run-relative paths")
        if free_pointer.get("sha256") != sha256_file(free_path):
            raise ValueError("completed free-running evaluation hash mismatch")
        if free_pointer.get("manifest_sha256") != sha256_file(free_manifest_path):
            raise ValueError("completed free-running manifest hash mismatch")
        free_evaluation = _read_json(free_path)
        free_manifest = _read_json(free_manifest_path)
        expected_free = {
            "mode": expected_mode,
            "anchors": EXPECTED_COUNTS["val"],
            "subjects": 505,
            "trajectories_per_anchor": 100,
        }
        for key, expected in expected_free.items():
            if free_evaluation.get(key) != expected:
                raise ValueError(f"completed free-running {key} must equal {expected!r}")
        for key in (
            "field_macro_lag1_variogram_score_p0_5",
            "relation_edge_macro_variogram_score_p0_5",
            "marginal_value_crps",
            "marginal_state_crps",
        ):
            summary = free_evaluation.get(key)
            if (
                not isinstance(summary, Mapping)
                or not math.isfinite(float(summary.get("subject_macro", math.nan)))
            ):
                raise ValueError(f"completed free-running structural metric is invalid: {key}")
        sample_schema_path = run_dir / "free_running/sample_schema.json"
        if (
            free_evaluation.get("sample_schema_path") != "sample_schema.json"
            or free_evaluation.get("sample_schema_sha256")
            != sha256_file(sample_schema_path)
        ):
            raise ValueError("completed free-running sample schema hash/pointer mismatch")
        free_shards = free_manifest.get("per_anchor_score_shards")
        if not isinstance(free_shards, list) or len(free_shards) != 2:
            raise ValueError("completed free-running manifest must contain two DDP shards")
        shard_anchors = 0
        shard_ranks: set[int] = set()
        for shard in free_shards:
            if not isinstance(shard, Mapping):
                raise ValueError("completed free-running shard manifest row is invalid")
            rank = int(shard.get("rank", -1))
            anchors = int(shard.get("anchors", -1))
            retained = int(shard.get("retained_audit_trajectories", -1))
            if rank not in {0, 1} or rank in shard_ranks or anchors < 1 or retained != anchors:
                raise ValueError("completed free-running shard rank/count identity is invalid")
            shard_ranks.add(rank)
            shard_anchors += anchors
            for path_key, hash_key in (
                ("per_anchor_score_path", "per_anchor_score_sha256"),
                ("audit_trajectory_sample_path", "audit_trajectory_sample_sha256"),
                ("progress_metrics_path", "progress_metrics_sha256"),
            ):
                relative = Path(str(shard.get(path_key) or ""))
                if relative.is_absolute() or not relative.name:
                    raise ValueError("completed free-running shard pointer is not portable")
                shard_path = (free_path.parent / relative).resolve()
                try:
                    shard_path.relative_to(free_path.parent.resolve())
                except ValueError as exc:
                    raise ValueError(
                        "completed free-running shard pointer escapes the run root"
                    ) from exc
                expected_sha = str(shard.get(hash_key) or "")
                if (
                    not shard_path.is_file()
                    or not SHA256_PATTERN.fullmatch(expected_sha)
                    or sha256_file(shard_path) != expected_sha
                ):
                    raise ValueError("completed free-running shard hash mismatch")
        if shard_ranks != {0, 1} or shard_anchors != EXPECTED_COUNTS["val"]:
            raise ValueError("completed free-running shard coverage is incomplete")
        coherence = free_evaluation.get("coherence") or {}
        if (
            float(coherence.get("rate", -1.0)) != 1.0
            or int(coherence.get("coherent_trajectories", -1))
            != EXPECTED_COUNTS["val"] * 100
        ):
            raise ValueError("completed free-running trajectories are not 100% coherent")
        free_identity = free_evaluation.get("identity")
        if not isinstance(free_identity, Mapping):
            raise ValueError("completed free-running evaluation lacks contract identity")
        for key, expected in evaluation_contract_identity.items():
            if free_identity.get(key) != expected:
                raise ValueError(f"completed free-running identity.{key} mismatch")
    elif free_pointer is not None:
        raise ValueError("route smoke run unexpectedly emitted production free-running output")
    model_identity = _read_json(run_dir / "model_identity.json")
    signature = str(model_identity.get("matched_design_signature") or "")
    parameter_count = int(model_identity.get("parameter_count", 0))
    if (
        model_identity.get("mode") != expected_mode
        or not SHA256_PATTERN.fullmatch(signature)
        or parameter_count < 1
    ):
        raise ValueError("completed V2 model identity lacks matched signature/parameter count")
    resolved_config = _read_json(run_dir / "resolved_config.json")
    resolved_sections: dict[str, dict[str, Any]] = {}
    for name in ("train", "dataset", "model"):
        section = resolved_config.get(name)
        if not isinstance(section, Mapping):
            raise ValueError(f"completed resolved config lacks {name}")
        resolved_sections[name] = dict(section)
    _validate_portable_config_yaml(run_dir, resolved_sections)
    validate_multires_event_v2_configs(
        resolved_sections["train"],
        resolved_sections["dataset"],
        resolved_sections["model"],
    )
    configured_max_steps = int(resolved_sections["train"]["training"]["max_steps"])
    expected_completed_steps = 4000 if require_free_running else 2
    if configured_max_steps != expected_completed_steps:
        raise ValueError("completed V2 run uses the wrong smoke/formal max_steps profile")
    optimizer_health_summary = validate_optimizer_health_summary(
        summary_path,
        metrics_path,
        training=resolved_sections["train"]["training"],
    )
    if int(optimizer_health_summary.get("canonical_steps", -1)) != expected_completed_steps:
        raise ValueError("completed V2 optimizer health coverage is incomplete")
    if objective_contract.get("objective") != resolved_sections["train"].get("objective"):
        raise ValueError("completed objective contract differs from the resolved train config")
    matched_train = json.loads(json.dumps(resolved_sections["train"]))
    matched_train.pop("mode", None)
    matched_train.pop("run_name", None)
    matched_train.pop("outputs", None)
    recomputed_signature = sha256_payload(
        {
            "train": matched_train,
            "dataset": resolved_sections["dataset"],
            "model": resolved_sections["model"],
        }
    )
    if recomputed_signature != signature:
        raise ValueError("completed matched design signature does not bind resolved configs")
    identity_hashes = _read_json(run_dir / "identity_hashes.json")
    expected_identity_hashes = {
        "train_config": sha256_payload(resolved_sections["train"]),
        "dataset_config": sha256_payload(resolved_sections["dataset"]),
        "model_config": sha256_payload(resolved_sections["model"]),
        "runtime": sha256_payload(identity),
        "contract_bundle": TARGET_AUTHORITY["contract_bundle_hash"],
        "normalization": normalization_sha256,
        "source_tree": source_tree_sha256,
        "source_identity": sha256_payload(source_identity),
        "git_commit": git_commit,
        "git_head_tree": git_head_tree,
        "matched_design": signature,
        "semantic_runtime": semantic_runtime_sha256,
    }
    for key, expected in expected_identity_hashes.items():
        if identity_hashes.get(key) != expected:
            raise ValueError(f"completed run identity_hashes.{key} mismatch")
    if manifest.get("identity_hashes") != identity_hashes:
        raise ValueError("completed run manifest identity hashes differ from persisted identity")
    best_pointer_path = run_dir / "best_checkpoint.json"
    best_pointer = _read_json(best_pointer_path)
    best_model_path = run_dir / "best_checkpoint/model.pt"
    best_model_sha256 = str(best_pointer.get("model_sha256") or "")
    if (
        best_pointer.get("schema_version")
        != "trauma_predict.multires_event_v2_best_checkpoint.v1"
        or best_pointer.get("path") != "best_checkpoint"
        or best_pointer.get("identity_hashes") != identity_hashes
        or not SHA256_PATTERN.fullmatch(best_model_sha256)
        or sha256_file(best_model_path) != best_model_sha256
        or _read_json(run_dir / "best_checkpoint/identity_hashes.json")
        != identity_hashes
    ):
        raise ValueError("completed best checkpoint is not bound to the run identity")
    best_step = best_pointer.get("step")
    if isinstance(best_step, bool) or not isinstance(best_step, int) or best_step < 1:
        raise ValueError("completed best checkpoint step is invalid")
    selected_model_identity = manifest.get("selected_model_identity")
    expected_selected_model_identity = {
        "schema_version": "trauma_predict.multires_event_v2_selected_model.v1",
        "selected_checkpoint_step": best_step,
        "selected_checkpoint_model_sha256": best_model_sha256,
        "selected_checkpoint_path": "best_checkpoint/model.pt",
        "best_checkpoint_manifest_path": "best_checkpoint.json",
        "best_checkpoint_manifest_sha256": sha256_file(best_pointer_path),
    }
    if selected_model_identity != expected_selected_model_identity:
        raise ValueError("completed selected-model identity differs from the best checkpoint")
    completed_training = manifest.get("training")
    if not isinstance(completed_training, Mapping):
        raise ValueError("completed run manifest lacks training completion identity")
    completed_step = completed_training.get("training_completed_step")
    if (
        isinstance(completed_step, bool)
        or not isinstance(completed_step, int)
        or completed_step != expected_completed_steps
        or completed_training.get("global_step") != expected_completed_steps
        or completed_training.get("max_steps") != expected_completed_steps
        or completed_training.get("scaler_skipped_steps") != 0
        or completed_step < best_step
        or completed_training.get("selected_checkpoint_step") != best_step
        or completed_training.get("selected_checkpoint_model_sha256")
        != best_model_sha256
        or evaluation.get("step") != best_step
        or (free_evaluation is not None and free_evaluation.get("step") != best_step)
    ):
        raise ValueError("completed evaluation is not bound to the selected checkpoint")
    model_manifest = _read_json(run_dir / "final_model/model_manifest.json")
    final_model_path = run_dir / "final_model/model.pt"
    final_model_sha256 = sha256_file(final_model_path)
    final_model_manifest_path = run_dir / "final_model/model_manifest.json"
    final_model_pointer = manifest.get("final_model")
    if (
        not isinstance(final_model_pointer, Mapping)
        or final_model_pointer.get("path") != "final_model/model.pt"
        or final_model_pointer.get("sha256") != final_model_sha256
        or final_model_pointer.get("manifest_path")
        != "final_model/model_manifest.json"
        or final_model_pointer.get("manifest_sha256")
        != sha256_file(final_model_manifest_path)
        or final_model_sha256 != best_model_sha256
    ):
        raise ValueError("completed run manifest does not bind the selected final model")
    if (
        model_manifest.get("mode") != expected_mode
        or model_manifest.get("model_file") != "final_model/model.pt"
        or model_manifest.get("model_sha256") != final_model_sha256
        or model_manifest.get("selected_checkpoint_step") != best_step
        or model_manifest.get("selected_checkpoint_model_sha256") != best_model_sha256
        or model_manifest.get("training_completed_step") != completed_step
        or model_manifest.get("identity_hashes") != identity_hashes
    ):
        raise ValueError("completed final model is not bound to the selected checkpoint and run")
    selected_evaluation_identity = {
        "source_tree_sha256": source_tree_sha256,
        "source_identity_sha256": sha256_payload(source_identity),
        "git_commit": git_commit,
        "git_head_tree": git_head_tree,
        "matched_design_signature": signature,
        "selected_checkpoint_step": best_step,
        "selected_checkpoint_model_sha256": best_model_sha256,
    }
    for label, row_identity in (
        ("teacher", teacher_identity),
        (
            "free-running",
            free_evaluation.get("identity") if free_evaluation is not None else None,
        ),
    ):
        if row_identity is None:
            continue
        for key, expected in selected_evaluation_identity.items():
            if row_identity.get(key) != expected:
                raise ValueError(f"completed {label} evaluation identity.{key} mismatch")
    source_comparison_identity = {
        "git_commit": git_commit,
        "git_head_tree": git_head_tree,
        "source_tree_sha256": source_tree_sha256,
    }
    return {
        "matched_design_signature": signature,
        "parameter_count": parameter_count,
        "input_normalization_sha256": normalization_sha256,
        "source_identity": source_comparison_identity,
        "source_identity_sha256": sha256_payload(source_identity),
        "contract_identity": contract_identity,
        "contract_identity_sha256": sha256_payload(contract_identity),
        "relation_contract_sha256": contract_identity[
            "relation_contract_sha256"
        ],
        "sidecar_schema_sha256": contract_identity["sidecar_schema_sha256"],
        "semantic_runtime_identity_sha256": semantic_runtime_sha256,
        "run_manifest_sha256": sha256_file(manifest_path),
        "optimizer_health_summary_sha256": summary_sha256,
        "metrics_sha256": metrics_sha256,
    }


def validate_promotion_run_roots(
    *,
    require_all: bool = True,
    require_attached: bool = False,
) -> dict[str, dict[str, Any]]:
    """Resolve and validate three independently persisted matched run roots."""

    validated: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    resolved_roots: dict[str, Path] = {}
    for mode in PROMOTION_MODES:
        env_name = PROMOTION_RUN_ROOT_ENV[mode]
        explicit = os.environ.get(env_name, "").strip()
        if not explicit:
            missing.append(f"{mode}=unset ({env_name})")
            continue
        run_dir = Path(explicit).expanduser().resolve()
        root_source = env_name
        if require_attached:
            try:
                run_dir.relative_to(KAGGLE_INPUT.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"promotion root {mode} must be an attached Kaggle input under "
                    f"{KAGGLE_INPUT}: {run_dir}"
                ) from exc
        resolved_roots[mode] = run_dir
        if not (run_dir / "SUCCESS").is_file():
            if require_all:
                missing.append(f"{mode}={run_dir} ({root_source})")
            continue
        validation = validate_completed_run(
            run_dir,
            expected_mode=mode,
            require_free_running=True,
        )
        validated[mode] = {
            "stage": mode,
            "mode": mode,
            "run_dir": str(run_dir),
            "run_root_source": root_source,
            "run_manifest": str(run_dir / "run_manifest.json"),
            **validation,
        }
    if missing:
        raise FileNotFoundError(
            "promotion requires immutable SUCCESS roots for every requested mode: "
            + "; ".join(missing)
        )
    if require_all and tuple(validated) != PROMOTION_MODES:
        absent = [mode for mode in PROMOTION_MODES if mode not in validated]
        raise FileNotFoundError(f"promotion run roots are incomplete: {absent}")
    if len(set(resolved_roots.values())) != len(resolved_roots):
        raise ValueError("block/trajectory/relational promotion run roots must be independent")
    assert_matched_promotion_identity(validated)
    return validated


def assert_matched_promotion_identity(
    validated: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require one design, parameterization, normalization, source, and contract identity."""

    unknown = sorted(set(validated) - set(PROMOTION_MODES))
    if unknown:
        raise ValueError(f"promotion received unknown modes: {unknown}")
    ordered = [mode for mode in PROMOTION_MODES if mode in validated]
    if not ordered:
        return
    roots: list[Path] = []
    for mode in ordered:
        row = validated[mode]
        if row.get("mode") != mode:
            raise ValueError(f"promotion run key/mode mismatch for {mode}")
        run_dir = str(row.get("run_dir") or "")
        if not run_dir:
            raise ValueError(f"promotion run {mode} lacks run_dir")
        roots.append(Path(run_dir).expanduser().resolve())
        signature = str(row.get("matched_design_signature") or "")
        normalization = str(row.get("input_normalization_sha256") or "")
        source_hash = str(row.get("source_identity_sha256") or "")
        contract_hash = str(row.get("contract_identity_sha256") or "")
        relation_hash = str(row.get("relation_contract_sha256") or "")
        sidecar_schema_hash = str(row.get("sidecar_schema_sha256") or "")
        runtime_hash = str(row.get("semantic_runtime_identity_sha256") or "")
        contract_identity = row.get("contract_identity")
        if (
            not SHA256_PATTERN.fullmatch(signature)
            or int(row.get("parameter_count", 0)) < 1
            or not SHA256_PATTERN.fullmatch(normalization)
            or not SHA256_PATTERN.fullmatch(source_hash)
            or not SHA256_PATTERN.fullmatch(contract_hash)
            or not SHA256_PATTERN.fullmatch(relation_hash)
            or not SHA256_PATTERN.fullmatch(sidecar_schema_hash)
            or not SHA256_PATTERN.fullmatch(runtime_hash)
            or not isinstance(row.get("source_identity"), Mapping)
            or not isinstance(contract_identity, Mapping)
            or contract_identity.get("relation_contract_sha256") != relation_hash
            or contract_identity.get("sidecar_schema_sha256") != sidecar_schema_hash
        ):
            raise ValueError(f"promotion run {mode} has incomplete comparison identity")
    if len(set(roots)) != len(roots):
        raise ValueError("block/trajectory/relational promotion run roots must be independent")

    baseline_mode = ordered[0]
    baseline = validated[baseline_mode]
    comparison_fields = {
        "matched design signature": "matched_design_signature",
        "parameter count": "parameter_count",
        "full input normalization SHA256": "input_normalization_sha256",
        "source tree and Git identity": "source_identity",
        "full source identity SHA256": "source_identity_sha256",
        "contract identity": "contract_identity",
        "full contract identity SHA256": "contract_identity_sha256",
        "relation contract SHA256": "relation_contract_sha256",
        "sidecar schema SHA256": "sidecar_schema_sha256",
        "semantic runtime identity SHA256": "semantic_runtime_identity_sha256",
    }
    for label, key in comparison_fields.items():
        mismatched = [
            mode for mode in ordered[1:] if validated[mode].get(key) != baseline.get(key)
        ]
        if mismatched:
            raise RuntimeError(
                f"matched promotion {label} mismatch: baseline={baseline_mode}, "
                f"different={mismatched}"
            )


def run_promotion(
    validated: Mapping[str, Mapping[str, Any]],
    attempt_dir: Path,
) -> dict[str, Any]:
    """Run the frozen promotion gate over already validated independent outputs."""

    missing = [mode for mode in PROMOTION_MODES if mode not in validated]
    if missing:
        raise ValueError(f"promotion requires all matched modes: {missing}")
    assert_matched_promotion_identity(validated)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {
        mode: Path(str(validated[mode]["run_dir"])).expanduser().resolve()
        for mode in PROMOTION_MODES
    }
    promotion_metric_contract = load_promotion_metric_contract(
        run_dirs["block"] / PORTABLE_RUN_ARTIFACTS["promotion_metric_contract"],
        expected_sha256=EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256,
    )
    atomic_write_json(
        attempt_dir / "promotion_inputs.json",
        {
            "schema_version": "trauma_predict.multires_event_v2_promotion_inputs.v1",
            "created_at": utc_now(),
            "promotion_metric_contract_sha256": (
                EXPECTED_PROMOTION_METRIC_CONTRACT_SHA256
            ),
            "runs": {mode: dict(validated[mode]) for mode in PROMOTION_MODES},
        },
    )
    promotion = evaluate_multires_event_v2_promotion(
        block_teacher_path=run_dirs["block"] / "val_per_anchor_joint_nll.jsonl",
        trajectory_teacher_path=run_dirs["trajectory"] / "val_per_anchor_joint_nll.jsonl",
        relational_teacher_path=run_dirs["relational"] / "val_per_anchor_joint_nll.jsonl",
        block_free_running_path=run_dirs["block"] / "free_running",
        trajectory_free_running_path=run_dirs["trajectory"] / "free_running",
        relational_free_running_path=run_dirs["relational"] / "free_running",
        promotion_metric_contract=promotion_metric_contract,
        expected_anchors=EXPECTED_COUNTS["val"],
        bootstrap_repetitions=2000,
        bootstrap_seed=20260713,
    )
    atomic_write_json(attempt_dir / "promotion.json", promotion)
    print(
        "MULTIRES_EVENT_V2_PROMOTION",
        json.dumps(
            {
                "promoted": promotion["promoted"],
                "winner": promotion["winner"],
                "path": str(attempt_dir / "promotion.json"),
                "sha256": sha256_file(attempt_dir / "promotion.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return promotion


def resolve_output_dir(config: Mapping[str, Any]) -> Path:
    value = str(config["outputs"]["output_dir"]).replace(
        "${TRAUMA_PREDICT_OUTPUT_ROOT}", str(OUTPUT_ROOT)
    )
    if "${" in value:
        raise ValueError(f"unexpanded environment variable in output path: {value}")
    return Path(value).resolve()


def print_failure_tail(path: Path) -> None:
    print(f"FAILURE_LOG_TAIL log={path}", flush=True)
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-FAILURE_TAIL_LINES:]:
        print(line, flush=True)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _base_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    counts = manifest.get("counts") or {}
    by_split = counts.get("selected_by_split") or {}
    return {
        "samples": int(counts.get("samples", -1)),
        "train": int(by_split.get("train", -1)),
        "val": int(by_split.get("val", -1)),
        "test": int(by_split.get("test", -1)),
        "shards": int(counts.get("completed_shards", -1)),
    }


def _target_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    counts = manifest.get("counts") or {}
    by_split = counts.get("by_split") or {}
    return {
        "samples": int(counts.get("samples", -1)),
        "train": int(by_split.get("train", -1)),
        "val": int(by_split.get("val", -1)),
        "test": int(by_split.get("test", -1)),
        "shards": int(counts.get("shards", -1)),
    }


def _file_hash_or_empty(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def _one_optional_archive(root: Path, names: tuple[str, ...]) -> Path | None:
    found = sorted(
        path for name in names for path in root.rglob(name) if path.is_file()
    )
    if len(found) > 1:
        raise RuntimeError(f"multiple candidate archives found for {names}: {found}")
    return found[0] if found else None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return payload


def _validate_portable_config_yaml(
    run_dir: Path,
    resolved_sections: Mapping[str, Mapping[str, Any]],
) -> None:
    bindings: dict[str, str] = {}
    for section, artifact_name in (
        ("train", "train_config"),
        ("dataset", "dataset_config"),
        ("model", "model_config"),
    ):
        path = run_dir / PORTABLE_RUN_ARTIFACTS[artifact_name]
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"portable {section} YAML must contain one mapping")
        expanded = _expand_portable_template(
            payload,
            resolved_sections[section],
            bindings,
            path=section,
        )
        if expanded != dict(resolved_sections[section]):
            raise ValueError(f"portable {section} YAML differs from resolved_config.json")


def _expand_portable_template(
    template: Any,
    resolved: Any,
    bindings: dict[str, str],
    *,
    path: str,
) -> Any:
    if isinstance(template, Mapping):
        if not isinstance(resolved, Mapping) or set(template) != set(resolved):
            raise ValueError(f"portable config mapping differs at {path}")
        return {
            key: _expand_portable_template(
                template[key], resolved[key], bindings, path=f"{path}.{key}"
            )
            for key in template
        }
    if isinstance(template, list):
        if not isinstance(resolved, list) or len(template) != len(resolved):
            raise ValueError(f"portable config list differs at {path}")
        return [
            _expand_portable_template(
                item, resolved[index], bindings, path=f"{path}[{index}]"
            )
            for index, item in enumerate(template)
        ]
    if isinstance(template, str) and ENV_PLACEHOLDER_PATTERN.search(template):
        if not isinstance(resolved, str):
            raise ValueError(f"portable config placeholder has non-string value at {path}")
        captured_variables: list[str] = []
        pattern_parts: list[str] = []
        position = 0
        for match in ENV_PLACEHOLDER_PATTERN.finditer(template):
            pattern_parts.append(re.escape(template[position : match.start()]))
            variable = match.group(1)
            if variable in bindings:
                pattern_parts.append(re.escape(bindings[variable]))
            else:
                pattern_parts.append("(.+?)")
                captured_variables.append(variable)
            position = match.end()
        pattern_parts.append(re.escape(template[position:]))
        matched = re.fullmatch("".join(pattern_parts), resolved)
        if matched is None:
            raise ValueError(f"portable config placeholder expansion differs at {path}")
        for variable, value in zip(
            captured_variables, matched.groups(), strict=True
        ):
            previous = bindings.setdefault(variable, value)
            if previous != value:
                raise ValueError(f"portable config variable {variable} is inconsistent")
        return resolved
    if type(template) is not type(resolved) or template != resolved:
        raise ValueError(f"portable config scalar differs at {path}")
    return template


def _git_result(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )


def _git_text(*args: str) -> str:
    result = _git_result(*args)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


if __name__ == "__main__":
    main()
