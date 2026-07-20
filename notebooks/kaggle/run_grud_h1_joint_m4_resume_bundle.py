from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import selectors
import subprocess
import sys
import tarfile
import time
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


RESUME_SCHEMA = "trauma_predict.grud_h1_joint_m4_resume_bundle.v1"
RESUME_DATASET_REF = "vanila111/trauma-predict-grud-h1-joint-m4-v2-resume-2500-bundle"
SCIENCE_SCHEMA = "trauma_predict.grud_h1_joint_m4_p100_bundle.v1"
SCIENCE_DATASET_REF = "vanila111/trauma-predict-grud-h1-joint-m4-v2-bundle"
RUNTIME_DATASET_REF = "vanila111/trauma-predict-relation-v2-p100-r9-bundle"
EXPECTED_RUNTIME_CONTRACT_SHA256 = (
    "aada1dee4ee21e02fd5c81ae97d441c38e72d770eec5398932ee295d08f8f2cc"
)
EXPECTED_RUNTIME_TORCH_VERSION = "2.10.0+cu126"
NOTEBOOK_REF = "vanila111/trauma-predict-grud-h1-joint-m4-v2-resume-2500"
RESUME_SOURCE_ROOT = Path("/kaggle/temp/grud_v2_resume_source")
RESUME_STATE_ROOT = Path("/kaggle/temp/grud_v2_resume_state")
OUTPUT_ROOT = Path("/kaggle/working/p100_grud_h1_joint_m4_v2_resume_2500")
LOG_ROOT = Path("/kaggle/working/logs")
EXPECTED_RESUME_STEP = 2500
EXPECTED_TARGET_STEP = 4000
EXPECTED_CHECKPOINT_SHA256 = (
    "ba5da75fe63808374916fd270f899e45f3a9c0c3452c85fdbe6edc5dfb233054"
)
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "28dac8094956b407cb8721a6e6b98f171a81867de79569f8ed121bef62021221"
)
EXPECTED_SOURCE_METRICS_SHA256 = (
    "422ba862a026c7c920cab2fcf93327b5365ae33e1decef16f3e9d992f1c9c72e"
)
EXPECTED_SOURCE_TRAINING_MANIFEST_SHA256 = (
    "21ec36a674d897fdbe9a565cec293a3823c0de3dda8fa9f5bec38e51882698cd"
)

_FRESH_BOOTSTRAP: ModuleType | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _safe_relative(value: Any, label: str) -> Path:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} is not a safe relative path")
    return relative


def _resolve_file(bundle: Path, row: Mapping[str, Any], label: str) -> Path:
    relative = _safe_relative(row.get("path"), f"{label}.path")
    path = bundle / relative
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is absent: {path}")
    if path.stat().st_size != int(row.get("size_bytes", -1)):
        raise ValueError(f"{label} size differs from its manifest")
    if sha256_file(path) != str(row.get("sha256") or ""):
        raise ValueError(f"{label} hash differs from its manifest")
    return path


def _find_manifest(
    root: Path,
    name: str,
    *,
    schema: str,
    dataset_ref: str,
) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob(name)):
        if path.is_symlink() or not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == schema and payload.get("dataset_ref") == dataset_ref:
            matches.append((path.parent.resolve(), payload))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one pre-bound {dataset_ref} Input, found {len(matches)}"
        )
    return matches[0]


def _extract_regular_ustar(archive: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"extraction destination already exists: {destination}")
    destination.mkdir(parents=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"archive member escapes destination: {member.name}") from exc
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"archive permits regular files only: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            with source, target.open("xb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load manifest-bound launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_resume_inputs(input_root: str | Path = "/kaggle/input") -> dict[str, str]:
    global _FRESH_BOOTSTRAP

    root = Path(input_root)
    science_bundle, science_manifest = _find_manifest(
        root,
        "grud_v2_bundle_manifest.json",
        schema=SCIENCE_SCHEMA,
        dataset_ref=SCIENCE_DATASET_REF,
    )
    if (
        science_manifest.get("fresh_start") is not True
        or int(science_manifest.get("target_step", -1)) != EXPECTED_TARGET_STEP
    ):
        raise ValueError("mounted science Dataset is not the frozen fresh baseline input")
    science_launcher = _resolve_file(
        science_bundle,
        _mapping(science_manifest.get("launcher"), "science.launcher"),
        "fresh science launcher",
    )
    fresh_bootstrap = _load_module(science_launcher, "grud_v2_fresh_bootstrap")
    science_state = fresh_bootstrap.bind_science_input(root)

    resume_bundle, resume_manifest = _find_manifest(
        root,
        "grud_v2_resume_bundle_manifest.json",
        schema=RESUME_SCHEMA,
        dataset_ref=RESUME_DATASET_REF,
    )
    if (
        resume_manifest.get("route") != "grud_h1_to_joint_m4_v2"
        or resume_manifest.get("run_name") != "p100_grud_h1_joint_m4_v2_resume_2500"
        or resume_manifest.get("notebook_ref") != NOTEBOOK_REF
        or resume_manifest.get("science_dataset_ref") != SCIENCE_DATASET_REF
        or resume_manifest.get("runtime_dataset_ref") != RUNTIME_DATASET_REF
        or int(resume_manifest.get("resume_step", -1)) != EXPECTED_RESUME_STEP
        or int(resume_manifest.get("target_step", -1)) != EXPECTED_TARGET_STEP
        or int(resume_manifest.get("new_optimizer_steps", -1)) != 1500
        or resume_manifest.get("rng_continuity")
        != "deterministic_reset_not_bitwise_equivalent"
    ):
        raise ValueError("resume bundle differs from the frozen 2500-to-4000 contract")
    source_archive = _resolve_file(
        resume_bundle,
        _mapping(resume_manifest.get("source_release"), "resume.source_release"),
        "resume source archive",
    )
    state_archive = _resolve_file(
        resume_bundle,
        _mapping(resume_manifest.get("resume_state"), "resume.resume_state"),
        "resume state archive",
    )
    launcher = _resolve_file(
        resume_bundle,
        _mapping(resume_manifest.get("launcher"), "resume.launcher"),
        "resume launcher",
    )
    if launcher.resolve() != Path(__file__).resolve():
        raise ValueError("executed resume launcher is not manifest-bound")

    _extract_regular_ustar(source_archive, RESUME_SOURCE_ROOT)
    _extract_regular_ustar(state_archive, RESUME_STATE_ROOT)
    checkpoint = RESUME_STATE_ROOT / "checkpoint-2500/checkpoint.pt"
    checkpoint_manifest = RESUME_STATE_ROOT / "checkpoint-2500/manifest.json"
    source_metrics = RESUME_STATE_ROOT / "source/metrics.jsonl"
    source_training_manifest = RESUME_STATE_ROOT / "source/training_manifest.json"
    locks = (
        (checkpoint, EXPECTED_CHECKPOINT_SHA256, "checkpoint"),
        (checkpoint_manifest, EXPECTED_CHECKPOINT_MANIFEST_SHA256, "checkpoint manifest"),
        (source_metrics, EXPECTED_SOURCE_METRICS_SHA256, "source metrics"),
        (
            source_training_manifest,
            EXPECTED_SOURCE_TRAINING_MANIFEST_SHA256,
            "source training manifest",
        ),
    )
    for path, expected, label in locks:
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"recovered {label} differs from the cancelled v4 output")

    _FRESH_BOOTSTRAP = fresh_bootstrap
    print(
        "GRUD_V2_RESUME_INPUT_OK "
        f"resume_bundle={resume_bundle} step=2500 target_step=4000 "
        f"checkpoint_sha256={EXPECTED_CHECKPOINT_SHA256}",
        flush=True,
    )
    return {
        "data_root": str(science_state["data_root"]),
        "normalization": str(science_state["normalization"]),
        "source_root": str(RESUME_SOURCE_ROOT),
        "checkpoint": str(checkpoint),
        "checkpoint_manifest": str(checkpoint_manifest),
        "source_metrics": str(source_metrics),
        "source_training_manifest": str(source_training_manifest),
    }


def install_p100_runtime() -> dict[str, str]:
    if _FRESH_BOOTSTRAP is None:
        raise RuntimeError("bind_resume_inputs must run before runtime installation")
    return dict(_FRESH_BOOTSTRAP.install_p100_runtime())


def _validate_checkpoint_artifact(
    checkpoint: Path,
    *,
    step: int,
    expected_sha256: str,
    require_exact_resume_state: bool,
) -> None:
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is absent: {checkpoint}")
    if sha256_file(checkpoint) != expected_sha256:
        raise ValueError(f"checkpoint hash differs at step {step}")
    manifest_path = checkpoint.with_name("manifest.json")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is absent at step {step}")
    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        f"checkpoint-{step} manifest",
    )
    expected_schema = (
        "trauma_predict.grud_h1_v2_checkpoint_manifest.v2"
        if require_exact_resume_state
        else "trauma_predict.grud_h1_v2_checkpoint_manifest.v1"
    )
    if (
        manifest.get("schema_version") != expected_schema
        or int(manifest.get("step", -1)) != step
        or manifest.get("checkpoint") != "checkpoint.pt"
        or manifest.get("checkpoint_sha256") != expected_sha256
        or bool(manifest.get("exact_resume_state", False))
        != require_exact_resume_state
    ):
        raise ValueError(f"checkpoint manifest identity differs at step {step}")


def _validate_checkpoint_payload(
    checkpoint: Path,
    *,
    step: int,
    require_exact_resume_state: bool,
) -> None:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint root is not a mapping at step {step}")
    expected_checkpoint_schema = (
        "trauma_predict.grud_h1_v2_checkpoint.v2"
        if require_exact_resume_state
        else "trauma_predict.grud_h1_v2_checkpoint.v1"
    )
    if (
        payload.get("schema_version") != expected_checkpoint_schema
        or payload.get("model_contract") != "grud_h1_joint_m4_v2"
        or int(payload.get("step", -1)) != step
        or not {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "grad_scaler_state_dict",
            "validation",
        }.issubset(payload)
    ):
        raise ValueError(f"checkpoint metadata differs at step {step}")
    if require_exact_resume_state:
        trainer = _mapping(payload.get("trainer_state"), f"checkpoint-{step}.trainer_state")
        sampler = _mapping(payload.get("sampler_state"), f"checkpoint-{step}.sampler_state")
        rng = _mapping(payload.get("rng_state"), f"checkpoint-{step}.rng_state")
        if (
            int(trainer.get("global_step", -1)) != step
            or int(trainer.get("microbatches_consumed_in_epoch", -1)) < 0
            or int(trainer.get("gradient_accumulation_steps", -1)) != 2
            or int(trainer.get("optimizer_steps_per_epoch", -1)) != 48
            or trainer.get("exact_rng_continuity_from_this_checkpoint") is not True
            or int(trainer.get("source_non_bitwise_resume_step", -1))
            != EXPECTED_RESUME_STEP
            or int(sampler.get("epoch", -1)) != int(trainer.get("epoch", -2))
            or not {"python", "torch_cpu", "torch_cuda"}.issubset(rng)
        ):
            raise ValueError(f"exact continuation state differs at step {step}")


def _audit_training_checkpoint_payloads(output_root: Path = OUTPUT_ROOT) -> None:
    import torch

    if (
        os.environ.get("TRAUMA_PREDICT_RUNTIME_LOCK_SHA256")
        != EXPECTED_RUNTIME_CONTRACT_SHA256
        or str(torch.__version__) != EXPECTED_RUNTIME_TORCH_VERSION
    ):
        raise RuntimeError("checkpoint payload audit is outside the hash-locked P100 runtime")
    manifest = _mapping(
        json.loads((output_root / "training_manifest.json").read_text(encoding="utf-8")),
        "training manifest",
    )
    selected = _mapping(
        manifest.get("selected_checkpoint"), "training_manifest.selected_checkpoint"
    )
    selected_step = int(selected.get("step", -1))
    selected_path = output_root / _safe_relative(
        selected.get("path"), "training_manifest.selected_checkpoint.path"
    )
    _validate_checkpoint_payload(
        selected_path,
        step=selected_step,
        require_exact_resume_state=selected_step > EXPECTED_RESUME_STEP,
    )
    if selected_step != EXPECTED_TARGET_STEP:
        final = _mapping(
            manifest.get("step_4000_checkpoint"),
            "training_manifest.step_4000_checkpoint",
        )
        _validate_checkpoint_payload(
            output_root
            / _safe_relative(
                final.get("path"), "training_manifest.step_4000_checkpoint.path"
            ),
            step=EXPECTED_TARGET_STEP,
            require_exact_resume_state=True,
        )
    print(
        "GRUD_V2_RESUME_CHECKPOINT_PAYLOAD_AUDIT_OK "
        f"torch={torch.__version__} selected_step={selected_step} final_step=4000",
        flush=True,
    )


def _validate_training_output(
    output_root: Path = OUTPUT_ROOT,
) -> tuple[Path, Path, str]:
    manifest_path = output_root / "training_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("completed resume training lacks training_manifest.json")
    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")), "training manifest"
    )
    if (
        manifest.get("schema_version")
        != "trauma_predict.grud_h1_v2_training_manifest.v1"
        or manifest.get("route") != "grud_h1_to_joint_m4_v2"
        or manifest.get("execution_mode") != "resume_2500_to_4000"
        or manifest.get("status") != "SUCCEEDED"
        or int(manifest.get("completed_step", -1)) != EXPECTED_TARGET_STEP
    ):
        raise ValueError("training manifest does not close the resumed 4000-step contract")
    lineage = _mapping(manifest.get("resume_lineage"), "training_manifest.resume_lineage")
    if (
        int(lineage.get("checkpoint_step", -1)) != EXPECTED_RESUME_STEP
        or lineage.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
        or lineage.get("rng_continuity")
        != "deterministic_reset_not_bitwise_equivalent"
    ):
        raise ValueError("completed training manifest lost its recovery lineage")

    selected_row = _mapping(
        manifest.get("selected_checkpoint"), "training_manifest.selected_checkpoint"
    )
    selected_step = int(selected_row.get("step", -1))
    if selected_step not in {2500, 3000, 3500, 4000}:
        raise ValueError("selected checkpoint step is outside the resumed save schedule")
    selected_relative = _safe_relative(
        selected_row.get("path"), "training_manifest.selected_checkpoint.path"
    )
    if selected_relative.as_posix() != f"checkpoint-{selected_step}/checkpoint.pt":
        raise ValueError("selected checkpoint identity differs from its manifest")
    selected = output_root / selected_relative
    selected_hash = str(selected_row.get("sha256") or "")
    _validate_checkpoint_artifact(
        selected,
        step=selected_step,
        expected_sha256=selected_hash,
        require_exact_resume_state=selected_step > EXPECTED_RESUME_STEP,
    )

    final_row = _mapping(
        manifest.get("step_4000_checkpoint"), "training_manifest.step_4000_checkpoint"
    )
    final_relative = _safe_relative(
        final_row.get("path"), "training_manifest.step_4000_checkpoint.path"
    )
    final_checkpoint = output_root / final_relative
    if final_relative.as_posix() != "checkpoint-4000/checkpoint.pt":
        raise ValueError("step-4000 checkpoint identity differs from its manifest")
    _validate_checkpoint_artifact(
        final_checkpoint,
        step=EXPECTED_TARGET_STEP,
        expected_sha256=str(final_row.get("sha256") or ""),
        require_exact_resume_state=True,
    )
    return selected, manifest_path, selected_hash


def run_training(
    resume_state: Mapping[str, str],
    runtime_environment: Mapping[str, str],
) -> None:
    source_root = Path(resume_state["source_root"])
    data_root = Path(resume_state["data_root"])
    script = source_root / "notebooks/kaggle/train_grud_h1_joint_m4_v2_resume.py"
    config = source_root / "configs/train/p100_grud_h1_joint_m4_v2_resume_2500.yaml"
    if script.is_symlink() or not script.is_file() or not config.is_file():
        raise FileNotFoundError("resume source release lacks the formal training entry")
    environment = dict(runtime_environment)
    inherited = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(source_root / "src"), inherited) if item
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TRAUMA_PREDICT_GRUD_H1_ROOT"] = str(data_root / "h1")
    environment["TRAUMA_PREDICT_V2_TARGET_ROOT"] = str(data_root / "target")
    environment["TRAUMA_PREDICT_GRUD_NORMALIZATION_PATH"] = str(
        resume_state["normalization"]
    )
    environment["TRAUMA_PREDICT_GRUD_RESUME_CHECKPOINT"] = str(
        resume_state["checkpoint"]
    )
    environment["TRAUMA_PREDICT_GRUD_RESUME_MANIFEST"] = str(
        resume_state["checkpoint_manifest"]
    )
    environment["TRAUMA_PREDICT_GRUD_RESUME_SOURCE_METRICS"] = str(
        resume_state["source_metrics"]
    )
    environment["TRAUMA_PREDICT_GRUD_RESUME_SOURCE_TRAINING_MANIFEST"] = str(
        resume_state["source_training_manifest"]
    )
    environment["TRAUMA_PREDICT_OUTPUT_ROOT"] = "/kaggle/working"
    environment["TRAUMA_PREDICT_REPO_ROOT"] = str(source_root)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    train_log = LOG_ROOT / "grud_v2_resume_training.log"
    command = [
        sys.executable,
        str(script),
        "--repo-root",
        str(source_root),
        "--config",
        str(config),
    ]
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("resume training process has no stdout pipe")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    tail: deque[str] = deque(maxlen=80)
    started = time.monotonic()
    last_heartbeat = started
    latest_step = EXPECTED_RESUME_STEP
    with train_log.open("w", encoding="utf-8") as log_handle:
        while process.poll() is None:
            ready = selector.select(timeout=5.0)
            for key, _ in ready:
                line = key.fileobj.readline()
                if not line:
                    continue
                log_handle.write(line)
                log_handle.flush()
                stripped = line.rstrip("\n")
                tail.append(stripped)
                if stripped.startswith("GRUD_V2_"):
                    print(stripped, flush=True)
                for token in stripped.split():
                    if token.startswith("step=") and token[5:].isdigit():
                        latest_step = max(latest_step, int(token[5:]))
            now = time.monotonic()
            if now - last_heartbeat >= 300:
                print(
                    "GRUD_V2_RESUME_HEARTBEAT "
                    f"phase=train elapsed_s={int(now-started)} step={latest_step} "
                    f"log_bytes={train_log.stat().st_size}",
                    flush=True,
                )
                last_heartbeat = now
        for line in process.stdout:
            log_handle.write(line)
            stripped = line.rstrip("\n")
            tail.append(stripped)
            if stripped.startswith("GRUD_V2_"):
                print(stripped, flush=True)
    returncode = process.wait()
    if returncode:
        print(
            "GRUD_V2_RESUME_FAILED",
            f"returncode={returncode}",
            *tail,
            sep="\n",
            flush=True,
        )
        raise RuntimeError("GRU-D step-2500 continuation failed")

    selected, manifest_path, expected_hash = _validate_training_output()
    audit = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--audit-training-output",
            str(OUTPUT_ROOT),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if audit.stdout:
        print(audit.stdout.rstrip(), flush=True)
    if audit.returncode:
        if audit.stderr:
            print(audit.stderr.rstrip(), flush=True)
        raise RuntimeError("isolated-runtime checkpoint payload audit failed")
    print(
        "GRUD_V2_RESUME_EXPORT_OK "
        f"checkpoint={selected} sha256={expected_hash} manifest={manifest_path}",
        flush=True,
    )
    print("GRUD_V2_RESUME_NOTEBOOK_FINISHED status=SUCCEEDED step=4000", flush=True)


def main() -> None:
    state = bind_resume_inputs()
    environment = install_p100_runtime()
    run_training(state, environment)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--audit-training-output":
        _audit_training_checkpoint_payloads(Path(sys.argv[2]).resolve())
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("usage: launcher.py [--audit-training-output OUTPUT_ROOT]")
