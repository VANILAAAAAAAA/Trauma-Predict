# File Index

Every tracked file in this repository must appear in this table. Run `python tools/update_file_index.py --check` before committing.

| Path | Area | Purpose | Data policy |
| --- | --- | --- | --- |
| `.env.example` | config | Environment variable template for local/Kaggle paths. | No secrets or real data paths. |
| `.gitignore` | repo | Blocks data, checkpoints, caches, and secrets from Git. | Protects restricted artifacts. |
| `README.md` | docs | Repository entry point and scope boundary. | Documents code-only policy. |
| `configs/accelerate/single_gpu.yaml` | config | Single-GPU fallback accelerator config. | No data. |
| `configs/accelerate/t4x2.yaml` | config | Kaggle T4 x2 accelerator config. | No data. |
| `configs/contracts/multires_event_v2/field_category_matrix_v1.csv` | config | Frozen 37-field channel/category authority used by Relation V2 axis validation. | Registry metadata only. |
| `configs/contracts/multires_event_v2/input_target_relation_edges_v2.csv` | config | Frozen 39-row history-input to future-output relation table. | Registry metadata only. |
| `configs/contracts/multires_event_v2/relation_evidence_registry_v2.json` | config | Evidence registry bound to every Relation V2 edge. | No patient rows. |
| `configs/contracts/multires_event_v2/target_target_relation_edges_v2.csv` | config | Frozen 52-row target-target relation table with edge-specific parameter keys. | Registry metadata only. |
| `configs/dataset/first_train.yaml` | config | First training dataset artifact paths and required sample fields. | Uses environment-variable paths only. |
| `configs/dataset/multires_event_v1_c4.yaml` | config | Frozen C4 multires event dataset identity, inventory, loader, and split contract. | Uses environment-variable paths only. |
| `configs/dataset/multires_event_v2_relation_v2_c4.yaml` | config | Joins the immutable V1 input base to the accepted r9 target sidecar for the strict Relation V2 route. | Uses environment-variable paths only. |
| `configs/dataset/multires_event_v2_c4_lab_affine_scale.json` | config | Train-subject-only affine scale for V2 laboratory likelihoods. | Aggregate scale metadata only. |
| `configs/dataset/multires_event_v2_c4_lab_affine_scale_r9.json` | config | Train-subject-only affine scale refit against the accepted r9 target authority. | Aggregate scale metadata only. |
| `configs/dataset/multires_event_v2_c4_standardized_primitive_scale.json` | config | Frozen standardized primitive scales for the V2 mixed-coordinate likelihood. | Aggregate scale metadata only. |
| `configs/dataset/multires_event_v2_c4_standardized_primitive_scale_r9.json` | config | Frozen r9 standardized primitive scales for the V2 mixed-coordinate likelihood. | Aggregate scale metadata only. |
| `configs/evaluation/multires_event_v2_relation_v2_metrics.json` | config | Report-only trajectory metric contract over the 23 registered cross-output edges. | No patient rows. |
| `configs/model/multires_event_v1.yaml` | config | Scratch hierarchical event Transformer and typed-head architecture contract. | No data. |
| `configs/model/multires_event_v1_supervision.json` | config | Model-side target overlay over the immutable 1,314-row canonical target. | Registry metadata only. |
| `configs/model/multires_event_v2_relation_v2.yaml` | config | Frozen 48,728,439-parameter Relation V2 architecture with mandatory target-target/input-target paths and 48-parameter input-only temporal fusion. | No data. |
| `configs/train/p100_stage_a_hour.yaml` | config | Stage A single-GPU/P100 HOUR values-only training config. | Uses environment-variable paths only. |
| `configs/train/t4x2_first_run.yaml` | config | Joint-baseline T4 x2 training config; not Stage A. | Uses environment-variable paths only. |
| `configs/train/t4x2_multires_event_v1_full.yaml` | config | Frozen 4,000-step T4 x2 multires baseline training and evaluation contract. | Uses environment-variable paths only. |
| `configs/train/t4x2_multires_event_v1_smoke.yaml` | config | Two-step T4 x2 DDP smoke contract run before the full multires baseline. | Uses environment-variable paths only. |
| `configs/train/p100_multires_event_v2_relation_v2.yaml` | config | Single authorized P100 Relation V2 identity with B64, raw 414-factor optimizer, and resumable 4,000-step contract. | Uses environment-variable paths only. |
| `configs/train/t4x2_smoke.yaml` | config | Joint-baseline smoke config; not Stage A. | Uses environment-variable paths only. |
| `configs/train/t4x2_stage_a_hour.yaml` | config | Stage A T4 x2 HOUR values-only training config with ventilation and `NEXT_24H` losses inactive. | Uses environment-variable paths only. |
| `configs/train/t4x2_stage_a_hour_smoke.yaml` | config | Stage A smoke config that proves HOUR values-only model/data/runtime wiring. | Uses environment-variable paths only. |
| `docs/DATA_POLICY.md` | docs | Allowed and forbidden repository content policy. | No data. |
| `docs/FILE_INDEX.md` | docs | Tracked-file index. | No data. |
| `docs/KAGGLE_RUNBOOK.md` | docs | Kaggle launch and output policy. | No data. |
| `docs/REPO_STRUCTURE.md` | docs | Directory structure and design rules. | No data. |
| `docs/TRAINING_STAGES.md` | docs | Stage A/B/C and joint-baseline training contract. | No data. |
| `notebooks/kaggle/README.md` | kaggle | Explains Kaggle launcher folder boundary. | No data. |
| `notebooks/kaggle/historical_v8_dataset_evidence.py` | kaggle | Retains v8 Dataset identity, hash, deterministic materialization, and log evidence without any training or promotion entrypoint. | Reads historical fixture or mounted bytes only; external actions fail closed. |
| `notebooks/kaggle/kernel-metadata-relation-v2-p100.template.json` | kaggle | Optional private P100 kernel metadata binding the frozen Relation V2 Dataset and notebook slug. | No embedded data or credentials. |
| `notebooks/kaggle/run_multires_event_v1.py` | kaggle | Pinned Kaggle launcher for data acquisition, preflight, DDP smoke, full training, and output verification. | Downloads or reads the private frozen artifact only at runtime. |
| `notebooks/kaggle/run_multires_event_v2.py` | kaggle | Small historical v8 launcher stub with no mode, promotion, config, or training API. | Stops immediately. |
| `notebooks/kaggle/run_relation_v2_p100_bundle.py` | kaggle | Hash-validates the private bundle and prior notebook output, then advances one formal P100 training/evaluation stage. | Reads the frozen private Dataset and writes resumable outputs only at runtime. |
| `notebooks/kaggle/run_relational_primary_bundle.py` | kaggle | Historical v8 bundle launcher retained as fail-closed evidence; it cannot launch Relation V2. | Stops before reading mounted inputs. |
| `notebooks/kaggle/run_stage_a_hour.py` | kaggle | Automated Stage A Kaggle launcher for preferred-encoder HOUR-only training. | Reads mounted private data only at runtime. |
| `notebooks/kaggle/scan_token_lengths.py` | kaggle | Scans shard input token lengths against the configured encoder window before training. | Reads mounted private data only at runtime. |
| `notebooks/kaggle/train_kaggle.py` | kaggle | Kaggle-compatible training entrypoint wrapper. | No data. |
| `notebooks/kaggle/train_multires_event_v1.ipynb` | kaggle | Two-cell Save & Run notebook pinned to the immutable multires baseline tag. | No embedded data. |
| `notebooks/kaggle/train_multires_event_v1.py` | kaggle | DDP training entrypoint and dry-run preflight for the multires route. | Reads the frozen private artifact only at runtime. |
| `notebooks/kaggle/train_relation_v2_p100.py` | kaggle | Single-process formal Relation V2 P100 training entrypoint. | Reads only manifest-bound runtime paths. |
| `notebooks/kaggle/train_relational_primary.py` | kaggle | Historical v8 entrypoint retained as fail-closed evidence; it does not resolve a training config. | Stops before importing training code. |
| `notebooks/kaggle/train_multires_event_v2.ipynb` | kaggle | Historical v8 multi-mode Notebook retained with an immediate fail-closed cell. | No embedded data and no active launch. |
| `notebooks/kaggle/train_multires_event_v2.py` | kaggle | Historical v8 mode/capacity entrypoint retained as fail-closed evidence. | Stops before importing training code. |
| `notebooks/kaggle/train_multires_event_v2_relational_primary.ipynb` | kaggle | Historical v8 Notebook retained with an immediate fail-closed cell; it is not the active P100 Relation V2 route. | No embedded patient data and no active launch. |
| `notebooks/kaggle/verify_multires_event_v2.ipynb` | kaggle | Historical v8 verification Notebook retained with an immediate fail-closed cell. | No embedded data and no active verification. |
| `notebooks/kaggle/train_full_first_run.ipynb` | kaggle | End-to-end Kaggle notebook for the joint-baseline run; not Stage A. | No data. |
| `notebooks/kaggle/train_stage_a_hour.ipynb` | kaggle | End-to-end Kaggle notebook for Stage A HOUR-only training. | No data. |
| `notebooks/kaggle/trauma_predict_relation_v2_p100_r9.ipynb` | kaggle | Thin UI-upload Notebook for the staged P100 Relation V2 run and authenticated private-Dataset fallback. | No embedded data, source download, or credentials. |
| `notebooks/kaggle/verify_private_dataset.ipynb` | kaggle | Kaggle notebook that verifies private Dataset mounting or API download before preflight. | No data. |
| `pyproject.toml` | packaging | Python package, optional dependencies, and test config. | No data. |
| `requirements-kaggle.txt` | packaging | Kaggle install requirements. | No data. |
| `requirements-multires-kaggle.txt` | packaging | Pinned direct dependencies for the multires Kaggle route. | No data. |
| `schemas/dataset_manifest.schema.json` | schema | Contract for generated dataset manifests. | Schema only. |
| `schemas/multires_event_dataset_manifest.schema.json` | schema | Contract for the frozen multires event dataset manifest. | Schema only. |
| `schemas/multires_event_normalization.schema.json` | schema | Contract for train-subject-only robust normalization statistics. | Schema only. |
| `schemas/multires_event_sample.schema.json` | schema | Contract for canonical multires event sample records. | Schema only. |
| `schemas/multires_event_supervision.schema.json` | schema | Contract for the model-side target overlay. | Schema only. |
| `schemas/multires_event_v2_dataset_manifest.schema.json` | schema | Contract for joined V1-base/V2-target dataset identity. | Schema only. |
| `schemas/multires_event_v2_target.schema.json` | schema | Contract for each six-block field-process target record. | Schema only. |
| `schemas/sample_manifest.schema.json` | schema | Contract for generated sample manifests. | Schema only. |
| `src/trauma_predict/__init__.py` | package | Package version. | No data. |
| `src/trauma_predict/cli.py` | package | CLI entry point for repository checks. | No data. |
| `src/trauma_predict/data/__init__.py` | package | Data utility namespace. | No data. |
| `src/trauma_predict/data/main_route.py` | package | Loads main-route records and batches HOUR side tensors for training. | No data. |
| `src/trauma_predict/data/main_route_contract.py` | package | Validates the standard textual V1 main-route record, HOUR tensors, and structured targets. | No data. |
| `src/trauma_predict/data/manifest.py` | package | Dataset manifest loading and validation helpers. | No data. |
| `src/trauma_predict/data/multires_event/__init__.py` | package | Public multires data contract, dataset, sampler, normalizer, collator, and runtime exports. | No data. |
| `src/trauma_predict/data/multires_event/collator.py` | package | Converts compact events and frozen target slots into aligned model tensors. | No data. |
| `src/trauma_predict/data/multires_event/contract.py` | package | Compiles supervision rules into fixed H1/M4 queries and derived F24 mappings. | Registry metadata only. |
| `src/trauma_predict/data/multires_event/dataset.py` | package | Lazy shard-backed multires dataset with model-side input filtering. | Reads private data only at runtime. |
| `src/trauma_predict/data/multires_event/normalization.py` | package | Fits and applies train-subject-only robust numeric normalization. | Reads private training data only at runtime. |
| `src/trauma_predict/data/multires_event/preflight.py` | package | Validates supervision, registry, and multires artifact identities before loading. | No data. |
| `src/trauma_predict/data/multires_event/sampler.py` | package | Subject-uniform train and duplicate-free distributed evaluation samplers. | No data. |
| `src/trauma_predict/data/multires_event_v2/__init__.py` | package | Public V2 data contract, dataset, collator, and preflight exports. | No data. |
| `src/trauma_predict/data/multires_event_v2/collator.py` | package | Converts aligned V1 input and accepted r9 process targets into typed tensors and gates. | Reads private records only at runtime. |
| `src/trauma_predict/data/multires_event_v2/contract.py` | package | Validates accepted r9 identities, arithmetic evidence, process support, and field order while retaining the embedded historical relation hash. | Registry metadata only. |
| `src/trauma_predict/data/multires_event_v2/dataset.py` | package | Exact sample-identity join over the immutable V1 base and accepted r9 target sidecar. | Reads private data only at runtime. |
| `src/trauma_predict/data/multires_event_v2/preflight.py` | package | Full-data V2 identity, count, shard-header, model, and batch preflight. | Reads private manifests and headers only at runtime. |
| `src/trauma_predict/data/multires_event_v2/relation_contract.py` | package | Strictly loads and hash-binds the 52+39 Relation V2 edge tables, channels, scopes, and evidence. | Registry metadata only. |
| `src/trauma_predict/data/preflight.py` | package | Validates generated training artifacts before Kaggle execution. | No data. |
| `src/trauma_predict/data/records.py` | package | Reads generated JSONL shard records for training. | No data. |
| `src/trauma_predict/data/splits.py` | package | Patient-level split invariant helpers. | No data. |
| `src/trauma_predict/eval/__init__.py` | package | Evaluation namespace. | No data. |
| `src/trauma_predict/eval/f24_composition.py` | package | Deterministically composes raw-unit F24 predictions from six predicted M4 blocks. | No data. |
| `src/trauma_predict/eval/metrics.py` | package | Basic metric aggregation helpers. | No data. |
| `src/trauma_predict/eval/multires_event.py` | package | Typed F24 diagnostics with field- and subject-macro aggregation. | No data. |
| `src/trauma_predict/eval/multires_event_v2.py` | package | Raw joint-NLL, subject-macro, and V2 decision-metric aggregation. | No data. |
| `src/trauma_predict/eval/multires_event_v2_free_running.py` | package | Ancestral rollout evaluation with 100-trajectory capacity proof, atomic chunk resume, sufficient statistics, and coherent trajectory export. | Writes derived predictions outside Git. |
| `src/trauma_predict/eval/multires_event_v2_metric_contract.py` | package | Validates the report-only Relation V2 metric contract and exact 23-edge cover. | No data. |
| `src/trauma_predict/eval/multires_event_v2_projections.py` | package | Deterministic five-tuple and physical projection runtime over sampled primitives. | No data. |
| `src/trauma_predict/eval/multires_event_v2_scale.py` | package | Loads and applies frozen V2 likelihood and reporting scales. | Aggregate scale metadata only. |
| `src/trauma_predict/modeling/__init__.py` | package | Modeling namespace. | No data. |
| `src/trauma_predict/modeling/main_route.py` | package | Encoder model with HourStateAdapter injection and structured prediction heads. | No data. |
| `src/trauma_predict/modeling/multires_event/__init__.py` | package | Public scratch multires model exports. | No data. |
| `src/trauma_predict/modeling/multires_event/decoder.py` | package | Fixed legal-query embedding and block-local future query decoder. | No data. |
| `src/trauma_predict/modeling/multires_event/embeddings.py` | package | Event, time-block, value, study-slot, and STATIC embedding modules. | No data. |
| `src/trauma_predict/modeling/multires_event/encoder.py` | package | Learned block-latent compressor and temporal trajectory encoder. | No data. |
| `src/trauma_predict/modeling/multires_event/heads.py` | package | Typed probabilistic output heads for all active loss families. | No data. |
| `src/trauma_predict/modeling/multires_event/model.py` | package | End-to-end scratch hierarchical event Transformer assembly. | No data. |
| `src/trauma_predict/modeling/multires_event_v2/config.py` | package | Validates the fixed V2 structured encoder-decoder architecture. | No data. |
| `src/trauma_predict/modeling/multires_event_v2/emissions.py` | package | Normalized mixed discrete/continuous process likelihoods and samplers. | No data. |
| `src/trauma_predict/modeling/multires_event_v2/field_state.py` | package | Typed field-process state embedding and autoregressive feedback assembly. | No data. |
| `src/trauma_predict/modeling/multires_event_v2/input_field_memory.py` | package | Builds 37 history-field states with final-H1 target bridges and learned block-geometry softmax pooling for eight input-only fields. | No data. |
| `src/trauma_predict/modeling/multires_event_v2/model.py` | package | End-to-end strict Relation V2 model with both registered relation paths always active. | No data. |
| `src/trauma_predict/modeling/multires_event_v2/relation_bias.py` | package | Edge-specific additive attention residuals keyed by the 91 frozen parameter keys. | No data. |
| `src/trauma_predict/modeling/multires_event_v2/rollout.py` | package | Autoregressive process sampling with deterministic projections. | No data. |
| `src/trauma_predict/modeling/multires_event_v2/trajectory.py` | package | Joint causal trajectory decoder with mandatory target-target and input-target relation scopes. | No data. |
| `src/trauma_predict/training/__init__.py` | package | Training namespace. | No data. |
| `src/trauma_predict/training/checkpoints.py` | package | Checkpoint retention helpers. | No data. |
| `src/trauma_predict/training/config.py` | package | YAML config loading and environment expansion. | No data. |
| `src/trauma_predict/training/main_route.py` | package | Hugging Face Trainer loop for main-route structured prediction. | No data. |
| `src/trauma_predict/training/multires_event.py` | package | DDP training, interval/final evaluation, resume identity, and artifact export for the multires route. | No data. |
| `src/trauma_predict/training/multires_event_loss.py` | package | Typed probabilistic losses and component-field-resolution macro aggregation. | No data. |
| `src/trauma_predict/training/multires_event_v2.py` | package | Single-P100 Relation V2 training with optimizer-health audit, staged checkpoint reopen, cached teacher evaluation, and resumable free-running integration. | Reads private data and writes outputs only at runtime. |
| `src/trauma_predict/training/multires_event_v2_loss.py` | package | Raw 414-factor joint canonical likelihood and primitive feedback contract. | No data. |
| `src/trauma_predict/training/observability.py` | package | Atomic JSON, rank-zero shared metrics, rank-local evidence, loss signals, and run heartbeat utilities. | No data. |
| `src/trauma_predict/training/runtime.py` | package | Shared training runtime helpers for logging, checkpoints, and snapshots. | No data. |
| `src/trauma_predict/training/stages.py` | package | Explicit Stage A/B/C and joint-baseline active-loss contracts. | No data. |
| `tests/helpers/multires_event_v2_best_checkpoint_worker.py` | tests | Two-process Gloo worker for the production best-checkpoint collective-order regression. | Synthetic zero-parameter checkpoint only. |
| `tests/helpers/multires_event_v2_rank_artifact_worker.py` | tests | Two-process Gloo worker for rank-local artifact success and failure-path regression. | Synthetic metadata only. |
| `tests/test_data_preflight.py` | tests | Tests generated artifact preflight checks with synthetic rows. | Synthetic records only. |
| `tests/test_manifest_contracts.py` | tests | Tests schema and manifest helper behavior. | Synthetic records only. |
| `tests/test_multires_event_contract.py` | tests | Tests target-overlay counts, semantics, and F24 mappings against the frozen registry. | Registry metadata only. |
| `tests/test_multires_event_data.py` | tests | Tests real-artifact filtering, sampling, normalization, and collator alignment. | Reads the immutable local artifact when mounted. |
| `tests/test_multires_event_kaggle_route.py` | tests | Tests notebook pinning, launcher order, dataset discovery, and shard extraction. | Synthetic metadata only. |
| `tests/test_multires_event_loss.py` | tests | Tests typed-loss numerical behavior, FP16 promotion, and duration censoring. | Synthetic tensors only. |
| `tests/test_multires_event_training.py` | tests | Tests training contract, aggregation, resume identity, and complete prediction export. | Synthetic tensors only. |
| `tests/test_multires_event_v2_checkpoint_identity.py` | tests | Tests checkpoint, optimizer, scheduler, scaler, and resume identity closure. | Synthetic tensors only. |
| `tests/test_multires_event_v2_formal_model.py` | tests | Tests the frozen Relation V2 formal parameter count and rejects removed alternative model identities. | Config metadata only. |
| `tests/test_multires_event_v2_formal_gradient_audit.py` | tests | Gated real-r9 CUDA FP16 audit of all 414 factors, all relation rows, temporal fusion rows, and legacy-checkpoint rejection. | Reads two persisted local samples and the retained historical checkpoint when mounted. |
| `tests/test_multires_event_v2_contract.py` | tests | Tests accepted r9 dataset, arithmetic-evidence, field-order, and contract identity validation. | Registry metadata and optional mounted manifests only. |
| `tests/test_multires_event_v2_data.py` | tests | Tests exact V1/r9 joins, collator targets, gates, and subject sampling. | Synthetic records and optional mounted manifests only. |
| `tests/test_multires_event_v2_emissions.py` | tests | Tests normalized process distributions, support, gates, and sampling. | Synthetic tensors only. |
| `tests/test_multires_event_v2_free_running.py` | tests | Tests seeded ancestral rollout, metrics, coherence, and artifacts. | Synthetic tensors only. |
| `tests/test_multires_event_v2_kaggle_route.py` | tests | Tests historical hosted-route failure closure while preserving Dataset identity, hash, materialization, and logging checks. | Synthetic metadata only. |
| `tests/test_multires_event_v2_loss.py` | tests | Tests exact raw 414-factor joint-NLL arithmetic and conditional composition. | Synthetic tensors only. |
| `tests/test_multires_event_v2_model.py` | tests | Tests strict model shapes, both edge-specific parameter banks, runtime override rejection, and cached rollout. | Synthetic tensors only. |
| `tests/test_multires_event_v2_p100_hosted_stages.py` | tests | Tests closed hosted stop schedules, step-250 checkpoint binding, free-running limits, and final-teacher row identity. | Synthetic metadata only. |
| `tests/test_multires_event_v2_projection_runtime.py` | tests | Tests physical projection closure and the report-only 23-edge Relation V2 metric contract. | Synthetic tensors and optional mounted contract only. |
| `tests/test_multires_event_v2_relation_contract_v2.py` | tests | Tests exact Relation V2 edge counts, channels, scopes, orientation, evidence, and mutation rejection. | Registry metadata only. |
| `tests/test_multires_event_v2_relation_v2_modeling.py` | tests | Tests field-history visibility, nonedge attention, all-edge gradients, and teacher/incremental equivalence. | Synthetic tensors only. |
| `tests/test_multires_event_v2_relation_v2_route.py` | tests | Tests the single hash-bound Relation V2 route and absence of runtime relation switches. | Config metadata only. |
| `tests/test_multires_event_v2_relations.py` | tests | Tests edge-specific bias, joint causal access, explicit scopes, and checkpoint key identity. | Registry metadata and synthetic tensors only. |
| `tests/test_multires_event_v2_training.py` | tests | Tests optimizer, health sequence, DDP batch, scheduler, authorization, and runtime contracts. | Synthetic tensors only. |
| `tests/test_multires_event_v2_trajectory.py` | tests | Tests the joint causal trajectory, generated feedback, and teacher/incremental cache equivalence. | Synthetic tensors only. |
| `tests/test_relational_primary_bundle.py` | tests | Tests mounted bundle discovery, no-copy dataset views, and source-extraction safety. | Synthetic files only. |
| `tests/test_relation_v2_p100_hosted_surfaces.py` | tests | Tests P100 notebook/builder/launcher identity, safe extraction, clean source release, checkpoint hashes, and chunk progress. | Synthetic files only. |
| `tests/test_repo_hygiene.py` | tests | Tests file index and forbidden repository paths. | No data. |
| `tests/test_training_main_route.py` | tests | Tests main-route config, label encoding, collator alignment, adapter shape, and checkpoint helpers. | Synthetic records only. |
| `tools/build_relational_primary_bundle.py` | tools | Historical v8 bundle builder retained as fail-closed evidence; it cannot package or resume Relation V2. | Stops without writing artifacts. |
| `tools/build_relation_v2_p100_bundle.py` | tools | Builds the clean source-bound private Kaggle Dataset bundle for the formal P100 Relation V2 route. | Packages only frozen derived artifacts; never raw MIMIC rows. |
| `tools/update_file_index.py` | tools | Validates that all tracked files appear in this index. | No data. |
| `uv.lock` | packaging | Exact dependency resolution used in semantic runtime identity. | No data. |
