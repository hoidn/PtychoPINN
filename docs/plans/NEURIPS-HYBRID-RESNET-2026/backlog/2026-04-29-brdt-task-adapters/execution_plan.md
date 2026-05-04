# BRDT Task Adapters Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. This item authorizes only the BRDT task-local loading, adapter, loss, and train/eval surfaces needed for the bounded four-row preflight. Do not register BRDT as a normal CDI generator, do not change the CDI/PtychoPINN generator contract, do not run the full BRDT four-row benchmark bundle here, do not promote BRDT into manuscript evidence, and do not create worktrees. If any sanity run becomes long-running, keep it under implementation ownership until the launched PID exits cleanly and the expected artifacts are freshly written; use `tmux` plus `ptycho311` when appropriate.

**Goal:** Add the task-local BRDT data-loading, Born-initialization, model-adapter, loss, and train/eval entry surfaces required so the later bounded four-row preflight can run classical Born backpropagation plus U-Net, FNO vanilla, and SRU/Hybrid-family rows under one shared contract.

**Architecture:** Keep BRDT entirely task-local under `scripts/studies/born_rytov_dt/`. Consume the locked operator and smoke-dataset contracts from the completed BRDT prerequisite artifacts, derive the first shared neural input representation as `born_init_image`, reuse existing model bodies only through ordinary real-channel adapters, and keep optimization semantics in a BRDT-specific training wrapper rather than the CDI generator registry or `PtychoPINN_Lightning` contract. The implementation splits cleanly into dataset/input preparation, model/loss/training surfaces, and durable contract/reporting surfaces.

**Tech Stack:** PATH `python`, PyTorch, `torch.utils.data`, existing `ptycho_torch` Hybrid/FNO components reused behind task-local wrappers, optional `neuralop` and `odtbrain` with controlled blocker handling, JSON/HDF5 manifests, pytest, compileall, Markdown summary docs.

---

## Selected Objective

- Implement backlog item `2026-04-29-brdt-task-adapters`.
- Consume the completed BRDT operator-validation and dataset-preflight authorities rather than redefining their contracts.
- Add the minimum runnable BRDT surfaces for:
  - classical Born backpropagation as a non-neural reference path;
  - U-Net with supervised plus Born consistency;
  - FNO vanilla with supervised plus Born consistency;
  - SRU-Net or Hybrid-family row with supervised plus Born consistency.
- Keep row metadata explicitly split into `model`, `training`, `input_mode`, `dataset_id`, `operator_version`, and `row_status`.
- Preserve the candidate-lane claim boundary: this item makes the later preflight runnable, but it does not itself become BRDT benchmark evidence.

## Scope

- Consume the binding BRDT authorities from:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/dataset_manifest.json`
- Reuse the checked-in dataset contract helper surface in `scripts/studies/born_rytov_dt/dataset_contract.py` for manifest parsing/building, train-only normalization, normalization inversion, and normalized-`q` operator guards instead of re-deriving those rules inside the new adapter code.
- Add task-local BRDT modules under `scripts/studies/born_rytov_dt/` for data loading, Born-initialization derivation, model adapters, loss/training configuration, and train/eval entrypoints.
- Add one authoritative test module, `tests/studies/test_born_rytov_dt_adapters.py`, that covers the new contract surfaces end to end.
- Add a durable checked-in adapter summary and index updates so the four-row preflight can consume the adapter contract without rereading backlog prose.

## Explicit Non-Goals

- Do not modify `ptycho_torch/model.py`, `ptycho_torch/train.py`, `ptycho_torch/generators/registry.py`, or register BRDT as a new CDI architecture.
- Do not change `ptycho_torch/physics/born_rytov_dt.py` unless a real operator-contract bug is discovered; if the locked operator authority is wrong, stop and open a separate narrow follow-up rather than drifting the contract here.
- Do not alter the completed BRDT smoke dataset files or rewrite the dataset preflight contract just to make adapters easier. Any Born initialization image or cache must be derived alongside the loader/runtime surface, not by silently mutating the canonical preflight dataset.
- Do not run the bounded four-row BRDT preflight item here. Small adapter sanity runs are allowed only to prove the new entrypoints; they must stay clearly below benchmark/preflight scope.
- Do not add Rytov mode, limited-angle rows, FFNO rows, spectral ResNet promotion rows, physics-only rows, multi-seed robustness, or paper-facing BRDT tables/figures.
- Do not compare direct-sinogram rows against Born-initialization-image rows. The first adapter contract must not broaden to mixed input modes.
- Do not label supervised plus Born consistency rows as `PINN-only`.
- Do not create `/home/ollie/Documents/neurips/` artifacts or manuscript prose.

## Steering, Roadmap, and Fairness Constraints

- Steering keeps CDI `lines128` and PDEBench CNS as the required manuscript pillars. BRDT remains additive candidate work only and cannot satisfy either pillar’s gate or support paper claims without a later reviewed amendment.
- The BRDT candidate-lane design binds the first neural input contract to `born_init_image`; direct sinogram input is optional later work and must not be mixed into the first bounded table.
- The first adapter scope must stay limited to the operator/data/adapter/four-row chain under roadmap phase `candidate-brdt-preflight`. Do not silently expand this item into a broad BRDT benchmark suite.
- Every row in the later bounded table must share the same input representation, split, normalization policy, angle mask, and loss family. This item must make that later enforcement possible by validating row metadata and run config now.
- The reviewer notes in the selected backlog item are binding:
  - do not register BRDT as a normal CDI generator;
  - do not compare a direct-sinogram row against a Born-initialization-image row in the same table;
  - do not label supervised plus Born consistency as `PINN-only`.
- Ordinary environment, import, or dependency failures must be diagnosed and retried. If `odtbrain` or `neuralop` is unavailable, record a clear row-level blocker or dependency gate instead of silently dropping the affected row or marking the whole backlog item blocked without a narrow fix attempt.

## Prerequisite Status

- The selected backlog prerequisite `2026-04-29-brdt-dataset-preflight` is satisfied by the completed summary `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md` and its machine-readable artifacts.
- The dataset-preflight summary in turn consumes the completed operator-validation authority; that dependency is already satisfied and must be read from `operator_validation.json`, not re-entered by hand.
- The initiative progress ledger records earlier core NeurIPS phases and does not yet track BRDT-specific completion separately. For this item, treat the completed BRDT operator/dataset backlog artifacts plus the selected backlog context as the authoritative prerequisite state.
- Because the canonical dataset preflight deliberately stopped before adapters and training surfaces, this item is the first place where the repo gains:
  - a shared BRDT row schema;
  - a born-init-image loader path;
  - train/eval entrypoints for the later bounded four-row preflight.

## Execution Rules

- Diagnose, fix, and rerun ordinary import, path, optional-dependency, test-harness, or CLI issues before considering the item blocked.
- Reserve `BLOCKED` for missing external resources, unrecoverable dependency or hardware failure after a documented narrow fix attempt, roadmap conflict, or an unresolved contract contradiction between the locked BRDT design/operator/dataset authorities.
- Keep any sanity artifacts under `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/` and keep scratch under `tmp/`; remove non-durable scratch before completion.
- If a derived Born-initialization cache is written, it must be clearly labeled as a derived adapter artifact, not as a mutation of the canonical preflight dataset.
- Long-running commands remain under implementation ownership until the tracked PID exits with code `0` and the expected output artifacts exist freshly. Do not launch duplicate runs against the same output root.

## Implementation Architecture

1. **Dataset And Input-Preparation Layer**
   - Owns consumption of the locked BRDT HDF5/manifest contract, collator behavior, and derivation or caching of the first shared `born_init_image` neural input from the observed sinogram.
2. **Model, Loss, And Row-Schema Layer**
   - Owns task-local ordinary `model(x) -> q_pred` adapters, the explicit separation of model label from training procedure, the unnormalize-before-physics loss wrapper, and controlled row-level blocker semantics for optional baselines.
3. **Train/Eval Entrypoints And Durable Reporting**
   - Owns the BRDT-specific training wrapper, evaluation/reporting entrypoints, invocation/provenance output, and the checked-in summary/index updates that the later four-row preflight will rely on.

## File and Artifact Targets

Mandatory contract outputs:

- `tests/studies/test_born_rytov_dt_adapters.py`
- `scripts/studies/born_rytov_dt/data.py`
- `scripts/studies/born_rytov_dt/models.py`
- `scripts/studies/born_rytov_dt/lightning_module.py`
- `scripts/studies/born_rytov_dt/run_config.py`
- `scripts/studies/born_rytov_dt/train.py`
- `scripts/studies/born_rytov_dt/evaluate.py`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md`

Likely support files to create or modify:

- `scripts/studies/born_rytov_dt/__init__.py`
- `scripts/studies/born_rytov_dt/dataset_contract.py` only if a narrow helper extension is required to expose the already locked manifest/normalization/operator-input contract without duplicating logic elsewhere
- `scripts/studies/born_rytov_dt/reporting.py`
- `scripts/studies/born_rytov_dt/metrics.py`
- `scripts/studies/born_rytov_dt/classical.py` or an equivalently named local backprop/initialization helper

Preferred packaging for adapter sanity artifacts:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/adapter_contract.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/fast_dev_run/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/fast_dev_run/invocation.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/fast_dev_run/invocation.sh`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/fast_dev_run/eval_summary.json`

Documentation discoverability updates required if the summary is written:

- `docs/index.md`
- `docs/studies/index.md`

Evidence-index updates:

- Because this item includes a sanity train/eval execution and writes result artifacts under the backlog artifact root, implementation must consult the roadmap evidence surfaces before closeout: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`, and `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`.
- If the sanity run produces a durable evaluated row, update the applicable surface before closeout and label it below manuscript grade (`feasibility_only` or equivalent).
- If no index update applies because the emitted artifacts remain pure fast-dev-run readiness evidence, say that explicitly in the implementation execution report rather than skipping the consultation silently.

Leave unchanged unless a separate approved follow-up authorizes it:

- `ptycho_torch/generators/registry.py`
- `ptycho_torch/model.py`
- `ptycho_torch/train.py`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `ptycho/model.py`
- `ptycho/diffsim.py`
- `ptycho/tf_helper.py`

## Mandatory Durable Output Contract

Before this item is considered complete, the checked-in summary plus task-local config/reporting surfaces must make the following durable and explicit:

- **Consumed authorities**
  - operator contract comes from `operator_validation.json`
  - dataset contract comes from `dataset_manifest.json`
  - dataset-contract helpers come from `scripts/studies/born_rytov_dt/dataset_contract.py`, specifically the locked manifest/normalization/operator-input helpers (`build_manifest`, `compute_train_normalization`, `normalize_q`, `unnormalize_q`, `reject_normalized_q_to_operator`) unless a narrow checked-in extension is required
  - no duplicated hand-maintained geometry constants outside the task-local contract helpers
- **First shared input mode**
  - `input_mode: born_init_image`
  - direct sinogram input explicitly rejected for the first bounded four-row contract
  - any optional confidence/mask channel recorded explicitly if used
- **Neural model contract**
  - `x` is image-like `B x C_in x 128 x 128`
  - output is `q_pred_norm` or `q_pred_physical` per run config
  - Hybrid/FNO/U-Net bodies are reused only through task-local adapters with ordinary real-channel semantics
- **Loss contract**
  - image loss and Born-consistency loss live in the BRDT task-local wrapper
  - the physics term always unnormalizes predicted `q` before calling the locked Born operator
  - default row label is `supervised + Born consistency`, not `PINN-only`
- **Row metadata schema**
  - required fields: `model`, `training`, `input_mode`, `dataset_id`, `operator_version`, `row_status`
  - row-status values at minimum distinguish `ready`, `blocked`, `feasibility_only`, and any later execution-owned status the four-row preflight needs
  - visible paper label and internal architecture ID remain distinct when the Hybrid-family row is presented as `SRU-Net`
- **Classical reference handling**
  - classical Born backprop path is represented in the same task-local row schema
  - if an optional dependency is missing, the classical row records a controlled row-level blocker instead of silently disappearing
  - if a local fallback backprop/adjoint initializer is used for `born_init_image`, the summary records that backend and its claim boundary
- **Entrypoint contract**
  - train/eval entrypoints accept dataset-manifest/operator-authority references and an output root
  - invocation/provenance artifacts are written for any sanity run
  - claim boundary remains adapter readiness only, not benchmark evidence

## Execution Tranches

### Tranche 1: Lock Dataset Consumption and Row Schema

**Purpose:** Freeze how BRDT consumes the completed smoke dataset and how later rows are described before building model or training code.

- [ ] Create `tests/studies/test_born_rytov_dt_adapters.py` first with failing coverage for:
  - loading the canonical BRDT smoke HDF5 splits and manifest without redefining geometry;
  - collating samples into a stable batch shape;
  - validating that the first supported neural `input_mode` is `born_init_image`;
  - validating the row metadata split between `model`, `training`, `input_mode`, `dataset_id`, `operator_version`, and `row_status`;
  - validating reuse of the locked `dataset_contract` normalization and normalized-`q` operator guard rather than duplicate local math;
  - rejecting any config that tries to mix direct-sinogram rows into the first bounded contract.
- [ ] Implement task-local config helpers in `scripts/studies/born_rytov_dt/run_config.py` that:
  - parse or define the first row roster (`classical_born_backprop`, `unet`, `fno_vanilla`, `hybrid_resnet` or `sru_net`);
  - separate visible labels from internal architecture/training IDs;
  - validate the allowed first-contract `input_mode`;
  - encode row-level blocker semantics for optional dependencies without broadening claim scope.
- [ ] Implement `scripts/studies/born_rytov_dt/data.py` so the loader consumes the canonical dataset manifest/HDF5 outputs through `scripts/studies/born_rytov_dt/dataset_contract.py` helpers and does not mutate them.
- [ ] Keep the dataset ID and operator version sourced from the machine-readable prerequisite artifacts, not from hard-coded duplicate strings.
- [ ] If adapter-specific parsing or metadata helpers are missing from `dataset_contract.py`, add only the narrow helper extension there and route the new loader/run-config code through it instead of copying normalization or operator-input logic into `data.py` or `run_config.py`.

**Verification**

- [ ] **Blocking:** run the newly added loader/config tests inside `tests/studies/test_born_rytov_dt_adapters.py` until the row schema and dataset-consumption contract pass.
- [ ] **Supporting:** run `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch` early for syntax/import feedback, but do not substitute it for the test gate.

### Tranche 2: Implement Born Initialization and Task-Local Model Adapters

**Purpose:** Make the shared `born_init_image` input mode and the required model bodies usable without touching the CDI generator registry.

- [ ] Add a task-local Born-initialization helper surface, preferably `scripts/studies/born_rytov_dt/classical.py`, that can derive `born_init_image` from the stored sinogram under the locked BRDT geometry.
- [ ] Prefer an ODTbrain-backed backprop path when available, but keep missing optional dependency handling explicit and controlled. If ODTbrain is unavailable, implement or document a narrow local backprop/adjoint fallback sufficient for the first adapter contract; do not silently pivot the first contract to direct sinogram input.
- [ ] Decide whether the derived initializer is produced on-the-fly, cached under the item artifact root, or both. The chosen path must preserve provenance and must not rewrite the canonical smoke dataset files.
- [ ] Implement `scripts/studies/born_rytov_dt/models.py` with a BRDT-specific ordinary real-channel adapter surface that supports at minimum:
  - `unet`
  - `fno_vanilla`
  - `hybrid_resnet` or `sru_net`
- [ ] Reuse existing model bodies only through task-local adapters. Suitable reuse references are:
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/pdebench_swe/models.py`
  - `scripts/studies/openfwi_flatvel_a/models.py`
  - `ptycho_torch/generators/README.md` scope-boundary notes
- [ ] Keep BRDT model adapters on ordinary `model(x) -> y` semantics and do not import them into the CDI generator registry.
- [ ] Extend `tests/studies/test_born_rytov_dt_adapters.py` to cover:
  - derived `born_init_image` tensor shape and dtype;
  - controlled behavior when the classical backend is unavailable;
  - successful construction and forward pass for U-Net, FNO vanilla, and SRU/Hybrid-family adapters on BRDT-shaped tensors.

**Verification**

- [ ] **Blocking:** the adapter test module must prove the first shared `born_init_image` contract is real and that the three neural adapters build and forward successfully.
- [ ] **Supporting:** if a classical fallback backend is approximate rather than ODTbrain-authored, write a small adapter-contract artifact or test note that records the backend name and why it is acceptable for feasibility-only preflight use.

### Tranche 3: Implement the BRDT Loss Wrapper and Train/Eval Entrypoints

**Purpose:** Add the task-local optimization and evaluation surfaces needed for the later bounded four-row preflight, while keeping this item below benchmark scope.

- [ ] Implement `scripts/studies/born_rytov_dt/lightning_module.py` as the BRDT-specific training wrapper or equivalently narrow task-local module that owns:
  - supervised image loss on `q`;
  - Born-consistency loss using the locked `BornRytovForward2D`;
  - unnormalize-before-physics behavior via `dataset_contract.unnormalize_q(...)` and `dataset_contract.reject_normalized_q_to_operator(...)`, not ad hoc duplicated math;
  - any positivity/TV terms approved by the candidate design;
  - explicit labeling of the training procedure as `supervised + Born consistency`.
- [ ] Implement `scripts/studies/born_rytov_dt/train.py` with a narrow training entrypoint for the later bounded four-row item.
  - It must accept a row/profile ID, dataset authority inputs, output root, and a small sanity-run mode.
  - It must not assume CDI generator registration or CDI-specific Lightning outputs.
- [ ] Implement `scripts/studies/born_rytov_dt/evaluate.py` with a narrow evaluation entrypoint that can:
  - run the classical reference path;
  - evaluate neural rows on BRDT-shaped data;
  - emit row metadata sufficient for the later four-row preflight to aggregate.
- [ ] Add task-local reporting/metrics helpers if needed so the evaluation entrypoint can emit a stable feasibility-only summary payload without becoming a paper table builder.
- [ ] Extend `tests/studies/test_born_rytov_dt_adapters.py` to cover:
  - the unnormalize-before-physics guard and normalized-`q` rejection path using the checked-in dataset-contract helpers;
  - loss-term routing on a tiny synthetic batch;
  - train/eval CLI argument validation;
  - row-status emission for completed and dependency-blocked paths.
- [ ] Run a tiny sanity execution after the tests pass:
  - a one-batch or fast-dev-run train step for one neural row;
  - a corresponding eval step or dry-run report write;
  - artifacts written under `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/`.
- [ ] Keep the sanity execution clearly labeled as adapter readiness only. It must not be summarized as the later four-row preflight.

**Verification**

- [ ] **Blocking:** the train/eval tests inside `tests/studies/test_born_rytov_dt_adapters.py` must pass before any sanity run is trusted.
- [ ] **Blocking:** one tiny sanity execution of the new entrypoints must complete and write fresh invocation/provenance plus a minimal summary payload before implementation hands off to the four-row preflight item.
- [ ] **Supporting:** if the sanity run needs a GPU or optional dependency not present locally, first attempt a narrow CPU fallback or dependency-gated path and record the resulting row-level blocker rather than escalating immediately to item-level `BLOCKED`.

### Tranche 4: Durable Summary, Discoverability, and Final Deterministic Gates

**Purpose:** Check in the adapter authority, keep discoverability current, and finish with the exact backlog checks.

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md` with:
  - the backlog item ID and prerequisite status;
  - the first shared `born_init_image` contract and any classical backend caveat;
  - the model/training/input/row-status schema;
  - the loss and unnormalize-before-physics contract;
  - the train/eval entrypoints and artifact locations;
  - the explicit claim boundary that this is still feasibility-only adapter readiness.
- [ ] Update `docs/index.md` and `docs/studies/index.md` so downstream BRDT work can discover the adapter summary alongside the operator and dataset authorities.
- [ ] Consult the roadmap-required NeurIPS evidence indexes before closeout. If the sanity run produced a durable evaluated row artifact beyond a pure fast-dev-run smoke, update the applicable index surfaces and keep the evidence tier explicitly below manuscript grade. If no update applies, record that no-op decision and rationale explicitly in the implementation execution report.
- [ ] Run the exact backlog-item required checks as the final deterministic gate:
  - `pytest -q tests/studies/test_born_rytov_dt_adapters.py`
  - `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`

**Verification**

- [ ] **Blocking:** `pytest -q tests/studies/test_born_rytov_dt_adapters.py`
- [ ] **Blocking:** `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
- [ ] **Supporting:** manually inspect the checked-in summary, evidence-index consultation outcome, and index updates or explicit execution-report no-op note to confirm they preserve the additive candidate-lane boundary and do not imply manuscript evidence or mixed input-mode fairness drift.

## Completion Criteria

- The repo has task-local BRDT loader, Born-initialization, model-adapter, loss, and train/eval surfaces sufficient for the later bounded four-row preflight to start from one shared contract.
- The first supported BRDT neural input mode is explicitly `born_init_image`, and the code rejects mixed direct-sinogram rows for the first bounded contract.
- Row metadata cleanly separates model identity from training procedure and records `input_mode`, `dataset_id`, `operator_version`, and `row_status`.
- The adapter stack reuses the checked-in `dataset_contract` helper surface for normalization and operator-input guards instead of duplicating that contract in new files.
- Any optional classical/FNO dependency issue is surfaced as a controlled row-level blocker or documented fallback, not as silent row omission.
- The exact required check commands pass.
- The checked-in summary plus evidence-index update or explicit execution-report no-op note leave BRDT clearly labeled as additive feasibility work, not manuscript evidence.
