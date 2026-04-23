# PDEBench Image-Suite Multi-GPU Enablement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable `torchrun`/DDP support to the local PDEBench `image128` study runners so Darcy and `2d_cfd_cns` training can scale to multi-GPU execution without changing the study-local data/model/artifact contract.

**Architecture:** Keep the PDEBench image-suite path separate from the main `ptycho_torch` Lightning stack. Introduce one shared distributed-runtime helper under `scripts/studies/pdebench_image128/`, wire both study runners to use it for launch detection, sampler setup, collectives, and rank-aware artifact writes, and keep evaluation/artifact emission on rank `0` only so the existing reporting surface stays single-writer and externally auditable. This retroactive plan assumes a partially implemented working tree and should be used as a reconciliation and verification plan, not as greenfield design.

**Tech Stack:** Python, PyTorch DDP, `torchrun`, JSON artifact manifests, pytest

---

## Initiative

- ID: `2026-04-22-pdebench-image-suite-multigpu`
- Title: PDEBench image-suite multi-GPU enablement
- Status: in_progress
- Spec/Source: `tmp/pdebench_multigpu_change_summary_2026-04-22.md`

## Compliance Matrix

- [ ] **Spec Constraint:** `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md` Stage B/C requirements still apply: shared adapter surface, focused pytest coverage, smoke/benchmark artifact contracts, and no Lightning-stack migration for this study path.
- [ ] **Project Workflow Constraint:** `docs/TESTING_GUIDE.md` requires `pytest -m integration` evidence for workflow changes, not only focused study selectors.
- [ ] **Interpreter Policy:** `docs/DEVELOPER_GUIDE.md` PYTHON-ENV-001 requires PATH `python` invocation, so all verification commands here use `python -m pytest`.
- [ ] **Repo Boundary Constraint:** `AGENTS.md` forbids unnecessary changes to the core ptycho physics/model modules; this initiative stays confined to `scripts/studies/pdebench_image128/*`, study tests, and plan artifacts.

## Spec Alignment

- **Normative Spec:** `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- **Key Clauses:**
  - keep PDEBench image-suite work as a study-local supervised runner rather than routing through `ptycho_torch` Lightning
  - preserve deterministic study manifests, normalization, metrics, and provenance surfaces
  - add focused tests around the shared image-suite adapter and runner contracts before relying on benchmark runs
  - treat missing verification evidence as a blocker rather than as implicit success

## Retroactive Context

The current working tree already contains a first-pass implementation in these files:

- `scripts/studies/pdebench_image128/distributed.py`
- `scripts/studies/pdebench_image128/darcy.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `tests/studies/test_pdebench_image128_distributed.py`
- `tests/studies/test_pdebench_image128_runner.py`

Review of that draft found one concrete regression to fold into execution of this plan:

- the DDP `train_batches` / `total_train_batches` counters are reduced after each epoch using the cumulative running value, so multi-epoch distributed runs overcount the total number of training batches written to metrics artifacts

This plan therefore covers both the intended multi-GPU feature and the follow-up reconciliation needed to make the artifact payload correct.

## Context Priming

- **Primary docs/specs to re-read before edits:**
  - `docs/index.md`
  - `docs/findings.md`
  - `docs/DEVELOPER_GUIDE.md`
  - `docs/TESTING_GUIDE.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- **Current implementation state:**
  - uncommitted runner/helper/test changes already exist in the working tree
  - this plan is retroactive and must reconcile those edits instead of assuming a fresh branch
- **Known blocker at plan-draft time:**
  - `python -m pytest --version` currently fails with `/usr/bin/python: No module named pytest`
  - until that is resolved, verification remains blocked and must be recorded explicitly in artifacts

## Documents Read

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_plan.md`
- `tmp/pdebench_multigpu_change_summary_2026-04-22.md`

## File Structure And Responsibilities

- `scripts/studies/pdebench_image128/distributed.py`
  - Shared runtime detection, process-group setup, rank-aware output-root preparation, sampler construction, collectives, and distributed metadata payload helpers.
- `scripts/studies/pdebench_image128/darcy.py`
  - Darcy study runner integration for distributed training, rank-0-only evaluation, and correct per-run metrics accounting.
- `scripts/studies/pdebench_image128/cfd_cns.py`
  - CNS study runner integration for distributed training, globally reduced supervised/physics summaries, and correct per-run metrics accounting.
- `tests/studies/test_pdebench_image128_distributed.py`
  - Focused helper tests for runtime initialization and distributed sampler construction.
- `tests/studies/test_pdebench_image128_runner.py`
  - Runner-level regression coverage for rank-aware artifact writes, distributed metadata emission, and corrected `train_batches` accounting.
- `.artifacts/2026-04-22-pdebench-image-suite-multigpu/`
  - Verification logs for focused selectors and required integration evidence.

## Architecture / Interfaces

- Components/boundaries touched:
  - shared distributed runtime helper
  - Darcy PDEBench study runner
  - CNS PDEBench study runner
  - study-local pytest coverage and plan evidence
- Primary data flow:
  - `torchrun` environment -> runtime initialization -> distributed sampler/model wrapping -> rank-0-only evaluation -> single-writer metrics/summary artifacts
- External interfaces/contracts impacted:
  - `scripts/studies/run_pdebench_image128_suite.py` launch behavior under `torchrun`
  - emitted `metrics_<profile>.json`, `comparison_summary.json`, and invocation/split manifests
  - study test selectors and archived verification logs

## Task 1: Lock The Distributed Runtime Boundary

**Files:**
- Create or modify: `scripts/studies/pdebench_image128/distributed.py`
- Test: `tests/studies/test_pdebench_image128_distributed.py`

- [ ] **Step 1: Assert the runtime contract in tests**

Add or confirm tests that cover:

- `torchrun` env detection from `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`
- backend selection: `nccl` for CUDA, `gloo` for CPU
- rank/local-rank/world-size capture
- training-loader construction with `DistributedSampler` when `distributed_enabled=True`
- rank-aware output-root preparation broadcast semantics

- [ ] **Step 2: Run the focused helper tests and confirm the missing cases fail first**

Run:

```bash
python -m pytest -q tests/studies/test_pdebench_image128_distributed.py
```

Expected before implementation or reconciliation:

- failure for any missing runtime-init or output-root behavior

- [ ] **Step 3: Implement or reconcile the helper module**

`distributed.py` must provide:

- `DistributedRuntime`
- `initialize_runtime(requested_device)`
- `prepare_output_root(output_root, allow_existing, runtime)`
- collective helpers for:
  - `broadcast_object`
  - `reduce_sum`
  - `reduce_sum_dict`
  - `max_cuda_memory_bytes`
- `build_training_loader(...)` returning `(loader, sampler)`
- `training_runtime_payload()` for artifact provenance
- `finalize()` to destroy the process group only when this runner initialized it

- [ ] **Step 4: Re-run the helper tests**

Run:

```bash
python -m pytest -q tests/studies/test_pdebench_image128_distributed.py
```

Expected:

- pass

## Task 2: Wire Darcy To DDP Without Changing The Study Contract

**Files:**
- Modify: `scripts/studies/pdebench_image128/darcy.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`

- [ ] **Step 1: Add or confirm Darcy runner regression tests**

The runner tests must prove:

- single-process readiness artifacts record `distributed_enabled=false` and `distributed_world_size=1`
- worker ranks do not write metrics or run evaluation
- benchmark budget device mismatch raises before profile execution
- `train_batches` in metrics equals the true global number of batches processed across all ranks and epochs

- [ ] **Step 2: Run the Darcy-focused selector and confirm failures expose the remaining gaps**

Run:

```bash
python -m pytest -q tests/studies/test_pdebench_image128_runner.py -k "darcy and (distributed or worker or train_batches)"
```

Expected before reconciliation:

- failure on the `train_batches` count if cumulative all-reduce is still used

- [ ] **Step 3: Reconcile the Darcy runner**

`darcy.py` must:

- initialize or receive `DistributedRuntime`
- wrap the model with `DistributedDataParallel` only when distributed mode is active
- replace the single-process output-root guard with `prepare_output_root(...)`
- build the training loader through `runtime.build_training_loader(...)`
- use rank-0-only evaluation and summary writing
- broadcast the rank-0 profile result back to workers
- record distributed metadata in metrics and summary outputs
- compute `train_batches` from per-epoch local increments rather than repeatedly all-reducing a cumulative running total

- [ ] **Step 4: Re-run the Darcy-focused selector**

Run:

```bash
python -m pytest -q tests/studies/test_pdebench_image128_runner.py -k "darcy and (distributed or worker or train_batches)"
```

Expected:

- pass

## Task 3: Wire CNS To DDP And Preserve Physics Reporting

**Files:**
- Modify: `scripts/studies/pdebench_image128/cfd_cns.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`

- [ ] **Step 1: Add or confirm CNS runner regression tests**

The CNS tests must prove:

- single-process readiness artifacts record distributed metadata
- worker ranks skip rank-0-only evaluation and artifact writes
- epoch summaries reduce supervised and physics totals globally
- `train_batches` equals the true global batch count across all epochs

- [ ] **Step 2: Run the CNS-focused selector and confirm failure if accounting is still wrong**

Run:

```bash
python -m pytest -q tests/studies/test_pdebench_image128_runner.py -k "cfd_cns and (distributed or worker or train_batches)"
```

Expected before reconciliation:

- failure on the batch-count regression if cumulative all-reduce remains

- [ ] **Step 3: Reconcile the CNS runner**

`cfd_cns.py` must:

- share the same runtime helper boundary as Darcy
- wrap every supported CNS profile in DDP when distributed mode is active
- build the training loader through the shared helper
- reduce epoch loss, supervised loss, physics total, and per-term physics aggregates globally
- keep train-split eval and held-out eval on rank `0` only
- attach distributed runtime metadata to metrics and comparison summary outputs
- compute `total_train_batches` from per-epoch batch counts instead of re-reducing the already-global cumulative total

- [ ] **Step 4: Re-run the CNS-focused selector**

Run:

```bash
python -m pytest -q tests/studies/test_pdebench_image128_runner.py -k "cfd_cns and (distributed or worker or train_batches)"
```

Expected:

- pass

## Task 4: Run The Required Verification Gates

**Files:**
- Test: `tests/studies/test_pdebench_image128_distributed.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`
- Test: `tests/studies/test_pdebench_image128_models.py`
- Artifact log root: `.artifacts/2026-04-22-pdebench-image-suite-multigpu/`

- [ ] **Step 1: Run the focused PDEBench selectors and capture a log**

Run:

```bash
mkdir -p .artifacts/2026-04-22-pdebench-image-suite-multigpu
python -m pytest -q \
  tests/studies/test_pdebench_image128_distributed.py \
  tests/studies/test_pdebench_image128_runner.py \
  tests/studies/test_pdebench_image128_models.py \
  | tee .artifacts/2026-04-22-pdebench-image-suite-multigpu/pytest_pdebench_image_suite_multigpu.log
```

Expected:

- all targeted study selectors pass

- [ ] **Step 2: Run the workflow integration gate required by repo policy**

Run:

```bash
python -m pytest -v -m integration \
  | tee .artifacts/2026-04-22-pdebench-image-suite-multigpu/pytest_integration.log
```

Expected:

- integration marker passes, or a concrete failing selector is captured for follow-up

- [ ] **Step 3: If `pytest` is unavailable, record the environment blocker explicitly**

Capture:

- `python -m pytest --version`
- the exact missing-module or import error
- the impacted selectors
- the return condition needed to resume verification

At the time this retroactive plan was drafted, the observed error was:

```text
/usr/bin/python: No module named pytest
```

Store that blocker note beside the plan or in the artifact root instead of silently treating focused test planning as sufficient evidence.

## Task 5: Close The Documentation Loop

**Files:**
- Modify: `docs/plans/2026-04-22-pdebench-image-suite-multigpu.md`
- Optional modify: `docs/findings.md`
- Optional modify: `docs/index.md`

- [ ] **Step 1: Update this plan with execution status**

After implementation or verification, record:

- which steps completed
- which selectors passed
- artifact log paths
- any remaining blockers

- [ ] **Step 2: Add a finding if the batch-count regression becomes durable project knowledge**

If the `train_batches` overcount bug required a code fix, add a concise entry to `docs/findings.md` so later distributed-study work does not repeat the same cumulative-all-reduce mistake.

- [ ] **Step 3: Keep documentation discoverable**

If this plan becomes the durable record for the multi-GPU change, ensure `docs/index.md` references it or the higher-level PDEBench plan links to it from the relevant tranche context.

## Verification Commands

```bash
python -m pytest -q tests/studies/test_pdebench_image128_distributed.py
python -m pytest -q tests/studies/test_pdebench_image128_runner.py -k "darcy and (distributed or worker or train_batches)"
python -m pytest -q tests/studies/test_pdebench_image128_runner.py -k "cfd_cns and (distributed or worker or train_batches)"
python -m pytest -q \
  tests/studies/test_pdebench_image128_distributed.py \
  tests/studies/test_pdebench_image128_runner.py \
  tests/studies/test_pdebench_image128_models.py
python -m pytest -v -m integration
```

## Completion Criteria

- [ ] Darcy and CNS study runners support `torchrun` multi-process training with a shared distributed helper and no duplicate artifact writes.
- [ ] Metrics and summary artifacts record distributed provenance and correct global batch counts across multi-epoch distributed runs.
- [ ] Focused PDEBench selectors pass and their logs are archived under `.artifacts/2026-04-22-pdebench-image-suite-multigpu/`.
- [ ] The repo-required integration marker is either passing with archived logs or blocked with an explicit captured environment failure.

## Workflow Compatibility Contract

When this plan is executed by a backlog or supervisor workflow:

- the execution unit is the whole reconciliation plan, not only the helper file
- completion requires:
  - the batch-count regression is fixed in both runners
  - focused study selectors pass or are explicitly blocked with captured evidence
  - the integration gate is passed or blocked with archived proof
- the artifact root for this plan remains `.artifacts/2026-04-22-pdebench-image-suite-multigpu/`

## Artifacts Index

- Reports root: `.artifacts/2026-04-22-pdebench-image-suite-multigpu/`
- Expected logs:
  - `pytest_pdebench_image_suite_multigpu.log`
  - `pytest_integration.log`
  - optional blocker note capturing missing-`pytest` environment evidence
