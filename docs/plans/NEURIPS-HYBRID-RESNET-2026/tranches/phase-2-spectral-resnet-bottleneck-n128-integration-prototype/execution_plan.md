# Phase 2 Spectral ResNet Bottleneck + N128 Integration Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `spectral_resnet_bottleneck_net` model family, validate it against `hybrid_resnet` on the existing fixed `N=128` grid-lines integration dataset/command contract, and wire the family into the PDEBench image-suite as a manual opt-in profile without changing the primary suite bundle.

**Architecture:** Add a new Torch generator family that keeps the Hybrid ResNet encoder/downsample/upsample shell but replaces the local-only bottleneck with a ResNet-local plus shared-factorized-spectral bottleneck. Prove the core generator and grid-lines runner path on the existing `N=128` integration dataset/runner contract first, then expose a manual PDEBench profile behind the existing suite gate instead of pushing the variant into standard benchmark bundles.

**Tech Stack:** Python 3.11 via PATH `python`, PyTorch, existing `ptycho_torch` generator registry, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/pdebench_image128/*`, pytest, tmux plus `ptycho311` for any long runs.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Tranche ID: `phase-2-spectral-resnet-bottleneck-n128-integration-prototype`
- Status: pending
- Date: 2026-04-20
- Scope owner: Roadmap Phase 2 optional architecture extension
- Source design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`
- Governing suite plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- N128 comparison contract sources:
  - `tests/torch/test_grid_lines_hybrid_resnet_integration.py`
  - `docs/COMMANDS_REFERENCE.md`
  - `scripts/studies/grid_lines_torch_runner.py`
- Experiment root: `/home/ollie/Documents/PtychoPINN/`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/`
- Manuscript artifact root: `/home/ollie/Documents/neurips/` (future Phase 5 root; do not create or write there in this tranche)

## Compliance Matrix

- [ ] **Repo Guidance:** Re-read `AGENTS.md`, `docs/index.md`, and `docs/findings.md` before edits; use PATH `python`; no worktrees; use tmux for long-running commands.
- [ ] **Design Contract:** The local branch inside `SpectralResnetBlock` must use the raw two-conv body pattern from `ResnetBlock`, not `ResnetBlock.forward()` with a nested residual add.
- [ ] **Design Contract:** `spectral_resnet_bottleneck_net` stays outside the `hybrid_resnet_*` namespace and remains manual `--profiles` opt-in for PDEBench until the primary suite rows are stable.
- [ ] **Comparison Contract:** The first N128 compare is a same-shell, changed-bottleneck row against `hybrid_resnet`; it is not required to be parameter matched.
- [ ] **Study Contract:** The prototype compare must use the same dataset/runner contract as `tests/torch/test_grid_lines_hybrid_resnet_integration.py`: `N=128`, `gridsize=1`, probe `datasets/Run1084_recon3_postPC_shrunk_3.npz`, `nimgs_train=2`, `nimgs_test=1`, `nphotons=1e9`, `probe_source=custom`, `probe_smoothing_sigma=0.5`, `probe_scale_mode=pad_extrapolate`, `set_phi=True`, no probe mask, and seed `3`.
- [ ] **Evidence Boundary:** The N128 integration comparison is architecture-validation evidence only. It does not satisfy the PDEBench primary-suite gate and cannot replace required `hybrid_resnet_base`, `fno_base`, or strong U-Net PDE rows.
- [ ] **CNS Contract:** If the new family is later compared on CNS, the active task contract still requires `fRMSE_low`, `fRMSE_mid`, and `fRMSE_high`; missing `fRMSE_*` plumbing is a blocker, not an allowed omission.

## Context Priming

Read before edits:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `tests/torch/test_grid_lines_hybrid_resnet_integration.py`
- `docs/COMMANDS_REFERENCE.md`
- `docs/model_baselines.md`
- `ptycho_torch/generators/hybrid_resnet.py`
- `ptycho_torch/generators/resnet_components.py`
- `ptycho_torch/generators/registry.py`
- `ptycho/config/config.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/pdebench_image128/run_config.py`

Relevant findings and constraints:

- `FNO-DEPTH-001` and `FNO-DEPTH-002`: deep spectral stacks can blow up params/VRAM if width or doubling logic is uncontrolled; keep the first bottleneck variant tightly bounded.
- `STABLE-CRASH-DEPTH-001`: deeper is not automatically more stable; capture parameter counts and keep the first comparison narrow.
- `GRIDLINES-OBJECT-BIG-001`, `GRIDLINES-PROBE-BIG-001`, `PROBE-MASK-DEFAULT-001`: the prototype compare must preserve the current Torch grid-lines forward-model contract.
- The integration fixture is intentionally tiny (`nimgs_train=2`, `nimgs_test=1`) and should be treated as a fast prototype compare only, not as paper-grade benchmark evidence.

## Files And Responsibilities

### New files

- Create: `ptycho_torch/generators/spectral_resnet_bottleneck.py`
  - `FactorizedSpectralConv2d`
  - raw local conv body helper matching the current `ResnetBlock` internals
  - `SpectralResnetBlock`
  - `SharedSpectralResnetBottleneck`
  - `SpectralResnetBottleneckGeneratorModule`
  - `SpectralResnetBottleneckGenerator`
- Create: `tests/torch/test_spectral_resnet_bottleneck.py`
  - shape tests
  - weight-sharing tests
  - local-branch/no-nested-residual tests
  - generator build/forward tests
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/spectral_resnet_bottleneck_n128_integration_summary.md`
  - durable implementation/comparison summary after execution

### Modified files

- Modify: `ptycho_torch/generators/registry.py`
  - register `spectral_resnet_bottleneck_net`
- Modify: `ptycho/config/config.py`
  - allow the new architecture string in model validation
- Modify: `scripts/studies/grid_lines_torch_runner.py`
  - accept the new architecture in CLI/config validation
  - thread the new bottleneck-specific config fields
- Modify: `tests/torch/test_grid_lines_hybrid_resnet_integration.py`
  - add or extend targeted N128 integration coverage for the new architecture path
- Modify: `scripts/studies/pdebench_image128/models.py`
  - add `SpectralResnetBottleneckImageModel`
  - add model-builder branch for `spectral_resnet_bottleneck_net`
- Modify: `scripts/studies/pdebench_image128/run_config.py`
  - add `spectral_resnet_bottleneck_base`
  - keep it out of `PRIMARY_*` and `READINESS_*` default bundles
- Modify: `tests/studies/test_pdebench_image128_models.py`
  - build/profile tests for the manual PDEBench profile
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`
  - update status/handoff notes after implementation if behavior or scope changes materially

## Phase A - Core Bottleneck Module And Unit Tests

### Task A1: Red Tests For The New Bottleneck Primitives

**Files:**
- Create: `tests/torch/test_spectral_resnet_bottleneck.py`
- Reference: `ptycho_torch/generators/resnet_components.py`
- Reference: `ptycho_torch/generators/hybrid_resnet.py`

- [ ] Write a failing shape-preservation test for `FactorizedSpectralConv2d`:
  - input shape `(2, 32, 32, 32)`
  - output shape must remain `(2, 32, 32, 32)`
- [ ] Write a failing test that `SharedSpectralResnetBottleneck` reuses one shared spectral operator across all bottleneck blocks.
- [ ] Write a failing test that `SpectralResnetBlock` uses one outer residual add only:
  - assert the local branch is not a literal `ResnetBlock`
  - assert the block exposes a raw local conv body rather than a nested residual wrapper
- [ ] Write a failing forward test for `SpectralResnetBottleneckGeneratorModule` at `N=128`, `gridsize=1`, `fno_blocks=4`, `hybrid_downsample_steps=2`.
- [ ] Run:
  - `python -m pytest tests/torch/test_spectral_resnet_bottleneck.py -q`
  - Expected: FAIL because the new module does not exist yet.

### Task A2: Minimal Generator Implementation

**Files:**
- Create: `ptycho_torch/generators/spectral_resnet_bottleneck.py`
- Modify: `ptycho_torch/generators/registry.py`

- [ ] Implement `FactorizedSpectralConv2d` with constant-shape `(B,C,H,W)` semantics.
- [ ] Implement the raw local conv body using the current `ResnetBlock` body pattern:
  - `ReflectionPad2d`
  - `Conv2d`
  - `InstanceNorm2d`
  - `GELU`
  - `ReflectionPad2d`
  - `Conv2d`
  - `InstanceNorm2d`
- [ ] Implement `SpectralResnetBlock` as:
  - `x_next = x + local_scale * local_conv_body(x) + spectral_gate * shared_spectral(x)`
- [ ] Implement `SharedSpectralResnetBottleneck` with:
  - `n_blocks`
  - shared spectral operator
  - shared or explicitly configured gate mode
  - shared bottleneck `layerscale`
- [ ] Implement `SpectralResnetBottleneckGeneratorModule` by mirroring the current Hybrid ResNet shell but swapping only the bottleneck.
- [ ] Register `spectral_resnet_bottleneck_net` in `ptycho_torch/generators/registry.py`.
- [ ] Re-run:
  - `python -m pytest tests/torch/test_spectral_resnet_bottleneck.py -q`
  - Expected: PASS.

## Phase B - Torch Config And Grid-Lines Runner Exposure

### Task B1: Config And CLI Acceptance

**Files:**
- Modify: `ptycho/config/config.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

- [ ] Add `spectral_resnet_bottleneck_net` to the allowed Torch architecture set in model validation.
- [ ] Add grid-lines runner validation for the new bottleneck config:
  - `spectral_bottleneck_blocks`
  - `spectral_bottleneck_modes`
  - `spectral_bottleneck_share_weights`
  - `spectral_bottleneck_gate_init`
  - `spectral_bottleneck_gate_mode`
- [ ] Write failing runner/config tests that:
  - accept the new architecture string
  - reject invalid bottleneck settings
  - preserve existing `hybrid_resnet` behavior
- [ ] Run the targeted runner/config tests and confirm they fail before implementation.
- [ ] Implement the minimal config plumbing and rerun the targeted tests until they pass.

### Task B2: Integration-Test Architecture Coverage

**Files:**
- Modify: `tests/torch/test_grid_lines_hybrid_resnet_integration.py`
- Reference: `docs/COMMANDS_REFERENCE.md`

- [ ] Keep the existing Hybrid ResNet integration command unchanged.
- [ ] Add a failing integration-style test or helper path that can execute the same N128 dataset/runner contract with `--architecture spectral_resnet_bottleneck_net`.
- [ ] Keep the dataset-generation contract unchanged from the existing integration test:
  - `N=128`
  - `gridsize=1`
  - probe `datasets/Run1084_recon3_postPC_shrunk_3.npz`
  - `nimgs_train=2`
  - `nimgs_test=1`
  - `nphotons=1e9`
  - `probe_source=custom`
  - `probe_smoothing_sigma=0.5`
  - `probe_scale_mode=pad_extrapolate`
  - `set_phi=True`
- [ ] Run:
  - `python -m pytest tests/torch/test_grid_lines_hybrid_resnet_integration.py -q`
  - Expected: FAIL before the new architecture path is implemented.
- [ ] Implement only the minimal runner/test changes needed and rerun the targeted test(s) until they pass.

## Phase C - Manual PDEBench Profile Wiring

### Task C1: Add The PDEBench Image-Suite Model Family

**Files:**
- Modify: `scripts/studies/pdebench_image128/models.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Modify: `tests/studies/test_pdebench_image128_models.py`

- [ ] Add `SpectralResnetBottleneckImageModel` that matches the current supervised PDEBench shell but swaps only the bottleneck.
- [ ] Add `spectral_resnet_bottleneck_base` to `run_config.py` with defaults:
  - `hidden_channels=32`
  - `fno_modes=12`
  - `fno_blocks=4`
  - `hybrid_downsample_steps=2`
  - `spectral_bottleneck_blocks=6`
  - `spectral_bottleneck_modes=12`
  - `spectral_bottleneck_share_weights=True`
  - `spectral_bottleneck_gate_init=0.1`
- [ ] Keep `spectral_resnet_bottleneck_base` out of:
  - `PRIMARY_DARCY_PROFILE_IDS`
  - `PRIMARY_CFD_CNS_PROFILE_IDS`
  - `READINESS_CFD_CNS_PROFILE_IDS`
- [ ] Write failing model-profile tests that:
  - build the new PDEBench profile
  - record parameter count
  - preserve the existing primary profile sets
- [ ] Run:
  - `python -m pytest tests/studies/test_pdebench_image128_models.py -q`
  - Expected: FAIL before implementation.
- [ ] Implement the model-builder/run-config changes and rerun until the tests pass.

### Task C2: Explicit PDEBench Gate Preservation

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md` only if behavior differs from the current written gate
- Optional modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md` only if the wording needs a selector clarification after implementation

- [ ] Confirm the implementation keeps the design gate intact:
  - manual `--profiles` opt-in only
  - no default PDEBench bundle changes
  - no claim that this row is now part of the required suite evidence
- [ ] If the code path or naming differs from the current design doc, update the doc before execution runs.

## Phase D - N128 Integration Comparison Execution

### Task D1: Reproduce The Existing N128 Integration Dataset And Baseline Contract

**Files:**
- Use: `scripts/studies/grid_lines_torch_runner.py`
- Use: `tests/torch/test_grid_lines_hybrid_resnet_integration.py`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/smoke/`

- [ ] Confirm the N128 integration dataset pair exists at:
  - `.artifacts/integration/grid_lines_hybrid_resnet/datasets/N128/gs1/train.npz`
  - `.artifacts/integration/grid_lines_hybrid_resnet/datasets/N128/gs1/test.npz`
- [ ] If those files do not exist, regenerate them using the same `GridLinesConfig` parameters from `tests/torch/test_grid_lines_hybrid_resnet_integration.py`.
- [ ] Run a cheap smoke for `hybrid_resnet` using the same dataset contract with `epochs=1` via the direct runner.
- [ ] Run a matching cheap smoke for `spectral_resnet_bottleneck_net` using the same dataset contract with `epochs=1` via the direct runner.
- [ ] Store smoke outputs and logs under the tranche artifact root.
- [ ] Do not interpret smoke metrics as performance evidence.

### Task D2: Scored N128 Integration Comparison

**Files:**
- Use: `scripts/studies/grid_lines_torch_runner.py`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/n128_compare/`

- [ ] Run the fixed-contract baseline with the same command family as the integration test and `docs/COMMANDS_REFERENCE.md`:
  - `python scripts/studies/grid_lines_torch_runner.py --output-dir .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/n128_compare/hybrid_base --architecture hybrid_resnet --train-npz .artifacts/integration/grid_lines_hybrid_resnet/datasets/N128/gs1/train.npz --test-npz .artifacts/integration/grid_lines_hybrid_resnet/datasets/N128/gs1/test.npz --N 128 --gridsize 1 --epochs 5 --batch-size 16 --infer-batch-size 16 --learning-rate 2e-4 --scheduler ReduceLROnPlateau --plateau-factor 0.5 --plateau-patience 2 --plateau-min-lr 1e-4 --plateau-threshold 0.0 --seed 3 --optimizer adam --weight-decay 0.0 --beta1 0.9 --beta2 0.999 --torch-loss-mode mae --torch-mae-pred-l2-match-target --output-mode real_imag --probe-source custom --no-probe-mask --fno-modes 12 --fno-width 32 --fno-blocks 4 --fno-cnn-blocks 2 --torch-logger mlflow`
- [ ] Run the candidate row with the same shell contract and the new architecture:
  - same dataset and command family as the baseline
  - same seed, epochs, optimizer, scheduler, and dataset contract
  - explicit bottleneck flags recorded in invocation artifacts
- [ ] Extract and record:
  - `amp_ssim`
  - MAE/SSIM pair from `metrics.json`
  - parameter counts
  - runtime
  - comparison PNG paths
- [ ] Interpret the result only as:
  - same-shell architecture validation on a fixed `N=128` single-dataset benchmark
  - same dataset and command family as the integration test, except for the new architecture string and bottleneck-specific flags
  - not as PDEBench evidence
  - not as a parameter-matched fairness result

### Task D3: Durable Comparison Summary

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/spectral_resnet_bottleneck_n128_integration_summary.md`

- [ ] Write a durable summary with:
  - commands used
  - exact N128 integration dataset-generation contract
  - exact runner invariants
  - baseline and candidate metrics
  - parameter counts
  - runtime
  - qualitative comparison PNG locations
  - explicit evidence boundary stating this is N128 integration architecture validation, not primary PDEBench evidence
- [ ] Record whether the result justifies spending PDEBench budget on the manual profile later.

## Phase E - Optional Next-Step Gate For PDEBench

### Task E1: Decide Whether To Spend PDEBench Budget

**Files:**
- Reference: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- Reference: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`

- [ ] Only after the N128 integration summary exists, decide one of:
  - stop: variant is not promising enough to justify PDEBench budget
  - hold: promising, but still blocked on the primary suite gate
  - proceed later: promising and ready for a future manual `--profiles spectral_resnet_bottleneck_base` PDEBench row once the suite gate is satisfied
- [ ] Do not run a PDEBench comparison from this plan unless a follow-up execution step explicitly confirms that the primary suite gate is already satisfied.

## Verification Commands

```bash
python -m pytest tests/torch/test_spectral_resnet_bottleneck.py -q
python -m pytest tests/torch/test_grid_lines_hybrid_resnet_integration.py -q
python -m pytest tests/studies/test_pdebench_image128_models.py -q
python -m pytest tests/torch/test_grid_lines_torch_runner.py -q
git diff --check
```

Long-running commands must use tmux and the `ptycho311` environment. For scored comparison runs, track the launched PID directly and confirm both exit code `0` and fresh artifacts.

## Completion Criteria

- [ ] `spectral_resnet_bottleneck_net` exists as a registered Torch generator family with targeted unit coverage.
- [ ] The grid-lines runner can run the new family on the existing N128 integration dataset contract without changing the existing Hybrid ResNet contract.
- [ ] `spectral_resnet_bottleneck_base` exists in the PDEBench image-suite config as a manual opt-in row and is absent from default primary bundles.
- [ ] A same-shell N128 integration baseline-vs-candidate comparison has been executed and summarized durably.
- [ ] The summary states clearly whether the result justifies later PDEBench budget, without overstating the N128 integration compare as suite evidence.

## Artifacts Index

- Reports root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/`
- Smoke subdir: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/smoke/`
- Comparison subdir: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/n128_compare/`
- Durable summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/spectral_resnet_bottleneck_n128_integration_summary.md`
