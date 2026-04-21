# Phase 2 PDEBench FFNO-Close Bottleneck CNS Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a bottleneck-only `ffno_bottleneck_net` family for the PDEBench image suite, lift the local/spectral/FFNO-close rows into the same canonical CNS skip-add shell, and run a capped same-shell CNS comparison that isolates the bottleneck.

**Architecture:** Refactor the supervised PDEBench shell in `scripts/studies/pdebench_image128/models.py` so the canonical CNS shell is shared across multiple bottleneck kinds. Add a new FFNO-close bottleneck module with factorized spectral mixing plus feedforward sublayers, keep it manual opt-in, then compare `hybrid_resnet_cns`, `spectral_resnet_bottleneck_base`, and `ffno_bottleneck_base` on the same capped CNS slice.

**Tech Stack:** Python 3.11 via PATH `python`, PyTorch, existing `ptycho_torch` generator modules, `scripts/studies/pdebench_image128/*`, pytest, compileall, tmux plus `ptycho311` for long runs.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Tranche ID: `phase-2-pdebench-ffno-bottleneck-cns-compare`
- Status: pending
- Date: 2026-04-21
- Scope owner: Roadmap Phase 2 optional architecture extension
- Source design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_design.md`
- Governing suite plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- Canonical CNS context:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Existing spectral bottleneck context:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`
  - `docs/backlog/paused/2026-04-20-pdebench-ffno-bottleneck-variant.md`
- Experiment root: `/home/ollie/Documents/PtychoPINN/`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/`
- Manuscript artifact root: `/home/ollie/Documents/neurips/` (future Phase 5 root; do not create or write there in this tranche)

## Compliance Matrix

- [ ] **Repo Guidance:** Re-read `AGENTS.md`, `docs/index.md`, and `docs/findings.md` before edits; use PATH `python`; do not create worktrees; use tmux for long-running commands.
- [ ] **Fairness Contract:** `hybrid_resnet_cns`, `spectral_resnet_bottleneck_base`, and `ffno_bottleneck_base` must share the same canonical CNS shell:
  - same lifter
  - same encoder blocks
  - same `hybrid_downsample_steps=2`
  - same skip-add policy
  - same decoder / upsampler
  - same output head
  - same loss / scheduler / split / metrics
  Only the bottleneck may differ.
- [ ] **Naming Contract:** The new family must stay outside `hybrid_resnet_*` and use the `ffno_bottleneck_*` namespace.
- [ ] **Evidence Boundary:** The new row remains manual opt-in. No default `PRIMARY_*` or `READINESS_*` profile bundle changes are allowed in this tranche.
- [ ] **Claim Boundary:** The implementation may claim only a bottleneck-only FFNO-close adaptation, not a paper-faithful full FFNO baseline.
- [ ] **CNS Metrics Contract:** The comparison must preserve existing CNS metrics and artifacts, including `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, and `fRMSE_high`.

## Context Priming

Read before edits:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_design.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`
- `docs/backlog/paused/2026-04-20-pdebench-ffno-bottleneck-variant.md`
- `docs/model_baselines.md`
- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/pdebench_image128/run_config.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/pdebench_image128/reporting.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`
- `ptycho_torch/generators/spectral_resnet_bottleneck.py`
- `ptycho_torch/generators/fno.py`

Relevant findings and constraints:

- The canonical CNS Hybrid row is already `hybrid_resnet_cns` with skip-add enabled; do not compare against an older no-skip shell by accident.
- The existing `spectral_resnet_bottleneck_base` is manual opt-in and must stay that way in this tranche.
- Parameter counts are captured before first forward in the PDEBench suite; any lazy parameter creation in the common shell or FFNO-close path will break metadata generation.
- The first CNS compare in this tranche is capped evidence only and must be treated as decision-support, not benchmark-complete evidence.

## Files And Responsibilities

### New files

- Create: `ptycho_torch/generators/ffno_bottleneck.py`
  - FFNO-close bottleneck primitives and generator module
- Create: `tests/torch/test_ffno_bottleneck.py`
  - unit coverage for the new bottleneck family
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`
  - durable implementation and capped-compare summary after execution

### Modified files

- Modify: `scripts/studies/pdebench_image128/models.py`
  - factor the common supervised CNS shell
  - keep `HybridResnetImageModel` behavior stable
  - lift the spectral wrapper into the common shell
  - add the FFNO-close wrapper path
- Modify: `scripts/studies/pdebench_image128/run_config.py`
  - add `ffno_bottleneck_base`
  - keep it manual opt-in
- Modify: `tests/studies/test_pdebench_image128_models.py`
  - common-shell shape and profile-set tests
- Modify: `tests/studies/test_pdebench_image128_runner.py`
  - optional runner/reporting expectations if profile metadata or summary handling needs extension
- Optional modify: `scripts/studies/pdebench_image128/reporting.py`
  - only if the comparison summary needs to record bottleneck kind or profile grouping more explicitly

## Phase A - Lock The Common-Shell Fairness Boundary

### Task A1: Write Red Tests For The Shared PDEBench Shell

**Files:**
- Modify: `tests/studies/test_pdebench_image128_models.py`
- Reference: `scripts/studies/pdebench_image128/models.py`
- Reference: `scripts/studies/pdebench_image128/run_config.py`

- [ ] Add a failing test that `hybrid_resnet_cns` still builds the canonical two-downsample, two-upsample, skip-add shell for CNS input/output shapes `(8,128,128) -> (4,128,128)`.
- [ ] Add a failing test that `spectral_resnet_bottleneck_base` can be built through the same shell and exposes the same skip-add topology keys `["d2", "d1"]`.
- [ ] Add a failing placeholder test for `ffno_bottleneck_base` with the same shell topology expectations.
- [ ] Add a failing profile-set test that confirms neither `spectral_resnet_bottleneck_base` nor `ffno_bottleneck_base` enters `PRIMARY_CFD_CNS_PROFILE_IDS`, `READINESS_CFD_CNS_PROFILE_IDS`, or `PRIMARY_DARCY_PROFILE_IDS`.
- [ ] Run:
  - `python -m pytest tests/studies/test_pdebench_image128_models.py -q`
  - Expected: FAIL because the common-shell spectral and FFNO-close paths do not exist yet.

### Task A2: Refactor The PDEBench Supervised Shell Without Changing `hybrid_resnet_cns`

**Files:**
- Modify: `scripts/studies/pdebench_image128/models.py`
- Test: `tests/studies/test_pdebench_image128_models.py`

- [ ] Extract or introduce a common supervised shell that owns:
  - `SpatialLifter`
  - encoder block stack
  - downsample stack
  - encoder tap plan
  - skip fusion plan
  - upsample stack
  - output projection
- [ ] Keep explicit eager parameter creation for skip projections and any bottleneck-specific projections so `describe_model()` can count parameters before first forward.
- [ ] Ensure `HybridResnetImageModel` continues to match the current `hybrid_resnet_cns` shell contract after the refactor.
- [ ] Re-run:
  - `python -m pytest tests/studies/test_pdebench_image128_models.py::test_hybrid_resnet_skip_add_builds_two_decoder_skip_fusions_for_cns_shape -q`
  - Expected: PASS before proceeding.
- [ ] Re-run the full targeted model-profile file:
  - `python -m pytest tests/studies/test_pdebench_image128_models.py -q`
  - Expected: the new common-shell red tests still fail, while existing Hybrid tests stay green.

## Phase B - Implement The FFNO-Close Bottleneck Module

### Task B1: Write Red Unit Tests For The New Bottleneck Family

**Files:**
- Create: `tests/torch/test_ffno_bottleneck.py`
- Reference: `ptycho_torch/generators/spectral_resnet_bottleneck.py`
- Reference: `ptycho_torch/generators/fno.py`

- [ ] Write a failing shape-preservation test for the new FFNO-close block on input `(2, 128, 32, 32)`.
- [ ] Write a failing test for the bottleneck stack that checks shared spectral-operator reuse when `ffno_bottleneck_share_weights=True`.
- [ ] Write a failing test that the block contains a feedforward path:
  - spectral mixing
  - expand `1x1`
  - `GELU`
  - project `1x1`
  - residual add
- [ ] Write a failing generator-module forward test for a `128x128` input using `hybrid_downsample_steps=2`.
- [ ] Run:
  - `python -m pytest tests/torch/test_ffno_bottleneck.py -q`
  - Expected: FAIL because the module does not exist yet.

### Task B2: Implement The FFNO-Close Bottleneck Primitives

**Files:**
- Create: `ptycho_torch/generators/ffno_bottleneck.py`
- Test: `tests/torch/test_ffno_bottleneck.py`

- [ ] Implement a shape-preserving FFNO-close block with:
  - factorized spectral mixing
  - optional explicit norm
  - `1x1` expand -> `GELU` -> `1x1` project feedforward path
  - outer residual add
- [ ] Reuse the existing factorized spectral operator if it fits cleanly; otherwise add a repo-local operator in this module instead of reaching for an external dependency.
- [ ] Implement a stacked bottleneck module with explicit control over:
  - `ffno_bottleneck_blocks`
  - `ffno_bottleneck_modes`
  - `ffno_bottleneck_share_weights`
  - `ffno_bottleneck_mlp_ratio`
  - `ffno_bottleneck_gate_init`
  - `ffno_bottleneck_norm`
- [ ] Implement a generator-module wrapper that matches the current Hybrid-style shell contract so it can be hosted in the common PDEBench shell.
- [ ] Re-run:
  - `python -m pytest tests/torch/test_ffno_bottleneck.py -q`
  - Expected: PASS.

## Phase C - Wire Spectral And FFNO-Close Bottlenecks Into The Same CNS Shell

### Task C1: Lift The Spectral Wrapper Into The Common Shell

**Files:**
- Modify: `scripts/studies/pdebench_image128/models.py`
- Test: `tests/studies/test_pdebench_image128_models.py`

- [ ] Replace the current standalone `SpectralResnetBottleneckImageModel` shell logic with the new common shell while preserving the `spectral_resnet_bottleneck_base` profile contract.
- [ ] Ensure the spectral wrapper now uses:
  - the same skip-add topology as `hybrid_resnet_cns`
  - the same upsampler path
  - the same output projection shape
- [ ] Re-run the new spectral common-shell test from Phase A.
- [ ] Confirm parameter counting still works before first forward by exercising `describe_model()` on the spectral profile.

### Task C2: Add The FFNO-Close PDEBench Wrapper And Profile

**Files:**
- Modify: `scripts/studies/pdebench_image128/models.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Modify: `tests/studies/test_pdebench_image128_models.py`

- [ ] Add `ffno_bottleneck_base` to `run_config.py` with an initial narrow config:
  - `base_model="ffno_bottleneck_net"`
  - `hidden_channels=32`
  - `fno_modes=12`
  - `fno_blocks=4`
  - `hybrid_downsample_steps=2`
  - `hybrid_skip_connections=True`
  - `hybrid_skip_style="add"`
  - `ffno_bottleneck_blocks=6`
  - `ffno_bottleneck_modes=12`
  - `ffno_bottleneck_share_weights=True`
  - `ffno_bottleneck_mlp_ratio=2.0`
  - `ffno_bottleneck_gate_init=0.1`
  - `evidence_scope="readiness-only"`
- [ ] Add a builder branch for `ffno_bottleneck_net`.
- [ ] Keep `ffno_bottleneck_base` out of all default required bundle lists.
- [ ] Re-run:
  - `python -m pytest tests/studies/test_pdebench_image128_models.py -q`
  - Expected: PASS.

## Phase D - Runner And Reporting Safety Checks

### Task D1: Extend Targeted Runner Coverage Only If Needed

**Files:**
- Modify: `tests/studies/test_pdebench_image128_runner.py`
- Optional modify: `scripts/studies/pdebench_image128/reporting.py`

- [ ] Check whether the runner or reporting layer assumes a closed set of model families beyond the current profile lists.
- [ ] If so, add a failing targeted test first.
- [ ] Implement the smallest change required so:
  - manual opt-in rows can run
  - comparison summaries still render correctly
  - required-primary profile logic remains unchanged
- [ ] Run:
  - `python -m pytest tests/studies/test_pdebench_image128_runner.py -q`
  - Expected: PASS.

### Task D2: Verify The Code Surface Compiles

**Files:**
- Use: `scripts/studies/pdebench_image128/*`
- Use: `ptycho_torch/generators/ffno_bottleneck.py`

- [ ] Run:
  - `python -m compileall -q scripts/studies/pdebench_image128 ptycho_torch/generators/ffno_bottleneck.py`
  - Expected: PASS with no output.

## Phase E - Capped Same-Shell CNS Comparison

### Task E1: Prove All Three Rows Build On The Same Capped CNS Slice

**Files:**
- Use: `scripts/studies/run_pdebench_image128_suite.py`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/`

- [ ] Use tmux with the `ptycho311` environment or equivalent PATH setup.
- [ ] Launch a capped CNS pilot using exactly these profiles:
  - `hybrid_resnet_cns`
  - `spectral_resnet_bottleneck_base`
  - `ffno_bottleneck_base`
- [ ] Start with the same non-toy capped contract already used for meaningful pilots:
  - `--task 2d_cfd_cns`
  - `--mode readiness`
  - `--epochs 10`
  - `--batch-size 4`
  - `--max-train-trajectories 512`
  - `--max-val-trajectories 64`
  - `--max-test-trajectories 64`
  - `--max-windows-per-trajectory 8`
  - `--device cuda`
  - `--num-workers 0`
- [ ] Track the exact launched PID and wait for that PID; do not use broad `pgrep -f` polling as the primary completion check.
- [ ] Treat the run as complete only if:
  - the tracked PID exits with `0`
  - `comparison_summary.json` exists under the chosen output root
  - per-profile metrics JSON files exist for all three rows

### Task E2: Render And Inspect Same-Shell PNGs

**Files:**
- Use existing comparison rendering utilities under `scripts/studies/pdebench_image128/`
- Artifact root from Task E1

- [ ] Generate at least:
  - one prediction gallery PNG
  - one error gallery PNG
- [ ] Ensure the galleries put the three rows beside the same ground truth sample.
- [ ] Copy the main gallery PNG into `tmp/` for easy human inspection.

### Task E3: Write The Durable Summary

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`

- [ ] Record:
  - code paths touched
  - verification commands and results
  - capped CNS command
  - artifact root
  - metrics for all three rows
  - explicit caveat that the result is capped decision-support evidence only
- [ ] If the FFNO-close row is obviously unstable or clearly dominated, say so plainly and do not imply future promotion.

## Verification Commands

```bash
python -m pytest tests/torch/test_ffno_bottleneck.py -q
python -m pytest tests/studies/test_pdebench_image128_models.py -q
python -m pytest tests/studies/test_pdebench_image128_runner.py -q
python -m compileall -q scripts/studies/pdebench_image128 ptycho_torch/generators/ffno_bottleneck.py
```

## Completion Criteria

- [ ] The repo contains a new manual `ffno_bottleneck_base` PDEBench family with targeted unit and builder coverage.
- [ ] `hybrid_resnet_cns`, `spectral_resnet_bottleneck_base`, and `ffno_bottleneck_base` can all build and run under the same canonical CNS skip-add shell.
- [ ] A capped CNS comparison run with those three rows completes successfully and produces metrics JSON plus comparison PNGs.
- [ ] The summary doc states the architectural scope and evidence boundary correctly.

## Artifacts Index

- Reports root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/`
- Durable summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`
- Latest run: `<output-root>/`
