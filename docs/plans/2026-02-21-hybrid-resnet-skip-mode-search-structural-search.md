# Hybrid ResNet Skip Connections + Mode Search - Structural Search Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional encoder-decoder skip connections to `hybrid_resnet`, add a reproducible mode×skip×width benchmark workflow for `N=128` and `N=256`, and define a staged structural-search extension for depth/downsampling/capacity/skip-design axes.

**Architecture:** Keep default behavior unchanged (`skip_connections=False`) to preserve current baselines/integration expectations. Implement additive skip fusion with lightweight `1x1` projection layers at decoder resolutions (`N/2`, `N`). Expose one boolean knob end-to-end (`hybrid_skip_connections`) through Torch-only runner/execution config + CLI (do not bridge this knob into TensorFlow/canonical model contracts), then run a deterministic sweep over `fno_modes × hybrid_skip_connections × fno_width` with fixed probe-mask/loss-normalization controls. Make dataset choice explicit via named dataset profiles so the same sweep can run on multiple failure-mode regimes. Execute Stage A in two steps (full grid on `N=128`, then top-K promotion to `N=256`), then add structural axes one stage at a time (B→E) with bounded per-stage run budgets. Promotion governance: keep broad sweeps single-seed (`seed=3` default), then run boundary seed reranks (`top-K + next 2`, seeds `11` and `17`) before every promotion, and promote by median rank across seeds. Governance decision for this initiative: new Stage C-E knobs stay Torch-only (runner/execution/model paths) unless a follow-up plan explicitly approves cross-backend bridge expansion.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, `pytest`, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (normative knob semantics, stage gates, ranking/promotion policy, and artifact contract).

## Scope

This split document owns Tasks 12-15 (structural-axis implementation/search stages and governance stop/go criteria).

## Shared Contracts

- Use canonical stage semantics and promotion policy from `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.
- Follow the Test Evidence Contract in `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-implementation-core.md` for every `pytest` selector in this document.
- Hub document: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`.

### Task 12: Stage C Search (Axis 2: Downsampling Schedule / Bottleneck Resolution + Downsampling Operator)

**Files:**
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho/config/config.py` (PyTorchExecutionConfig only; no canonical `ModelConfig` edits)
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_fno_generators.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `ptycho_torch/generators/README.md`
- Regenerate output: `docs/development/TEST_SUITE_INDEX.md` via `scripts/tools/generate_test_index.py`

**Step 0: Topology Refactor Checkpoint (required before C/E axis expansion)**

Refactor `HybridResnetGeneratorModule` skip-fusion wiring to be topology-driven:
- derive encoder/decoder stage metadata from configured downsample schedule,
- resolve skip fusion points from metadata (resolution/stage mapping), not hard-coded layer indices,
- keep behavior identical to current two-step path when `hybrid_downsample_steps=2`.

Add focused tests before Stage C sweep execution:
- `hybrid_skip_connections=True` forward pass works for `hybrid_downsample_steps=1` and `2`,
- fusion-point selection is metadata-driven (no brittle assumptions tied to exactly two downsamples),
- output shape contract remains unchanged across schedules.

**Step 1: Add `hybrid_downsample_steps` config/CLI plumbing**

Scope guard for this stage:
- add Stage-C knobs to Torch execution/model paths only.
- in `ptycho/config/config.py`, touch `PyTorchExecutionConfig` only.
- do not add new canonical `ModelConfig` keys, config-bridge mappings, or `params.cfg` emissions for these knobs in this initiative.
- add explicit forwarding in `ptycho_torch/workflows/components.py` so these execution-only knobs reach `create_training_payload(..., overrides=...)`.

Add model field:
```python
hybrid_downsample_steps: int = 2
```
Generator uses it to choose `1` (`N->N/2`) vs `2` (`N->N/4`) downsample steps.

Also add:
```python
hybrid_downsample_op: Literal["stride_conv", "avgpool_conv", "blurpool_conv"] = "stride_conv"
```
Generator uses this to select operator family for each downsample stage.

**Step 2: Add RED/GREEN tests**

Cover:
- valid range `[1,2]` for current implementation
- output shape invariance
- runner Torch execution/model propagation
- invalid `hybrid_downsample_op` rejected
- output shape invariance for each `hybrid_downsample_op`
- branch-distinctness: `stride_conv`, `avgpool_conv`, and `blurpool_conv` must exercise distinct operator code paths (for example module-type assertions plus non-identical forward outputs under fixed seed/input).
- workflow forwarding: `_train_with_lightning` must pass `hybrid_downsample_steps` and `hybrid_downsample_op` into factory overrides.

Run:
```bash
pytest tests/torch/test_fno_generators.py -k "hybrid_downsample_steps" -v
pytest tests/torch/test_fno_generators.py -k "hybrid_downsample_op" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "downsample_steps or downsample_op or torch_only" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "workflow and downsample and factory" -v
```
Expected: PASS.

**Step 3: Execute Stage C runs**

Sub-stage C1 (schedule): use `--downsample-schedule-values 1,2` while fixing all other structural axes to promoted Stage B configs loaded via `--promotion-source-summary <stageB_n128_summary.csv>`.

Sub-stage C2 (operator): lock best schedule from C1 (from `--promotion-source-summary <stageC1_n128_summary.csv>`), then run:
- `--downsample-op-values stride_conv,avgpool_conv,blurpool_conv`

Stage budget:
- N=128: max 12 runs for C1 + max 12 runs for C2
- N=256: top 4 from C2 only
- Before selecting top-4 for `N=256`, apply the boundary seed-rerank policy on the C2 `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.

**Step 4: Documentation sync for Stage-C knobs**

- `docs/CONFIGURATION.md`: add `hybrid_downsample_steps` and `hybrid_downsample_op` rows (defaults, valid values, constraints).
- `docs/workflows/pytorch.md`: add CLI examples and guardrail notes for `--downsample-schedule-values` / `--downsample-op-values`, including explicit Torch-only scope.
- `ptycho_torch/generators/README.md`: describe downsampling semantics and operator differences.
- no `docs/specs/spec-ptycho-config-bridge.md` changes for Stage-C knobs.
- regenerate `docs/development/TEST_SUITE_INDEX.md` via:
  - `python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md`

**Step 5: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add ptycho_torch/generators/hybrid_resnet.py ptycho/config/config.py ptycho_torch/config_params.py ptycho_torch/workflows/components.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_generators.py tests/torch/test_grid_lines_torch_runner.py docs/CONFIGURATION.md docs/workflows/pytorch.md ptycho_torch/generators/README.md docs/development/TEST_SUITE_INDEX.md
git commit -m "feat+docs(torch): add hybrid_resnet downsample controls with torch-only scope"
```

---

### Task 13: Stage D Search (Axes 3 + 4: Capacity and Decoder Depth)

**Files:**
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho/config/config.py` (PyTorchExecutionConfig only; no canonical `ModelConfig` edits)
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_fno_generators.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `ptycho_torch/generators/README.md`
- Regenerate output: `docs/development/TEST_SUITE_INDEX.md` via `scripts/tools/generate_test_index.py`

**Step 1: Axis 3 (capacity) sub-stage**

Scope guard for this stage:
- keep new Stage-D knob plumbing Torch-only in execution/model paths.
- in `ptycho/config/config.py`, touch `PyTorchExecutionConfig` only.
- do not add new config-bridge/spec mappings for newly introduced Stage-D knobs in this initiative.
- add explicit forwarding in `ptycho_torch/workflows/components.py` for Stage-D execution-only knobs.

Evaluate one capacity knob at a time:
- `max_hidden_channels` values: `none,256,512`, or
- `resnet_width` values: `none,192,256` (only if divisibility constraints pass).

Keep `resnet_blocks` fixed for this sub-stage.

**Step 2: Axis 4 (decoder depth) sub-stage**

Add/plumb:
```python
hybrid_resnet_blocks: int = 6
```
Sweep `4,6,8` using best capacity setting from Step 1.

**Step 3: Add/execute Task-13 tests (required by test-evidence contract)**

Run:
```bash
pytest tests/torch/test_fno_generators.py -k "hybrid_resnet_blocks or max_hidden_channels or resnet_width" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet_blocks or max_hidden_channels or resnet_width or torch_only" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "workflow and (hybrid_resnet_blocks or max_hidden_channels or resnet_width) and factory" -v
```
Expected: PASS.
Also capture matching `--collect-only` and execution logs under `${REPORT_DIR}`.

**Step 4: Run bounded stage budgets**

Budget rule:
- N=128: max 18 runs per sub-stage
- N=256: top 4 only
- Before selecting top-4 for `N=256`, apply the boundary seed-rerank policy on the D2 `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.

**Step 5: Documentation sync for Stage-D knobs**

- `docs/CONFIGURATION.md`: document `hybrid_resnet_blocks` (new) and any capacity-option constraints used in this stage.
- `docs/workflows/pytorch.md` and `ptycho_torch/generators/README.md`: add usage guidance and constraints with explicit Torch-only scope.
- no `docs/specs/spec-ptycho-config-bridge.md` changes for Stage-D knobs in this initiative.
- regenerate `docs/development/TEST_SUITE_INDEX.md` via:
  - `python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md`

**Step 6: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add ptycho_torch/generators/hybrid_resnet.py ptycho/config/config.py ptycho_torch/config_params.py ptycho_torch/workflows/components.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_generators.py tests/torch/test_grid_lines_torch_runner.py docs/CONFIGURATION.md docs/workflows/pytorch.md ptycho_torch/generators/README.md docs/development/TEST_SUITE_INDEX.md
git commit -m "feat+docs(torch): add hybrid_resnet capacity/depth controls with torch-only scope"
```

---

### Task 14: Stage E Search (Axis 5: Skip-Connection Design)

**Files:**
- Modify: `ptycho_torch/generators/hybrid_resnet.py`
- Modify: `ptycho/config/config.py` (PyTorchExecutionConfig only; no canonical `ModelConfig` edits)
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_fno_generators.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `ptycho_torch/generators/README.md`
- Regenerate output: `docs/development/TEST_SUITE_INDEX.md` via `scripts/tools/generate_test_index.py`

**Step 1: Add `hybrid_skip_style` enum**

Scope guard for this stage:
- keep new Stage-E knob plumbing Torch-only in execution/model paths.
- in `ptycho/config/config.py`, touch `PyTorchExecutionConfig` only.
- do not add new config-bridge/spec mappings for newly introduced Stage-E knobs in this initiative.
- add explicit forwarding in `ptycho_torch/workflows/components.py` for `hybrid_skip_style`.

Add model field:
```python
hybrid_skip_style: Literal["add", "concat", "gated_add"] = "add"
```

Implementation guidance:
- `add`: current behavior
- `concat`: concat then `1x1` projection
- `gated_add`: learnable scalar/channel gate initialized near zero

**Step 2: Add RED/GREEN tests**

Cover:
- shape contract unchanged for each style
- invalid style rejected
- propagation through runner Torch execution/model paths with explicit no-bridge assertions
- branch-distinctness: `add`, `concat`, and `gated_add` each execute distinct fusion logic and produce non-identical outputs for a fixed seed/input.
- workflow forwarding: `_train_with_lightning` must pass `hybrid_skip_style` into factory overrides.

Run:
```bash
pytest tests/torch/test_fno_generators.py -k "skip_style" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "skip_style or torch_only" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "workflow and skip_style and factory" -v
```

**Step 3: Execute Stage E**

Run skip styles on best config from Stage D, budget:
- N=128: all 3 styles
- N=256: top 2 styles only
- Before selecting top-2 for `N=256`, apply the boundary seed-rerank policy on the Stage-E `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.

Constraint:
- Stage E must reuse the topology-driven fusion-point mechanism introduced in Task 12 Step 0; do not introduce new hard-coded skip tap indices.

**Step 4: Documentation sync for Stage-E knob**

- `docs/CONFIGURATION.md`: add `hybrid_skip_style` row with allowed enum values and default.
- `docs/workflows/pytorch.md` and `ptycho_torch/generators/README.md`: document behavior/constraints of `add|concat|gated_add` with explicit Torch-only scope.
- no `docs/specs/spec-ptycho-config-bridge.md` changes for Stage-E knobs in this initiative.
- regenerate `docs/development/TEST_SUITE_INDEX.md` via:
  - `python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md`

**Step 5: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add ptycho_torch/generators/hybrid_resnet.py ptycho/config/config.py ptycho_torch/config_params.py ptycho_torch/workflows/components.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_generators.py tests/torch/test_grid_lines_torch_runner.py docs/CONFIGURATION.md docs/workflows/pytorch.md ptycho_torch/generators/README.md docs/development/TEST_SUITE_INDEX.md
git commit -m "feat+docs(torch): add hybrid_resnet skip-style variants with torch-only scope"
```

---

### Task 15: Stage Governance and Stop/Go Criteria

**Files:**
- Modify: `docs/studies/index.md`
- Modify: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`

**Step 1: Define promotion criteria between stages**

Use the canonical policy in:
- `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (Section 6)

Use explicit gates:
- must beat previous stage baseline on amplitude `MAE` and `MSE`
- no catastrophic phase regression (phase SSIM drop > 0.03 vs stage baseline)
- runtime/params within budget envelope recorded in summary
- promotion source must be robustness-validated (`top-K + next 2` reranked with seeds `{3,11,17}` and promoted by median rank across seeds)
- stage-promotion governance: Stage A is not complete until `docs/studies/index.md` contains the verified `hybrid-resnet-mode-skip-sweep` entry (Task 7 checks pass).

**Step 2: Define hard stop conditions**

Use a **pause-and-diagnose** gate (not immediate abandonment):
- Trigger a pause when either condition is met:
  - two consecutive stages show `<1%` median relative gain on the primary metric **and** the seed-rerank confidence interval overlaps zero (seeds `{3,11,17}`),
  - all new-stage candidates regress on both amplitude MAE and MSE at `N=256` **and** the same directional regression is present in the `N=128` robustness summary.
- Before final stop on an axis, run one bounded rescue mini-sweep on that same axis (for example targeted high-mode/width slice) and re-evaluate gates.
- If the rescue mini-sweep still fails gates, pause further expansion on that axis and carry at least one hedge candidate into the next stage for low-budget monitoring.

**Step 3: Commit**

```bash
git add docs/studies/index.md docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md
git commit -m "docs(studies): add staged structural-search governance for hybrid_resnet sweep"
```
