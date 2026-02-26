# Hybrid ResNet Skip Connections + Mode Search - Structural Search Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional encoder-decoder skip connections to `hybrid_resnet`, add a reproducible mode×skip×width benchmark workflow for `N=128` and `N=256`, and define a staged structural-search extension for depth/downsampling/capacity/skip-design axes.

**Architecture:** Keep default behavior unchanged (`skip_connections=False`) to preserve current baselines/integration expectations. Implement additive skip fusion with lightweight `1x1` projection layers at decoder resolutions (`N/2`, `N`). Expose one boolean knob end-to-end (`hybrid_skip_connections`) through Torch-only runner/execution config + CLI (do not bridge this knob into TensorFlow/canonical model contracts), then run a deterministic sweep over `fno_modes × hybrid_skip_connections × fno_width` with fixed probe-mask/loss-normalization controls. Make dataset choice explicit via named dataset profiles so the same sweep can run on multiple failure-mode regimes. Execute Stage A in two steps (full grid on `N=128`, then feasible Pareto-ranked top-K promotion to `N=256`), then add structural axes one stage at a time (B→E) with bounded per-stage run budgets. Promotion governance: keep broad sweeps single-seed (`seed=3` default), then run boundary seed reranks (`top-K + next 2`, seeds `11` and `17`) before every promotion, and promote by median Pareto rank across seeds. Governance decision for this initiative: new Stage C-E knobs stay Torch-only (runner/execution/model paths) unless a follow-up plan explicitly approves cross-backend bridge expansion.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, `pytest`, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (normative knob semantics, stage gates, ranking/promotion policy, and artifact contract).

## Scope

This split document owns Tasks 12-15 (structural-axis implementation/search stages and governance stop/go criteria).

## Shared Contracts

- Use canonical stage semantics and promotion policy from `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.
- Follow the Test Evidence Contract in `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-implementation-core.md` for every `pytest` selector in this document.
- Hub document: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`.
- Cleanup posture: treat the design-doc retention/cleanup contract as operationally critical guidance across Stages C-E, even when enforcement remains soft.

### Strong-Advisory Cleanup Contract (Stages C-E)

- Every Stage C/D/E execution SHOULD aggressively prune heavyweight byproducts once required summaries/manifests are persisted.
- Pruned runs SHOULD NOT retain avoidable heavyweight training/runtime artifacts (for example checkpoints, per-epoch logs, transient caches, bulky intermediates).
- Exceptions MUST be explicit, justified, and time-bounded in execution logs.
- Reviews SHOULD default to `REVISE` when avoidable heavy retention is present without explicit approved exception.
- Disk pressure and sustained growth SHOULD be treated as blocking operational risk in stage-governance narratives.

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

Sub-stage C1 (schedule): use `--downsample-schedule-values 1,2` while fixing all other structural axes to one promoted Stage-B anchor config loaded via `--promotion-source-summary <stageB_anchor_n128_summary.csv>`.
Record provenance with `--stage-id C --substage-id C1`.

Sub-stage C2 (operator): lock best schedule from C1 (from `--promotion-source-summary <stageC1_anchor_n128_summary.csv>`), then run:
- `--downsample-op-values stride_conv,avgpool_conv,blurpool_conv`
Record provenance with `--stage-id C --substage-id C2`.

Stage budget:
- N=128: max 12 runs for C1 + max 12 runs for C2
- N=256: top 4 feasible Pareto-ranked candidates from C2 only
- Before selecting top-4 for `N=256`, apply the boundary seed-rerank policy on the C2 `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.
- After each Stage C invocation, perform heavy-pruning verification and log any retained heavy paths with explicit justification.

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

### Task 13: Stage D Search (Axes 3 + 4 + 4b: Encoder Branch Capacity, Global Capacity, and Decoder Depth)

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

**Step 1: Axis 3 sub-stages (encoder branch capacity decoupling)**

Scope guard for this stage:
- keep new Stage-D knob plumbing Torch-only in execution/model paths.
- in `ptycho/config/config.py`, touch `PyTorchExecutionConfig` only.
- do not add new config-bridge/spec mappings for newly introduced Stage-D knobs in this initiative.
- add explicit forwarding in `ptycho_torch/workflows/components.py` for Stage-D execution-only knobs.

Add/plumb:
```python
hybrid_encoder_conv_hidden_channels: Optional[int] = None
hybrid_encoder_spectral_hidden_channels: Optional[int] = None
```

Semantics:
- `None` preserves current behavior (branch width equals stage width).
- Positive integer values set internal branch widths while keeping additive shape contracts by projecting each branch back to stage width before merge.
- Reject invalid values (`<=0`) with actionable errors.

Run sub-stage D1 (conv branch):
- sweep `--encoder-conv-hidden-values none,48,64` with `--encoder-spectral-hidden-values none`.
- record provenance with `--stage-id D --substage-id D1`.

Run sub-stage D2 (spectral branch):
- lock best D1 setting, then sweep `--encoder-spectral-hidden-values none,48,64`.
- record provenance with `--stage-id D --substage-id D2`.

**Step 2: Axis 4 sub-stage (global capacity)**

Evaluate one global capacity knob at a time on best D2 settings:
- `max_hidden_channels` values: `none,256,512`, or
- `resnet_width` values: `none,192,256` (only if divisibility constraints pass; reject values not divisible by 4).

Keep `resnet_blocks` fixed for this sub-stage.
Record provenance with `--stage-id D --substage-id D3`.

**Step 3: Axis 4b sub-stage (decoder depth)**

Add/plumb:
```python
hybrid_resnet_blocks: int = 6
```
Sweep `4,6,8` using best capacity setting from Step 2.
Record provenance with `--stage-id D --substage-id D4`.

**Step 4: Add/execute Task-13 tests (required by test-evidence contract)**

Run:
```bash
pytest tests/torch/test_fno_generators.py -k "hybrid_encoder_conv_hidden_channels or hybrid_encoder_spectral_hidden_channels or hybrid_resnet_blocks or max_hidden_channels or resnet_width" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "hybrid_encoder_conv_hidden_channels or hybrid_encoder_spectral_hidden_channels or hybrid_resnet_blocks or max_hidden_channels or resnet_width or torch_only" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "workflow and (hybrid_encoder_conv_hidden_channels or hybrid_encoder_spectral_hidden_channels or hybrid_resnet_blocks or max_hidden_channels or resnet_width) and factory" -v
pytest tests/torch/test_fno_generators.py -k "invalid and (hybrid_encoder_conv_hidden_channels or hybrid_encoder_spectral_hidden_channels or resnet_width)" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "invalid and (hybrid_encoder_conv_hidden_channels or hybrid_encoder_spectral_hidden_channels or resnet_width)" -v
```
Expected: PASS.
Also capture matching `--collect-only` and execution logs under `${REPORT_DIR}`.

Coverage expectations for new branch-capacity knobs:
- independent branch effects under fixed seed/input (conv vs spectral width changes produce distinct outputs),
- additive shape invariance and default parity when both knobs are `None`,
- explicit invalid-value rejection coverage (`<=0` for branch-width knobs, non-divisible `resnet_width`),
- workflow forwarding into `create_training_payload(..., overrides=...)`.

**Step 5: Run bounded stage budgets**

Budget rule:
- N=128: max 18 runs per sub-stage (D1-D4) and max `12` GPU-hours per sub-stage.
- N=256: top 4 feasible Pareto-ranked candidates only and max `16` GPU-hours per sub-stage.
- Before selecting top-4 for `N=256`, apply the boundary seed-rerank policy on the D4 `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.
- After each Stage D invocation, perform heavy-pruning verification and log any retained heavy paths with explicit justification.

**Step 6: Documentation sync for Stage-D knobs**

- `docs/CONFIGURATION.md`: document `hybrid_encoder_conv_hidden_channels`, `hybrid_encoder_spectral_hidden_channels`, `hybrid_resnet_blocks`, and any capacity-option constraints used in this stage.
- `docs/workflows/pytorch.md` and `ptycho_torch/generators/README.md`: add usage guidance and constraints with explicit Torch-only scope.
- no `docs/specs/spec-ptycho-config-bridge.md` changes for Stage-D knobs in this initiative.
- regenerate `docs/development/TEST_SUITE_INDEX.md` via:
  - `python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md`

**Step 7: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add ptycho_torch/generators/hybrid_resnet.py ptycho/config/config.py ptycho_torch/config_params.py ptycho_torch/workflows/components.py scripts/studies/grid_lines_torch_runner.py tests/torch/test_fno_generators.py tests/torch/test_grid_lines_torch_runner.py docs/CONFIGURATION.md docs/workflows/pytorch.md ptycho_torch/generators/README.md docs/development/TEST_SUITE_INDEX.md
git commit -m "feat+docs(torch): add hybrid_resnet branch-capacity and depth controls with torch-only scope"
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
- `gated_add`: `y = decoder + g * skip_proj` with learnable gate `g` initialized to `0.0` (identity-safe start)

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
- N=256: top 2 feasible Pareto-ranked styles only
- Before selecting top-2 for `N=256`, apply the boundary seed-rerank policy on the Stage-E `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.
- After each Stage E invocation, perform heavy-pruning verification and log any retained heavy paths with explicit justification.

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
- promotion ordering must follow the canonical feasible-Pareto policy (Section 6 in the design doc): objectives are `amp_mae`, `amp_mse`, and `train_wall_time_sec` after feasibility filters.
- no catastrophic phase regression (`phase_ssim_drop_vs_baseline <= max_phase_ssim_drop`, default `0.03`)
- train-time/params within budget envelope recorded in summary (`train_wall_time_sec`, `model_params`)
- inference SLA satisfied for promoted candidates (`inference_time_s <= 60` at `N=128`, `<= 240` at `N=256`)
- promotion source must be robustness-validated (`top-K + next 2` reranked with seeds `{3,11,17}` and promoted by median Pareto rank across seeds)
- baseline MAE/MSE regressions are handled as stop/go diagnostics (Step 2), not a per-candidate hard pre-filter before Pareto ranking.
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
