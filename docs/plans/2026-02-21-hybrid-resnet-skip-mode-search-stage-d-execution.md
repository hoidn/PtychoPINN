# Hybrid ResNet Skip Connections + Mode Search - Stage D Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute Stage D (branch capacity, global capacity, and decoder-depth structural search) and produce required artifacts under the canonical promotion policy.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.

## Scope

This split document owns Task 13 only (Stage-D implementation/search and artifacts).

## Shared Contracts

- Use canonical stage semantics and promotion policy from `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.
- Follow the Test Evidence Contract in `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-implementation-core.md` for every `pytest` selector in this document.
- Hub document: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`.
- Stage-D transition-anchor source (single upstream artifact): producer is Stage-C2 `N=128` `promotion/champion_anchor_summary.csv` (single-row robust champion selected from Stage-C2 `promotion/summary_seed_robust.csv`), consumer field is `--promotion-source-summary`.
- Stage-D transition-anchor fail-closed rule: if the Stage-C2 champion anchor source is missing, has zero rows, or has more than one row, stop Stage-D execution and report the missing/ambiguous source; do not substitute `promotion/summary_seed_robust.csv`, `promotion/stage_anchor_summary.csv`, or `promotion/default_baselines.csv`.
- Between consecutive Stage-D run invocations, delete repo-root `memoized_data/` before launching the next run command (`rm -rf memoized_data/`).
- Global epoch floor: all Stage-D runs MUST use at least `10` training epochs per run (`--epochs-n128 >= 10`, `--epochs-n256 >= 10`) unless an approved exception is recorded.
- Non-canonical rule: outputs generated below the epoch floor MUST NOT be used as promotion sources.
- Per-profile baseline discoverability rule: Stage-D artifacts MUST include `promotion/default_baselines.csv` and `promotion/default_baselines.md` with exactly one true-default baseline row per active `(N, dataset_profile)` combination.
- N=256 dual-profile rule: canonical `N=256` evaluation/promotion runs MUST include both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256`.
- Canonical Stage-D `N=128` progression source MUST be `<stage_c2_n128_root>/promotion/champion_anchor_summary.csv`, selected from Stage-C2 `promotion/summary_seed_robust.csv` and passed via `--promotion-source-summary`.
- Default-control runs stay in the baseline-comparison lane and MUST NOT be used as Stage-D transition anchors.
- Baseline-lane separation rule: `promotion/default_baselines.csv|.md` remains baseline/default evidence only and MUST NOT be used as Stage-D transition-anchor source.

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
hybrid_encoder_conv_hidden_scale: float = 1.0
hybrid_encoder_spectral_hidden_scale: float = 1.0
```

Semantics:
- `1.0` preserves current behavior (branch width equals stage width).
- Positive finite scale values resolve internal branch widths per block while keeping additive shape contracts by projecting each branch back to stage width before merge.
- Resolution rule (required for determinism): `resolved_width = max(1, round(stage_channels * scale))`.
- Persist both configured scale values and resolved branch widths in manifest/summary rows for auditability.
- Reject invalid scale values (`<=0` or non-finite) with actionable errors.

Run sub-stage D1 (conv branch) on the Stage-C2 champion anchor source (`--promotion-source-summary <stage_c2_n128_root>/promotion/champion_anchor_summary.csv`):
- sweep `--encoder-conv-hidden-scale-values 0.5,1,2` with `--encoder-spectral-hidden-scale-values 1`.
- record provenance with `--stage-id D --substage-id D1`.

Run sub-stage D2 (spectral branch):
- lock best D1 setting, then sweep `--encoder-spectral-hidden-scale-values 0.5,1,2`.
- record provenance with `--stage-id D --substage-id D2`.

**Step 2: Axis 4 sub-stage (global capacity)**

Evaluate one global capacity knob at a time on best D2 champion settings:
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
pytest tests/torch/test_fno_generators.py -k "hybrid_encoder_conv_hidden_scale or hybrid_encoder_spectral_hidden_scale or hybrid_resnet_blocks or max_hidden_channels or resnet_width" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "hybrid_encoder_conv_hidden_scale or hybrid_encoder_spectral_hidden_scale or hybrid_resnet_blocks or max_hidden_channels or resnet_width or torch_only" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "workflow and (hybrid_encoder_conv_hidden_scale or hybrid_encoder_spectral_hidden_scale or hybrid_resnet_blocks or max_hidden_channels or resnet_width) and factory" -v
pytest tests/torch/test_fno_generators.py -k "invalid and (hybrid_encoder_conv_hidden_scale or hybrid_encoder_spectral_hidden_scale or resnet_width)" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "invalid and (hybrid_encoder_conv_hidden_scale or hybrid_encoder_spectral_hidden_scale or resnet_width)" -v
```
Expected: PASS.
Also capture matching `--collect-only` and execution logs under `${REPORT_DIR}`.

Coverage expectations for new branch-capacity knobs:
- independent branch effects under fixed seed/input (conv vs spectral scale changes produce distinct outputs),
- additive shape invariance and default parity when both scales are `1.0`,
- explicit invalid-value rejection coverage (`<=0` or non-finite for branch-scale knobs, non-divisible `resnet_width`),
- explicit scale-to-resolved-width mapping coverage (`0.5,1,2` over representative stage widths),
- workflow forwarding into `create_training_payload(..., overrides=...)`.

**Step 5: Run bounded stage budgets**

Budget rule:
- N=128: max 18 runs per sub-stage (D1-D4) and max `12` GPU-hours per sub-stage.
- N=256: top 4 feasible Pareto-ranked candidates only and max `16` GPU-hours per sub-stage.
- N=256 promotion evidence must be aggregated across both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256` profile results.
- Epoch floor: all Stage-D runs MUST use `--epochs-n128 >= 10` and `--epochs-n256 >= 10`.
- Before selecting top-4 for `N=256`, apply the boundary seed-rerank policy on the D4 `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.
- Emit one-row `<stage_d4_n128_root>/promotion/champion_anchor_summary.csv` from D4 `promotion/summary_seed_robust.csv`; this is the only canonical Stage-E transition anchor artifact.
- Between consecutive Stage D invocations (including seed-rerank reruns and promotion runs), run `rm -rf memoized_data/`.
- After each Stage D invocation, perform heavy-pruning verification and log any retained heavy paths with explicit justification.
- Persist Stage D apples-to-apples baseline artifacts:
  - `outputs/<stage_d_root>/promotion/apples_to_apples_baseline.csv`
  - `outputs/<stage_d_root>/promotion/apples_to_apples_baseline.md`
  - `outputs/<stage_d_root>/promotion/default_baselines.csv`
  - `outputs/<stage_d_root>/promotion/default_baselines.md`
  - Include baseline run details for the exact epoch budget and dataset profile used by D-stage comparisons.
  - Gate: `default_baselines.*` must contain exactly one true-default baseline row per active `(N, dataset_profile)` combination.

**Step 6: Documentation sync for Stage-D knobs**

- `docs/CONFIGURATION.md`: document `hybrid_encoder_conv_hidden_scale`, `hybrid_encoder_spectral_hidden_scale`, `hybrid_resnet_blocks`, and any capacity-option constraints used in this stage.
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
