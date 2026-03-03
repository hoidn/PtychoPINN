# Hybrid ResNet Skip Connections + Mode Search - Stage C Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute Stage C (downsampling schedule/operator structural search) and produce the required artifacts under the canonical promotion policy.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.

## Scope

This split document owns Task 12 only (Stage-C implementation/search and artifacts).

## Shared Contracts

- Use canonical stage semantics and promotion policy from `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.
- Follow the Test Evidence Contract in `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-implementation-core.md` for every `pytest` selector in this document.
- Hub document: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`.
- Stage-C C1 transition-anchor source (single upstream artifact): producer is Stage-B `N=128` `promotion/champion_anchor_summary.csv` (single-row robust champion selected from Stage-B `promotion/summary_seed_robust.csv`), consumer field is `--promotion-source-summary`.
- Stage-C C2 transition-anchor source (single upstream artifact): producer is Stage-C1 `N=128` `promotion/champion_anchor_summary.csv` (single-row robust champion selected from Stage-C1 `promotion/summary_seed_robust.csv`), consumer field is `--promotion-source-summary`.
- Stage-C transition-anchor fail-closed rule: if either required champion anchor source is missing, has zero rows, or has more than one row, stop Stage-C execution and report the missing/ambiguous source; do not substitute `promotion/summary_seed_robust.csv`, `promotion/stage_anchor_summary.csv`, or `promotion/default_baselines.csv`.
- Between consecutive Stage-C run invocations, delete repo-root `memoized_data/` before launching the next run command (`rm -rf memoized_data/`).
- Global epoch floor: all Stage-C runs MUST use at least `10` training epochs per run (`--epochs-n128 >= 10`, `--epochs-n256 >= 10`) unless an approved exception is recorded.
- Non-canonical rule: outputs generated below the epoch floor MUST NOT be used as promotion sources.
- Per-profile baseline discoverability rule: Stage-C artifacts MUST include `promotion/default_baselines.csv` and `promotion/default_baselines.md` with exactly one true-default baseline row per active `(N, dataset_profile)` combination.
- Baseline-lane separation rule: `promotion/default_baselines.csv|.md` remains baseline/default evidence only and MUST NOT be used as Stage-C transition-anchor source.
- N=256 dual-profile rule: canonical `N=256` evaluation/promotion runs MUST include both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256`.

---

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

Pre-step baseline requirement (mandatory before C1/C2 interpretation):
- Run true-default baseline on the same `custom_npz_pair_n128` dataset pair and same epoch budget as Stage C (`--epochs-n128 10`), and persist baseline outputs under a dedicated path.
- Do not accept any "better than baseline" interpretation in Stage C until this baseline run is complete and documented.

Sub-stage C1 (schedule): use `--downsample-schedule-values 1,2` while fixing all other structural axes to one Stage-B champion anchor config loaded via `--promotion-source-summary <stage_b_n128_root>/promotion/champion_anchor_summary.csv`.
Record provenance with `--stage-id C --substage-id C1`.

Sub-stage C2 (operator): lock best schedule from C1 (from `--promotion-source-summary <stage_c1_n128_root>/promotion/champion_anchor_summary.csv`), then run:
- `--downsample-op-values stride_conv,avgpool_conv,blurpool_conv`
Record provenance with `--stage-id C --substage-id C2`.
Stage-C downstream handoff: Stage-C2 `N=128` seed-rerank collation MUST emit one-row `<stage_c2_n128_root>/promotion/champion_anchor_summary.csv`; this is the only canonical Stage-D transition anchor artifact.

Stage budget:
- N=128: max 12 runs for C1 + max 12 runs for C2
- N=256: top 4 feasible Pareto-ranked candidates from C2 only
- N=256 promotion evidence must be aggregated across both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256` profile results.
- Epoch floor: all Stage-C runs MUST use `--epochs-n128 >= 10` and `--epochs-n256 >= 10`.
- Before selecting top-4 for `N=256`, apply the boundary seed-rerank policy on the C2 `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.
- Between consecutive Stage C invocations (including seed-rerank reruns and promotion runs), run `rm -rf memoized_data/`.
- After each Stage C invocation, perform heavy-pruning verification and log any retained heavy paths with explicit justification.
- Persist Stage C baseline-comparison artifacts:
  - `outputs/<stage_c_root>/promotion/apples_to_apples_baseline.csv`
  - `outputs/<stage_c_root>/promotion/apples_to_apples_baseline.md`
  - `outputs/<stage_c_root>/promotion/default_baselines.csv`
  - `outputs/<stage_c_root>/promotion/default_baselines.md`
  - Gate: only rows marked `apples_to_apples=true` may be used in promotion/governance narrative.
  - Gate: `default_baselines.*` must contain exactly one true-default baseline row per active `(N, dataset_profile)` combination.

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
