# Hybrid ResNet Skip Connections + Mode Search - Stage E Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute Stage E (skip-connection style structural search) and produce required artifacts under the canonical promotion policy.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.

## Scope

This split document owns Task 14 only (Stage-E implementation/search and artifacts).

## Shared Contracts

- Use canonical stage semantics and promotion policy from `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.
- Follow the Test Evidence Contract in `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-implementation-core.md` for every `pytest` selector in this document.
- Hub document: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`.
- Between consecutive Stage-E run invocations, delete repo-root `memoized_data/` before launching the next run command (`rm -rf memoized_data/`).
- Global epoch floor: all Stage-E runs MUST use at least `10` training epochs per run (`--epochs-n128 >= 10`, `--epochs-n256 >= 10`) unless an approved exception is recorded.
- Non-canonical rule: outputs generated below the epoch floor MUST NOT be used as promotion sources.
- Per-profile baseline discoverability rule: Stage-E artifacts MUST include `promotion/default_baselines.csv` and `promotion/default_baselines.md` with exactly one true-default baseline row per active `(N, dataset_profile)` combination.
- Stage-E default-baseline provenance rule: when Stage-E candidate governance enforces `skip=on` and therefore blocks true-default (`skip=off`) candidate execution, `promotion/default_baselines.*` MUST source true-default rows from canonical Stage-D `promotion/default_baselines.csv`, preserve original `baseline_run_id` values verbatim, and include provenance text in `seed_policy` (no synthetic run IDs).
- N=256 dual-profile rule: canonical `N=256` evaluation/promotion runs MUST include both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256`.
- Canonical Stage-E `N=128` progression source MUST be the single-row Stage-D4 champion anchor selected from Stage-D4 `promotion/summary_seed_robust.csv` and passed via `--promotion-source-summary`.
- Default-control runs stay in the baseline-comparison lane and MUST NOT be used as Stage-E transition anchors.

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

Run skip styles on Stage-D champion config (`--promotion-source-summary <stageD4_champion_anchor_n128_summary.csv>`), budget:
- N=128: all 3 styles
- N=256: top 2 feasible Pareto-ranked styles only
- N=256 promotion evidence must be aggregated across both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256` profile results.
- Epoch floor: all Stage-E runs MUST use `--epochs-n128 >= 10` and `--epochs-n256 >= 10`.
- Before selecting top-2 for `N=256`, apply the boundary seed-rerank policy on the Stage-E `N=128` source summary (`top-K + next 2`, seeds `11` and `17`) and promote from the resulting robustness summary.
- Between consecutive Stage E invocations (including seed-rerank reruns and promotion runs), run `rm -rf memoized_data/`.
- After each Stage E invocation, perform heavy-pruning verification and log any retained heavy paths with explicit justification.
- Persist Stage E apples-to-apples baseline artifacts:
  - `outputs/<stage_e_root>/promotion/apples_to_apples_baseline.csv`
  - `outputs/<stage_e_root>/promotion/apples_to_apples_baseline.md`
  - `outputs/<stage_e_root>/promotion/default_baselines.csv`
  - `outputs/<stage_e_root>/promotion/default_baselines.md`
  - Gate: only comparisons recorded as `apples_to_apples=true` are valid evidence for style promotion conclusions.
  - Gate: `default_baselines.*` must contain exactly one true-default baseline row per active `(N, dataset_profile)` combination.
  - Provenance gate: if Stage-E cannot run true-default candidates because of `skip=on` enforcement, `default_baselines.*` MUST reuse canonical Stage-D true-default rows (same `baseline_run_id`, no fabricated aliases) and document the cross-stage source in `seed_policy`.

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
