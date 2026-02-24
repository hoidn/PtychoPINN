# Hybrid ResNet Skip Connections + Mode Search - Stage A Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional encoder-decoder skip connections to `hybrid_resnet`, add a reproducible mode×skip×width benchmark workflow for `N=128` and `N=256`, and define a staged structural-search extension for depth/downsampling/capacity/skip-design axes.

**Architecture:** Keep default behavior unchanged (`skip_connections=False`) to preserve current baselines/integration expectations. Implement additive skip fusion with lightweight `1x1` projection layers at decoder resolutions (`N/2`, `N`). Expose one boolean knob end-to-end (`hybrid_skip_connections`) through Torch-only runner/execution config + CLI (do not bridge this knob into TensorFlow/canonical model contracts), then run a deterministic sweep over `fno_modes × hybrid_skip_connections × fno_width` with fixed probe-mask/loss-normalization controls. Make dataset choice explicit via named dataset profiles so the same sweep can run on multiple failure-mode regimes. Execute Stage A in two steps (full grid on `N=128`, then top-K promotion to `N=256`), then add structural axes one stage at a time (B→E) with bounded per-stage run budgets. Promotion governance: keep broad sweeps single-seed (`seed=3` default), then run boundary seed reranks (`top-K + next 2`, seeds `11` and `17`) before every promotion, and promote by median rank across seeds. Governance decision for this initiative: new Stage C-E knobs stay Torch-only (runner/execution/model paths) unless a follow-up plan explicitly approves cross-backend bridge expansion.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, `pytest`, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (normative knob semantics, stage gates, ranking/promotion policy, and artifact contract).

## Scope

This split document owns Tasks 9-11 (full Stage-A execution handoff and Stage-B launch commands/artifacts).

## Shared Contracts

- Use promotion and robustness rules from `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` Section 6.
- Follow the Test Evidence Contract in `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-implementation-core.md` when running any `pytest` selectors.
- Hub document: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`.

### Task 9: Final Full Sweep Command (Hand-off)

**Files:**
- Modify: none

**Step 1: Run full Stage-A grid at N=128 (single-seed exploration)**

```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 12,16,24,32,48 \
  --skip-values off,on \
  --widths 32,48,64 \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1,fly001_external_n128_top_bottom_v1 \
  --fly001-external-train-npz <path/to/fly001_n128_train_top_half.npz> \
  --fly001-external-test-npz <path/to/fly001_n128_test_bottom_half.npz> \
  --epochs-n128 20 \
  --top-k-n256 6 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

**Step 2: Boundary seed-rerank before promotion (mandatory)**

From `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/summary.csv`, build the boundary candidate set:
- `top-K + next 2` (for this command, `K=6`, so rerank top 8),
- if fewer than 8 eligible candidates remain after guardrails, rerank all eligible candidates.

Rerun each boundary candidate at `N=128` with seeds `11` and `17` (same confounder settings, single-value `--modes/--skip-values/--widths` per candidate).

Template:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes <mode> \
  --skip-values <off_or_on> \
  --widths <width> \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1,fly001_external_n128_top_bottom_v1 \
  --fly001-external-train-npz <path/to/fly001_n128_train_top_half.npz> \
  --fly001-external-test-npz <path/to/fly001_n128_test_bottom_half.npz> \
  --epochs-n128 20 \
  --top-k-n256 0 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/seed_rerank/<candidate_id>_seed<seed> \
  --seed <11_or_17> \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

After all rerank jobs finish, run an explicit aggregation pass to materialize the consolidated robustness summary:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id A \
  --aggregate-seed-rerank-root outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/seed_rerank \
  --source-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/summary.csv \
  --top-k-n256 6 \
  --emit-robust-promotion-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv
```
Expected:
- promotion summary `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv` exists,
- ranking in that summary uses median rank across seeds `{3,11,17}`.

**Step 3: Promote robust top-K and run N=256**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --ns 256 \
  --promotion-source-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv \
  --dataset-profiles-n256 cameraman256_halfsplit_v1 \
  --epochs-n256 40 \
  --top-k-n256 6 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_full_n256_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: promoted `N=256` run set is driven by `summary_seed_robust.csv` (not raw single-seed summary).

**Step 4: Add targeted N=256 high-mode probe (diagnostic)**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 32,48 \
  --skip-values off,on \
  --widths 48 \
  --ns 256 \
  --allow-n256-direct-diagnostic \
  --dataset-profiles-n256 cameraman256_halfsplit_v1 \
  --epochs-n256 20 \
  --top-k-n256 0 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_n256_highmode_probe_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: explicit N=256-only high-frequency sensitivity evidence (diagnostic-only; not used for stage promotion state).

**Step 5: Confirm artifacts**

Run:
```bash
find outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221 -maxdepth 3 -type f | rg "summary.csv|summary.md|summary_seed_robust.csv"
```
Expected: Stage-A `N=128` summary and robustness summary are present.

Run:
```bash
find outputs/hybrid_resnet_mode_skip_sweep_full_n256_20260221 -maxdepth 2 -type f | rg "sweep_manifest.json|summary.csv|summary.md"
```
Expected: promoted `N=256` aggregate artifacts are present.

Run:
```bash
cat outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv | head
```
Expected: robust summary header/rows include seeded promotion ranking fields.

**Step 6: Commit**

No commit (execution-only handoff).

---

### Task 10: Add Structural-Axis Hooks to Sweep Runbook (No Cartesian Explosion)

**Files:**
- Modify: `scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py`
- Modify: `tests/studies/test_hybrid_resnet_mode_skip_sweep.py`
- Modify: `docs/studies/index.md`

**Step 1: Add CLI hooks for staged axes with safe defaults**

Follow knob semantics in:
- `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (Section 4)

Add optional arguments (defaults preserve Stage A behavior):
- `--fno-blocks-values` (default: `4`)
- `--downsample-schedule-values` (default: `2`)  # number of encoder downsample steps
- `--downsample-op-values` (default: `stride_conv`)  # `stride_conv|avgpool_conv|blurpool_conv`
- `--encoder-conv-hidden-values` (default: `none`)  # maps to `hybrid_encoder_conv_hidden_channels`
- `--encoder-spectral-hidden-values` (default: `none`)  # maps to `hybrid_encoder_spectral_hidden_channels`
- `--max-hidden-values` (default: `none`)  # maps to `max_hidden_channels`
- `--resnet-width-values` (default: `none`)
- `--resnet-blocks-values` (default: `6`)
- `--skip-style-values` (default: `add`)  # `add|concat|gated_add`
- `--dataset-profiles-n128` (default: `integration_grid_lines_n128_v1`)
- `--dataset-profiles-n256` (default: `cameraman256_halfsplit_v1`)
- `--cameraman-dp` (required when `N=256` is selected with `cameraman256_halfsplit_v1`)
- `--cameraman-para` (required when `N=256` is selected with `cameraman256_halfsplit_v1`)
- `--fly001-external-train-npz`
- `--fly001-external-test-npz`
- `--custom-n128-train-npz`
- `--custom-n128-test-npz`
- `--custom-n256-train-npz`
- `--custom-n256-test-npz`
- `--stage-id` (default: `A`; allowed: `A|B|C|D|E`)
- `--promotion-source-summary` (default: empty; required for stages `B-E`, and always required when stage `B-E` runs with `--ns 256` only)
- `--allow-n256-direct-diagnostic` (default off; only valid with `--ns 256 --top-k-n256 0`)
- `--aggregate-seed-rerank-root` (optional; enables robustness-summary collation mode)
- `--source-summary` (required with `--aggregate-seed-rerank-root`)
- `--emit-robust-promotion-summary` (required with `--aggregate-seed-rerank-root`)

Execution-path note:
- do not widen wrapper `--N` in this initiative; keep wrapper at `N in {64,128}` and route all `N=256` sweeps through `grid_lines_torch_runner.py` pathing.

Ensure `--stage-id` is persisted to manifest/summary (`A|B|C|D|E`).
Persist dataset provenance in manifest:
- profile ids
- resolved input paths and which path-based args were emitted (`--train-data/--test-data` vs `--train-npz/--test-npz`)
- SHA256 for train/test NPZ (or source HDF5 pair where applicable)

**Step 2: Add matrix builder constraints**

Implement guardrails:
- exactly one structural axis may vary per stage B-E
- Stage A varies only `{modes, widths, skip on/off}`
- For stages B-E, non-active structural axes are inherited from `--promotion-source-summary`; multi-value lists on non-active axes are rejected.
- For Stage D branch-capacity work, treat `encoder_conv_hidden`, `encoder_spectral_hidden`, `max_hidden/resnet_width`, and `resnet_blocks` as separate serial sub-stages; only one may carry multiple values in one invocation.
- For stages B-E, reject `--ns 256`-only invocations when `--promotion-source-summary` is missing.
- Reject `cameraman256_halfsplit_v1` profile usage for active `N=256` runs unless both `--cameraman-dp` and `--cameraman-para` are provided.
- Reject `custom_npz_pair_n128` unless both `--custom-n128-train-npz` and `--custom-n128-test-npz` are provided for active `N=128` runs.
- Reject `custom_npz_pair_n256` unless both `--custom-n256-train-npz` and `--custom-n256-test-npz` are provided for active `N=256` runs.
- Reject `--allow-n256-direct-diagnostic` unless `--ns 256` and `--top-k-n256 0`.
- In seed-rerank aggregation mode (`--aggregate-seed-rerank-root`):
  - require `--source-summary` and `--emit-robust-promotion-summary`,
  - fail if any required seed rows are missing for boundary candidates,
  - emit consolidated robustness summary with per-seed ranks + median-rank promotion order.
- raise actionable error if multiple structural axes contain >1 value in one stage

**Step 3: Add/adjust tests**

Add explicit invocation-provenance assertions in
`tests/studies/test_hybrid_resnet_mode_skip_sweep.py`:
- `invocation.json` is created at the run root with expected script path/argv/parsed args.
- `invocation.sh` is created at the run root and contains a reconstructible command.
- cleanup behavior is validated:
  - first successful run per `(stage_id, N, dataset_profile)` is retained as `full_anchor`.
  - subsequent successful runs in same tuple are `pruned`.
  - required retained artifacts and `cleanup_report.json` still exist with `retention_tier`.
- promotion-source validation is strict:
  - missing/unknown `summary_schema_version` fails with actionable error.
  - required summary columns missing fails before any stage execution.
- seed-rerank collation validation is explicit:
  - boundary candidate coverage for seeds `{3,11,17}` is complete,
  - `summary_seed_robust.csv` is emitted with median-rank ordering and consumed by promotion-enabled `N=256` commands.
- confounder provenance is enforced:
  - persisted summary rows and manifest always include `probe_mask_enabled` and `torch_mae_pred_l2_match_target`.
- wrapper/runner parity:
  - new sweep knobs shared across both leaf CLIs are validated to parse and pass through on both wrapper and runner code paths.
  - branch-capacity knobs (`--encoder-conv-hidden-values`, `--encoder-spectral-hidden-values`) are validated on runner path and manifest/summary persistence.
  - MAE normalization toggle naming parity is enforced:
    - canonical shared flags: `--torch-mae-pred-l2-match-target` and `--no-torch-mae-pred-l2-match-target`,
    - wrapper keeps backward-compatible alias `--torch-no-mae-pred-l2-match-target` mapped to the same destination.
  - runner-only knobs and N=256 execution paths are validated in runner/runbook tests.
- seed-robust promotion tests are explicit:
  - verify boundary candidate construction (`top-K + next 2`) against fixture summaries,
  - verify median-rank aggregation across seeds `{3,11,17}`,
  - verify promoted set is selected from robustness ranking, not raw seed-3 ranking.

Run:
```bash
pytest tests/studies/test_hybrid_resnet_mode_skip_sweep.py -k "invocation or cleanup or matrix or guardrail or stage_id" -v
```
Expected: PASS.

**Step 4: Commit**

```bash
python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
git add scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py tests/studies/test_hybrid_resnet_mode_skip_sweep.py docs/studies/index.md docs/development/TEST_SUITE_INDEX.md
git commit -m "feat(studies): add staged structural-axis hooks to hybrid_resnet sweep runbook"
```

---

### Task 11: Stage B Search (Axis 1: `fno_blocks`)

**Files:**
- Modify: none (execution + artifacts)

**Step 1: Run Stage B at N=128 on promoted Stage A configs**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --promotion-source-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv \
  --fno-blocks-values 4,5,6 \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1,fly001_external_n128_top_bottom_v1 \
  --fly001-external-train-npz <path/to/fly001_n128_train_top_half.npz> \
  --fly001-external-test-npz <path/to/fly001_n128_test_bottom_half.npz> \
  --epochs-n128 20 \
  --top-k-n256 0 \
  --output-root outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: `summary.md` includes `stage_id=B` and `fno_blocks` column.
Non-axis knobs (`modes`, `skip`, `width`) come from `--promotion-source-summary`; do not re-sweep them in Stage B.

**Step 2: Promote top-K and run N=256**

Before this step, run the boundary seed-rerank policy on `outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221/summary.csv`:
- rerank `top-K + next 2` at `N=128` with seeds `11` and `17`,
- produce `outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221/promotion/summary_seed_robust.csv`,
- use that robustness summary as promotion source.

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --promotion-source-summary outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221/promotion/summary_seed_robust.csv \
  --ns 256 \
  --dataset-profiles-n256 cameraman256_halfsplit_v1 \
  --epochs-n256 40 \
  --top-k-n256 6 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --output-root outputs/hybrid_resnet_stageB_fno_blocks_n256_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

**Step 3: Commit**

No commit (execution-only stage).

---
