# Hybrid ResNet Skip Connections + Mode Search - Stage A Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional encoder-decoder skip connections to `hybrid_resnet`, add a reproducible mode×skip×width benchmark workflow for `N=128` and `N=256`, and define a staged structural-search extension for depth/downsampling/capacity/skip-design axes.

**Architecture:** Keep default behavior unchanged (`skip_connections=False`) to preserve current baselines/integration expectations. Implement additive skip fusion with lightweight `1x1` projection layers at decoder resolutions (`N/2`, `N`). Expose one boolean knob end-to-end (`hybrid_skip_connections`) through Torch-only runner/execution config + CLI (do not bridge this knob into TensorFlow/canonical model contracts), then run a deterministic sweep over `fno_modes × hybrid_skip_connections × fno_width` with fixed probe-mask/loss-normalization controls. Make dataset choice explicit via named dataset profiles so the same sweep can run on multiple failure-mode regimes. Execute Stage A in two steps (full grid on `N=128`, then feasible Pareto-ranked top-K promotion to `N=256`), then add structural axes one stage at a time (B→E) with bounded per-stage run budgets. Promotion governance: keep broad sweeps single-seed (`seed=3` default), then run boundary seed reranks (`top-K + next 2`, seeds `11` and `17`) before every promotion, and promote by median Pareto rank across seeds. Governance decision for this initiative: new Stage C-E knobs stay Torch-only (runner/execution/model paths) unless a follow-up plan explicitly approves cross-backend bridge expansion.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, `pytest`, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (normative knob semantics, stage gates, ranking/promotion policy, and artifact contract).

## Scope

This split document owns Tasks 9-11 (full Stage-A execution handoff and Stage-B launch commands/artifacts).

Execution order for this split document is:
1. Task 10 (implement/validate runbook CLI hooks and guardrails).
2. Task 9 (run Stage-A execution commands).
3. Task 11 (run Stage-B execution commands).

## Shared Contracts

- Use promotion and robustness rules from `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` Section 6.
- Follow the Test Evidence Contract in `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-implementation-core.md` when running any `pytest` selectors.
- Hub document: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`.
- Task-ordering precondition: do not execute Task 9 commands until runbook implementation work is complete (Tasks 0-8 from `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-implementation-core.md` plus Task 10 in this document, or equivalent merged implementation already present in the working tree).
- Cleanup posture: follow the design-doc retention/cleanup contract as operationally critical guidance, even if orchestration enforcement is soft.
- Epoch floor: all Stage-A/Stage-B runs MUST use at least `10` epochs (`--epochs-n128 >= 10`, `--epochs-n256 >= 10`) unless an approved exception is recorded in the execution log.
- Non-canonical rule: outputs generated below the epoch floor MUST NOT be used as promotion sources; rerun from earliest violating stage if such outputs were consumed.
- Apples-to-apples baseline rule: any "better than baseline/default/anchor" claim in this plan MUST compare runs with the same dataset profile, epoch budget, and seed policy.
- Stage-A specific baseline rule:
  - For canonical Stage-A evaluation/promotion at `N=128`, use the true-default row (`modes=12`, `skip=off`, `width=32`) from `integration_grid_lines_n128_v1` as the baseline comparator.
  - Stage-A promotion/governance claims MUST use the same seed policy as candidate ranking (`seed=3` for exploration, `{3,11,17}` median-rank context for robustness/promotion).
- Baseline documentation rule: Stage-A execution artifacts MUST include a compact baseline table with baseline run id(s), candidate run id(s), and metrics (`amp_ssim`, `amp_mae`, `amp_mse`, `phase_ssim`, `train_wall_time_sec`, `inference_time_s`) plus an explicit `apples_to_apples=true|false` flag per comparison row.
- Per-profile baseline discoverability rule: Stage-A/Stage-B execution artifacts MUST include `promotion/default_baselines.csv` and `promotion/default_baselines.md` with exactly one true-default baseline row per active `(N, dataset_profile)` combination, including the full default-model tuple and `baseline_run_id`.
- N=256 dual-profile rule: canonical `N=256` evaluation/promotion commands in this plan MUST include both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256`; single-profile `N=256` commands are diagnostic-only.

### Strong-Advisory Cleanup Contract (Stage A)

- After each Stage-A command, implementation SHOULD aggressively prune heavyweight byproducts as soon as required summary artifacts are persisted.
- Pruned runs SHOULD NOT retain avoidable heavyweight directories/files (for example checkpoints, per-epoch logs, transient caches, large intermediate recon artifacts).
- Between every consecutive run invocation in Tasks 9 and 11, delete repo-root `memoized_data/` before launching the next run command (`rm -rf memoized_data/`).
- If heavy retention is temporarily required, execution logs MUST explicitly state why, how long it will remain, and what follow-up cleanup is expected.
- Review outcomes SHOULD default to `REVISE` when avoidable heavy retention is observed without explicit approved exception.
- Disk growth and near-capacity conditions SHOULD be treated as blocking operational risk in execution/review narratives.

### Task 9: Final Full Sweep Command (Post-Implementation Hand-off)

**Files:**
- Modify: none

**Step 0: Verify runbook supports required Stage-A flags**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py --help | rg "promotion-objectives|max-train-seconds-n128|max-train-seconds-n256|max-inference-seconds-n128|max-inference-seconds-n256|max-phase-ssim-drop|max-model-params|substage-id"
```
Expected: all required flags are present before any Task 9 run command executes.

Before each Task 9 run command block (Step 1, each rerank command in Step 2, Step 2 aggregation, Step 3, and Step 4), run:
```bash
rm -rf memoized_data/
```

**Step 1: Run full Stage-A grid at N=128 (single-seed exploration)**

```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 12,16,24,32,48 \
  --skip-values off,on \
  --widths 32,48,64 \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1 \
  --epochs-n128 20 \
  --top-k-n256 6 \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-train-seconds-n128 2700 \
  --max-inference-seconds-n128 60 \
  --max-phase-ssim-drop 0.03 \
  --max-model-params 300000000 \
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
  --dataset-profiles-n128 integration_grid_lines_n128_v1 \
  --epochs-n128 20 \
  --top-k-n256 0 \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-train-seconds-n128 2700 \
  --max-inference-seconds-n128 60 \
  --max-phase-ssim-drop 0.03 \
  --max-model-params 300000000 \
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
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-phase-ssim-drop 0.03 \
  --top-k-n256 6 \
  --emit-stage-anchor-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/stage_anchor_summary.csv \
  --emit-robust-promotion-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv
```
Expected:
- promotion summary `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv` exists,
- stage anchor summary `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/stage_anchor_summary.csv` exists and contains exactly one true-default control row (`modes=12`, `skip=off`, `width=32`, `fno_blocks=4`, `downsample_schedule=2`, `downsample_op=stride_conv`, `encoder_conv_hidden=none`, `encoder_spectral_hidden=none`, `max_hidden=none`, `resnet_width=none`, `resnet_blocks=6`, `skip_style=add`) for baseline-comparison lane,
- champion anchor summary `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/champion_anchor_summary.csv` is materialized as a single-row selection from `summary_seed_robust.csv`,
- champion-anchor selection tie-break is deterministic: higher `amp_ssim`, then lower `train_wall_time_sec`, then lower `model_params`,
- ranking in that summary uses median Pareto rank across seeds `{3,11,17}` on feasible candidates.

Deterministic one-row champion-anchor materialization (required before Stage B `N=128`):
```bash
python - <<'PY'
import csv
from pathlib import Path

src = Path("outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv")
dst = Path("outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/champion_anchor_summary.csv")

with src.open(newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("empty robust summary")

def _is_feasible(row):
    value = row.get("is_feasible")
    if value in (None, ""):
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

feasible = [row for row in rows if _is_feasible(row)]
if feasible:
    rows = feasible

def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

def _params(row):
    return _as_float(row.get("model_params", row.get("params")), 1e18)

rows.sort(
    key=lambda row: (
        -_as_float(row.get("amp_ssim"), -1e18),
        _as_float(row.get("train_wall_time_sec"), 1e18),
        _params(row),
    )
)

with dst.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerow(rows[0])

print(dst)
PY
```

**Step 3: Promote robust top-K and run N=256**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --ns 256 \
  --promotion-source-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv \
  --dataset-profiles-n256 cameraman256_halfsplit_v1,custom_npz_pair_n256 \
  --epochs-n256 40 \
  --top-k-n256 6 \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-train-seconds-n256 9000 \
  --max-inference-seconds-n256 240 \
  --max-phase-ssim-drop 0.03 \
  --max-model-params 300000000 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --custom-n256-train-npz <path/to/lines_n256_train.npz> \
  --custom-n256-test-npz <path/to/lines_n256_test.npz> \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_full_n256_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

Archived canonical-pair note: later canonical reruns resolved these placeholder `custom_npz_pair_n256` inputs to `outputs/hybrid_resnet_structural_rerun_20260226T110719Z/datasets/custom_npz_builder_n256/datasets/N256/gs1/train.npz` and `.../test.npz`. Their embedded metadata identifies them as `grid_lines_workflow` synthetic lines data (`N=256`, `gridsize=1`, `nimgs_train=2`, `nimgs_test=1`, `nphotons=1e9`, `size=392`, `offset=4`, `outer_offset_train=8`, `outer_offset_test=20`, `probe_source=custom`, relative coordinates). The original generation invocation was not archived, so the exact command remains unknown.

Expected: promoted `N=256` run set is driven by feasible Pareto-ranked `summary_seed_robust.csv` (not raw single-seed summary) and evaluates each promoted candidate on both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256`.

**Step 4: Add targeted N=256 high-mode probe (diagnostic)**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 32,48 \
  --skip-values off,on \
  --widths 48 \
  --ns 256 \
  --allow-n256-direct-diagnostic \
  --dataset-profiles-n256 cameraman256_halfsplit_v1,custom_npz_pair_n256 \
  --epochs-n256 20 \
  --top-k-n256 0 \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-train-seconds-n256 9000 \
  --max-inference-seconds-n256 240 \
  --max-phase-ssim-drop 0.03 \
  --max-model-params 300000000 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --custom-n256-train-npz <path/to/lines_n256_train.npz> \
  --custom-n256-test-npz <path/to/lines_n256_test.npz> \
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

**Step 5B: Produce and persist apples-to-apples baseline evidence**

Create:
- `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/apples_to_apples_baseline.csv`
- `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/apples_to_apples_baseline.md`

Required columns:
- `dataset_profile`
- `baseline_run_id`
- `candidate_run_id`
- `baseline_amp_ssim`
- `candidate_amp_ssim`
- `baseline_amp_mae`
- `candidate_amp_mae`
- `baseline_amp_mse`
- `candidate_amp_mse`
- `baseline_phase_ssim`
- `candidate_phase_ssim`
- `baseline_train_wall_time_sec`
- `candidate_train_wall_time_sec`
- `baseline_inference_time_s`
- `candidate_inference_time_s`
- `seed_policy`
- `apples_to_apples`
- `notes`

Gate:
- Promotion/governance conclusions are invalid unless every cited "better than baseline" comparison appears in this table with `apples_to_apples=true`.

**Step 5C: Persist default baseline registry (required)**

Create:
- `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/default_baselines.csv`
- `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/default_baselines.md`
- `outputs/hybrid_resnet_mode_skip_sweep_full_n256_20260221/promotion/default_baselines.csv`
- `outputs/hybrid_resnet_mode_skip_sweep_full_n256_20260221/promotion/default_baselines.md`

Required rows:
- exactly one true-default baseline row for each active `(N, dataset_profile)` combination in this plan:
  - `N=128`: `integration_grid_lines_n128_v1`
  - `N=256`: `cameraman256_halfsplit_v1`, `custom_npz_pair_n256`

Required columns:
- `N`, `dataset_profile`, `baseline_run_id`, `stage_id`, `substage_id`, `seed_policy`, `epochs`
- default tuple fields:
  - `modes`, `skip`, `width`, `fno_blocks`, `downsample_schedule`, `downsample_op`
  - `encoder_conv_hidden`, `encoder_spectral_hidden`, `encoder_conv_hidden_scale`, `encoder_spectral_hidden_scale`
  - `max_hidden`, `resnet_width`, `resnet_blocks`, `skip_style`
- baseline metrics: `amp_ssim`, `amp_mae`, `amp_mse`, `phase_ssim`, `train_wall_time_sec`, `inference_time_s`

Gate:
- Promotion/governance conclusions are invalid unless all active `(N, dataset_profile)` combinations have exactly one discoverable true-default baseline row.

**Step 5A: Strong-advisory heavy-pruning verification**

Run:
```bash
du -sh outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221 outputs/hybrid_resnet_mode_skip_sweep_full_n256_20260221 outputs/hybrid_resnet_mode_skip_sweep_n256_highmode_probe_20260221
find outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221 outputs/hybrid_resnet_mode_skip_sweep_full_n256_20260221 outputs/hybrid_resnet_mode_skip_sweep_n256_highmode_probe_20260221 -type d \( -name checkpoints -o -name lightning_logs -o -name mlruns \)
```
Expected:
- retained artifacts are limited to reproducibility-critical outputs and explicit anchor exceptions,
- any remaining heavy directories are explicitly justified in execution logs with a cleanup follow-up.

**Step 6: Commit**

No commit (execution-only handoff).

---

### Task 10: Add Structural-Axis Hooks to Sweep Runbook (No Cartesian Explosion, Prerequisite for Task 9 Commands)

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
- `--encoder-conv-hidden-scale-values` (default: `1`)  # maps to `hybrid_encoder_conv_hidden_scale`
- `--encoder-spectral-hidden-scale-values` (default: `1`)  # maps to `hybrid_encoder_spectral_hidden_scale`
- `--max-hidden-values` (default: `none`)  # maps to `max_hidden_channels`
- `--resnet-width-values` (default: `none`)
- `--resnet-blocks-values` (default: `6`)
- `--skip-style-values` (default: `add`)  # `add|concat|gated_add`
- `--dataset-profiles-n128` (default: `integration_grid_lines_n128_v1`)
- `--dataset-profiles-n256` (default: `cameraman256_halfsplit_v1`)
- canonical plan note: for `N=256` promotion/evaluation commands, pass `cameraman256_halfsplit_v1,custom_npz_pair_n256` explicitly.
- `--cameraman-dp` (required when `N=256` is selected with `cameraman256_halfsplit_v1`)
- `--cameraman-para` (required when `N=256` is selected with `cameraman256_halfsplit_v1`)
- `--fly001-external-train-npz`
- `--fly001-external-test-npz`
- diagnostic-only note: `fly001_external_n128_top_bottom_v1` and its path args are for non-canonical diagnostics and MUST NOT be used for promotion/evaluation evidence.
- `--custom-n128-train-npz`
- `--custom-n128-test-npz`
- `--custom-n256-train-npz`
- `--custom-n256-test-npz`
- `--stage-id` (default: `A`; allowed coarse stage values: `A|B|C|D|E`; pair with `--substage-id` for C/D traceability)
- `--substage-id` (default: `none`; allowed: `none|C1|C2|D1|D2|D3|D4`)
- `--promotion-source-summary` (default: empty; required for stages `B-E`, and always required when stage `B-E` runs with `--ns 256` only)
- `--allow-n256-direct-diagnostic` (default off; only valid with `--ns 256 --top-k-n256 0`)
- `--aggregate-seed-rerank-root` (optional; enables robustness-summary collation mode)
- `--source-summary` (required with `--aggregate-seed-rerank-root`)
- `--emit-stage-anchor-summary` (required with `--aggregate-seed-rerank-root`; writes the one-row Stage-A true-default control summary for baseline lane; canonical downstream progression must use champion anchor summary derived from `summary_seed_robust.csv`)
- `--emit-robust-promotion-summary` (required with `--aggregate-seed-rerank-root`)
- `--promotion-objectives` (default: `amp_ssim,train_wall_time_sec`)
- `--max-train-seconds-n128` (default: `2700`)
- `--max-train-seconds-n256` (default: `9000`)
- `--max-inference-seconds-n128` (default: `60`)
- `--max-inference-seconds-n256` (default: `240`)
- `--max-phase-ssim-drop` (default: `0.03`)
- `--max-model-params` (default: `300000000`)

Compatibility note:
- keep `--encoder-conv-hidden-values` / `--encoder-spectral-hidden-values` as diagnostic-only aliases if needed for backward compatibility, but canonical Stage-D runs in this plan MUST use scale-valued axes.

Execution-path note:
- do not widen wrapper `--N` in this initiative; keep wrapper at `N in {64,128}` and route all `N=256` sweeps through `grid_lines_torch_runner.py` pathing.

Ensure `--stage-id` and `--substage-id` are persisted to manifest/summary.
Sub-stage rules:
- `--substage-id` must be `C1|C2` when `--stage-id C`.
- `--substage-id` must be `D1|D2|D3|D4` when `--stage-id D`.
- `--substage-id` must be `none` for `--stage-id A|B|E`.
Persist dataset provenance in manifest:
- profile ids
- resolved input paths and which path-based args were emitted (`--train-data/--test-data` vs `--train-npz/--test-npz`)
- SHA256 for train/test NPZ (or source HDF5 pair where applicable)

**Step 2: Add matrix builder constraints**

Implement guardrails:
- exactly one structural axis may vary per stage B-E
- Stage A varies only `{modes, widths, skip on/off}`
- For stages B-E, non-active structural axes are inherited from `--promotion-source-summary`; multi-value lists on non-active axes are rejected.
- For stages B-E `N=128` sweeps, `--promotion-source-summary` must resolve exactly one anchor row (single row total or one `is_stage_anchor=true` row); reject multi-anchor fan-out.
- For Stage D branch-capacity work, treat `encoder_conv_hidden_scale`, `encoder_spectral_hidden_scale`, `max_hidden/resnet_width`, and `resnet_blocks` as separate serial sub-stages; only one may carry multiple values in one invocation.
- Reject invalid `stage_id/substage_id` combinations (including missing sub-stage id for `C`/`D` runs).
- Pruning implementation MUST target real heavyweight artifacts produced by training/runtime paths, not placeholder files only.
- For stages B-E, reject `--ns 256`-only invocations when `--promotion-source-summary` is missing.
- Reject `resnet_width` sweep values that are not `none` and not divisible by 4 (or non-positive).
- Reject encoder branch-scale sweep values that are non-finite or `<=0`.
- Reject promotion candidates with `phase_ssim_drop_vs_baseline > max_phase_ssim_drop`.
- Reject promotion candidates that violate feasibility caps (`max_model_params`, `max_train_seconds_*`, `max_inference_seconds_*`).
- Reject `cameraman256_halfsplit_v1` profile usage for active `N=256` runs unless both `--cameraman-dp` and `--cameraman-para` are provided.
- Reject `custom_npz_pair_n128` unless both `--custom-n128-train-npz` and `--custom-n128-test-npz` are provided for active `N=128` runs.
- Reject `custom_npz_pair_n256` unless both `--custom-n256-train-npz` and `--custom-n256-test-npz` are provided for active `N=256` runs.
- For canonical non-diagnostic `N=256` invocations, require both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256` in `--dataset-profiles-n256`.
- Reject `--allow-n256-direct-diagnostic` unless `--ns 256` and `--top-k-n256 0`.
- In seed-rerank aggregation mode (`--aggregate-seed-rerank-root`):
  - require `--source-summary`, `--emit-robust-promotion-summary`, and `--emit-stage-anchor-summary`,
  - fail if any required seed rows are missing for boundary candidates,
  - emit consolidated robustness summary with per-seed Pareto ranks + median Pareto-rank promotion order.
- raise actionable error if multiple structural axes contain >1 value in one stage

**Step 3: Add/adjust tests**

Add explicit invocation-provenance assertions in
`tests/studies/test_hybrid_resnet_mode_skip_sweep.py`:
- `invocation.json` is created at the run root with expected script path/argv/parsed args.
- `invocation.sh` is created at the run root and contains a reconstructible command.
- cleanup behavior is validated:
  - first successful run per `(stage_id, substage_id, N, dataset_profile)` is retained as `full_anchor`.
  - subsequent successful runs in same tuple are `pruned`.
  - required retained artifacts and `cleanup_report.json` still exist with `retention_tier`.
- promotion-source validation is strict:
  - missing/unknown `summary_schema_version` fails with actionable error.
  - required summary columns missing fails before any stage execution.
- Pareto/feasibility validation is explicit:
  - summary rows include `train_wall_time_sec`, phase-guardrail fields, feasibility flags, and Pareto rank columns.
  - promotion ordering is derived from feasible Pareto ranks, not single scalar quality-only sort.
- seed-rerank collation validation is explicit:
  - boundary candidate coverage for seeds `{3,11,17}` is complete,
  - `summary_seed_robust.csv` is emitted with median Pareto-rank ordering and consumed by promotion-enabled `N=256` commands,
  - `stage_anchor_summary.csv` is emitted as default-control evidence and contains exactly one true-default tuple row (including `skip=off`),
  - `champion_anchor_summary.csv` is emitted as the single-row canonical downstream stage anchor selected from robust summary with deterministic tie-break (`amp_ssim`, then `train_wall_time_sec`, then `model_params`).
- numeric guardrail validation is explicit:
  - invalid `resnet_width` values fail fast at CLI parsing/validation with actionable errors.
- confounder provenance is enforced:
  - persisted summary rows and manifest always include `probe_mask_enabled` and `torch_mae_pred_l2_match_target`.
- wrapper/runner parity:
  - new sweep knobs shared across both leaf CLIs are validated to parse and pass through on both wrapper and runner code paths.
  - branch-capacity knobs (`--encoder-conv-hidden-scale-values`, `--encoder-spectral-hidden-scale-values`) are validated on runner path and manifest/summary persistence, including deterministic scale-to-resolved-width mapping metadata.
  - MAE normalization toggle naming parity is enforced:
    - canonical shared flags: `--torch-mae-pred-l2-match-target` and `--no-torch-mae-pred-l2-match-target`,
    - wrapper keeps backward-compatible alias `--torch-no-mae-pred-l2-match-target` mapped to the same destination.
  - runner-only knobs and N=256 execution paths are validated in runner/runbook tests.
- seed-robust promotion tests are explicit:
  - verify boundary candidate construction (`top-K + next 2`) against fixture summaries,
  - verify median Pareto-rank aggregation across seeds `{3,11,17}`,
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

**Step 0: Resolve single Stage-A champion-anchor summary for Stage-B N=128**

Use the one-row champion-anchor artifact selected from Stage-A robust summary:
- `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/champion_anchor_summary.csv`

Selection rule: rank-1 robust feasible candidate from `summary_seed_robust.csv`, with deterministic tie-break (higher `amp_ssim`, then lower `train_wall_time_sec`, then lower `model_params`).
Keep the true-default Stage-A control anchor only in baseline artifacts (`promotion/default_baselines.csv|.md` and/or control-anchor diagnostics), not as Stage-B canonical source.

**Step 1: Run Stage B at N=128 on Stage-A champion anchor config**

Before each Task 11 run command block (Step 1 and Step 2), run:
```bash
rm -rf memoized_data/
```

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --promotion-source-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/champion_anchor_summary.csv \
  --fno-blocks-values 3,4 \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1 \
  --epochs-n128 20 \
  --top-k-n256 0 \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-train-seconds-n128 2700 \
  --max-inference-seconds-n128 60 \
  --max-phase-ssim-drop 0.03 \
  --max-model-params 300000000 \
  --output-root outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: `summary.md` includes `stage_id=B`, `substage_id=none`, and `fno_blocks` column.
Non-axis knobs (`modes`, `skip`, `width`) come from `--promotion-source-summary`, so canonical Stage-B runs inherit the Stage-A champion context.

**Step 2: Promote feasible Pareto-ranked top-K and run N=256**

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
  --dataset-profiles-n256 cameraman256_halfsplit_v1,custom_npz_pair_n256 \
  --epochs-n256 40 \
  --top-k-n256 6 \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-train-seconds-n256 9000 \
  --max-inference-seconds-n256 240 \
  --max-phase-ssim-drop 0.03 \
  --max-model-params 300000000 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --custom-n256-train-npz <path/to/lines_n256_train.npz> \
  --custom-n256-test-npz <path/to/lines_n256_test.npz> \
  --output-root outputs/hybrid_resnet_stageB_fno_blocks_n256_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

**Step 3: Commit**

No commit (execution-only stage).

---
