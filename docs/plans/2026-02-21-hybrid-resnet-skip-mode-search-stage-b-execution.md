# Hybrid ResNet Skip Connections + Mode Search - Stage B Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute Stage B (`fno_blocks` structural axis) using the Stage-A control anchor and the same promotion/robustness policy used throughout this initiative.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.

## Scope

This split document owns Task 11 only (Stage-B execution commands and artifacts).

## Shared Contracts

- Use promotion and robustness rules from `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` Section 6.
- Hub document: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`.
- Stage ordering precondition: run Task 11 only after Stage-A control-anchor outputs exist.
- Epoch floor: all Stage-B runs MUST use at least `10` epochs (`--epochs-n128 >= 10`, `--epochs-n256 >= 10`) unless an approved exception is recorded.
- Non-canonical rule: outputs generated below the epoch floor MUST NOT be used as promotion sources.
- Between consecutive Stage-B run invocations, delete repo-root `memoized_data/` before launching the next command (`rm -rf memoized_data/`).
- Per-profile baseline discoverability rule: Stage-B artifacts MUST include `promotion/default_baselines.csv` and `promotion/default_baselines.md` with exactly one true-default baseline row per active `(N, dataset_profile)` combination.
- N=256 dual-profile rule: canonical `N=256` evaluation/promotion runs MUST include both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256`.

---

### Task 11: Stage B Search (Axis 1: `fno_blocks`)

**Files:**
- Modify: none (execution + artifacts)

**Step 0: Resolve single Stage-A control-anchor summary for Stage-B N=128**

Use the one-row control-anchor artifact emitted in Task 9 Step 2:
- `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/stage_anchor_summary.csv`

Do not pass full top-K robustness summary as the Stage-B `N=128` source.

**Step 1: Run Stage B at N=128 on Stage-A control anchor config**

Before each Task 11 run command block (Step 1 and Step 2), run:
```bash
rm -rf memoized_data/
```

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --promotion-source-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/stage_anchor_summary.csv \
  --fno-blocks-values 3,4 \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1,fly001_external_n128_top_bottom_v1 \
  --fly001-external-train-npz <path/to/fly001_n128_train_top_half.npz> \
  --fly001-external-test-npz <path/to/fly001_n128_test_bottom_half.npz> \
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
Non-axis knobs (`modes`, `skip`, `width`) come from `--promotion-source-summary`; canonical Stage-B runs must therefore inherit `skip=off` from the Stage-A control anchor.

**Step 2: Promote feasible Pareto-ranked top-K and run N=256**

Before this step, run the boundary seed-rerank policy on `outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221/summary.csv`:
- rerank `top-K + next 2` at `N=128` with seeds `11` and `17`,
- produce `outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221/promotion/summary_seed_robust.csv`,
- use that robustness summary as promotion source.
- ensure `outputs/hybrid_resnet_stageB_fno_blocks_n128_20260221/promotion/default_baselines.csv` exists before promotion and contains exactly one true-default baseline row per active `N=128` profile.

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

After this run completes, emit `outputs/hybrid_resnet_stageB_fno_blocks_n256_20260221/promotion/default_baselines.csv` and `.md` with exactly one true-default baseline row per active `N=256` profile.

**Step 3: Commit**

No commit (execution-only stage).

---
