# Hybrid ResNet Skip Connections + Mode Search - Stage B Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute Stage B (`fno_blocks` structural axis) using the Stage-A robust champion anchor and the same promotion/robustness policy used throughout this initiative.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.

## Scope

This split document owns Task 11 only (Stage-B execution commands and artifacts).

## Stage Plan Links

- Upstream Plan: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-stage-a-execution.md`
- Downstream Plan: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-stage-c-execution.md`

## Shared Contracts

- Use promotion and robustness rules from `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` Section 6.
- Hub document: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`.
- Stage ordering precondition: run Task 11 only after Stage-A robust outputs exist (`promotion/summary_seed_robust.csv`) and a single-row Stage-A champion anchor summary has been materialized.
- Stage-B transition-anchor source (single upstream artifact): producer is Stage-A `N=128` `promotion/champion_anchor_summary.csv` (single-row robust champion selected from Stage-A `promotion/summary_seed_robust.csv`), consumer field is `--promotion-source-summary`.
- Stage-B transition-anchor fail-closed rule: if the Stage-A champion anchor source is missing, has zero rows, or has more than one row, stop Stage-B execution and report the missing/ambiguous source; do not substitute `promotion/summary_seed_robust.csv`, `promotion/stage_anchor_summary.csv`, or `promotion/default_baselines.csv`.
- Stage-B downstream handoff contract: Stage-B `N=128` seed-rerank collation MUST emit one-row `promotion/champion_anchor_summary.csv`; this is the only canonical Stage-C transition anchor artifact.
- Epoch floor: all Stage-B runs MUST use at least `10` epochs (`--epochs-n128 >= 10`, `--epochs-n256 >= 10`) unless an approved exception is recorded.
- Non-canonical rule: outputs generated below the epoch floor MUST NOT be used as promotion sources.
- Between consecutive Stage-B run invocations, delete repo-root `memoized_data/` before launching the next command (`rm -rf memoized_data/`).
- Per-profile baseline discoverability rule: Stage-B artifacts MUST include `promotion/default_baselines.csv` and `promotion/default_baselines.md` with exactly one true-default baseline row per active `(N, dataset_profile)` combination.
- Baseline-lane separation rule: `promotion/default_baselines.csv|.md` remains baseline/default evidence only and MUST NOT be used as Stage-B transition-anchor source.
- N=256 dual-profile rule: canonical `N=256` evaluation/promotion runs MUST include both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256`.

---

### Task 11: Stage B Search (Axis 1: `fno_blocks`)

**Files:**
- Modify: none (execution + artifacts)

**Step 0: Resolve single Stage-A champion-anchor summary for Stage-B N=128**

Use the one-row champion-anchor artifact selected from Stage-A robust summary:
- `<stage_a_n128_root>/promotion/champion_anchor_summary.csv`

Selection rule for this one-row file: rank-1 robust feasible candidate from `summary_seed_robust.csv`, breaking ties deterministically by (1) higher `amp_ssim`, (2) lower `train_wall_time_sec`, (3) lower `model_params`.
Keep the true-default Stage-A control anchor only in baseline artifacts (`promotion/default_baselines.csv|.md` and/or control-anchor diagnostics), not as Stage-B canonical source.

**Step 1: Run Stage B at N=128 on Stage-A champion anchor config**

Before each Task 11 run command block (Step 1 and Step 2), run:
```bash
rm -rf memoized_data/
```

Run:
```bash
STAGE_A_CHAMPION_ANCHOR="<stage_a_n128_root>/promotion/champion_anchor_summary.csv"
STAGE_B_N128_ROOT="<stage_b_n128_root>"

python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --promotion-source-summary "${STAGE_A_CHAMPION_ANCHOR}" \
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
  --output-root "${STAGE_B_N128_ROOT}" \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: `summary.md` includes `stage_id=B`, `substage_id=none`, and `fno_blocks` column.
Non-axis knobs (`modes`, `skip`, `width`) come from `--promotion-source-summary`, so canonical Stage-B runs inherit the Stage-A champion context.

**Step 2: Promote feasible Pareto-ranked top-K and run N=256**

Before this step, run the boundary seed-rerank policy on `${STAGE_B_N128_ROOT}/summary.csv`:
- rerank `top-K + next 2` at `N=128` with seeds `11` and `17`,
- produce `${STAGE_B_N128_ROOT}/promotion/summary_seed_robust.csv`,
- use that robustness summary as promotion source.
- emit `${STAGE_B_N128_ROOT}/promotion/champion_anchor_summary.csv` as the single-row Stage-B champion anchor for downstream Stage-C consumption via `--promotion-source-summary`.
- ensure `${STAGE_B_N128_ROOT}/promotion/default_baselines.csv` exists before promotion and contains exactly one true-default baseline row per active `N=128` profile.

Run:
```bash
STAGE_B_N128_ROOT="<stage_b_n128_root>"
STAGE_B_ROBUST_SUMMARY="${STAGE_B_N128_ROOT}/promotion/summary_seed_robust.csv"
STAGE_B_N256_ROOT="<stage_b_n256_root>"
CAMERAMAN_DP="<path/to/cameraman256_dp.hdf5>"
CAMERAMAN_PARA="<path/to/cameraman256_para.hdf5>"

python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --promotion-source-summary "${STAGE_B_ROBUST_SUMMARY}" \
  --ns 256 \
  --dataset-profiles-n256 cameraman256_halfsplit_v1,custom_npz_pair_n256 \
  --epochs-n256 40 \
  --top-k-n256 6 \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-train-seconds-n256 9000 \
  --max-inference-seconds-n256 240 \
  --max-phase-ssim-drop 0.03 \
  --max-model-params 300000000 \
  --cameraman-dp "${CAMERAMAN_DP}" \
  --cameraman-para "${CAMERAMAN_PARA}" \
  --custom-n256-train-npz <path/to/lines_n256_train.npz> \
  --custom-n256-test-npz <path/to/lines_n256_test.npz> \
  --output-root "${STAGE_B_N256_ROOT}" \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

After this run completes, emit `${STAGE_B_N256_ROOT}/promotion/default_baselines.csv` and `.md` with exactly one true-default baseline row per active `N=256` profile.

**Step 3: Commit**

No commit (execution-only stage).

---
