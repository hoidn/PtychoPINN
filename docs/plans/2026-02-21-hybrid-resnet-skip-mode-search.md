# Hybrid ResNet Skip Connections + Mode Search Implementation Plan (Hub)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional encoder-decoder skip connections to `hybrid_resnet`, add a reproducible mode×skip×width benchmark workflow for `N=128` and `N=256`, and define a staged structural-search extension for depth/downsampling/capacity/skip-design axes.

**Architecture:** Keep default behavior unchanged (`skip_connections=False`) to preserve current baselines/integration expectations. Implement additive skip fusion with lightweight `1x1` projection layers at decoder resolutions (`N/2`, `N`). Expose one boolean knob end-to-end (`hybrid_skip_connections`) through Torch-only runner/execution config + CLI (do not bridge this knob into TensorFlow/canonical model contracts), then run a deterministic sweep over `fno_modes × hybrid_skip_connections × fno_width` with fixed probe-mask/loss-normalization controls. Make dataset choice explicit via named dataset profiles so the same sweep can run on multiple failure-mode regimes. Execute Stage A in two steps (full grid on `N=128`, then top-K promotion to `N=256`), then add structural axes one stage at a time (B→E) with bounded per-stage run budgets.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, `pytest`, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` (normative knob semantics, stage gates, ranking/promotion policy, and artifact contract).

## Plan Split

- `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-implementation-core.md`
  - Owns Tasks 0-8 (preflight, RED/GREEN implementation, docs sync, verification/smoke).
  - Owns the Test Evidence Contract for this initiative.
- `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-stage-a-execution.md`
  - Owns Tasks 9-11 (full Stage-A handoff and Stage-B execution commands).
- `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-structural-search.md`
  - Owns Tasks 12-15 (Stages C-E structural-axis work and governance).

## Global Epoch Contract

- All stage execution runs (`A` through `E`) MUST use at least `10` training epochs per run (`--epochs-n128 >= 10`, `--epochs-n256 >= 10`) unless an approved exception is recorded in the execution log with rationale.
- Any stage outputs produced with fewer than `10` epochs are non-canonical for promotion/governance decisions.
- If a downstream stage consumed non-canonical (<10 epoch) upstream outputs, rerun from the earliest violating stage and regenerate all dependent downstream artifacts.

## Apples-to-Apples Baseline Contract

- Any claim that a candidate is "better than baseline/default/anchor" MUST use an apples-to-apples comparator:
  - same dataset profile and exact data sources (same NPZ/HDF5 paths and split),
  - same epoch budget,
  - same stage scope and objective set,
  - same seed policy (single-seed comparisons use seed `3`; promotion comparisons use seeds `{3,11,17}` and median-rank governance).
- Cross-stage or cross-epoch comparisons (for example Stage C `10` epochs vs Stage A `1` epoch) are non-canonical and MUST NOT be used for promotion/governance claims.
- Each stage execution package MUST include a baseline evidence artifact (CSV/Markdown table) that records baseline run id(s) and metrics (`amp_ssim`, `amp_mae`, `amp_mse`, `phase_ssim`, `train_wall_time_sec`, `inference_time_s`) and explicitly marks whether each comparison is apples-to-apples.
- Each stage execution package MUST also include `promotion/default_baselines.csv` and `promotion/default_baselines.md` with exactly one true-default baseline row per active `(N, dataset_profile)` combination, including the full default-model parameter tuple and baseline run id for discoverability.

## N=256 Dual-Profile Contract

- Canonical `N=256` evaluation and promotion MUST run on both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256` (lines/structured NPZ pair supplied by caller).
- `N=256` results produced on only one of these profiles are diagnostic-only and MUST NOT be used as canonical promotion evidence.

## Canonical Anchor Contract

- Canonical structural search starts from true `hybrid_resnet` defaults at Stage B `N=128` (the Stage-A control anchor), not from the Stage-A promoted winner.
- Required Stage-A control anchor tuple:
  - `modes=12`, `skip=off`, `width=32`, `fno_blocks=4`
  - `downsample_schedule=2`, `downsample_op=stride_conv`
  - `encoder_conv_hidden=none`, `encoder_spectral_hidden=none`
  - `max_hidden=none`, `resnet_width=none`, `resnet_blocks=6`, `skip_style=add`
- Stage-A promoted winners remain valid for Stage-A `N=256` promotion/rerank and optional diagnostics.

## Progress Checklist

- [ ] Task 0-8 complete in `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-implementation-core.md`
- [ ] Task 9-11 complete in `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-stage-a-execution.md`
- [ ] Task 12-15 complete in `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-structural-search.md`

## Session Log

| Date (UTC) | Scope | Status | Evidence Paths | Commit(s) |
| --- | --- | --- | --- | --- |
| 2026-02-24 | Plan hygiene + contracts | Completed | `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`, `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` | `9c477c4e`, `cf2dee67` |
| 2026-02-24 | Plan split (hub + 3 subplans) | Completed | `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`, `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-implementation-core.md`, `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-stage-a-execution.md`, `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-structural-search.md` | pending |

## Do Next (Update Every Session)

- Execute Tasks 0-8 using the implementation-core split plan.
- After each session, update the relevant split plan checklist and append one row here with evidence paths.
- Keep stage ranking/promotion behavior aligned to the companion design doc.

## Stage Promotion Governance (Task 15)

Promotion gates between stages:
- Apply feasible-Pareto ordering with `amp_ssim` as the primary promotion metric and `train_wall_time_sec` as the efficiency objective.
- Enforce feasibility filters before promotion:
  - `phase_ssim_drop_vs_baseline <= max_phase_ssim_drop` (default `0.03`)
  - train/runtime and parameter budgets captured in summary rows (`train_wall_time_sec`, `model_params`)
  - inference SLA: `inference_time_s <= 60` at `N=128`, `<= 240` at `N=256`
- Require robustness validation for promotion sources:
  - boundary set = `top-K + next 2`
  - rerank seeds = `{3,11,17}`
  - promote by median Pareto rank across seeds
- Baseline MAE/MSE regressions are treated as diagnostics; promotion ordering is driven by amplitude SSIM.
- Stage A completion requires a verified `hybrid-resnet-mode-skip-sweep` entry in `docs/studies/index.md`.

Hard stop conditions (pause-and-diagnose):
- Trigger pause when two consecutive stages show `<1%` median relative gain and rerank confidence intervals overlap zero.
- Trigger pause when all new-stage candidates regress on amplitude SSIM at `N=256` and the same direction appears in `N=128` robustness summaries.
- Before final axis stop, run one bounded rescue mini-sweep for the same axis; if still failing, pause further expansion on that axis and carry at least one hedge candidate into the next stage with low-budget monitoring.
