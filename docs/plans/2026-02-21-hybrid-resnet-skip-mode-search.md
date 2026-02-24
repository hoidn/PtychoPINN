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
