# Backlog Dependency Index

This index records the current dependency relations for the active NeurIPS
backlog queue. It is the human-readable companion to backlog frontmatter such as
`prerequisites` and `related_roadmap_phases`.

Current source of truth:

- strategic order: [steering.md](../steering.md)
- roadmap gates: [2026-04-20-neurips-hybrid-resnet-submission-roadmap.md](../plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md)
- deterministic active gate: [roadmap_gate.json](roadmap_gate.json)
- queue items: [`docs/backlog/active/`](active/) and [`docs/backlog/in_progress/`](in_progress/)

## Rules

- A backlog item gets a `prerequisites` entry only when another backlog item or
  completed tranche must land first.
- Reuse of existing CNS anchors, shell-lock decisions, or adapter support is
  treated as an already-satisfied study precondition, not as a backlog-to-backlog
  dependency.
- Parallel items on the same roadmap phase are explicitly listed as such so the
  selector does not invent a false serial order.
- `docs/backlog/roadmap_gate.json` is applied before provider selection. If no
  active item satisfies that gate and the missing work is already authorized by
  the roadmap, the workflow may draft a new active backlog item instead of
  selecting later-phase work.

## Current Queue Graph

| Item | Status | Roadmap Phase | Dependency Relation | Notes |
|---|---|---|---|---|
| [2026-04-28-pdebench-cns-spectral-modes24-convergence-compare.md](active/2026-04-28-pdebench-cns-spectral-modes24-convergence-compare.md) | `active` | `phase-2-pdebench-128x128-image-suite` | Depends on completed modes-32 compare | Reruns `12/12` and new `24/24` spectral rows at `1024 / 128 / 128` with batch size `16` under the same convergence-oriented budget, so mode-count interpretation is not confounded by old under-converged batch-4 rows. |
| [2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap.md](in_progress/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap.md) | `in_progress` | `phase-2-pdebench-128x128-image-suite` | Depends on completed Hybrid-spectral architecture ablation | Reruns only the `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_shared_blocks10` finalists at `2048 / 256 / 256` to compare `512 -> 1024 -> 2048` capped scaling deltas. |
| [2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space.md](active/2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space.md) | `active` | `phase-2-pdebench-128x128-image-suite` | Depends on completed Hybrid-spectral architecture and CNS FFNO-conv follow-ups | Phase 2 CNS-only split of the former mixed CNS/CDI parameter-space item. Runs a staged one-axis-at-a-time architecture-space study under the capped CNS decision-support contract. |
| [2026-04-27-cdi-ffno-generator-lines-best-config.md](in_progress/2026-04-27-cdi-ffno-generator-lines-best-config.md) | `in_progress` | `phase-3-cdi-anchor-regeneration` | CDI/ptycho generator lane; no active backlog prerequisite | Uses the study-indexed best lines configuration and must not reuse CNS evidence as CDI evidence. Currently blocked by roadmap routing unless Phase 3 is explicitly opened. |
| [2026-04-27-hybrid-spectral-ffno-parameter-space-cdi.md](active/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on CDI FFNO generator baseline | Phase 3 CDI-only split of the former mixed CNS/CDI parameter-space item. The deterministic Phase 2 gate should exclude it until Phase 3 is explicitly opened. |
| [2026-03-13-lines256-experiment-history-summary-input.md](active/2026-03-13-lines256-experiment-history-summary-input.md) | `active` | n/a | Separate non-NeurIPS queue branch | Does not block or unlock the CNS benchmark queue. |

## Implications For Selection

- The immediate roadmap-consistent Phase 2 CNS queue now has one active
  follow-up lane, one in-progress follow-up lane, and one lower-priority staged
  architecture-space lane: the converged-budget `12/12` versus `24/24`
  mode-count compare, the in-progress `2048 / 256 / 256` Hybrid-spectral
  finalist scaling check, and the CNS-only Hybrid-spectral-to-FFNO
  parameter-space split.
- The FFNO-as-CDI-generator lane is separate from CNS and belongs to the CDI
  evidence track. It should not be selected while the current roadmap routing
  still requires remaining Phase 2 PDEBench work.
- The broader Hybrid-spectral-to-FFNO parameter-space study has been split by
  roadmap phase. The CNS half is selectable under the Phase 2 gate after its
  CNS prerequisites; the CDI half remains future-gated on Phase 3 and the CDI
  FFNO generator baseline.
- The Markov history-1, modes-32, Hybrid-spectral architecture, FFNO local-conv,
  paper-default GNOT, and author-FFNO equal-footing lanes are completed. The
  modes-24 convergence item may use the completed modes-32 run as context, but
  it should not treat that old item as still-runnable active work.
- For the independently runnable lanes, the selector should choose between them
  using steering and roadmap value, not a fabricated prerequisite chain.
- If a future backlog item truly requires another backlog item to land first,
  add that item id to `prerequisites` and update this index in the same patch.
