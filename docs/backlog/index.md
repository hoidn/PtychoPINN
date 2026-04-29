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
- `docs/backlog/roadmap_gate.json` is applied before provider selection. The
  current gate is a Phase 2 PDEBench plus Phase 3 CDI-preparation selection
  window: Phase 2 PDEBench evidence remains preferred, and Phase 3 CDI
  preparation may be selected as useful parallel work without counting as
  completed Phase 2 evidence. If no active item satisfies the current window and
  the missing work is already authorized by the roadmap, the workflow may draft
  a new active backlog item instead of selecting later-phase work.

## Current Queue Graph

| Item | Status | Roadmap Phase | Dependency Relation | Notes |
|---|---|---|---|---|
| [2026-04-28-pdebench-cns-spectral-modes24-convergence-compare.md](in_progress/2026-04-28-pdebench-cns-spectral-modes24-convergence-compare.md) | `in_progress` | `phase-2-pdebench-128x128-image-suite` | Depends on completed modes-32 compare | Reruns `12/12` and new `24/24` spectral rows at `1024 / 128 / 128` with batch size `16` under the same convergence-oriented budget, so mode-count interpretation is not confounded by old under-converged batch-4 rows. |
| [2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap.md](done/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap.md) | `done` | `phase-2-pdebench-128x128-image-suite` | Depends on completed Hybrid-spectral architecture ablation | Reran only the `spectral_resnet_bottleneck_base` and `spectral_resnet_bottleneck_shared_blocks10` finalists at `2048 / 256 / 256`; base remained stronger on aggregate while shared-blocks10 kept narrower mid/high-frequency advantages. |
| [2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence.md](active/2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence.md) | `active` | `phase-2-pdebench-128x128-image-suite` | Depends on completed Hybrid-spectral architecture ablation and 2048cap scaling follow-up | Reruns only `spectral_resnet_bottleneck_shared_blocks10` at the `1024 / 128 / 128` cap for a longer budget because its 40-epoch 1024cap train curve was still dropping sharply. |
| [2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space.md](active/2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space.md) | `active` | `phase-2-pdebench-128x128-image-suite` | Depends on completed Hybrid-spectral architecture and CNS FFNO-conv follow-ups | Phase 2 CNS-only split of the former mixed CNS/CDI parameter-space item. Runs a staged one-axis-at-a-time architecture-space study under the capped CNS decision-support contract. |
| [2026-04-27-cdi-ffno-generator-lines-best-config.md](active/2026-04-27-cdi-ffno-generator-lines-best-config.md) | `active` | `phase-3-cdi-anchor-regeneration` | CDI/ptycho generator lane; no active backlog prerequisite | Uses the study-indexed best Lines128 configuration needed by the paper benchmark design and must not reuse CNS evidence as CDI evidence. Eligible as Phase 3 CDI-preparation work, but it does not satisfy the Phase 2 PDEBench evidence gate. |
| [2026-04-29-cdi-lines128-paper-benchmark-harness.md](active/2026-04-29-cdi-lines128-paper-benchmark-harness.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on CDI FFNO generator baseline | Adds the shared Lines128 paper-benchmark wrapper/harness, contract-reconstruction preflight artifact, FNO/seed decision manifest, and metric-schema downgrade gate. Does not launch the full benchmark. |
| [2026-04-29-cdi-lines128-paper-benchmark-execution.md](active/2026-04-29-cdi-lines128-paper-benchmark-execution.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on CDI FFNO generator baseline and Lines128 paper-benchmark harness | Runs the full four-row Lines128 paper-quality CDI benchmark and publishes metrics tables, visual reconstruction comparisons, provenance manifests, and durable summaries. |
| [2026-04-29-cdi-lines128-supervised-equivalent-rows.md](active/2026-04-29-cdi-lines128-supervised-equivalent-rows.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on complete Lines128 CDI benchmark execution | Adds supervised-training equivalents for CNN, FNO, SCR, and FFNO under the locked Lines128 contract so the paper can separate model-body effects from PINN forward-model training effects. |
| [2026-04-27-hybrid-spectral-ffno-parameter-space-cdi.md](active/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on CDI FFNO generator baseline | Phase 3 CDI-only split of the former mixed CNS/CDI parameter-space item. It becomes eligible only after the CDI FFNO generator baseline completes. |
| [2026-03-13-lines256-experiment-history-summary-input.md](active/2026-03-13-lines256-experiment-history-summary-input.md) | `active` | n/a | Separate non-NeurIPS queue branch | Does not block or unlock the CNS benchmark queue. |

## Implications For Selection

- The immediate roadmap-consistent Phase 2 CNS queue now has one in-progress
  mode-count lane plus two active follow-up lanes: the converged-budget `12/12`
  versus `24/24` mode-count compare, the longer-convergence `1024cap`
  shared-blocks10 rerun, and the lower-priority CNS-only
  Hybrid-spectral-to-FFNO parameter-space split.
- The completed `2048 / 256 / 256` Hybrid-spectral finalist scaling check
  motivates the longer-convergence shared-blocks10 item, but the new item is a
  narrow convergence probe rather than a new scaling sweep.
- The FFNO-as-CDI-generator lane is separate from CNS and belongs to the CDI
  evidence track. It is now selectable as Phase 3 CDI-preparation work when it
  is the most useful parallel item, but completion does not close any remaining
  Phase 2 PDEBench evidence requirement.
- The Lines128 paper-quality CDI benchmark queue is intentionally serial:
  `2026-04-27-cdi-ffno-generator-lines-best-config` must land first,
  `2026-04-29-cdi-lines128-paper-benchmark-harness` then makes the shared
  wrapper/harness, contract reconstruction, FNO/seed decision, and metric schema
  executable, and `2026-04-29-cdi-lines128-paper-benchmark-execution` runs the
  paper benchmark only after those prerequisites are complete.
- The Lines128 harness and execution items are Phase 3 CDI work. They are
  allowed by the current deterministic gate as CDI-track work, but they do not
  count as Phase 2 PDEBench evidence and should not be used to close the PDEBench
  image-suite gate.
- The supervised Lines128 CDI extension is deliberately downstream of the
  complete PINN-trained CDI table. It should reuse the locked contract and label
  rows as architecture plus training procedure, not reopen the primary benchmark
  definition.
- The broader Hybrid-spectral-to-FFNO parameter-space study has been split by
  roadmap phase. The CNS half is selectable under the Phase 2 gate after its
  CNS prerequisites; the CDI half remains blocked until the CDI FFNO generator
  baseline completes.
- The Markov history-1, modes-32, Hybrid-spectral architecture, FFNO local-conv,
  paper-default GNOT, and author-FFNO equal-footing lanes are completed. The
  modes-24 convergence item may use the completed modes-32 run as context, but
  it should not treat that old item as still-runnable active work.
- For the independently runnable lanes, the selector should choose between them
  using steering and roadmap value, not a fabricated prerequisite chain.
- If a future backlog item truly requires another backlog item to land first,
  add that item id to `prerequisites` and update this index in the same patch.
