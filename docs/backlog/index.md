# Backlog Dependency Index

This index records the current dependency relations for the active NeurIPS
backlog queue. It is the human-readable companion to backlog frontmatter such as
`prerequisites` and `related_roadmap_phases`.

Current source of truth:

- strategic order: [steering.md](../steering.md)
- roadmap gates: [2026-04-20-neurips-hybrid-resnet-submission-roadmap.md](../plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md)
- queue items: [`docs/backlog/active/`](active/) and [`docs/backlog/in_progress/`](in_progress/)

## Rules

- A backlog item gets a `prerequisites` entry only when another backlog item or
  completed tranche must land first.
- Reuse of existing CNS anchors, shell-lock decisions, or adapter support is
  treated as an already-satisfied study precondition, not as a backlog-to-backlog
  dependency.
- Parallel items on the same roadmap phase are explicitly listed as such so the
  selector does not invent a false serial order.

## Current Queue Graph

| Item | Status | Roadmap Phase | Dependency Relation | Notes |
|---|---|---|---|---|
| [2026-04-21-pdebench-cns-markov-history1-compare.md](active/2026-04-21-pdebench-cns-markov-history1-compare.md) | `active` | `phase-2-pdebench-128x128-image-suite` | Independent ablation lane; no active backlog prerequisite | Depends only on already-available `history_len=2` capped anchors for comparison. Must rerun the full four-row shell on `history_len=1`. |
| [2026-04-21-pdebench-cns-spectral-modes32-compare.md](active/2026-04-21-pdebench-cns-spectral-modes32-compare.md) | `active` | `phase-2-pdebench-128x128-image-suite` | Independent ablation lane; no active backlog prerequisite | Depends only on the existing `12/12` spectral anchor family, which is already available. |
| [2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation.md](in_progress/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation.md) | `in_progress` | `phase-2-pdebench-128x128-image-suite` | Independent ablation lane; no active backlog prerequisite | Assumes the fixed canonical CNS shell (`skip-add` + `pixelshuffle`) and existing shared/noshare anchor evidence. Separate from Markov and modes studies. |
| [2026-04-27-pdebench-ffno-convolutional-features-cns.md](active/2026-04-27-pdebench-ffno-convolutional-features-cns.md) | `active` | `phase-2-pdebench-128x128-image-suite` | FFNO-family CNS extension; no active backlog prerequisite | Builds on completed authored-FFNO and FFNO-close context, but is not blocked by any remaining active item. |
| [2026-04-27-cdi-ffno-generator-lines-best-config.md](active/2026-04-27-cdi-ffno-generator-lines-best-config.md) | `active` | `phase-3-cdi-anchor-regeneration` | CDI/ptycho generator lane; no active backlog prerequisite | Uses the study-indexed best lines configuration and must not reuse CNS evidence as CDI evidence. |
| [2026-04-27-hybrid-spectral-ffno-parameter-space-cns-cdi.md](active/2026-04-27-hybrid-spectral-ffno-parameter-space-cns-cdi.md) | `active` | `phase-2-pdebench-128x128-image-suite`, `phase-3-cdi-anchor-regeneration` | Blocked on narrower architecture and FFNO follow-ups | Depends on `2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation`, `2026-04-27-pdebench-ffno-convolutional-features-cns`, and `2026-04-27-cdi-ffno-generator-lines-best-config`. |
| [2026-03-13-lines256-experiment-history-summary-input.md](active/2026-03-13-lines256-experiment-history-summary-input.md) | `active` | n/a | Separate non-NeurIPS queue branch | Does not block or unlock the CNS benchmark queue. |

## Implications For Selection

- Most active items remain independently runnable, but
  `2026-04-27-hybrid-spectral-ffno-parameter-space-cns-cdi` is intentionally
  blocked on narrower architecture and FFNO follow-ups.
- The Phase 2 CNS queue currently has **four immediate ablation/extension
  lanes** (`history_len=1`, `modes32`, hybrid-spectral architecture, and
  FFNO-with-convolutional-features), with the hybrid-spectral architecture lane
  already in progress.
- The FFNO-as-CDI-generator lane is separate from CNS and belongs to the CDI
  evidence track.
- The broader Hybrid-spectral-to-FFNO parameter-space study is intentionally
  blocked on the narrower CNS architecture, CNS FFNO-conv, and CDI FFNO
  generator items.
- The paper-default GNOT rerun and author-FFNO equal-footing compare are now
  completed external-baseline lanes, so they should not stay in the active
  queue or be treated as prerequisites for the remaining CNS items.
- For the independently runnable lanes, the selector should choose between them
  using steering and roadmap value, not a fabricated prerequisite chain.
- If a future backlog item truly requires another backlog item to land first,
  add that item id to `prerequisites` and update this index in the same patch.
