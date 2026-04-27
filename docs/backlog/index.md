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
| [2026-03-13-lines256-experiment-history-summary-input.md](active/2026-03-13-lines256-experiment-history-summary-input.md) | `active` | n/a | Separate non-NeurIPS queue branch | Does not block or unlock the CNS benchmark queue. |

## Implications For Selection

- There is **no current active backlog item whose runnable status depends on
  another active backlog item**.
- The Phase 2 CNS queue currently has **three independent ablation lanes**
  (`history_len=1`, `modes32`, and hybrid-spectral architecture), with the
  hybrid-spectral architecture lane already in progress.
- The paper-default GNOT rerun and author-FFNO equal-footing compare are now
  completed external-baseline lanes, so they should not stay in the active
  queue or be treated as prerequisites for the remaining CNS items.
- Because the remaining CNS lanes are parallel, the selector should choose
  between them using steering and roadmap value, not a fabricated prerequisite
  chain.
- If a future backlog item truly requires another backlog item to land first,
  add that item id to `prerequisites` and update this index in the same patch.
