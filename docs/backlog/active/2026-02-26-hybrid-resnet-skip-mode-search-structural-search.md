---
priority: 30
plan_path: docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-structural-search.md
check_commands:
  - test -f docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-structural-search.md
  - rg -q "### Task 12: Stage C Search" docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-structural-search.md
---

# Backlog Item: Hybrid ResNet Skip/Mode Search Structural Search

Implement and maintain the structural search contract in:
`docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-structural-search.md`

Execution mode:
- one plan slice at a time
- implementation-vs-plan review gate required
- fix loop required on `REVISE`
