---
priority: 18
plan_path: docs/plans/2026-03-13-lines256-experiment-history-summary-input.md
check_commands:
  - pytest --collect-only tests/studies/test_lines_256_arch_improvement_workflow.py -q
  - pytest tests/studies/test_lines_256_arch_improvement_workflow.py -v
  - PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml --dry-run --stream-output
---

# Backlog Item: Add Lines 256 Experiment History Summary Input

## Objective
- Feed the `lines_256` experiment agent a compact derived history summary so it can avoid repeating recent failed ideas without reading the raw TSV ledger directly.

## Scope
- Add a deterministic summary artifact derived from `state/lines_256_arch_improvement/results.tsv`.
- Inject that summary into the experiment and crash-debug steps in both workflow variants.
- Keep the accepted-state contract intact and leave the ledger format unchanged.

## Notes for Reviewer
- Do not inject the full TSV ledger into prompts unless the bounded-summary approach is proven insufficient.
- Keep the summary small and deterministic; this is a prompt-efficiency and search-quality improvement, not a logging redesign.
- Preserve behavior for empty-history sessions and freshly initialized ledgers.
