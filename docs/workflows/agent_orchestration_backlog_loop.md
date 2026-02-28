# Agent-Orchestration Backlog Loop Runbook

This runbook explains how to kick off and monitor the long-running backlog-driven implementation workflow:

- Workflow file: `workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`
- Prompt set: `prompts/workflows/backlog_plan_loop/*.md`

Execution model:
- one workflow run processes backlog items repeatedly
- for each backlog item, the full plan is executed in one unit
- each approved item is moved `docs/backlog/active -> docs/backlog/done` and then committed
- run exits only when `docs/backlog/active` has no remaining `.md` items (or on failure)

## 1) Preconditions

Run from `~/Documents/tmp/PtychoPINN`.

Required tooling:
- `codex` CLI installed and authenticated.
- `agent-orchestration` repo available at `~/Documents/agent-orchestration`.

Quick checks:

```bash
cd ~/Documents/tmp/PtychoPINN
command -v codex python
python -c "import sys; print(sys.version)"
```

## 2) Queue Contract (Backlog Items)

The workflow selects the next item from `docs/backlog/active/*.md` by:
- lowest `priority` (integer), then
- lexical filename order.

Each active backlog item must have YAML frontmatter with:
- `plan_path` (must point to an existing plan under `docs/plans/`)
- `priority` (integer)
- `check_commands` (non-empty YAML list; targeted checks, not full test suite)

Example:

```md
---
priority: 10
plan_path: docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md
check_commands:
  - pytest tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet" -q
---
```

If you only want a subset of work items, keep only those items in `docs/backlog/active/` and move others to `docs/backlog/paused/`.

## 3) Start in tmux (Recommended)

Because the loop is long-running, start it in tmux.

```bash
tmux new -s ptychopinn-backlog-loop
cd ~/Documents/tmp/PtychoPINN
mkdir -p logs state artifacts/work artifacts/checks artifacts/review artifacts/fixes docs/backlog/active docs/backlog/paused docs/backlog/done

export PYTHONPATH=~/Documents/agent-orchestration
python -m orchestrator.cli.main run \
  workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml \
  --debug \
  --step-summaries \
  --summary-mode async \
  --summary-provider claude_sonnet_summary \
  2>&1 | tee logs/backlog-loop-$(date -u +%Y%m%dT%H%M%SZ).log
```

Detach from tmux: `Ctrl-b` then `d`

Reattach later:

```bash
tmux attach -t ptychopinn-backlog-loop
```

## 4) What the Workflow Does

For each selected backlog item, it runs:
1. Select item + resolve `plan_path` + load `check_commands`
2. Capture a git baseline for this cycle
3. Initialize review cycle counter
4. Prepare cycle-scoped execute-session output path
5. Execute the full plan (provider: codex)
6. Prepare cycle-scoped checks-log output path
7. Run targeted checks
8. Prepare cycle-scoped review-report output path
9. Review implementation vs plan (`APPROVE` or `REVISE`)
10. If `REVISE`:
    - gate on max cycles
    - prepare cycle-scoped fix-session output path
    - run full-plan fix pass (provider: codex)
    - increment cycle and loop back through checks + review
11. On success, move backlog item `active -> done`
12. Run a post-approval commit step (provider: codex)
13. Recount active queue and continue until no active items remain

## 5) Monitoring

Run artifacts are under `.orchestrate/runs/<run_id>/`.

- State ledger: `.orchestrate/runs/<run_id>/state.json`
- Step logs: `.orchestrate/runs/<run_id>/logs/`
- Step summaries (when enabled): `.orchestrate/runs/<run_id>/summaries/`
- Prompt audit logs (because `--debug`):
  - `ExecutePlan.prompt.txt`
  - `ReviewImplVsPlan.prompt.txt`
  - `FixIssues.prompt.txt` (when that step runs)
  - `CommitOnApprove.prompt.txt` (when approval path runs)

Workflow state/output files in repo root:
- `state/execution_session_log_path.txt` (pointer for current execute/fix session log target)
- `state/check_log_path.txt` (pointer for current checks log target)
- `state/code_review_path.txt` (pointer for current review report target)
- `state/review_decision.txt`
- `state/review_cycle.txt`
- `state/commit_sha_path.txt`

Cycle-scoped human artifacts are written under:
- `artifacts/work/runs/<run_id>/c*-{execute,fix}-session.md`
- `artifacts/checks/runs/<run_id>/c*-checks.log`
- `artifacts/review/runs/<run_id>/c*-review.md`

## 6) Resume After Interruption

If interrupted, resume with the run ID:

```bash
cd ~/Documents/tmp/PtychoPINN
export PYTHONPATH=~/Documents/agent-orchestration
python -m orchestrator.cli.main resume <run_id> --debug \
  --step-summaries --summary-mode async --summary-provider claude_sonnet_summary
```

Quick way to resume the latest run:

```bash
cd ~/Documents/tmp/PtychoPINN
export PYTHONPATH=~/Documents/agent-orchestration
run_id=$(ls -1dt .orchestrate/runs/* | head -n1 | xargs -n1 basename)
python -m orchestrator.cli.main resume "$run_id" --debug \
  --step-summaries --summary-mode async --summary-provider claude_sonnet_summary \
  2>&1 | tee -a logs/backlog-loop-resume-${run_id}.log
```

Resume options:
- `--debug`: keep prompt/state debug artifacts and backups.
- `--repair`: attempt recovery from `state.json.step_*.bak` if state is corrupted.
- `--force-restart`: ignore existing state and start a new run from current queue state.

## 7) Safety Notes

- Run from `~/Documents/tmp/PtychoPINN` so relative paths resolve correctly.
- The workflow will move approved backlog items to `docs/backlog/done/`.
- `check_commands` should stay targeted and fast; avoid full-suite commands unless necessary.
