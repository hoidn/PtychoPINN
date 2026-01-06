# Debugging Loop Prompt — PtychoPINN Parity & Trace Discipline

Purpose
- Use this prompt when `input.md` sets Mode=`Parity`/`Debug`, an acceptance selector fails, or supervisor requests evidence collection.
- Drive first-divergence analysis between TensorFlow and PyTorch backends using trace artifacts.

Pre-flight
1. `timeout 30 git pull --rebase` (abort + `git pull --no-rebase` on timeout; resolve conflicts, resume with `timeout 30 git rebase --continue --no-edit`).
2. Read `input.md`, the focused `docs/fix_plan.md` item, and prior artifacts under `plans/active/<initiative-id>/reports/`.
3. Confirm the ledger item is `in_progress` and already lists reproduction commands.

Required references
- Spec parity & tracing shards: `docs/specs/spec-ptycho-conformance.md`, `docs/specs/spec-ptycho-tracing.md`
- Physics/runtime guardrails: `docs/specs/spec-ptycho-core.md`, `docs/specs/spec-ptycho-runtime.md`
- Architecture guides: `docs/architecture.md`, `docs/architecture_torch.md`, `docs/architecture_tf.md`
- Testing guidance: `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`
- Agent workflow: `CLAUDE.md`, `AGENTS.md`

Loop policy
- Two messages maximum: one short preamble before running commands, one final report.
- One fix-plan item per loop. Update Attempts History with `Metrics:`, `Artifacts:`, `First Divergence:` (or `n/a` if not found) before ending.
- Gather evidence first, then edit code. If the required parity harness is missing, pause and add a fix-plan TODO to author it.
- Documentation after fix: If the root cause is likely to recur, add an entry to `docs/debugging/TROUBLESHOOTING.md` in the same loop (Symptom, Root Cause, Solution). Only create a follow-up task in `docs/fix_plan.md` when in-loop documentation isn't feasible.

Procedure
1. **Reproduce** the failing selector(s) exactly as listed in `input.md` (selectors should come from `docs/TESTING_GUIDE.md` / `docs/specs/spec-ptycho-conformance.md`). Capture console output to the artifact directory.
2. **Callchain snapshot (optional)**: If the failure locus is unclear, run `prompts/callchain.md` to map candidate functions and tap points before modifying code.
3. **Trace instrumentation**: follow `docs/specs/spec-ptycho-tracing.md` to log paired traces (TensorFlow vs PyTorch). Collect values for targeted variables (e.g., forward model outputs, loss components, probe/object tensors).
4. **Isolation**: identify the first divergent value/operation. Document variable names, file:line, and the magnitude of divergence.
5. **Fix**: implement the smallest change that addresses the root cause. Do not weaken tests or thresholds.
6. **Metrics & visuals**: recompute correlation, MSE, RMSE, max|Δ|, sum ratios, and generate diff heatmaps. Store under the artifact directory (`metrics.json`, `diff_heatmap.png`, `trace.log`).
7. **Validation**: rerun targeted pytest selectors. Run the full suite once only if instructed and targeted tests pass.
8. **Ledger update**: append Attempts History with metrics, artifact paths, divergence summary, and next actions. If parity is not restored, keep `Status: in_progress` and outline follow-up steps.

Hard guardrails
- Never modify acceptance thresholds to make tests pass.
- Respect runtime guardrails: maintain vectorized execution, avoid unnecessary device transfers, keep dtype-neutral operations.
- Do not commit runtime artifacts; store them in the initiative reports directory only.

Output checklist (final message)
- Problem statement referencing spec lines implemented or investigated.
- Files/commits touched.
- Trace findings and first divergence.
- Metrics table (correlation, MSE, RMSE, max|Δ|, sum ratios).
- Pytest commands executed with results.
- Fix plan updates performed.
- Next action if parity is still incomplete.
