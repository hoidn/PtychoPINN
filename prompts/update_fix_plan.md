# Fix Plan Update Prompt (Structure & Update Semantics)

Purpose
- Provide a single, strict prompt for updating `fix_plan.md` (path: `docs/fix_plan.md`).
- Enforce “one item per loop” as “one item attempted per loop” — never silently drop failed attempts; keep the item active and add useful diagnostic context for future iterations.

When To Use
- At the start and end of every loop (build or debugging). If the loop is a debugging loop, first follow `prompts/debug.md`, then use this prompt to update `fix_plan.md`.

Core Principles
- One item per loop means: pick exactly one high‑value item and ATTEMPT it. If it fails, record the failure, first divergence, artifacts, and hypotheses; keep it active (do not mark complete).
- No‑cheating: do not change tests or thresholds; reflect reality in the plan. Anyone reading `fix_plan.md` should be able to reproduce the state and continue the work.
- Deterministic trace: always include reproduction commands, trace artifacts, and quantitative metrics.

Canonical Structure of docs/fix_plan.md

1) Header
- Title, last updated timestamp, owner (optional).
- Current active focus (1–2 sentences).

2) Index (optional)
- Short table of items (ID → Title, Priority, Status).

3) Items (one section per item)
Each item SHALL use the following template. When updating, keep bullet text concise (aim for ≤6 lines per bullet group) to preserve readability.

```
## [ID] <Concise Title>
- Spec/AT: <e.g., AT‑PARALLEL‑002 Pixel Size Independence>
- Priority: High | Medium | Low
- Status: pending | in_progress | blocked | done
- Owner/Date: <name>/<YYYY‑MM‑DD>
- Reproduction (C & PyTorch):
  * C: <full CLI>
  * PyTorch: <full CLI>
  * Shapes/ROI: <detector size, pixel size, ROI if used>
- First Divergence (if known): <variable + file:line>
- Attempts History (append per loop):
  * [YYYY‑MM‑DD] Attempt #N — Result: success|partial|failed; Summary (1–3 lines)
  * Metrics: corr=..., RMSE=..., MSE=..., max|Δ|=..., sums (C/Py) & ratio=..., peaks if relevant
  * Artifacts: plans/active/<ID>/reports/<timestamp>/{c_trace.log,py_trace.log,diff_heatmap.png,summary.json}
  * Observations/Hypotheses: bullets (ranked)
  * Next Actions: 1–3 surgical steps for the next loop
- Risks/Assumptions: bullets
- Exit Criteria (quote thresholds from spec): bullets
```

4) Completed (optional archive)
- Move fully completed/validated items here (keep metrics & artifact links). For very long plans, you MAY create an `archive/<date>_fix_plan_archive.md` file and move closed items there (include a summary + cross-reference).

Update Semantics (what to do each loop)

A) At Loop Start
- Pick exactly one item (highest value or unblocked). Mark `Status: in_progress`.
- Append an entry to “Attempts History” with the planned approach and current environment (device, dtype, seeds).
- Ensure reproduction commands are present and current.

B) During Loop (as you discover)
- If you find the FIRST DIVERGENCE (via traces), record `First Divergence: <variable + file:line>` and link artifacts.
- Update Observations/Hypotheses and Next Actions as you narrow scope.

C) Loop Outcome — Success
- Update the latest “Attempts History” entry with finalized metrics and artifacts.
- Set `Status: done`.
- Ensure Exit Criteria are explicitly satisfied and quoted.

D) Loop Outcome — Failure or Partial
- DO NOT mark done. Keep the item active.
- Update the latest “Attempts History” with:
  * Result: failed|partial, precise symptom (what still fails)
  * Metrics and visuals
  * First Divergence (or “not found”)
  * Most plausible hypotheses and what was ruled out
  * Concrete Next Actions for the next loop
- If truly blocked (external dependency, missing data), set `Status: blocked` and state why. Otherwise, return to `pending`.

E) Never “Gloss Over” Failed Attempts
- The plan is a ledger. If an attempt did not resolve the issue, record it with enough detail that a future loop can reproduce and continue.

F) Sorting & Archival
- Keep “in_progress” and “High” priority items at the top. Archive only “done” items (either in a dedicated Completed section or an `archive/<date>_fix_plan_archive.md` file), and when archiving condense the summary to a few lines while preserving key metrics and artifact links.
- Enforce a hard **1000-line maximum** for `docs/fix_plan.md`. Before saving, run `wc -l docs/fix_plan.md`; if the count is ≥1000, STOP the loop and move the least relevant or completed material to the `archive/` ledger (e.g., `archive/2025-XX-XX_fix_plan_archive.md` with a short summary + link) before committing. Never push a fix_plan.md longer than 999 lines.
- When pruning a bloated plan, preserve essential context. If the starting file exceeds 1000 lines, target a trimmed size around ~500 lines (not dramatically less) so active work, recent attempts, and high-value history stay in the main ledger. Document any major removals in the archive.

Example (abbreviated)

```
## [AT-WORKFLOW-001] Integration Workflow Regression
- Spec/AT: AT‑WORKFLOW‑001
- Priority: High
- Status: in_progress
- Owner/Date: alex/2025‑09‑29
- Reproduction:
  * CLI: `python -m unittest tests.test_integration_workflow`
  * Dataset: `datasets/fly/fly001_transposed.npz`
  * Shapes/ROI: full dataset (64×64 diffraction, gridsize=1)
- First Divergence: pending — integration workflow fails during model load (see Latest Attempt)
- Attempts History:
  * [2025‑09‑29] Attempt #1 — Result: failed. Training subprocess exited 1; model bundle missing `params.json`.
    Metrics: train subprocess log captured; no evaluation metrics produced.
    Artifacts: plans/active/AT-WORKFLOW-001/reports/2025-09-29T120300Z/{pytest.log,train_stdout.txt}
    Observations/Hypotheses: packaging step no longer writes `params.json`; CLI flag `--save_params` dropped in recent refactor.
    Next Actions: inspect `scripts/training/train.py` for the artifact write, add regression test for bundle contents, confirm `tests/test_integration_workflow` passes.
- Risks/Assumptions: none
- Exit Criteria: integration workflow passes with bundle containing `wts.h5.zip` and `params.json`; reproduction command exits 0 twice consecutively.
```

Routing Notes
- For debugging loops, always use `prompts/debug.md` for the loop itself. Then, apply this prompt to update `docs/fix_plan.md` precisely as above.
- Never remove an item because an attempt failed; update its Attempts History and Next Actions, and keep it active.

Commit Guidance
- Changes to `docs/fix_plan.md` should summarize the attempt succinctly, include metrics and artifact paths, and reflect the true outcome. Do not add runtime artifacts themselves to the repo unless the project standards say so.

End of Prompt
