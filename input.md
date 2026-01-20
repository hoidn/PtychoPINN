**Summary**: Document NaN debugging completion and close out DEBUG-SIM-LINES-DOSE-001 initiative for its core scope.
**Focus**: DEBUG-SIM-LINES-DOSE-001 — Initiative cleanup and documentation (NaN debugging COMPLETE)
**Branch**: paper
**Mapped tests**: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
**Artifacts**: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103144Z/`

**Do Now**
This is a documentation and cleanup handoff. The NaN debugging scope of DEBUG-SIM-LINES-DOSE-001 is COMPLETE. The tasks below finalize the initiative documentation.

- Create: Create the new artifacts directory:
  ```bash
  mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103144Z/
  ```

- Document: Write a final summary to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103144Z/final_summary.md` with:
  1. **Initiative Outcome**: NaN debugging COMPLETE
  2. **Root Cause**: CONFIG-001 violation (stale params.cfg not synced before training/inference)
  3. **Fix Applied**: C4f added update_legacy_dict() calls in pipeline.py and plan-local runner
  4. **Verification Evidence**: All 4 scenarios (gs1_ideal, gs1_custom, gs2_ideal, gs2_custom) train without NaN
  5. **Hypotheses Resolved**: H-CONFIG ✅ CONFIRMED; H-PROBE-IDEAL-REGRESSION ❌ RULED OUT; H-GRIDSIZE-NUMERIC ❌ RULED OUT
  6. **Remaining Issue**: Amplitude bias (~3-6x) is a separate quality issue for future investigation
  7. **Key Artifacts**: Links to reports/2026-01-20T160000Z/ (gs1/gs2 ideal runs) and reports/2026-01-20T102300Z/ (B0f isolation test)

- Update: Add a knowledge base entry to `docs/findings.md` documenting the CONFIG-001 enforcement requirement for sim_lines_4x workflows:
  ```markdown
  | SIM-LINES-CONFIG-001 | 2026-01-20 | sim_lines_4x, CONFIG-001, NaN, training | The sim_lines_4x pipeline and any plan-local runners MUST call `update_legacy_dict(params.cfg, config)` before training or inference to prevent NaN failures. Without this sync, legacy modules read stale gridsize/intensity values causing training collapse. Fix applied in C4f (scripts/studies/sim_lines_4x/pipeline.py + DEBUG-SIM-LINES-DOSE-001 runner). | plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103144Z/final_summary.md | Resolved |
  ```

- Guard: Run the CLI smoke test to ensure all sim_lines imports remain healthy:
  ```bash
  AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
    2>&1 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103144Z/pytest_cli_smoke.log
  ```

**How-To Map**
1. The final_summary.md should be a standalone document that someone can read to understand what this initiative accomplished.
2. The findings.md entry follows the existing table format (see PINN-CHUNKED-001 for reference).
3. Keep the documentation concise — the detailed evidence is in the artifact directories.

**Pitfalls To Avoid**
1. Do not modify any production code — this is documentation-only.
2. Do not touch the existing report directories (2026-01-20T160000Z, 2026-01-20T102300Z) — only create the new 2026-01-20T103144Z directory.
3. The findings.md entry should be appended to the existing table, not replace any entries.

**If Blocked**
- If the pytest smoke test fails, capture the error and note it in the final_summary.md — the documentation can still be completed.
- Record any blockers in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103144Z/blocked.log`.

**Findings Applied (Mandatory)**
- CONFIG-001 — This initiative CONFIRMS that CONFIG-001 enforcement is critical for sim_lines_4x workflows; the finding being added documents this specific application.

**Pointers**
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` — Full implementation plan with all checklist items
- `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md` — Turn-by-turn summaries
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/` — gs1_ideal/gs2_ideal verification runs
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T102300Z/` — B0f isolation test (gs1_custom)
- `docs/findings.md` — Knowledge base where the new entry should be added

**Next Up (optional)**
After this documentation is complete, the supervisor will:
1. Mark DEBUG-SIM-LINES-DOSE-001 as "done" in docs/fix_plan.md
2. Select the next focus from the pending initiatives (likely FIX-DEVICE-TOGGLE-001 or PARALLEL-API-INFERENCE)
