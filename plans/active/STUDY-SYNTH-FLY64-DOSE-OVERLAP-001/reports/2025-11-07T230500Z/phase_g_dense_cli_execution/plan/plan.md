# Phase G Dense Pipeline Evidence — Execution Plan (2025-11-07T230500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense evidence)  
**Action Type:** Planning (supervisor loop)  
**Target State:** ready_for_implementation

---

## Context

- `prepare_hub()` helper + `--clobber` flag landed in 2025-11-07T210500Z loop; pytest coverage and docs are GREEN.  
- Dense Phase C→G pipeline evidence is still missing; prior runs stalled due to stale Phase C outputs.  
- `validate_phase_c_metadata()` and `summarize_phase_g_outputs()` are implemented with tests but need real artifacts to validate on.  
- Dense manifest + metrics will unlock comparison analysis required for Phase G write-up.

## Objectives

1. Exercise the dense pipeline (`dose=1000`, `view=dense`) end-to-end with the fresh hub (`--clobber`) and capture CLI transcript + generated artifacts under this hub.  
2. Extend pytest coverage with a lightweight `--collect-only` smoke test to guard the CLI command wiring without running Phase C/G workloads.  
3. Post-run, invoke `validate_phase_c_metadata()` and `summarize_phase_g_outputs()` to confirm artifacts satisfy metadata + metrics requirements; persist derived summaries.  
4. Update documentation/ledger with metrics snapshot, CLI evidence, and findings adherence.

## Deliverables

- New pytest test `test_run_phase_g_dense_collect_only_generates_commands` covering CLI dry-run behavior.  
- CLI transcript `cli/phase_g_dense_pipeline.log` plus resulting manifest/metrics under `analysis/`.  
- Verification logs for `validate_phase_c_metadata` + `summarize_phase_g_outputs`.  
- Refreshed summary.md, docs/fix_plan.md attempt entry, galph_memory.md notes, and (if selectors change) TESTING_GUIDE/Test Suite Index updates.

## Task Breakdown

1. **TDD — Extend CLI coverage**  
   - Add `_import_main()` helper (similar to existing loader) to load `main()` from the orchestrator script.  
   - Author `test_run_phase_g_dense_collect_only_generates_commands` that invokes `main()` with `--collect-only` against a tmp hub and asserts commands are emitted without creating Phase C outputs.  
   - Capture RED log (`AttributeError` or missing helper) if needed; GREEN once assertions pass.

2. **Dense Pipeline Execution**  
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.  
  - Run `python plans/.../bin/run_phase_g_dense.py --hub <this hub> --dose 1000 --view dense --splits train test --clobber | tee cli/phase_g_dense_pipeline.log`.  
   - Monitor runtime; on failure capture traceback to `analysis/blocker.log` and stop.

3. **Post-run Validation**  
   - Invoke `python plans/.../bin/run_phase_g_dense.py --hub ... --collect-only` after successful run to capture dry-run commands (optional) and tee to `analysis/collect_commands.log`.  
   - Execute `python - <<'PY'` shim or dedicated helper to call `validate_phase_c_metadata` and `summarize_phase_g_outputs`; store logs in `analysis/validate_and_summarize.log`.  
   - Ensure metrics summary files (`metrics_summary.json/md`) exist and reference them in summary.md.

4. **Documentation & Ledger Updates**  
   - Update summary/summary.md with Turn Summary block + key metrics.  
   - Append Attempt entry to docs/fix_plan.md, refresh docs/TESTING_GUIDE.md + TEST_SUITE_INDEX if the new pytest test is added, and log findings in galph_memory.md.

## Findings & Guardrails Reinforced

- POLICY-001 — maintain torch dependency expectations (PyTorch may be tapped for baseline).  
- CONFIG-001 — ensure legacy bridge remains intact; CLI helper already handles this.  
- DATA-001 — validate NPZ metadata via guard after run.  
- TYPE-PATH-001 — keep all path interactions normalized.  
- OVERSAMPLING-001 — dense overlap parameters must remain unchanged during rerun.

## Pitfalls & Mitigations

- Do not delete hub contents without recording archival path; rely on `prepare_hub` for cleanup.  
- CLI run is long (~minutes); abort and log blocker at first failure instead of re-running blindly.  
- Tests must avoid real dataset paths—use tmp directories and monkeypatch slow calls if needed.  
- Ensure new pytest selector collects >0 tests; include collect-only proof.  
- Keep AUTHORITATIVE_CMDS_DOC exported for all pytest/CLI invocations.
