Summary: Wire post-verify automation into `run_phase_g_dense.py`, update the orchestrator tests, then run the counted dense Phase C→G pipeline (post-verify on) into the 2025-11-12 hub with verifier/highlights evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Plan Document: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Reports Hub (active): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/
Hub Change Justification: reuse hub until the counted dense run lands (analysis/ + cli/ still empty).
Mapped tests:
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
- pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_hooks -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — add a default-on post-verify hook (with `--skip-post-verify`) that calls `verify_dense_pipeline_artifacts.py` and `check_dense_highlights_match.py`, tees their logs to `analysis/verify_dense_stdout.log` and `analysis/check_dense_highlights.log`, emits the JSON report, prints hub-relative success paths, and includes the commands in `--collect-only` output (POLICY-001, TYPE-PATH-001, TEST-CLI-001, PREVIEW-PHASE-001).
- Implement: tests/study/test_phase_g_dense_orchestrator.py::{test_run_phase_g_dense_collect_only_generates_commands,test_run_phase_g_dense_post_verify_hooks} — extend the collect-only expectations for the new commands and add a monkeypatched test that asserts the post-verify invocations fire with the correct hub/log/report paths (DATA-001, TEST-CLI-001).
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log (post-verify enabled) to produce the counted Phase C→G bundle plus verifier/highlights outputs.
- Validate: pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify -vv | tee "$HUB"/collect/pytest_collect_orchestrator_post_verify.log, then run pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_hooks -vv | tee "$HUB"/green/pytest_phase_g_dense_post_verify.log (archive any RED run under `$HUB`/red/).
- Validate: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/verification_report.json --dose 1000 --view dense | tee "$HUB"/analysis/verify_dense_stdout.log && python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights.log (rerun `ssim_grid.py --hub "$HUB"` if summary drift is detected).
- Document: Update `$HUB/summary/summary.md` with MS-SSIM/MAE deltas, preview verdict, CLI/log/test references; refresh docs/fix_plan.md + galph_memory with this loop’s evidence, and record any doc/test registry edits (TEST-CLI-001, STUDY-001, TYPE-PATH-001).

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier
2. Edit `plans/active/.../bin/run_phase_g_dense.py`: add argparse flag `--skip-post-verify`, thread the boolean through `main()`, list the verifier/highlights commands inside the collect-only printout, and after the SSIM grid helper call `run_command` for:
   - `python plans/.../bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/verification_report.json --dose 1000 --view dense` → log `analysis/verify_dense_stdout.log`
   - `python plans/.../bin/check_dense_highlights_match.py --hub "$HUB"` → log `analysis/check_dense_highlights.log`
   Ensure the success banner prints hub-relative paths for the logs/report and re-run `generate_artifact_inventory(hub)` after the post-verify commands.
3. Update `tests/study/test_phase_g_dense_orchestrator.py`:
   - Extend `test_run_phase_g_dense_collect_only_generates_commands` assertions for the new CLI/log strings (`verify_dense_pipeline_artifacts.py`, `check_dense_highlights_match.py`, `verification_report.json`, `check_dense_highlights.log`).
   - Add `test_run_phase_g_dense_post_verify_hooks` that monkeypatches `run_command` to capture command tuples, runs `main()` with `--splits train test --dose 1000 --view dense --hub <tmp> --post-verify`, and asserts the captured commands include the verifier/highlights invocations with hub-relative log paths.
4. pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify -vv | tee "$HUB"/collect/pytest_collect_orchestrator_post_verify.log (must show the new test).
5. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_hooks -vv; tee stdout to /tmp/pytest_phase_g_dense_post_verify.log and copy RED/GREEN outputs into `$HUB`/red/ or `$HUB`/green/.
6. python plans/.../bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense_stdout.log (post-verify on). Monitor `cli/` for per-phase logs and `analysis/` for new summaries + verification artifacts.
7. python plans/.../bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/verification_report.json --dose 1000 --view dense | tee "$HUB"/analysis/verify_dense_stdout.log (rerun if automation fails); python plans/.../bin/check_dense_highlights_match.py --hub "$HUB" | tee "$HUB"/analysis/check_dense_highlights.log (rerun `ssim_grid.py --hub "$HUB"` before rechecking if values drift).
8. Update `$HUB/summary/summary.md` with MS-SSIM/MAE deltas (phase only), preview guard status, CLI/log/test evidence, and verification JSON pointers; log outcomes in docs/fix_plan.md + galph_memory; ensure artifact inventory + summary mention new files.

Pitfalls To Avoid
- Skipping `AUTHORITATIVE_CMDS_DOC` export will break Phase D/E legacy bridges (CONFIG-001).
- Do not hardcode absolute paths in success banners, logs, or artifact inventory (TYPE-PATH-001).
- Keep post-verify automation optional via `--skip-post-verify`; default must remain enabled but debuggable.
- Ensure pytest selectors are updated and logged under `$HUB`/{collect,red,green}/ to satisfy TEST-CLI-001.
- Capture `analysis/verification_report.json`, `analysis/check_dense_highlights.log`, and `analysis/ssim_grid_summary.md` in the artifact inventory; missing entries violate DATA-001.
- Phase C metadata guard must continue to run; do not bypass `validate_phase_c_metadata()` when adding automation (PHASEC-METADATA-001).
- The highlights checker must still enforce phase-only preview content; do not weaken PREVIEW-PHASE-001 when wiring automation.
- Keep CLI logs for new commands under `cli/` or `analysis/` as prescribed—no `/tmp` or absolute host paths.
- Avoid launching multiple orchestrator instances simultaneously; wait for the counted run to finish before reruns.

If Blocked
- If `run_phase_g_dense.py` fails before Phase C completes, capture `analysis/blocker.log`, note the failing command + return code in docs/fix_plan.md and galph_memory, and halt; do not delete partial outputs.
- If post-verify automation flakes (e.g., missing preview or JSON drift), rerun the helper (`ssim_grid.py`) and document the error string verbatim in `$HUB`/analysis, then mark the ledger entry blocked until evidence lands.
- If pytest selectors fail to collect (0 tests), fix the test definitions immediately; if unresolved, log the failure in `docs/fix_plan.md` (Attempts History) and keep the artifacts under `$HUB`/red/.

Findings Applied (Mandatory)
- POLICY-001 — PyTorch is mandatory for the Phase E/Phase F steps invoked by the orchestrator; ensure env includes torch>=2.2.
- CONFIG-001 — Maintain the `update_legacy_dict(params.cfg, config)` ordering by exporting AUTHORITATIVE_CMDS_DOC before pipeline commands.
- DATA-001 — Verifier enforces the canonical NPZ/JSON contracts; post-verify hook must run to keep evidence trustworthy.
- TYPE-PATH-001 — All CLI logs, summaries, and inventory entries must be hub-relative.
- STUDY-001 — Report MS-SSIM ±0.000 and MAE ±0.000000 deltas for vs_Baseline and vs_PtyChi.
- TEST-CLI-001 — Preserve collect/red/green pytest logs plus CLI transcripts for every command added to the automation.
- PREVIEW-PHASE-001 — Highlights checker/SSIM grid summary must continue to enforce phase-only previews with actionable failures.
- PHASEC-METADATA-001 — Phase C metadata validation still blocks the pipeline; do not bypass the guard when wiring automation.

Pointers
- docs/TESTING_GUIDE.md:331 — Phase G verifier + helper requirements and CLI logging expectations.
- docs/development/TEST_SUITE_INDEX.md:64 — Phase G tests roster and selector format (update after adding new orchestrator tests).
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:199 — Phase G objectives + deliverables for this initiative.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/plan/plan.md — Current plan detailing post-verify automation + run steps.
- docs/findings.md (POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, STUDY-001, TEST-CLI-001, PREVIEW-PHASE-001, PHASEC-METADATA-001) — Authoritative guardrails referenced above.

Next Up (optional)
- Run the sparse view variant once dense evidence is green.

Doc Sync Plan (tests added/renamed this loop)
- After the new orchestrator tests pass, update docs/TESTING_GUIDE.md §5 “Phase G Orchestrator Evidence” with the post-verify automation flag + selectors, and append the new test row to docs/development/TEST_SUITE_INDEX.md. Archive the `pytest --collect-only` log under `$HUB/collect/` and mention the update in docs/fix_plan.md before closing the loop.

Mapped Tests Guardrail
- `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify -vv` must collect the new test; save the log to `$HUB`/collect/pytest_collect_orchestrator_post_verify.log.

Hard Gate
- If any mapped selector collects 0 after these changes (e.g., orchestrator test filtered away), do not mark the loop complete; either repair the selector or document the block with artifact links and a ledger update.
