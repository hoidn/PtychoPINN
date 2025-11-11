Summary: Document the preview guard updates, then rerun the dense Phase C→G pipeline with verifier/highlight evidence stored under the new 2025-11-11T012044Z hub.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv; pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T012044Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract)
- Implement: docs/TESTING_GUIDE.md::Phase G Delta Metrics Persistence — describe the JSON/highlights/preview trio (MS-SSIM ±0.000, MAE ±0.000000, preview is four phase-only lines under analysis/metrics_delta_highlights_preview.txt) and reference PREVIEW-PHASE-001 + TYPE-PATH-001 so the spec matches commit 783c32aa.
- Implement: docs/development/TEST_SUITE_INDEX.md::Phase G entry — add the new preview guard selector (`pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude -vv`), summarize its RED fixture (amplitude contamination) + GREEN expectations, and point to the hub evidence path.
- Validate: run `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude -vv`, `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv`, `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`, and `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv`, archiving stdout/stderr to `$HUB`/{green,collect} for PREVIEW-PHASE-001 evidence.
- Execute: `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` then `export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T012044Z/phase_g_dense_full_execution_real_run`; ensure `$HUB`/{analysis,cli,collect,green,red,summary} exist and run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log`, leaving every per-phase CLI log + helper log under `$HUB`/cli/.
- Verify: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log`, `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" |& tee "$HUB"/analysis/highlights_check.log`, and rerun `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights_post_run.log` so the guardrail selector collects after the dense run as well.
- Document: capture MS-SSIM/MAE deltas (phase emphasis), preview guard status, verifier/highlights results, CLI log inventory, and doc updates inside `$HUB`/summary/summary.md; update docs/fix_plan.md + docs/findings.md (close PREVIEW-PHASE-001 once preview evidence is archived) and append the Turn Summary + artifact links.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T012044Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude -vv | tee "$HUB"/green/pytest_preview_guard_green.log
5. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv | tee "$HUB"/green/pytest_highlights_complete.log
6. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_digest.log
7. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights.log
8. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
9. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" |& tee "$HUB"/analysis/highlights_check.log
11. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights_post_run.log
12. Update docs/TESTING_GUIDE.md (Phase G section) + docs/development/TEST_SUITE_INDEX.md (Phase G entry) with the preview rules + selector details; record deltas with `git diff -- docs/TESTING_GUIDE.md docs/development/TEST_SUITE_INDEX.md` for summary.md references.
13. Summarize MS-SSIM/MAE deltas, CLI/verifier/highlight evidence, doc updates, and findings status inside "$HUB"/summary/summary.md; refresh docs/fix_plan.md + docs/findings.md; append the Turn Summary block.

Pitfalls To Avoid
- Do not run the dense pipeline without exporting AUTHORITATIVE_CMDS_DOC first (CONFIG-001 guard rail).
- Keep `$HUB` paths POSIX-relative in all JSON/summary files to satisfy TYPE-PATH-001.
- Treat Phase C NPZ outputs as read-only; rerun generators rather than editing data (DATA-001).
- Ensure every CLI log filename keeps its dose/view suffix (TEST-CLI-001) before copying into `$HUB`/cli/.
- Capture every pytest/log command output to the hub; PREVIEW-PHASE-001 requires reproducible evidence.
- Monitor `run_phase_g_dense.py` for `[n/8]` stalls; if a phase fails, stop immediately and archive the failing log instead of rerunning blindly.
- Highlight preview + JSON deltas must retain ±0.000 / ±0.000000 precision; verify before updating docs to avoid spec drift.
- When updating docs, cite the initiative ID so readers can trace PREVIEW-PHASE-001 history.
- Do not delete the prior 2025-11-11T005802Z hub; it’s a planning artifact referenced in earlier attempts.
- Avoid touching core physics/TensorFlow modules (ptycho/model.py, diffsim.py, tf_helper.py); focus stays on docs/tests/pipeline orchestration.

If Blocked
- If the dense pipeline aborts (e.g., Phase F reconstruction failure), capture `$HUB`/cli/run_phase_g_dense.log plus the specific phase log, summarize the failure in `$HUB`/summary/summary.md`, and update docs/fix_plan.md with the blocker signature; stop further steps.
- If pytest selectors fail unexpectedly, keep the failing log under `$HUB`/red/ with the traceback, reference PREVIEW-PHASE-001 in summary.md, and mark the attempt blocked in docs/fix_plan.md before requesting guidance.
- If doc updates cannot reconcile with actual behavior, document the mismatch in `$HUB`/summary/summary.md`, leave comments in the docs referencing TODO + PREVIEW-PHASE-001, and halt before the dense run.

Findings Applied (Mandatory)
- POLICY-001 — Dense pipeline touches PtyChi LSQML (PyTorch backend); keep torch>=2.2 available during the run.
- CONFIG-001 — Export AUTHORITATIVE_CMDS_DOC so legacy params.cfg consumers stay synchronized before Phase C/D/E helpers execute.
- DATA-001 — Treat Phase C NPZ datasets as canonical inputs; use validators rather than manual edits.
- TYPE-PATH-001 — Keep all recorded paths POSIX-relative (e.g., analysis/metrics_delta_highlights_preview.txt) inside JSON/summary payloads.
- STUDY-001 — Report MS-SSIM/MAE deltas with explicit ± signs and emphasize phase metrics in the summary.
- TEST-CLI-001 — Record the real CLI filenames (dose/view suffixes + helper logs) so the verifier/tests enforce complete bundles.
- PREVIEW-PHASE-001 — Preview artifacts must remain phase-only; archive validator metadata + pytest logs proving the guard blocks amplitude contamination.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:309 — preview guard implementation (`preview_phase_only` + metadata checks).
- tests/study/test_phase_g_dense_artifacts_verifier.py:1921 — `test_verify_dense_pipeline_highlights_preview_contains_amplitude` RED fixture ensuring amplitude contamination fails.
- docs/TESTING_GUIDE.md:330 — Phase G delta persistence section that still documents ±0.000 formatting for all deltas and lacks preview guidance.
- docs/development/TEST_SUITE_INDEX.md:62 — Phase G orchestrator entry to expand with the preview guard selector.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:1150 — Orchestrator main path that produces highlights/preview artifacts for dense evidence.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py:12 — Helper that cross-checks highlights vs preview output files.

Next Up (optional)
1. After dense evidence lands, replicate the preview guard flow for the sparse-overlap pipeline to close the study.
2. Extend `check_dense_highlights_match.py` to emit a CSV diff for faster regression comparisons.

Doc Sync Plan
- Update `docs/TESTING_GUIDE.md` (§Phase G Delta Metrics Persistence) and `docs/development/TEST_SUITE_INDEX.md` (Phase G entry) after the code/tests stay green; cite PREVIEW-PHASE-001 and include the new selector.
- Run `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv` again after code/doc updates (log stored at `$HUB`/collect/pytest_collect_highlights_post_run.log) and reference the log in summary.md plus docs/TESTING_GUIDE.md §2/test registry if names change.

Mapped Tests Guardrail
- `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv` must collect (>0) before and after the dense run; if collection breaks, stop immediately, fix the selector (or document the block), and do not mark the loop complete until collection succeeds.
