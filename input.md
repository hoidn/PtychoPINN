Summary: Harden the dense highlights verifier for phase-only previews, sync the docs, then rerun the dense Phase C→G pipeline under the 2025-11-11T005802Z hub with full verifier evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude -vv; pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv; pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv; pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T005802Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract)
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py::validate_metrics_delta_highlights — enforce preview-phase-only formatting (reject any preview line containing "amplitude" or missing the phase-only prefix), record violations in new metadata (`preview_phase_only`, `preview_format_errors`), and keep TYPE-PATH-001 compliant error payloads.
- Implement: tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude — add a RED→GREEN test that injects amplitude text into the preview file and asserts the validator fails with the new metadata fields; update the GREEN coverage (`...highlights_complete`) to expect empty `preview_format_errors`.
- Validate: run `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude -vv`, `pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv`, `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv`, and `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv`, archiving logs under `$HUB`/{red,green,collect}.
- Execute: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md and HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T005802Z/phase_g_dense_full_execution_real_run; ensure `$HUB`/{analysis,cli,collect,green,red,summary} exist; run `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log` and wait for `[8/8]` SUCCESS.
- Verify: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log`, `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" |& tee "$HUB"/analysis/highlights_check.log`, and `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights_post_run.log` for evidence.
- Document: update `$HUB`/summary/summary.md with MS-SSIM/MAE (phase emphasis) deltas, CLI/verifier status, highlight preview guard results, and artifact inventory counts; refresh docs/TESTING_GUIDE.md and docs/development/TEST_SUITE_INDEX.md for the preview artifact + new selector, then update docs/fix_plan.md and galph Turn Summary with links + lessons.

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T005802Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{analysis,cli,collect,green,red,summary}
4. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude -vv | tee "$HUB"/red/pytest_preview_guard_red.log || true
5. Edit plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py and tests/study/test_phase_g_dense_artifacts_verifier.py per Do Now (preview-phase guard + new RED fixture) and capture doc updates after tests pass.
6. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_preview_contains_amplitude -vv | tee "$HUB"/green/pytest_preview_guard_green.log
7. pytest tests/study/test_phase_g_dense_artifacts_verifier.py::test_verify_dense_pipeline_highlights_complete -vv | tee "$HUB"/green/pytest_highlights_complete.log
8. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_digest.log
9. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights_pre_run.log
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$HUB"/cli/run_phase_g_dense.log
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py --hub "$HUB" --report "$HUB"/analysis/pipeline_verification.json |& tee "$HUB"/analysis/verifier_cli.log
12. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$HUB" |& tee "$HUB"/analysis/highlights_check.log
13. pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv | tee "$HUB"/collect/pytest_collect_highlights_post_run.log
14. Update docs/TESTING_GUIDE.md (Phase G delta section) and docs/development/TEST_SUITE_INDEX.md (Phase G entry) to describe the preview artifact + new selector; capture diffs referencing the new guard.
15. Summarize MS-SSIM/MAE deltas, guard status, CLI evidence, and artifact inventory counts in "$HUB"/summary/summary.md; update docs/fix_plan.md + galph Turn Summary with artifact links.

Pitfalls To Avoid
- Do not allow preview lines to include "amplitude" or extra tokens (verification must fail loudly on mixed content).
- Keep AUTHORITATIVE_CMDS_DOC exported before running orchestrator/verifier to satisfy CONFIG-001.
- Maintain POSIX-relative paths in inventory and metadata (TYPE-PATH-001); avoid `Path.resolve()` when serializing JSON entries.
- When editing tests, re-run the RED fixture before claiming GREEN evidence (TDD guard).
- The dense pipeline is long-running; monitor `[n/8]` progress and capture blockers immediately instead of rerunning blindly.
- Never edit Phase C NPZ outputs or generated CLI logs; rely on validators and re-runs for correctness.
- CLI log filenames must retain the dose/view suffixes spelled out in TEST-CLI-001; do not collapse them into generic names.
- Keep doc updates aligned with actual behavior (MAE precision ±0.000000, preview artifact now required) to avoid spec drift.

If Blocked
- If the new preview-phase guard keeps failing, archive the failing pytest log under `$HUB`/red/ with the assertion message, capture the offending preview snippet, and update docs/fix_plan.md Attempts History plus summary.md with the failure signature before pausing the loop.
- If the dense pipeline aborts mid-phase, stop immediately, save `$HUB`/cli/run_phase_g_dense.log` plus the specific phase log, and log the blocker (phase name, exit code) inside `$HUB`/summary/summary.md and docs/fix_plan.md; enter blocked status only after recording evidence.

Findings Applied (Mandatory)
- POLICY-001 — PtyChi reconstruction inside run_phase_g_dense.py still depends on torch>=2.2; keep the environment torch-enabled.
- CONFIG-001 — Export AUTHORITATIVE_CMDS_DOC so legacy consumers observe synchronized params before data/model work.
- DATA-001 — Phase C datasets remain authoritative; diagnose via validators instead of editing NPZ contents.
- TYPE-PATH-001 — Inventory, JSON metadata, and preview/highlights paths must remain POSIX-relative for downstream tooling.
- STUDY-001 — Report MS-SSIM/MAE deltas with explicit signs and phase emphasis inside summary.md + doc updates.
- TEST-CLI-001 — CLI log + preview/highlights guards must enforce real filename patterns and sentinel strings so regressions fail fast.

Pointers
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:309 — current `validate_metrics_delta_highlights` gap (no preview-phase-only enforcement).
- tests/study/test_phase_g_dense_artifacts_verifier.py:1756 — existing RED test for missing preview file (extend nearby to cover amplitude contamination).
- docs/TESTING_GUIDE.md:331 — Phase G delta persistence spec needs precision + preview updates.
- docs/development/TEST_SUITE_INDEX.md:62 — Phase G orchestrator test registry entry to expand with the new selector.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:1151 — call site where `persist_delta_highlights` returns `delta_summary` referenced by verifier.

Next Up (optional)
1. After dense evidence lands, queue the sparse-view pipeline run plus verifier coverage using the same preview guard.
2. Extend `check_dense_highlights_match.py` to emit a CSV/JSON summary for highlights vs preview parity to aid regression diffing.

Doc Sync Plan (Conditional)
- Once the preview guard + tests pass, run `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv` (logs already mapped) and update `docs/TESTING_GUIDE.md` §"Phase G Delta Metrics Persistence" plus `docs/development/TEST_SUITE_INDEX.md` to describe the preview artifact + new selector. Archive the collect-only log under `$HUB`/collect/ and reference the doc diffs in summary.md.

Mapped Tests Guardrail
- `pytest --collect-only tests/study/test_phase_g_dense_artifacts_verifier.py -vv` must collect (>0 tests). If collection breaks after changes, stop and fix (or document the block) before marking the loop complete; no downgrade without justification.
