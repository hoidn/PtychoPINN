Summary: Extend Phase C metadata coverage and rerun the dense Phase C→G pipeline to capture real metrics evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_dose_overlap_generation.py::test_generate_dataset_for_dose_handles_metadata_splits -vv
  - pytest tests/study/test_dose_overlap_generation.py -k metadata_pickle_guard -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: tests/study/test_dose_overlap_generation.py::test_generate_dataset_for_dose_handles_metadata_splits (prove Stage 5 loads metadata-bearing splits via MetadataManager) and adjust studies/fly64_dose_overlap/generation.py::generate_dataset_for_dose if validator still receives `_metadata`
- Validate: pytest tests/study/test_dose_overlap_generation.py -k "metadata_splits or metadata_pickle_guard" -vv
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
- Execute: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log
- Run: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/analysis/metrics_summary.json --highlights plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/analysis/aggregate_highlights.txt --output plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.md | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.log
- Capture: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/summary/summary.md and docs/fix_plan.md with MS-SSIM/MAE deltas, pipeline exit code, and artifact pointers.
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T230500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{plan,collect,red,green,cli,analysis,summary}
4. pytest tests/study/test_dose_overlap_generation.py::test_generate_dataset_for_dose_handles_metadata_splits -vv | tee "$HUB"/red/pytest_metadata_splits.log  # expect RED until validator sees filtered dict
5. pytest tests/study/test_dose_overlap_generation.py -k metadata_pickle_guard -vv | tee "$HUB"/collect/pytest_metadata_pickle_guard.log
6. Apply code/test updates per Implement bullet, keeping MetadataManager usage and Path normalization intact.
7. pytest tests/study/test_dose_overlap_generation.py -k "metadata_splits or metadata_pickle_guard" -vv | tee "$HUB"/green/pytest_metadata_suite.log
8. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee "$HUB"/green/pytest_highlights_preview.log
9. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber | tee "$HUB"/cli/run_phase_g_dense_cli.log
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --highlights "$HUB"/analysis/aggregate_highlights.txt --output "$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest.log
11. Record MS-SSIM/MAE deltas plus any failures in "$HUB"/summary/summary.md and mirror the same update into docs/fix_plan.md Attempts History.

Pitfalls To Avoid:
- Keep `_metadata` out of validator inputs; fail loudly if MetadataManager is bypassed.
- Do not run the pipeline without `--clobber`; stale Phase C outputs will trigger prepare_hub guard.
- Ensure AUTHORITATIVE_CMDS_DOC stays exported for every pytest/pipeline shell.
- Avoid touching protected core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Capture RED → GREEN logs inside the new hub; no evidence outside `plans/active/`.
- Watch GPU saturation; if the rerun stalls, log `pgrep -fl run_phase_g_dense.py` results before relaunching.
- Maintain Path objects when handing off to validators (TYPE-PATH-001 compliance).
- Do not introduce new dependencies or alter environment settings.
- Preserve highlights output (`aggregate_highlights.txt`) for digest input.
- Treat any non-zero exit from pipeline or digest script as a blocker and log immediately.

If Blocked:
- Store failing pytest output under "$HUB"/red/ with selector in filename; summarize traceback in docs/fix_plan.md.
- If pipeline aborts, keep CLI logs, note return code, and capture offending command in `$HUB`/analysis/blocker.log before halting.
- If digest inputs are missing, snapshot `$HUB` directory tree via `find "$HUB" -maxdepth 2` into red/ and mark attempt blocked in fix_plan + galph_memory.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency acknowledged; no skipping torch-backed phases/commands.
- CONFIG-001 — Export AUTHORITATIVE_CMDS_DOC before pytest/pipeline invocations.
- DATA-001 — Enforce MetadataManager for all NPZ IO and confirm validator receives canonical structures.
- TYPE-PATH-001 — Normalize Path usage in generate_dataset_for_dose and downstream helpers.
- OVERSAMPLING-001 — Monitor neighbor_count ≥ gridsize² when reviewing validator output and logs.

Pointers:
- studies/fly64_dose_overlap/generation.py:150 — Stage 5 validation block using MetadataManager.
- tests/study/test_dose_overlap_generation.py:260 — Existing metadata guard tests to extend with metadata splits coverage.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:664 — Phase commands + reporting helper invocation.
- docs/TESTING_GUIDE.md:268 — Phase G orchestrator workflow and command guardrails.
- docs/findings.md:2 — Knowledge base index (POLICY-001 / DATA-001 / TYPE-PATH-001 references).

Next Up (optional):
- Stage the sparse-view dense pipeline run once dense evidence is captured.

Doc Sync Plan (Conditional):
- After GREEN tests, run `pytest tests/study/test_dose_overlap_generation.py -k "metadata_splits or metadata_pickle_guard" --collect-only -vv | tee "$HUB"/green/pytest_metadata_collect.log` and update docs/TESTING_GUIDE.md §2.4 plus docs/development/TEST_SUITE_INDEX.md with the new selector label (`metadata_splits`).

Mapped Tests Guardrail:
- Ensure the metadata selectors collect ≥1 test via `--collect-only` (log saved at `$HUB`/green/pytest_metadata_collect.log`); do not proceed if collection returns 0.

Hard Gate:
- Do not mark the focus done until the dense pipeline exits 0, metrics_summary.json + aggregate_highlights.txt exist under `$HUB`/analysis/, `metrics_digest.md` is generated without failure banner, and both summary.md + docs/fix_plan.md capture MS-SSIM/MAE deltas with artifact links.
