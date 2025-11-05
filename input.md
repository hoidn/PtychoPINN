Summary: Make Phase C simulation resilient to metadata-bearing NPZ files so the dense pipeline can finish and produce metrics evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_dose_overlap_generation.py::test_build_simulation_plan_handles_metadata_pickle_guard -vv
  - pytest tests/study/test_dose_overlap_generation.py::test_load_data_for_sim_handles_metadata_pickle_guard -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: scripts/simulation/simulate_and_save.py::load_data_for_sim (metadata-aware base NPZ load) and studies/fly64_dose_overlap/generation.py::build_simulation_plan (reuse MetadataManager for n_images inspection)
- Validate: pytest tests/study/test_dose_overlap_generation.py -k "metadata_pickle_guard" -vv
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
- Execute: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log
- Run: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run/analysis/metrics_summary.json --highlights plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run/analysis/aggregate_highlights.txt --output plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.md | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.log
- Capture: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run/summary/summary.md with MS-SSIM/MAE deltas, pipeline exit code, and log references; append the same outcomes to docs/fix_plan.md Attempts History.
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. export HUB=plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T210500Z/phase_g_dense_full_execution_real_run
3. mkdir -p "$HUB"/{plan,red,green,cli,analysis,summary}
4. pytest tests/study/test_dose_overlap_generation.py -k "metadata_pickle_guard" -vv | tee "$HUB"/red/pytest_phase_c_metadata_guard.log  # expect RED until loaders switch to MetadataManager
5. Edit scripts/simulation/simulate_and_save.py::load_data_for_sim and studies/fly64_dose_overlap/generation.py::build_simulation_plan to load via MetadataManager; ensure tests reference new helper names.
6. pytest tests/study/test_dose_overlap_generation.py -k "metadata_pickle_guard" -vv | tee "$HUB"/green/pytest_phase_c_metadata_guard.log
7. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee "$HUB"/green/pytest_highlights_preview.log
8. tail -n 40 plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log >> "$HUB"/plan/previous_cli_tail.log
9. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber | tee "$HUB"/cli/run_phase_g_dense_cli.log
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$HUB"/analysis/metrics_summary.json --highlights "$HUB"/analysis/aggregate_highlights.txt --output "$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest.log
11. Update "$HUB"/summary/summary.md and docs/fix_plan.md with exit status, MS-SSIM/MAE deltas, and artifact links.

Pitfalls To Avoid:
- Do not fall back to raw np.load with allow_pickle=False in any new code paths; rely on MetadataManager everywhere.
- Keep authoring/tests under plans/active scope; avoid touching core stable modules (ptycho/model.py, ptycho/diffsim.py, ptycho/tf_helper.py).
- Always export AUTHORITATIVE_CMDS_DOC before pytest or pipeline commands (CONFIG-001).
- Capture RED/GRAY evidence in "$HUB"/red before applying fixes; rerun for GREEN in "$HUB"/green.
- Ensure new tests stay pure pytest (no unittest mix) and name them with `metadata_pickle_guard` for selector reuse.
- When editing scripts, preserve Path usage to satisfy TYPE-PATH-001.
- Watch for lingering pipeline processes; use `pgrep -fl run_phase_g_dense.py` if runs stall before relaunching.
- Keep all outputs inside "$HUB"; no stray artifacts at repo root.
- Treat missing metrics files or non-zero pipeline exit as blockers and log under "$HUB"/red.
- Do not upgrade/install packages; environment is frozen for this focus.

If Blocked:
- Store failing pytest output in "$HUB"/red/ with the selector in the filename and note the traceback in docs/fix_plan.md.
- If the pipeline fails again, save CLI logs under "$HUB"/red/, capture exit code, and document the failure signature plus next debugging hypothesis in docs/fix_plan.md and galph_memory.md.
- For missing metrics artifacts, snapshot directory listings via `ls` into "$HUB"/red/`ls_analysis.txt` and mark the attempt blocked.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch >=2.2 dependency enforced; ensure imports succeed before long runs.
- CONFIG-001 — AUTHORITATIVE_CMDS_DOC export precedes pytest/pipeline commands.
- DATA-001 — Maintain DATA-001 contract when reading/writing NPZ files and surface validator failures immediately.
- TYPE-PATH-001 — Keep path parameters as Path objects when passing between helpers.
- OVERSAMPLING-001 — Verify Phase C outputs maintain K ≥ C; log if validator reports violations.

Pointers:
- docs/fix_plan.md:4 — Active initiative ledger entry.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:722 — Phase C→G command orchestration.
- studies/fly64_dose_overlap/generation.py:64 — build_simulation_plan loader currently using np.load.
- scripts/simulation/simulate_and_save.py:37 — load_data_for_sim to be refactored for MetadataManager.
- tests/study/test_dose_overlap_generation.py:28 — Existing Phase C tests to extend with metadata guards.
- docs/TESTING_GUIDE.md:268 — Phase G orchestrator workflow and AUTHORITATIVE guardrails.

Next Up (optional):
- After dense evidence lands, stage sparse-view pipeline rerun with the same metadata-safe loaders.

Doc Sync Plan (Conditional):
- After GREEN tests, run `pytest tests/study/test_dose_overlap_generation.py -k "metadata_pickle_guard" --collect-only -vv | tee "$HUB"/green/pytest_metadata_collect.log` and append the new selectors to docs/TESTING_GUIDE.md §2.4 and docs/development/TEST_SUITE_INDEX.md under Phase C tests.

Mapped Tests Guardrail:
- Ensure `pytest tests/study/test_dose_overlap_generation.py -k "metadata_pickle_guard" -vv` collects both new tests; if collection fails, fix names before proceeding.

Hard Gate:
- Do not mark this loop complete until the dense pipeline exits 0, metrics_summary.json + aggregate_highlights.txt exist under "$HUB"/analysis/, the digest script exits 0 without failure banner, and docs/fix_plan.md plus "$HUB"/summary/summary.md capture MS-SSIM/MAE deltas with artifact links.
