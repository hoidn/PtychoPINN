Summary: Add aggregate metrics to Phase G summaries and rerun the dense pipeline on a clean hub to capture real evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution/

Do Now (hard validity contract):
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::summarize_phase_g_outputs — compute per-model aggregate metrics (mean & best MS-SSIM amplitude/phase, mean MAE amplitude/phase) and emit them in both JSON (`aggregate_metrics`) and Markdown (`## Aggregate Metrics`) outputs.
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution/cli/phase_g_dense_pipeline.log
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): python - <<'PY' | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution/analysis/validate_and_summarize.log
from pathlib import Path
import importlib.util

script_path = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py")
spec = importlib.util.spec_from_file_location("run_phase_g_dense", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

hub = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution")
module.validate_phase_c_metadata(hub)
module.summarize_phase_g_outputs(hub)
PY

How-To Map:
  1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  2. mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution/{analysis,cli,collect,docs,green,red,summary}
  3. Implement aggregate metrics in `summarize_phase_g_outputs` (preserve existing per-job data; add deterministic ordering) and extend the Markdown writer with a `## Aggregate Metrics` section summarizing mean/best MS-SSIM + mean MAE per model.
  4. Update `tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs` to fabricate fixtures that cover aggregates, asserting both JSON (`aggregate_metrics`) and Markdown section contents.
  5. Run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution/green/pytest_summarize_green.log`
  6. Run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution/green/pytest_metadata_guard_green.log`
  7. Run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution/green/pytest_collect_only_green.log`
  8. Capture selector inventory with `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution/collect/pytest_phase_g_orchestrator_collect.log`
  9. Execute dense pipeline: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution/cli/phase_g_dense_pipeline.log`
 10. After success, run the embedded validation shim from Do Now to capture guard + aggregate summary logs.
 11. Update `summary/summary.md` with Turn Summary, aggregate highlights, metrics pointers, and link to CLI/analysis logs; sync docs (`docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`) once GREEN.

Pitfalls To Avoid:
  - Do not mutate Phase C NPZ contents while computing aggregates; operate on CSV outputs only.
  - Keep aggregate ordering deterministic (sorted by model, metric) to avoid flaky diffs.
  - Ensure float formatting uses consistent precision (retain raw floats in JSON, format Markdown to 3 decimals max).
  - Monitor CLI runtime; abort immediately on failure and capture traceback in `analysis/blocker.log`.
  - Preserve previous hubs; never reuse 2025-11-07T230500Z directories when generating new evidence.
  - Maintain AUTHORITATIVE_CMDS_DOC export for every pytest/CLI invocation.
  - Avoid hard-coded absolute paths in tests; rely on `tmp_path` and helper loaders.
  - Do not skip doc/test registry updates if new selectors collect; mark blocked if aggregates fail before docs update.

If Blocked:
  - Stop work, tee failing pytest/CLI output to `analysis/blocker.log`, add blocker context to `summary/summary.md`, and document the issue in docs/fix_plan.md + galph_memory.md before ending the loop.

Findings Applied (Mandatory):
  - POLICY-001 — Dense pipeline invokes PyTorch baselines; ensure environment retains torch dependency and CLI handles backend switches.
  - CONFIG-001 — CLI helper must continue bridging params.cfg prior to Phase C; aggregates cannot bypass initialization order.
  - DATA-001 — Guard + summaries confirm NPZ metadata contract on regenerated outputs.
  - TYPE-PATH-001 — Normalize hubs/paths inside script and tests (Path.resolve, Path operations).
  - OVERSAMPLING-001 — Dense overlap configuration must stay unchanged; report any deviation detected in CLI output.

Pointers:
  - docs/findings.md:8
  - docs/findings.md:10
  - docs/findings.md:14
  - docs/findings.md:21
  - docs/findings.md:17
  - docs/TESTING_GUIDE.md:215
  - docs/development/TEST_SUITE_INDEX.md:62
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:194
  - tests/study/test_phase_g_dense_orchestrator.py:100
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution/plan/plan.md:1

Next Up (optional):
  - When dense evidence is green, mirror aggregate logic for sparse view to complete Phase G coverage.

Doc Sync Plan (Conditional):
  - After GREEN, rerun `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv` (already captured in step 8), update docs/TESTING_GUIDE.md §2 and docs/development/TEST_SUITE_INDEX.md with the aggregate-aware selector wording, and reference the new collect-only log in summary.md.
