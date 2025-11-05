Summary: Integrate the Phase G reporting helper into the dense orchestrator and execute the dose 1000 dense pipeline to capture real metrics evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics_missing_model_fails -vv
  - pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T050500Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract):
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — append a reporting-helper command (using run_command) that generates `analysis/aggregate_report.md` with CLI log coverage in both collect-only and real execution paths.
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics_missing_model_fails -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T050500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T050500Z/phase_g_dense_full_execution_real_run/cli/phase_g_dense_pipeline.log
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python - <<'PY' | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T050500Z/phase_g_dense_full_execution_real_run/analysis/validate_and_summarize.log
from pathlib import Path
import importlib.util

script_path = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py")
spec = importlib.util.spec_from_file_location("run_phase_g_dense", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

hub = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T050500Z/phase_g_dense_full_execution_real_run")
module.validate_phase_c_metadata(hub)
module.summarize_phase_g_outputs(hub)
PY

How-To Map:
  1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  2. mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T050500Z/phase_g_dense_full_execution_real_run/{analysis,cli,collect,docs,green,red,summary}
  3. Update `tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands` to assert the new reporting-helper command appears in collect-only output; run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee .../red/pytest_orchestrator_collect_only_red.log` to capture RED.
  4. Modify `plans/active/.../bin/run_phase_g_dense.py::main` to append the reporting-helper command details (TYPE-PATH-001) and, after `summarize_phase_g_outputs`, call `run_command` to generate `analysis/aggregate_report.md` with CLI log `analysis/aggregate_report_cli.log`.
  5. Re-run the orchestrator collect-only selector for GREEN: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_collect_only_generates_commands -vv | tee .../green/pytest_orchestrator_collect_only_green.log`.
  6. Run guard selectors to ensure no regressions: 
     - `pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv | tee .../green/pytest_summarize_green.log`
     - `pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv | tee .../green/pytest_report_helper_green.log`
     - `pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics_missing_model_fails -vv | tee .../green/pytest_report_helper_missing_green.log`
  7. Capture selector inventory with `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > .../collect/pytest_phase_g_orchestrator_collect.log`.
  8. Execute the dense pipeline with clobber (Do Now command) and confirm `analysis/aggregate_report_cli.log` + `analysis/aggregate_report.md` are generated automatically by the orchestrator; abort and document if any command fails.
  9. Re-run the validation shim (Do Now snippet) to verify guards remain GREEN and metrics summary matches expectations.
 10. Inspect `analysis/aggregate_report.md` and record key deltas in `summary/summary.md`, linking to CLI logs and Markdown report.
 11. Update `docs/TESTING_GUIDE.md` §Phase G reporting, `docs/development/TEST_SUITE_INDEX.md` Phase G entries, and `docs/fix_plan.md` Attempts History to reflect the automated report flow and evidence.

Pitfalls To Avoid:
  - Do not run the reporting helper manually after integration; rely on orchestrator output to avoid duplicate artifacts.
  - Keep log and report paths under the hub; no writes outside `plans/active/.../reports/2025-11-08T050500Z/...`.
  - Ensure metrics summary exists before invoking the helper; guard the command with actionable error messaging.
  - Maintain AUTHORITATIVE_CMDS_DOC export for every pytest/CLI invocation.
  - Watch for long-running Phase E/F jobs; capture partial logs if aborting due to failure.
  - Avoid hard-coded absolute paths in tests—use `Path` utilities for deterministic normalization.
  - Do not skip guard selectors even if pipeline succeeds; DATA-001 validation is mandatory.
  - Preserve prior hubs; never delete or overwrite earlier timestamps.

If Blocked:
  - Tee failing pytest/CLI output to `analysis/blocker.log`, note the blocker in `summary/summary.md`, and document the issue plus exit status in docs/fix_plan.md and galph_memory.md before ending the loop.

Findings Applied (Mandatory):
  - POLICY-001 — Dense pipeline invokes PyTorch baseline tooling; keep torch import failures visible (docs/findings.md:8).
  - CONFIG-001 — Maintain update_legacy_dict ordering inside orchestrator helpers (docs/findings.md:10).
  - DATA-001 — Phase C metadata validation guarantees NPZ contract conformance (docs/findings.md:14).
  - TYPE-PATH-001 — Normalize helper/log paths using Path objects throughout (docs/findings.md:21).
  - OVERSAMPLING-001 — Dense overlap parameters stay fixed; scrutinize deltas that violate expectations (docs/findings.md:17).

Pointers:
  - docs/findings.md:8
  - docs/findings.md:10
  - docs/findings.md:14
  - docs/findings.md:17
  - docs/findings.md:21
  - docs/TESTING_GUIDE.md:284
  - docs/development/TEST_SUITE_INDEX.md:63
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T050500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:531
  - tests/study/test_phase_g_dense_orchestrator.py:1

Next Up (optional):
  - Execute sparse view pipeline once dense evidence lands.

Doc Sync Plan (Conditional):
  - After GREEN, refresh `docs/TESTING_GUIDE.md` Phase G reporting subsection and `docs/development/TEST_SUITE_INDEX.md` to mention automated aggregate-report generation with new selector evidence; archive updated excerpts under this hub.
