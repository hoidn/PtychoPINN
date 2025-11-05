Summary: Execute dense Phase C→G pipeline with aggregates and ship a reporting helper that surfaces per-model deltas for Turn Summary authoring.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv
  - pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run/

Do Now (hard validity contract):
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py::main — add a reporting helper that reads `metrics_summary.json`, emits aggregate tables, and computes per-model deltas (PtychoPINN vs Baseline/PtyChi) in stdout/Markdown with deterministic ordering.
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run/cli/phase_g_dense_pipeline.log
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): python - <<'PY' | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run/analysis/validate_and_summarize.log
from pathlib import Path
import importlib.util

script_path = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py")
spec = importlib.util.spec_from_file_location("run_phase_g_dense", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

hub = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run")
module.validate_phase_c_metadata(hub)
module.summarize_phase_g_outputs(hub)
PY
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run/analysis/metrics_summary.json --output plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run/analysis/aggregate_report.md | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run/analysis/aggregate_report.log

How-To Map:
  1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  2. mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run/{analysis,cli,collect,docs,green,red,summary}
  3. Following TDD, draft `tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics` using fixture `metrics_summary.json` content (PtychoPINN/Baseline/PtyChi aggregates) to encode expected tables + deltas; run once to capture RED log under `red/pytest_report_helper_red.log`.
  4. Implement `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py::main` with argparse, deterministic ordering, 3-decimal formatting, stdout + optional Markdown emission, and non-zero exit when required models missing.
  5. Run `pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run/green/pytest_report_helper_green.log`
  6. Re-run guard selectors to ensure regressions absent: 
     `pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv | tee .../green/pytest_summarize_green.log`
     and
     `pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv | tee .../green/pytest_metadata_guard_green.log`
  7. Capture selector inventory with `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > .../collect/pytest_phase_g_orchestrator_collect.log`
  8. Execute dense pipeline with clobber as in Do Now step; ensure log saved to `cli/phase_g_dense_pipeline.log`. Abort on failure and record blocker if needed.
  9. Run validation shim (Do Now snippet) to confirm guard + summarizer succeed; stdout tee already configured.
 10. Invoke reporting helper CLI to produce Markdown summary and archive stdout (Do Now command).
 11. Update `summary/summary.md` with Turn Summary + aggregate highlights, include links to CLI/analysis logs and Markdown report.
 12. Sync docs (`docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`, `docs/fix_plan.md`) and note new selector/test in registry once everything is GREEN.

Pitfalls To Avoid:
  - Dense pipeline can exceed 20 minutes; monitor progress and capture partial logs if interrupted.
  - Reporting helper must not mutate JSON; treat files read-only and write outputs under the hub.
  - Keep float formatting consistent: raw floats in JSON, 3-decimal strings in Markdown/stdout.
  - Ensure models `PtychoPINN`, `Baseline`, and `PtyChi` are present; fail loudly if deltas cannot be computed.
  - Do not reuse or delete prior hubs; rely on `prepare_hub` and timestamped directories.
  - Maintain AUTHORITATIVE_CMDS_DOC export for every pytest/CLI invocation.
  - Avoid hard-coded absolute paths in new tests; use `tmp_path` and `Path` utilities.
  - Capture RED evidence for new test before implementation; include log path in summary.
  - Skip doc/test registry updates only if loop blocked; otherwise document rationale in Attempts History.

If Blocked:
  - Stop immediately, tee failing pytest/CLI output to `analysis/blocker.log`, summarize the blocker in `summary/summary.md`, and document the issue in docs/fix_plan.md plus galph_memory.md before ending the loop.

Findings Applied (Mandatory):
  - POLICY-001 — Dense pipeline invokes PyTorch baseline tooling; commands preserve torch>=2.2 requirement.
  - CONFIG-001 — Orchestrator guard still bridges params.cfg before model work; reporting helper stays read-only.
  - DATA-001 — Phase C metadata validation remains part of post-run checks to ensure NPZ contracts hold.
  - TYPE-PATH-001 — All helper/test paths normalized via `Path`; new script should resolve inputs/outputs.
  - OVERSAMPLING-001 — Dense overlap parameters remain fixed; reporting helper should flag unexpected deltas.

Pointers:
  - docs/findings.md:8
  - docs/findings.md:10
  - docs/findings.md:14
  - docs/findings.md:21
  - docs/findings.md:17
  - docs/TESTING_GUIDE.md:215
  - docs/development/TEST_SUITE_INDEX.md:62
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T030500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:44
  - tests/study/test_phase_g_dense_orchestrator.py:100

Next Up (optional):
  - Mirror reporting helper for sparse view once dense evidence and docs land.

Doc Sync Plan (Conditional):
  - After GREEN, rerun `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv` (captured in step 7), archive log, and update `docs/TESTING_GUIDE.md` §2 plus `docs/development/TEST_SUITE_INDEX.md` to mention the new reporting helper test/selector with evidence links.
