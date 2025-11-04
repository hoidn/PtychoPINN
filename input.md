Summary: Capture dense Phase G evidence by extending the orchestrator to emit a metrics summary and running the full Phase C→G pipeline (dose=1000, train/test).
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T110500Z/phase_g_dense_execution/

Do Now (hard validity contract):
  - Implement: tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs — author failing test covering orchestrator summary helper expectations (summary files + metric extraction).
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::summarize_phase_g_outputs — add Path-safe manifest validation + metrics summary emission (`analysis/metrics_summary.json`/`.md`) and invoke from `main()` after successful pipeline execution.
  - Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T110500Z/phase_g_dense_execution --dose 1000 --view dense --splits train test

How-To Map:
  1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  2. Create `tests/study/test_phase_g_dense_orchestrator.py` (or extend existing file) with `test_summarize_phase_g_outputs` that builds a temp hub containing `analysis/comparison_manifest.json`, per-job `comparison_metrics.csv`, and asserts that `summarize_phase_g_outputs()` writes JSON/Markdown summaries with MS-SSIM + MAE entries for each model/split. Run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv` expecting an ImportError/AssertionError (RED) and archive to `red/pytest_red.log`.
  3. Update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py`: add `summarize_phase_g_outputs(hub: Path)` that (a) loads `analysis/comparison_manifest.json`, (b) raises `RuntimeError` if `n_failed > 0` or metrics CSV missing, (c) parses each CSV via `csv.DictReader` into `{model: metric}` dictionaries (handle amplitude/phase/value columns), and (d) writes deterministic JSON + Markdown tables. Call helper from `main()` after all commands succeed. Ensure all new filesystem paths go through `Path(...)`.
  4. Rerun `pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv` (GREEN) and store log at `green/pytest_green.log`.
  5. Execute the dense pipeline: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T110500Z/phase_g_dense_execution --dose 1000 --view dense --splits train test`. Confirm `cli/` logs, `analysis/comparison_manifest.json`, `analysis/metrics_summary.json`, `analysis/metrics_summary.md`, and `data/phase_{c,d,e,f}/` outputs populate the hub.
  6. If the run succeeds, update `summary/summary.md` with key MS-SSIM/MAE numbers. If it fails, capture traceback in `analysis/blocker.log` and halt.

Pitfalls To Avoid:
  - Do not modify core stable modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
  - Keep all new filesystem interactions Path-based (TYPE-PATH-001) and inside the hub.
  - Fail fast when manifest reports `n_failed > 0`; do not silently continue.
  - Ensure the summary helper handles both amplitude/phase and scalar metrics; avoid dropping rows.
  - Leave orchestrator collect-only behavior unchanged.
  - Avoid reusing previous hubs; this loop must write only to `2025-11-07T110500Z/phase_g_dense_execution/`.
  - Do not delete existing artifacts from earlier attempts.

If Blocked:
  - Record failure signature in `analysis/blocker.log`, keep relevant CLI log under `cli/`, update `summary/summary.md` with the blocker context, and notify supervisor via docs/fix_plan.md + galph_memory.md entry.

Findings Applied (Mandatory):
  - POLICY-001 — PyTorch remains installed; TensorFlow backend still drives training CLI.
  - CONFIG-001 — Legacy bridge order untouched; orchestrator summary must not mutate params.cfg.
  - DATA-001 — Verify generated datasets remain contract compliant when summarizing metrics.
  - OVERSAMPLING-001 — Dense view spacing thresholds stay at f_overlap=0.7; no parameter tweaks.
  - TYPE-PATH-001 — Normalize every new path (manifests, CSVs, output summaries) via `Path`.

Pointers:
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T110500Z/phase_g_dense_execution/plan/plan.md
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix/summary.md
  - docs/TESTING_GUIDE.md#phase-g-study-selectors
  - docs/development/TEST_SUITE_INDEX.md

Next Up (optional): Run sparse view pipeline once dense evidence + summaries are green.

Doc Sync Plan (tests updated this loop):
  - pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > collect/pytest_collect.log
  - Update docs/TESTING_GUIDE.md §2 (Phase G study selectors) to include new selector.
  - Update docs/development/TEST_SUITE_INDEX.md to list `tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs`.

Mapped Tests Guardrail: `tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs` collects exactly one test; ensure selector stays active.

Hard Gate: Do not mark attempt complete unless pipeline finishes without blocker **and** `analysis/metrics_summary.json`/`.md` reflect both train/test splits; otherwise treat as blocked and escalate.
