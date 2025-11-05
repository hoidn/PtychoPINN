Summary: Rerun the Phase C→G dense pipeline on a clean hub by adding a `prepare_hub` helper with `--clobber`, covering it with tests, and capturing full CLI evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_detects_stale_outputs -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_clobbers_previous_outputs -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_canonical_transform -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T210500Z/phase_g_dense_execution_rerun/

Do Now (hard validity contract):
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_detects_stale_outputs — add failing pytest case that seeds a fake hub containing existing Phase C outputs and asserts `prepare_hub(..., clobber=False)` raises `RuntimeError` with actionable guidance.
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::prepare_hub — create helper that normalizes paths, detects stale hub contents, and either raises or moves/deletes them based on a new `--clobber` flag.
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_clobbers_previous_outputs — add positive-path test proving `prepare_hub(..., clobber=True)` produces a clean hub and archives/deletes prior data.
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_detects_stale_outputs -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_clobbers_previous_outputs -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_canonical_transform -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T210500Z/phase_g_dense_execution_rerun --dose 1000 --view dense --splits train test --clobber

How-To Map:
  1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  2. mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T210500Z/phase_g_dense_execution_rerun/{red,green,collect,cli,analysis}
  3. Run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_detects_stale_outputs -vv` and tee output to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T210500Z/phase_g_dense_execution_rerun/red/pytest_prepare_hub_red.log` to capture the expected RuntimeError.
  4. Extend `tests/study/test_phase_g_dense_orchestrator.py` with helper imports and new tests covering `prepare_hub` failure/cleanup paths; implement `prepare_hub` and `--clobber` flag in `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py`.
  5. Re-run the mapped pytest selectors one by one, capturing GREEN logs under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T210500Z/phase_g_dense_execution_rerun/green/pytest_prepare_hub_green.log`, `pytest_metadata_guard_green.log`, and `pytest_summary_helper_green.log`.
  6. Execute `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T210500Z/phase_g_dense_execution_rerun/collect/pytest_phase_g_orchestrator_collect.log`.
  7. Launch the dense pipeline rerun with `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T210500Z/phase_g_dense_execution_rerun --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T210500Z/phase_g_dense_execution_rerun/cli/phase_g_dense_pipeline.log`.
  8. If the pipeline succeeds, ensure metrics/manifests emitted by the script remain under the hub’s `analysis/` directory; if it fails, capture the traceback in `analysis/blocker.log` and stop.
  9. Update `summary/summary.md`, `docs/fix_plan.md`, `docs/TESTING_GUIDE.md` §2, `docs/development/TEST_SUITE_INDEX.md`, and `galph_memory.md` with outcomes, selector references, and artifact paths.

Pitfalls To Avoid:
  - Do not delete hub contents unless `--clobber` is explicitly passed; default must be read-only.
  - Keep cleanup logic TYPE-PATH-001 compliant (`Path.resolve()` before filesystem ops).
  - Preserve prior evidence by archiving instead of trashing when practical; document retention in RuntimeError text.
  - New tests must use `tmp_path` and avoid touching real study directories.
  - Ensure RuntimeError messages are stable for pytest matching and mention both the stale path and the `--clobber` remedy.
  - Export `AUTHORITATIVE_CMDS_DOC` before running pytest/CLI commands.
  - Monitor pipeline duration; if a long-running phase fails, stop immediately and record the failure signature rather than re-running blindly.
  - Never mutate NPZ payloads during validation; `prepare_hub` must only touch directory scaffolding.
  - Confirm mapped selectors still collect ≥1 test; fix immediately if collection drops.
  - When moving old hubs, avoid crossing filesystem boundaries that would discard metadata permissions.

If Blocked:
  - Stop work, save the failing pytest or CLI output to the hub (`analysis/blocker.log` plus relevant logs), update `summary/summary.md` with the blocker context, and document the block in `docs/fix_plan.md` and `galph_memory.md` before ending the loop.

Findings Applied (Mandatory):
  - POLICY-001 — PyTorch dependency policy: pipeline script must stay backend-neutral while maintaining mandatory torch install context (docs/findings.md:8).
  - CONFIG-001 — Respect legacy bridge sequencing; helper cannot bypass `update_legacy_dict` (docs/findings.md:10).
  - DATA-001 — Enforce NPZ metadata contract without mutating payloads (docs/findings.md:14).
  - TYPE-PATH-001 — Normalize filesystem paths when cleaning/archiving hub contents (docs/findings.md:21).
  - OVERSAMPLING-001 — Maintain dense view overlap parameters during rerun (docs/findings.md:17).

Pointers:
  - docs/findings.md:8
  - docs/findings.md:10
  - docs/findings.md:14
  - docs/findings.md:21
  - specs/data_contracts.md:215
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:120
  - tests/study/test_phase_g_dense_orchestrator.py:200

Next Up (optional):
  - After dense evidence is archived, extend the orchestrator summary to aggregate sparse view metrics for parity.

Doc Sync Plan (Conditional):
  - After GREEN, run `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T210500Z/phase_g_dense_execution_rerun/collect/pytest_phase_g_orchestrator_collect.log` and update `docs/TESTING_GUIDE.md` §2 plus `docs/development/TEST_SUITE_INDEX.md` with the new `prepare_hub` selectors.
