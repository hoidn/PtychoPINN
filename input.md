Summary: Land the Phase C metadata guard with TDD and prove it via targeted pytest plus a fresh dense orchestrator run.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T170500Z/phase_c_metadata_guard/

Do Now (hard validity contract):
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata — add failing pytest case that fabricates Phase C NPZ outputs without `_metadata` and asserts `RuntimeError` mentioning `_metadata`.
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::validate_phase_c_metadata — introduce guard helper that normalizes hub paths, ensures both splits exist, loads metadata via `MetadataManager.load_with_metadata`, and checks for `transpose_rename_convert`.
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — invoke the guard immediately after Phase C completes (skip when `--collect-only`), logging success/failure.
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T170500Z/phase_c_metadata_guard --dose 1000 --view dense --splits train test

How-To Map:
  1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  2. mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T170500Z/phase_c_metadata_guard/{red,green,collect,cli,analysis}
  3. Run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv` and tee output to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T170500Z/phase_c_metadata_guard/red/pytest_guard_red.log` to capture the expected failure signature.
  4. Implement `validate_phase_c_metadata` helper plus `main()` hook in the orchestrator script, ensuring TYPE-PATH-001 normalization and actionable RuntimeError text.
  5. Re-run the pytest selector for GREEN, teeing stdout to `.../green/pytest_guard_green.log`, and collect proof via `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > .../collect/pytest_phase_g_orchestrator_collect.log`.
  6. Execute the dense orchestrator CLI (`python plans/.../bin/run_phase_g_dense.py --hub .../phase_c_metadata_guard --dose 1000 --view dense --splits train test`) with env guard exported; tee stdout/stderr to `.../cli/phase_g_orchestrator.log`.
  7. If the CLI run fails, capture the guard traceback in `analysis/blocker.log`; otherwise archive any generated metrics summaries under `analysis/`.
  8. Update `summary/summary.md`, docs/fix_plan.md (Attempt entry), and galph_memory.md with outcomes and artifacts; append new durable findings to docs/findings.md only if we codify a metadata policy.

Pitfalls To Avoid:
  - Guard must be skipped when `--collect-only` to keep dry runs fast.
  - Do not mutate or delete Phase C outputs; guard should only read metadata.
  - Ensure RuntimeError message contains `_metadata` so pytest match stays stable.
  - Log guard success/failure to stdout for CLI traceability.
  - Keep path handling compliant with TYPE-PATH-001; rely on `Path.resolve()`.
  - Avoid redundant large NPZ loads; only inspect metadata headers.
  - Guard must allow both TensorFlow and PyTorch downstream paths (POLICY-001).
  - Preserve DATA-001 array layout; no reshaping in tests or guard.
  - Record pytest + CLI logs under the artifacts hub; no repo-root debris.
  - Confirm mapped selectors still collect ≥1 test after edits; fix immediately if not.

If Blocked:
  - Capture failure signature in `analysis/blocker.log` with offending NPZ path, keep CLI log, and document the block in docs/fix_plan.md plus galph_memory.md before pausing.

Findings Applied (Mandatory):
  - POLICY-001 — Guard must work alongside TensorFlow default backend while respecting PyTorch requirement.
  - CONFIG-001 — No bypass of `update_legacy_dict`; guard is read-only metadata verification.
  - DATA-001 — Guard enforces canonical NPZ contract without altering content.
  - OVERSAMPLING-001 — No change to dense overlap parameters during guard work.
  - TYPE-PATH-001 — Normalize filesystem paths before IO to avoid string-based errors.

Pointers:
  - docs/findings.md:8 (POLICY-001 dependency policy)
  - docs/findings.md:10 (CONFIG-001 sequencing)
  - docs/findings.md:14 (DATA-001 data format contract)
  - docs/findings.md:17 (OVERSAMPLING-001 gridsize constraints)
  - docs/findings.md:21 (TYPE-PATH-001 path normalization guardrail)
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:340 (Phase C command wiring)
  - tests/study/test_phase_g_dense_orchestrator.py:1 (existing orchestrator test harness)
  - ptycho/metadata.py:26 (MetadataManager API for metadata loading)

Next Up (optional):
  - Harden orchestrator summary generation once metadata guard ships.

Doc Sync Plan (Conditional):
  - After GREEN, run `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T170500Z/phase_c_metadata_guard/collect/pytest_phase_g_orchestrator_collect.log` and update docs/TESTING_GUIDE.md §2 plus docs/development/TEST_SUITE_INDEX.md with the new guard selector.
