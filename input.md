Summary: Enforce the Phase C metadata guard to require canonical transformation history and prove it with a dense orchestrator run.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_canonical_transform -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T190500Z/phase_c_metadata_guard_followup/

Do Now (hard validity contract):
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_canonical_transform — add failing pytest case that fabricates Phase C NPZ outputs with `_metadata` but missing `transpose_rename_convert`, expecting `RuntimeError` mentioning the missing transformation.
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata — add positive-path test using `MetadataManager.save_with_metadata` to embed a transformation record and assert the guard passes.
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::validate_phase_c_metadata — require `transpose_rename_convert` in `metadata["data_transformations"]` for each split; raise a RuntimeError referencing `_metadata` and the missing transformation when absent.
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_canonical_transform -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T190500Z/phase_c_metadata_guard_followup --dose 1000 --view dense --splits train test

How-To Map:
  1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  2. mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T190500Z/phase_c_metadata_guard_followup/{red,green,collect,cli,analysis}
  3. Run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_canonical_transform -vv` and tee output to `.../red/pytest_guard_transform_red.log` to capture the expected failure.
  4. Extend `tests/study/test_phase_g_dense_orchestrator.py` with helper(s) that write NPZ files via `MetadataManager.save_with_metadata`, add the new RED test, add the positive-path test, and update fixtures as needed.
  5. Update `validate_phase_c_metadata` in the orchestrator script to enforce `transpose_rename_convert` in `metadata["data_transformations"]` (list membership, case-sensitive) and emit actionable RuntimeError text.
  6. Re-run the three mapped pytest selectors, capturing GREEN logs under `.../green/` and `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > .../collect/pytest_phase_g_guard_collect.log`.
  7. Execute the dense orchestrator CLI (`python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub .../phase_c_metadata_guard_followup --dose 1000 --view dense --splits train test`) while teeing output to `.../cli/phase_g_dense_guard.log`; if it fails, copy the traceback into `analysis/blocker.log`.
  8. On success, archive any metrics/manifests produced into the hub’s `analysis/` directory; if blocked, leave the pipeline outputs intact for inspection.
  9. Update `summary/summary.md`, `docs/fix_plan.md` Attempts History, `docs/TESTING_GUIDE.md` §2, `docs/development/TEST_SUITE_INDEX.md`, and `galph_memory.md` with outcomes and selector references.

Pitfalls To Avoid:
  - Do not load NPZ fixtures with `allow_pickle=False`; rely on `MetadataManager.save_with_metadata`.
  - Guard must stay read-only; never mutate or delete Phase C files during checks.
  - RuntimeError text should include both `_metadata` and `transpose_rename_convert` for stable pytest matching.
  - Ensure the new tests clean up tmp artifacts and avoid writing outside `tmp_path`.
  - Keep guard logs backend-neutral (POLICY-001) and avoid TensorFlow-specific assumptions.
  - Preserve TYPE-PATH-001 normalization (`Path.resolve()`) when touching filesystem paths.
  - Prevent redundant large array loads; only inspect metadata headers.
  - Confirm mapped selectors still collect ≥1 test; repair immediately if collection drops.
  - CLI run should target the new hub only; do not reuse prior hubs to avoid overwriting evidence.
  - If CLI fails, capture `analysis/blocker.log` before retrying and do not rerun blindly.

If Blocked:
  - Stop, record the failure signature (pytest or CLI) in `analysis/blocker.log`, keep logs in place, and document the block in `summary/summary.md`, `docs/fix_plan.md`, and `galph_memory.md` before ending the loop.

Findings Applied (Mandatory):
  - POLICY-001 — Guard and tests must remain backend-neutral while honoring mandatory PyTorch dependency.
  - CONFIG-001 — Guard must not bypass the params.cfg bridge; read-only metadata validation only.
  - DATA-001 — NPZ contract (including `_metadata`) enforced without mutating payloads.
  - TYPE-PATH-001 — All filesystem interactions use `Path.resolve()` to avoid string path bugs.
  - OVERSAMPLING-001 — Ensure dense view parameters remain unchanged when running the pipeline evidence.

Pointers:
  - docs/findings.md:8 (POLICY-001 PyTorch requirement context)
  - docs/findings.md:10 (CONFIG-001 legacy bridge sequencing)
  - docs/findings.md:14 (DATA-001 NPZ contract expectations)
  - docs/findings.md:21 (TYPE-PATH-001 path normalization guardrail)
  - specs/data_contracts.md:215 (`_metadata` contract details)
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:137 (guard implementation site)
  - tests/study/test_phase_g_dense_orchestrator.py:200 (existing orchestrator guard tests scaffold)

Next Up (optional):
  - After guard passes with dense evidence, extend CLI summary generation checks for sparse view coverage.

Doc Sync Plan (Conditional):
  - After GREEN, run `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T190500Z/phase_c_metadata_guard_followup/collect/pytest_phase_g_guard_collect.log` and update `docs/TESTING_GUIDE.md` §2 plus `docs/development/TEST_SUITE_INDEX.md` with the new guard selectors.
