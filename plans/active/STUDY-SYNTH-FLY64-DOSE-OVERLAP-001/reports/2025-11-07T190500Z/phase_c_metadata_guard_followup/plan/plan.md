# Phase C Metadata Guard Follow-up — Plan (2025-11-07T190500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense evidence)  
**Action Type:** Planning (supervisor loop)  
**State Target:** ready_for_implementation

---

## Context

- The metadata guard (`validate_phase_c_metadata`) now prevents Phase G from executing when Phase C NPZ outputs omit the `_metadata` blob. The guard shipped with RED→GREEN pytest evidence and full-suite coverage (attempt 2025-11-07T170500Z+exec).  
- Phase C canonicalization tools (`transpose_rename_convert`, `generate_patches`) were refactored to preserve metadata and now record transformation history. However, the guard only checks whether `_metadata` exists; it does **not** confirm that the canonicalization transformations ran.  
- We need stronger guarantees before re-running the multi-hour dense orchestrator jobs: (1) enforce that each Phase C split records a `transpose_rename_convert` transformation, and (2) capture a real orchestrator CLI run proving the guard passes with current datasets.  
- Test registry updates for the new guard selector are still outstanding; we must incorporate them once the enhanced guard ships.

## Objectives

1. Extend `validate_phase_c_metadata` to require that each Phase C NPZ has a `data_transformations` entry containing `transpose_rename_convert`. Raise an actionable `RuntimeError` if the record is missing.  
2. Strengthen pytest coverage: add a RED test where metadata exists but lacks the required transformation (expect guard failure) and a GREEN path that includes both `_metadata` and `transpose_rename_convert`.  
3. Run the dense Phase G orchestrator (`--dose 1000 --view dense --splits train test`) to capture CLI evidence that the enhanced guard passes against freshly generated Phase C outputs.  
4. Update documentation artifacts (`summary.md`, `docs/fix_plan.md`, testing registry entries) with references to the new selectors and CLI log.

## Deliverables

- Updated guard implementation in `plans/active/.../bin/run_phase_g_dense.py` enforcing transformation history.  
- Expanded pytest coverage with RED/GREEN logs and `--collect-only` proof stored under this loop’s artifacts.  
- CLI transcript demonstrating a successful dense orchestrator run with the guard enabled (or blocker log if it fails).  
- Refreshed `summary.md`, `docs/fix_plan.md`, and (after GREEN) `docs/TESTING_GUIDE.md` + `docs/development/TEST_SUITE_INDEX.md` entries documenting the guard selectors.

## Tasks

1. **TDD — RED:**  
   - Add a helper in `tests/study/test_phase_g_dense_orchestrator.py` to fabricate Phase C NPZs with metadata but **without** `transpose_rename_convert`.  
   - Add `test_validate_phase_c_metadata_requires_canonical_transform` expecting `RuntimeError` mentioning `transpose_rename_convert`.  
   - Capture the failure log at `reports/.../red/pytest_guard_transform_red.log`.
2. **Implementation:**  
   - Update `validate_phase_c_metadata` to parse `metadata["data_transformations"]`, confirm it is a list, and assert that it contains an entry (string or dict name) equal to `transpose_rename_convert`.  
   - Ensure error messages cite both `_metadata` and the missing transformation for stable pytest matching.  
   - Add a small helper factory inside the test to emit a positive metadata sample (with transformation record) for reuse.  
3. **TDD — GREEN & Collection:**  
   - Extend the test suite with a positive-path assertion (e.g., `test_validate_phase_c_metadata_accepts_valid_metadata`) that fabricates metadata with the required record and confirms no exception.  
   - Re-run targeted selectors (`test_validate_phase_c_metadata_requires_metadata`, new transformation test, positive-path test) capturing GREEN logs plus `pytest --collect-only` output under `reports/.../green/` and `collect/`.  
4. **CLI Evidence:**  
   - Execute `AUTHORITATIVE_CMDS_DOC=... python plans/.../bin/run_phase_g_dense.py --hub <this-loop-hub> --dose 1000 --view dense --splits train test`.  
   - Tee stdout/stderr to `cli/phase_g_dense_guard.log`; on failure, write traceback to `analysis/blocker.log` and mark attempt blocked.  
5. **Documentation & Ledger:**  
   - Summarize outcomes in `summary/summary.md` with Turn Summary block (also copied to CLI reply).  
   - Update `docs/fix_plan.md` Attempts History and `galph_memory.md`.  
   - After GREEN and CLI success, update `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` with the new guard selector; archive `pytest --collect-only` log as proof.

## Guardrails & Findings Applied

- POLICY-001 — Guard must remain backend-neutral.  
- DATA-001 — Do not mutate NPZ payloads during validation.  
- CONFIG-001 — Guard remains read-only; no params.cfg mutations.  
- TYPE-PATH-001 — Keep all path operations via `Path.resolve()`.  
- OVERSAMPLING-001 — Guard changes must not alter dense/sparse grouping parameters.

## Pitfalls

- Do not rely on pickle-unsafe loads in tests; use `MetadataManager.save_with_metadata`.  
- Ensure tests clean up temporary files; avoid writing outside `tmp_path`.  
- Keep RuntimeError messages deterministic for pytest matching.  
- CLI run may take time; monitor for guard failures before Phase D/E steps launch.  
- Update registries only after selectors are proven to collect (>0 tests) post-change.
