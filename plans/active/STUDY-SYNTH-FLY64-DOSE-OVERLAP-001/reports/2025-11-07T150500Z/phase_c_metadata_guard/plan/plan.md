# Phase C Metadata Guard — Plan (2025-11-07T150500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense evidence)

---

## Objectives

1. Add an explicit validation step after Phase C in the dense orchestrator to ensure metadata survives canonicalization (guards against `_metadata` regressions detected this morning).
2. Exercise the new guard via targeted TDD: require metadata on both train/test NPZ outputs, and capture failure signature when metadata is missing.
3. Re-run the dense pipeline after the guard lands to confirm Phase C passes validation and the rest of the pipeline proceeds.
4. Update initiative ledger (`docs/fix_plan.md`) and artifacts with guard evidence, including RED→GREEN pytest logs and orchestrator transcript.

## Deliverables

- New pytest coverage in `tests/study/test_phase_g_dense_orchestrator.py` covering `validate_phase_c_metadata` guard (RED/ GREEN logs + collect-only proof).
- Updated `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py` featuring:
  - `validate_phase_c_metadata()` helper invoked after Phase C command succeeds.
  - Guard requirements: each split NPZ must exist, load via `MetadataManager.load_with_metadata`, require metadata present plus at least one transformation entry named `transpose_rename_convert`.
  - Clear RuntimeError message when guard fails (captured in RED test).
- Dense pipeline log under this hub (`cli/phase_g_orchestrator.log`) showing guard success with metadata preserved.
- Updated summary noting guard rationale, RED message, GREEN evidence, and pipeline status.

## Tasks

1. **TDD – RED**
   - Extend `tests/study/test_phase_g_dense_orchestrator.py` with parametrized case that stubs Phase C outputs without metadata (use tmp_path fixtures) and asserts `validate_phase_c_metadata` raises RuntimeError mentioning `_metadata`.
   - Capture RED run: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv` (expect failure before implementation).

2. **Implementation**
   - Implement `validate_phase_c_metadata` helper in orchestrator script using `MetadataManager.load_with_metadata`.
   - Call guard immediately after Phase C command completes (before Phase D command is enqueued) so later stages never see invalid datasets.
   - Ensure guard writes helpful log lines (print validations) and raises RuntimeError with actionable message on failure.

3. **Validation – GREEN**
   - Re-run pytest selector (GREEN) and collect-only proof.
   - Execute orchestrator with dense configuration using new guard; log outputs to hub; confirm guard passes (metadata found, transformations >=1).

4. **Ledger & Docs**
   - Summarize guard addition + evidence in `summary/summary.md`.
   - Update `docs/fix_plan.md` attempts with metadata guard entry and reference this timestamp.
   - Note guard in `docs/findings.md` only if new durable lesson emerges.

## Findings To Honor
- CONFIG-001 — Ensure no params.cfg mutations happen inside guard.
- DATA-001 — Continue enforcing canonical array layout (metadata check must not mutate arrays).
- TYPE-PATH-001 — Normalize all filesystem paths via `Path`.
- POLICY-001 — Guard must work for both TensorFlow (Phase C) and downstream PyTorch phases.
- TYPE-PATH-001 — maintain path normalization when loading NPZs.

## Pitfalls
- Do NOT delete Phase C outputs when guard fails; emit blocker log for reruns.
- Avoid loading full diffraction stacks multiple times; guard should only inspect metadata.
- Keep RuntimeError message stable for pytest assertion.
- Ensure orchestrator `--collect-only` still works (guard should be skipped in dry mode).

