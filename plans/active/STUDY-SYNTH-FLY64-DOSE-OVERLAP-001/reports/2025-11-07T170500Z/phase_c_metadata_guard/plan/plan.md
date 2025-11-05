# Phase C Metadata Guard — Plan (2025-11-07T170500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense evidence)  
**Action Type:** Planning (supervisor loop)  
**State Target:** ready_for_implementation

---

## Context

- Phase C dataset canonicalization recently failed because `_metadata` and transformation history were stripped when `transpose_rename_convert` rewrote arrays. Dense orchestrator can silently continue with metadata-free NPZs, leading to incorrect Phase E manifest telemetry and Phase G metrics.  
- Current orchestrator script (`bin/run_phase_g_dense.py`) chains Phase C → Phase G without inspecting metadata. Tests only cover `summarize_phase_g_outputs`.  
- We must insert a metadata gate immediately after Phase C to fail fast, plus add TDD coverage so regressions surface in CI before long pipeline runs.

## Objectives

1. Author a RED test in `tests/study/test_phase_g_dense_orchestrator.py` that simulates metadata-free Phase C outputs and asserts `validate_phase_c_metadata()` raises a `RuntimeError` mentioning `_metadata`.  
2. Implement `validate_phase_c_metadata()` in the orchestrator script, using `MetadataManager.load_with_metadata` to confirm both train/test NPZs exist, include metadata, and record at least one `transpose_rename_convert` transformation.  
3. Invoke the guard from `main()` immediately after the Phase C command succeeds (skipping when `--collect-only` is set) so later phases never run on bad datasets.  
4. Re-run targeted pytest selectors plus the dense orchestrator CLI to capture RED→GREEN evidence, CLI transcript, and update ledger artifacts.

## Deliverables

- New pytest coverage for the metadata guard (RED and GREEN logs + `--collect-only` proof).  
- Guard helper and main-hook landed in `bin/run_phase_g_dense.py` with TYPE-PATH-001 compliant path handling and actionable error text.  
- Dense orchestrator log file under this hub confirming guard success on a real run; blocker log on failure.  
- Updated `summary/summary.md`, `docs/fix_plan.md`, and `galph_memory.md` documenting guard rationale, evidence, and findings references.

## Tasks

1. **TDD (RED):** Extend orchestrator tests with `_import_validate_phase_c_metadata()` helper and add `test_validate_phase_c_metadata_requires_metadata` that creates dummy NPZ outputs lacking `_metadata`, expecting `RuntimeError` with `_metadata` in the message. Capture RED pytest log at `red/pytest_guard_red.log`.  
2. **Implementation:**  
   - Add `validate_phase_c_metadata()` helper that normalizes hub paths, enumerates expected split NPZs (`train`, `test`), loads each via `MetadataManager.load_with_metadata`, verifies metadata is not `None`, and checks `data_transformations` contains `transpose_rename_convert`.  
   - Call the helper from `main()` after `Phase C` command (skip when `args.collect_only` is true). Emit informative `print()` lines on success/failure.  
3. **Validation (GREEN):** Re-run targeted pytest selector, capture GREEN log at `green/pytest_guard_green.log`, and store `pytest --collect-only` output under `collect/`. Execute the full dense orchestrator CLI (`--hub .../phase_c_metadata_guard --dose 1000 --view dense --splits train test`) teeing output to `cli/phase_g_orchestrator.log`.  
4. **Ledger & Findings:** Summarize outcomes in `summary/summary.md`, update `docs/fix_plan.md` with Attempt entry, append durable lessons to `docs/findings.md` only if a new metadata policy emerges.

## Guardrails & Findings Applied

- POLICY-001 — Keep orchestrator backend-neutral; guard must run in TensorFlow-only environments.  
- CONFIG-001 — Guard must not mutate `params.cfg` or bypass configuration sequencing.  
- DATA-001 — Preserve canonical NPZ layout; guard is read-only.  
- OVERSAMPLING-001 — Do not adjust overlap/grouping parameters.  
- TYPE-PATH-001 — Normalize all filesystem paths with `Path.resolve()` before IO.

## Pitfalls

- Do not delete or overwrite Phase C outputs on guard failure; instead, raise `RuntimeError` and log a blocker.  
- Skip guard when `--collect-only` or when Phase C failed (command failure already halts pipeline).  
- Ensure RuntimeError text is stable for pytest match (include `_metadata`).  
- Avoid double-loading large NPZs—restrict to metadata extraction.  
- Capture logs/artifacts only inside this hub; no repo-root clutter.
