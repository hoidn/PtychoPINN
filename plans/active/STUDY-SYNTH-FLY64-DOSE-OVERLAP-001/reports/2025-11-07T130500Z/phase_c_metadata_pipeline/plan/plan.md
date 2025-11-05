# Phase C Metadata Pipeline Hardening — Plan (2025-11-07T130500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense evidence) → unblock Phase C canonicalization  
**Loop Type:** Planning → Implementation (TDD)

---

## Objectives

1. Reproduce the Phase C failure triggered by `_metadata` object arrays during canonicalization (ValueError from `np.load(..., allow_pickle=False)`).
2. Add regression tests covering metadata-preserving workflows for `transpose_rename_convert` and `generate_patches` so metadata-aware saving/loading is contractually enforced.
3. Update canonicalization and patch-generation tools to load NPZ inputs via `MetadataManager`, drop/transform `_metadata` safely, and re-save outputs with updated transformation records while keeping array dtypes intact.
4. Re-run the dense dose=1000 orchestrator pipeline to confirm Phase C completes and the pipeline progresses past the previous blocker; capture fresh CLI logs + summary notes.

---

## Deliverables

- New pytest coverage (likely under `tests/tools/`) validating:
  - `transpose_rename_convert` handles NPZ files that include `_metadata` and preserves JSON metadata with a canonicalization transformation record.
  - `generate_patches` loads metadata-bearing files and writes patches while propagating metadata (recording `tool=generate_patches_tool.py`).
- Updated implementations:
  - `scripts/tools/transpose_rename_convert_tool.py` — metadata-aware loading/saving + TYPE-PATH-001 compliance.
  - `scripts/tools/generate_patches_tool.py` — metadata-aware loading/saving + transformation update.
- Dense orchestrator hub populated at `reports/2025-11-07T130500Z/phase_c_metadata_pipeline/` with:
  - `plan/plan.md` (this file)
  - `summary/summary.md` capturing blocker reproduction + fix evidence
  - `cli/phase_c_generation.log` (new run) demonstrating Phase C success
  - `analysis/metrics_todo.md` noting pending Phase D→G reruns if pipeline is halted later
- `docs/fix_plan.md` Attempt entry describing the metadata regression and the TDD remediation path.

---

## Task Breakdown

1. **TDD Setup**
   - Craft minimal NPZ fixture with metadata via `MetadataManager.save_with_metadata` (diffraction/objectGuess/etc.).
   - Author failing test(s) asserting canonicalization/patche generation crash on current code (expect `ValueError: Object arrays cannot be loaded when allow_pickle=False`). Capture RED log.

2. **Implementation**
   - Refactor `transpose_rename_convert` to:
     - Use `MetadataManager.load_with_metadata` for ingestion (allowing `_metadata`).
     - Skip `_metadata` during array iteration, but preserve metadata when saving via `MetadataManager.save_with_metadata` (add transformation record `operation='canonicalize_dataset'`).
   - Refactor `generate_patches` to:
     - Load via `MetadataManager.load_with_metadata`.
     - Update metadata with transformation `operation='generate_patches'` and persist via `MetadataManager.save_with_metadata`.
     - Ensure array iteration ignores `_metadata` and enforces DATA-001 keys.

3. **Validation**
   - Re-run new pytest selectors (GREEN) + collect-only proof.
   - Execute orchestrator command:
     ```bash
     export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
     python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
         --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T130500Z/phase_c_metadata_pipeline \
         --dose 1000 \
         --view dense \
         --splits train test
     ```
   - Archive CLI log(s); if later phases still block, document blocker in `analysis/blocker.log` and summary.

4. **Documentation + Ledger Updates**
   - Update `summary/summary.md` with RED/observed failure, GREEN fix evidence, and pipeline status.
   - Amend `docs/fix_plan.md` Attempt history; append new lesson to `docs/findings.md` if metadata handling emerges as new pattern.
   - Ensure `docs/TESTING_GUIDE.md` + `docs/development/TEST_SUITE_INDEX.md` reference any new selectors (after GREEN).

---

## Findings / Policies To Honor

- DATA-001 — Maintain canonical array keys/dtypes while adding metadata support.
- CONFIG-001 — Ensure tooling continues to respect legacy bridge order (no params.cfg side effects).
- TYPE-PATH-001 — All new paths must be normalized with `Path`.
- OVERSAMPLING-001 — Do not alter dense/sparse overlap thresholds while debugging.
- POLICY-001 — Preserve PyTorch dependency expectations (even though Phase C is TF-specific).

---

## Pitfalls / Guardrails

- Avoid dropping metadata entirely; instead, propagate it with transformation records for auditability.
- Keep tests lightweight (small arrays) to avoid long runtimes.
- Do not mutate core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Ensure orchestrator hub isolation—no writes back to previous timestamp directories.
- If orchestrator still fails, halt after logging blocker; do not delete intermediate files.
