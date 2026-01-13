# Implementation Plan (Draft) — Batched Object Reassembly

## Initiative
- ID: FIX-OBJECT-REASSEMBLY-BATCH-001
- Title: Batched object reassembly for stitching (compare_models + shared helpers)
- Owner: Codex
- Spec Owner: specs/compare_models_spec.md
- Status: pending

## Goals
- Add a batched object-reassembly path that preserves `--stitch-crop-size` semantics.
- Eliminate OOMs during large-scale stitching in `scripts/compare_models.py` while keeping outputs consistent.

## Phases Overview
- Phase A — Discovery & test strategy: confirm call paths, define API and test plan.
- Phase B — Implementation: add batched object reassembly and wire into compare_models.
- Phase C — Verification: tests + regression validation on a representative dataset.

## Exit Criteria
1. Object reassembly supports `batch_size` and preserves `M` crop semantics.
2. `scripts/compare_models.py` uses the batched object reassembly path for stitching when enabled.
3. Targeted pytest selector(s) pass and capture equivalence for small inputs.
4. Test registry synchronized: `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` reflect any new/changed tests; `pytest --collect-only` logs for documented selectors are saved under `plans/active/FIX-OBJECT-REASSEMBLY-BATCH-001/reports/<timestamp>/`. Do not close the initiative if any selector marked "Active" collects 0 tests.

## Compliance Matrix (Mandatory)
- [ ] **Spec Constraint:** `specs/compare_models_spec.md` — stitching uses `--stitch-crop-size` and enforces shape checks.
- [ ] **Fix-Plan Link:** `docs/fix_plan.md — Row [FIX-OBJECT-REASSEMBLY-BATCH-001]`.
- [ ] **Finding/Policy ID:** `REASSEMBLY-BATCH-001`, `XLA-VECTORIZE-001`, `BUG-TF-REASSEMBLE-001`.

## Spec Alignment
- **Normative Spec:** `specs/compare_models_spec.md`
- **Key Clauses:** Stitching uses `--stitch-crop-size`, chunking/batching knobs, error handling.

## Context Priming (read before edits)
- Primary docs/specs to re-read: `docs/index.md`, `specs/compare_models_spec.md`, `docs/architecture_inference.md`, `docs/TESTING_GUIDE.md`, `docs/debugging/QUICK_REFERENCE_PARAMS.md`
- Required findings/case law: `REASSEMBLY-BATCH-001`, `XLA-VECTORIZE-001`, `BUG-TF-REASSEMBLE-001`
- Related telemetry/attempts: `gs2_generalization_study_20260112_162025/study_run.log` (OOM in compare_models stitching)
- Data dependencies to verify: `datasets/fly64/fly64_top_half_shuffled.npz` (large test set)

## Phase A — Discovery & Test Strategy
### Checklist
- [ ] A0: **Nucleus / Test-first gate:** capture the compare_models OOM traceback and confirm call path.
- [ ] A1: Enumerate object reassembly call sites (`reassemble_position` and consumers).
- [ ] A2: Define batched object reassembly API (new helper or `reassemble_position(..., batch_size=...)`) that preserves `M`.
- [ ] A3: Draft test strategy via `plans/templates/test_strategy_template.md` and link it in `docs/fix_plan.md`.

### Notes & Risks
- Preserve `--stitch-crop-size` semantics; avoid silent shape changes.
- Avoid known XLA vectorized translate path failures on large patch counts.

## Phase B — Implementation
### Checklist
- [ ] B1: Implement batched object reassembly in `ptycho/tf_helper.py` with explicit `batch_size` and `M` crop handling.
- [ ] B2: Update `reassemble_position` to route to the batched path when `batch_size` is set or patch count exceeds a threshold.
- [ ] B3: Update `scripts/compare_models.py` to pass a batch size for object reassembly (new CLI flag or reuse existing chunk size).
- [ ] B4: Validate output parity on small inputs (numerical tolerance).

### Notes & Risks
- Keep dtype expectations intact (complex64 patches, float64 offsets).

## Phase C — Verification
### Checklist
- [ ] C1: Add a pytest regression comparing batched vs non-batched object reassembly on a small synthetic case.
- [ ] C2: Run a scaled comparison (or reduced test) to verify no OOM and that `comparison_metrics.csv` is produced.
- [ ] C3: Update test registries + collect-only logs if new tests are added.

### Notes & Risks
- CI/GPU constraints: keep tests small and deterministic.

## Artifacts Index
- Reports root: `plans/active/FIX-OBJECT-REASSEMBLY-BATCH-001/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`
