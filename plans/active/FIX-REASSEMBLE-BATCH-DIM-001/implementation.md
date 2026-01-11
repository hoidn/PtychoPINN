# Implementation Plan: Preserve Batch Dimension in Batched Reassembly

## Initiative
- ID: FIX-REASSEMBLE-BATCH-DIM-001
- Title: Preserve batch dimension in batched patch reassembly
- Owner: Codex
- Spec Owner: docs/specs/spec-ptycho-workflow.md (Reassembly Requirements)
- Status: done

## Goals
- Ensure `_reassemble_position_batched` returns one canvas per batch item instead of collapsing the batch.
- Keep `ReassemblePatchesLayer` output consistent with its documented shape.
- Add/adjust regression coverage to prevent batch-collapse regressions.

## Exit Criteria
1. `_reassemble_position_batched` returns `(B, padded_size, padded_size, 1)` for batched paths.
2. `ReassemblePatchesLayer` preserves batch dimension for gridsize > 1.
3. `tests/study/test_dose_overlap_comparison.py::test_pinn_reconstruction_reassembles_full_train_split` passes.

## Compliance Matrix (Mandatory)
- [ ] **Spec Constraint:** `docs/specs/spec-ptycho-workflow.md` (Reassembly Requirements)
- [ ] **Spec Constraint:** `specs/spec-inference-pipeline.md` §7 (reassemble_position contracts)
- [ ] **Finding/Policy ID:** CONFIG-001 (legacy params sync), POLICY-001 (torch required, no changes needed)

## Phase A — Batch-Preserving Reassembly Fix
### Checklist
- [x] A1: Update `_reassemble_position_batched` to accumulate per-sample canvases (ptycho/tf_helper.py).
- [x] A2: Align `ReassemblePatchesLayer` output shape metadata (ptycho/custom_layers.py).
- [x] A3: Update regression test to assert batch dimension is preserved.
- [x] A4: Treat `padded_size=None` as unset in batched reassembly setup (avoid passing None to `_reassemble_position_batched`).
- [x] A5: Run targeted pytest and archive log.

## Artifacts Index
- Logs: `.artifacts/FIX-REASSEMBLE-BATCH-DIM-001/`
