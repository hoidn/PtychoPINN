# Phase D.C C4 Documentation Update Summary

**Initiative:** ADR-003-BACKEND-API
**Phase:** D.C — Inference CLI Thin Wrapper (Task C4)
**Date:** 2025-10-20
**Mode:** Docs — Documentation Alignment
**Agent:** Ralph (Attempt #52)

## Objective

Update `docs/workflows/pytorch.md` §12 (inference execution flags) to reflect Phase D.C C3 thin wrapper implementation, ensuring flag defaults, helper delegation, artifact outputs, and example commands match the actual CLI behavior.

## Files Modified

### 1. `docs/workflows/pytorch.md` (lines 354-393)

**Changes Applied:**

1. **Flag Defaults Table (lines 358-363):**
   - Updated `--accelerator` default from `'cpu'` to `'auto'` (matches `ptycho_torch/inference.py:472`)
   - Expanded accelerator choices to include `'auto'`, `'cuda'`, `'mps'` in description
   - Updated `--inference-batch-size` default from `1` to `None` (reuses training batch_size per `inference.py:492`)
   - Added explicit description noting auto-detection behavior (cuda if available, else cpu)

2. **Deprecation Notice (lines 365-366):**
   - Aligned wording with training section deprecation policy
   - Added explicit Phase E removal timeline reference ("will be removed in Phase E (post-ADR acceptance)")
   - Maintains consistency with `resolve_accelerator()` warning behavior

3. **Helper-Based Configuration Flow (lines 368-374):**
   - Added new subsection documenting inference delegation to shared helpers
   - Lists `resolve_accelerator()`, `build_execution_config_from_args()`, `validate_paths()` with descriptions
   - Documents `_run_inference_and_reconstruct()` helper location and responsibilities (checkpoint loading, Lightning prediction, PNG artifact saving)
   - Confirms CONFIG-001 compliance via factory-driven `update_legacy_dict()` call
   - Notes parity with training CLI architecture per Phase D.C blueprint

4. **Example CLI Command (lines 376-386):**
   - Added complete inference example using minimal dataset fixture (`tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`)
   - Demonstrates `--accelerator cpu`, `--quiet`, `--n_images 64` usage
   - Follows same format as training example for consistency

5. **Expected Output Artifacts (lines 388-390):**
   - Documents amplitude/phase PNG outputs per `save_individual_reconstructions()` contract (`inference.py:622-633`)
   - Provides clear artifact path expectations for users

6. **Evidence Reference (line 392):**
   - Cites Phase D.C C3 GREEN test results (9/9 passing)
   - Points readers to `tests/torch/test_cli_inference_torch.py` for delegation contract tests

## Gap Analysis Cross-Check

All five gaps identified in `docs_gap_analysis.md` have been addressed:

| Gap ID | Description | Resolution | Evidence |
|--------|-------------|------------|----------|
| #1 | Flag defaults drifted | Updated table defaults to `'auto'` and `None` | Lines 360, 362 |
| #2 | Helper flow undocumented | Added "Helper-Based Configuration Flow" subsection | Lines 368-374 |
| #3 | Example usage missing | Added CLI example with minimal fixture | Lines 376-386 |
| #4 | Artifact expectations undocumented | Added "Expected Output Artifacts" bullets | Lines 388-390 |
| #5 | Deprecation timeline misaligned | Updated deprecation notice with Phase E reference | Lines 365-366 |

## Tests Run

**None** — This is a documentation-only loop per `input.md` directive (Mode: Docs, tests: none).

## References

- **Spec:** `specs/ptychodus_api_spec.md` §4.6 (model persistence), §4.8 (backend selection), §7 (CLI flags)
- **Implementation:** `ptycho_torch/inference.py:381-640` (cli_main + _run_inference_and_reconstruct)
- **Helpers:** `ptycho_torch/cli/shared.py` (validate_paths, resolve_accelerator, build_execution_config_from_args)
- **Tests:** `tests/torch/test_cli_inference_torch.py` (TestInferenceCLIThinWrapper — 5/5 delegation tests GREEN)
- **Blueprint:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/inference_refactor.md`
- **Gap Analysis:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123820Z/phase_d_cli_wrappers_inference_docs/docs_gap_analysis.md`

## Exit Criteria Validation

- [x] Inference flag defaults match `inference.py:472,492` (auto, None)
- [x] Helper delegation documented with function names + responsibilities
- [x] Example command uses minimal fixture + shows current flag syntax
- [x] Artifact outputs documented (amplitude/phase PNGs)
- [x] Deprecation messaging consistent with training section + Phase E timeline noted
- [x] All gap analysis items from `docs_gap_analysis.md` addressed
- [x] No production code modified (docs-only loop)

## Next Steps

1. Mark `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md` row C4 as `[x]` with completion notes
2. Refresh `summary.md` in same directory with C4 completion narrative
3. Append Attempt #52 to `docs/fix_plan.md` referencing this artifact directory
4. Proceed to Phase D.D (smoke evidence + hygiene) per plan

## Artifact Inventory

- `docs_gap_analysis.md` (1.2 KB) — Pre-existing gap identification from previous loop
- `docs_update_summary.md` (this file, 4.5 KB) — C4 completion documentation

---

**Status:** Phase D.C C4 COMPLETE — Inference CLI documentation aligned with thin wrapper implementation.
