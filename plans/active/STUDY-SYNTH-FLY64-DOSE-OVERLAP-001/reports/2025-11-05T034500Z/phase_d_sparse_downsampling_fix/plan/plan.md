# Phase D Sparse Downsampling Fix — Plan

## Context
- Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Phase: D (overlap filtering) unblocker for Phase F sparse LSQML runs
- Problem: Sparse overlap generation currently raises `ValueError` when acceptance rate <10% because all positions violate the 102.4 px threshold. Real Phase C coordinates are dense but contain viable subsets if we downsample greedily. Need to add spacing-aware sampler so Phase D can emit sparse NPZs instead of aborting.

## Goal
Implement a deterministic greedy spacing selector in `generate_overlap_views` so sparse view generation emits DATA-001 compliant NPZs even when raw coordinates are dense, provided a subset satisfies the threshold. Capture TDD evidence with a RED test that mirrors the current failure (line-spaced positions @64 px) and GREEN run after implementation. Preserve guardrails for truly unsalvageable inputs.

## Tasks
| ID | Description | State | Notes |
| --- | --- | --- | --- |
| D7.1 | Author failing test `tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_sparse_downsamples` constructing coords spaced at 64 px (all points fail individually; subset should pass). Expect `generate_overlap_views` to succeed and emit metadata with `n_accepted > 0`. | [ ] | Mirrors real sparse failure reported in Attempt #20 CLI logs. Use tmp_path to fabricate NPZs. |
| D7.2 | Implement greedy spacing selector helper in `studies/fly64_dose_overlap/overlap.py` and invoke when initial acceptance <0.1. Recompute metrics post-selection and annotate metadata with `selection_strategy`. | [ ] | Keep module pure (CONFIG-001) and maintain deterministic ordering (lexsort on coords). |
| D7.3 | Re-run targeted pytest (`tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_sparse_downsamples`, `-k overlap`) capturing logs under `reports/2025-11-05T034500Z/phase_d_sparse_downsampling_fix/{red,green,collect}`. | [ ] | Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before running commands. |
| D7.4 | Archive summary with metrics + acceptance counts in `docs/summary.md` and update `docs/fix_plan.md` Attempt #85 once GREEN. | [ ] | Note whether greedy selection changed dataset sizes and highlight compliance with DATA-001 / OVERSAMPLING-001. |

## Findings to Observe
- CONFIG-001: Keep overlap filtering pure.
- DATA-001: Validate filtered NPZs post-selection via existing validator.
- POLICY-001: No change (PyTorch dependency unaffected).
- OVERSAMPLING-001: Ensure neighbor_count metadata remains intact for Phase E.

## Exit Criteria
1. New pytest covers greedy sparse selection and passes with GREEN log archived.
2. `generate_overlap_views` emits sparse NPZs for 64 px spaced coordinates without raising `ValueError`.
3. Metadata records acceptance counts and selection strategy.
4. CLI guard still raises descriptive error when even greedy selection yields <10% acceptance (existing test updated as needed).
5. docs/fix_plan.md Attempt #85 documents work with artifact links to this hub.
