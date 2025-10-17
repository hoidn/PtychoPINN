# INTEGRATE-PYTORCH-001 — Dataloader DATA-001 Triage (2025-10-17T223200Z)

## Context & References
- **Initiative:** INTEGRATE-PYTORCH-001
- **Focus:** Restore DATA-001 compliance for PyTorch memory-mapped dataloader.
- **Key Specs:**
  - `specs/data_contracts.md` — Canonical NPZ schema (`diffraction` amplitude key, complex64 `Y`).
  - `specs/ptychodus_api_spec.md` §4.5 — Ptychodus reconstructor contract; requires canonical datasets.
  - `docs/workflows/pytorch.md` §4 — PyTorch dataset expectations mirror TensorFlow workflow.
- **Existing Evidence:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T221500Z/phase_e_parity_summary.md` (PyTorch integration failure log).
- **Reproduction:** `pytest tests/torch/test_integration_workflow_torch.py -vv`

## Findings Ledger
- `docs/findings.md#DATA-001` — Canonical NPZ key must be `diffraction`; legacy `diff3d` only valid pre-conversion.

## Hypotheses & Triage
| # | Hypothesis | Evidence & Triage Outcome | Status |
| - | ---------- | ------------------------- | ------ |
| H1 | PyTorch dataloader only checks for legacy `diff3d` key, triggering ValueError when canonical `diffraction` is present. | Verified via static inspection (`ptycho_torch/dataloader.py:40-66, 533-538`) — loader reads `diff3d` exclusively and raises `ValueError` when absent. Integration log shows identical exception. | **Confirmed** |
| H2 | Downstream memory-map writer cannot ingest amplitude data unless it is rounded to integers, so loader intentionally searches for legacy intensity `diff3d`. | Refuted: rounding logic at `dataloader.py:533` converts loaded array to `torch.float32` after `np.load(...)['diff3d']`. This can operate on amplitude data without integer assumption; no other module enforces integer-only path. Switching to `diffraction` retains same dtype. | **Refuted** |
| H3 | Dataset directory lacks converted canonical files, so failure stems from missing preprocessing rather than loader bug. | Refuted: TensorFlow baseline uses the exact same NPZ (`datasets/Run1084_recon3_postPC_shrunk_3.npz`) successfully (`phase_e_tf_baseline.log`). Dataset inspected manually (`np.load(...).keys()`) exposes `diffraction`, `Y`, `objectGuess`, `probeGuess`, etc. | **Refuted** |

## Next Confirming Step
- **Confidence:** High (H1 confirmed via code review + reproducible failure).
- **Next Action:** Implement loader fallback that prefers `diffraction` (DATA-001 canonical) and gracefully falls back to `diff3d` for legacy datasets. Add regression coverage to ensure both keys work and that ValueError surfaces only when neither key is present.

## Suggested Implementation Hooks
1. Update `calculate_length()` helper to attempt `diffraction` first, capturing shape/dtype validation per spec. Log or raise informative error when neither key found.
2. Update `memory_map_data()` (and any other call sites such as `_load_probe_object`) to reuse shared helper retrieving diffraction stack with canonical preference.
3. Extend `tests/torch/test_integration_workflow_torch.py` by asserting that DATA-001 fixtures load successfully once fix lands; add a lighter-weight unit/fixture test (e.g., new pytest in `tests/torch/test_dataloader.py`) that mocks NPZ contents for both key variants.
4. Re-run `pytest tests/torch/test_integration_workflow_torch.py -vv` to confirm green parity; capture logs under a new timestamped directory.

## Artifact Notes
- Store follow-up test logs under `plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/` per initiative discipline.
- Update docs/fix_plan.md with new `[INTEGRATE-PYTORCH-001-DATALOADER]` entry tracking this defect and referencing this triage document as Attempt #0.
