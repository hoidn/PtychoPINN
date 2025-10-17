# Supervisor Summary — Phase C.C1 Prep

**Date:** 2025-10-17T07:36:40Z
**Initiative:** INTEGRATE-PYTORCH-001
**Focus:** Phase C.C1 — Implement `RawDataTorch` adapter (torch-optional green phase)

## Context
- Red-phase tests for RawData and DataContainer parity are complete (`reports/2025-10-17T071836Z/`).
- Phase C checklist updated: C.B2/C.B3 marked done; implementation plan C2 now ✅.
- Next unblocker is delivering a thin wrapper that delegates to `ptycho.raw_data.RawData.generate_grouped_data()` while preserving torch-optional behavior.

## Key Requirements
1. **Wrapper Location:** Prefer new module `ptycho_torch/raw_data_bridge.py` (per phase plan).
2. **Configuration:** Pull `TrainingConfig`/`ModelConfig` from config bridge; call `update_legacy_dict(params.cfg, config)` before delegating.
3. **Delegation:** Reuse `RawData.from_coords_without_pc()` to build TF RawData, then forward all arguments to `generate_grouped_data()`.
4. **Torch Optionality:** Only import torch lazily; default to NumPy outputs so tests pass without torch installed.
5. **Return Contract:** Match keys/shapes from TensorFlow baseline — `diffraction`, `X_full`, `coords_offsets`, `coords_relative`, `nn_indices`, `Y` when available (reference `data_contract.md` §2).
6. **Testing:** Targeted selector `pytest tests/torch/test_data_pipeline.py -k raw_data -vv` must pass and produce green log.
7. **Artifacts:** Archive implementation notes and pytest output under this directory (see below).

## Deliverables for Ralph
- `ptycho_torch/raw_data_bridge.py` (new module) exposing `RawDataTorch` class with torch-optional behavior.
- Any necessary exports (`ptycho_torch/__init__.py`) so tests can import the wrapper.
- Updated `tests/torch/test_data_pipeline.py` to instantiate wrapper inside tests (replace `pytest.fail` with real assertions once wrapper exists).
- Targeted pytest log capturing green run.
- Short `implementation_notes.md` documenting design decisions (delegation strategy, dtype handling, torch fallback).

## Artifact Targets
Store outputs under this directory after implementation:
- `implementation_notes.md`
- `pytest_raw_data_green.log`
- Optional diffs or helper scripts (`raw_data_bridge_diff.md` etc.)

## References
- `plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md` — Phase C checklist (IDs C.C1–C.C4)
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md` — Canonical contract (§2)
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T071836Z/notes.md` — Red-phase guidance + failure messages
- `tests/torch/test_data_pipeline.py` — Failing tests to satisfy (focus on RawData harness for this loop)

## Next Steps After C.C1
- Phase C.C2: Implement `PtychoDataContainerTorch` (use same test module) — defer until RawData wrapper is green.
- Coordinate with TEST-PYTORCH-001 to ensure fixtures align once adapters exist.

**Action State Recommendation:** Move to `[ready_for_implementation]` for C.C1 once Ralph is briefed via `input.md`.
