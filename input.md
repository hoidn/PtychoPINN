Summary: Implement the torch-optional RawDataTorch wrapper so the RawData parity test goes green.
Mode: TDD
Focus: INTEGRATE-PYTORCH-001 — Phase C.C1 RawDataTorch adapter
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_data_pipeline.py -k raw_data -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T073640Z/{supervisor_summary.md,implementation_notes.md,pytest_raw_data_green.log}
Do Now:
1. INTEGRATE-PYTORCH-001 (C.C1 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md + C3 @ plans/active/INTEGRATE-PYTORCH-001/implementation.md) — add `ptycho_torch/raw_data_bridge.py` exposing torch-optional `RawDataTorch`, wire exports, and update `tests/torch/test_data_pipeline.py::test_raw_data_torch_matches_tensorflow` to exercise it (replace the `pytest.fail` placeholder) while preserving NumPy fallback; tests: none.
2. INTEGRATE-PYTORCH-001 (C.C1 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md) — run `pytest tests/torch/test_data_pipeline.py -k raw_data -vv` and save stdout/stderr to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T073640Z/pytest_raw_data_green.log`; tests: pytest tests/torch/test_data_pipeline.py -k raw_data -vv.
3. INTEGRATE-PYTORCH-001 (C.C1 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md) — capture implementation rationale (delegation path, dtype handling, torch availability guard) in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T073640Z/implementation_notes.md` and append the new attempt to `docs/fix_plan.md`; tests: none.
If Blocked: Record the failing command output and current adapter state in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T073640Z/blocked.md`, note the blocker in docs/fix_plan.md Attempts History, and flag it for the next supervisor loop.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md:44 — C.C1 is the gating task for Phase C; completing it unlocks the remaining data pipeline work.
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:64 — C3 ties RawDataTorch implementation directly to the main execution plan.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md:58 — defines the exact grouped-data contract the wrapper must satisfy.
- tests/torch/test_data_pipeline.py:103 — failing test encodes the acceptance criteria; going green proves parity.
- docs/findings.md:11 — DATA-001 warns that Y dtype must stay complex64, guiding how grouped outputs are handled.
How-To Map:
- Create `ptycho_torch/raw_data_bridge.py` with a `RawDataTorch` class that constructs a TensorFlow `RawData` via `RawData.from_coords_without_pc` and forwards `generate_grouped_data` parameters; guard torch imports with `try/except ImportError` so the module stays importable without torch.
- Ensure the wrapper accepts configuration dataclasses or overrides from the config bridge, calling `update_legacy_dict(params.cfg, TrainingConfig(...))` before delegating; reuse existing fixtures in `tests/torch/test_data_pipeline.py` for params setup.
- Export the wrapper from `ptycho_torch/__init__.py` so tests can `from ptycho_torch.raw_data_bridge import RawDataTorch` without touching torch-only modules.
- In the test, instantiate the wrapper using arrays from `minimal_raw_data`, call `generate_grouped_data` with the same arguments as the TensorFlow baseline, and compare keys, shapes, and dtypes with `np.testing.assert_allclose` / `assert`.
- Run `pytest tests/torch/test_data_pipeline.py -k raw_data -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T073640Z/pytest_raw_data_green.log` to capture the passing log once parity holds.
- Summarize any non-trivial choices (e.g., dtype normalization, sequential_sampling handling) plus follow-ups for C.C2 in `implementation_notes.md` before updating docs/fix_plan.md.
Pitfalls To Avoid:
- Do not import torch at module scope; keep the wrapper usable when torch is absent.
- Avoid reimplementing grouping logic—delegate to `ptycho.raw_data.RawData` to maintain parity.
- Always call `update_legacy_dict` before invoking RawData functions or tests will silently corrupt `params.cfg`.
- Preserve dtype/shape fidelity (float32 diffraction, int32 nn_indices, complex64 Y) per data contract.
- Keep tests torch-optional: guard optional tensor assertions with the `TORCH_AVAILABLE` flag already defined in the module.
- Do not modify existing fixtures beyond what’s required for the wrapper; they are shared across other parity tests.
- Ensure the new module lives under `ptycho_torch/` and does not touch stable core files (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Capture artifacts exactly at the specified path; supervisors rely on deterministic names.
- When updating docs/fix_plan.md, only append the new attempt—do not rewrite prior history.
- Keep the RawDataTorch API surface minimal (constructor + `generate_grouped_data`) to avoid premature scope creep.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md:44
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:64
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md:58
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T071836Z/notes.md:1
- tests/torch/test_data_pipeline.py:103
- ptycho/raw_data.py:365
- docs/findings.md:11
Next Up:
- Phase C.C2 — Implement `PtychoDataContainerTorch` once RawDataTorch parity is green.
