Summary: Capture torch-optional red tests for RawDataTorch and data-container parity.
Mode: TDD
Focus: INTEGRATE-PYTORCH-001 — Phase C.B2+C.B3 torch-optional parity tests
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_data_pipeline.py -k raw_data -vv (expected red); pytest tests/torch/test_data_pipeline.py -k data_container -vv (expected red)
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T071836Z/{notes.md,pytest_raw_data_red.log,pytest_data_container_red.log}
Do Now:
1. INTEGRATE-PYTORCH-001 (Phase C.B2 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md) — add `test_raw_data_torch_matches_tensorflow` to `tests/torch/test_data_pipeline.py` per blueprint ROI and fixtures, then run `pytest tests/torch/test_data_pipeline.py -k raw_data -vv` (expect fail) and save log to the artifact directory; tests: pytest tests/torch/test_data_pipeline.py -k raw_data -vv (expected red).
2. INTEGRATE-PYTORCH-001 (Phase C.B3 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md) — extend the same module with `test_data_container_shapes_and_dtypes` validating torch-optional container outputs, run `pytest tests/torch/test_data_pipeline.py -k data_container -vv` (expect fail), and archive the log alongside `notes.md` summarizing failure signatures; tests: pytest tests/torch/test_data_pipeline.py -k data_container -vv (expected red).
If Blocked: Document the blocker and observed output in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T071836Z/blocked.md`, update docs/fix_plan.md Attempts History, and notify supervisor in the next loop.
Priorities & Rationale:
- specs/data_contracts.md:1 — defines canonical NPZ keys/shapes the RawData parity test must enforce.
- specs/ptychodus_api_spec.md:78 — details RawData/PtychoDataContainer lifecycle requirements consumed by Ptychodus.
- docs/architecture.md:68 — illustrates loader pipeline sequence to mirror in parity assertions.
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md:24 — checklist C.B2/C.B3 defining this loop’s deliverables.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/test_blueprint.md:1 — torch-optional pytest structure and ROI guidance.
How-To Map:
- Create `tests/torch/test_data_pipeline.py` (or extend if already present) using the whitelist pattern from `tests/conftest.py` so collection succeeds without torch.
- Import fixtures/helpers lazily: keep module imports limited to numpy/pathlib; build TensorFlow RawData reference via `ptycho.raw_data.RawData.from_file` only inside the test body.
- Reuse ROI constants from `data_contract.md` §7 (sample count, gridsize) and document any adjustments in `notes.md`.
- For C.B2, compare grouped keys (`diffraction`, `coords_offsets`, `coords_relative`, etc.) between TensorFlow RawData output and the placeholder PyTorch wrapper (expect failure because wrapper absent); assert placeholders with `pytest.fail` once divergence observed to lock expectations.
- For C.B3, scaffold assertions for container attributes (`X`, `Y`, `coords_nominal`, dtypes) referencing expectations in specs/ptychodus_api_spec.md §4; leave calls to yet-unimplemented wrappers raising `NotImplementedError` to maintain red state.
- After each pytest run, copy stdout/stderr into `pytest_raw_data_red.log` and `pytest_data_container_red.log`; record failure summaries and outstanding tasks in `notes.md`.
Pitfalls To Avoid:
- Do not implement RawDataTorch or container adapters yet—tests only.
- Avoid top-level `import torch`; rely on blueprint skip mechanisms.
- Keep ROI minimal (≤ 3 groups) to prevent long runs; do not load entire dataset.
- Do not silence failures with broad try/except; let pytest capture the red state.
- Ensure artifact filenames match those listed above for traceability.
- Do not mutate existing config bridge tests or params fixtures.
- Skip adding new dependencies; use existing utilities under `ptycho/` for reference data.
- Do not convert tests to unittest style; stay with native pytest functions and fixtures.
- Avoid committing passing placeholders—the goal is failing assertions documenting gaps.
- Remember to reference `update_legacy_dict` in notes so configuration expectations stay front-of-mind.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md:24
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/test_blueprint.md:1
- specs/data_contracts.md:1
- specs/ptychodus_api_spec.md:78
- docs/architecture.md:68
- ptycho/raw_data.py:365
- tests/conftest.py:25
Next Up:
- Phase C.C1 — implement `RawDataTorch` adapter once red tests document the required behaviour.
