# Phase G Dense Pipeline Recovery Plan (2025-11-08T150500Z)

## Context
- Attempted Phase C→G dense pipeline execution via `bin/run_phase_g_dense.py --clobber` failed during Phase C validation.
- CLI log (`.../cli/run_phase_g_dense_cli.log`) shows Stage 5 TypeError: `validate_dataset_contract()` received unexpected keyword argument `dataset_path`.
- Validator signature was refactored earlier in Phase B to accept in-memory data dicts; orchestration still calls old interface.

## Findings
1. `generate_dataset_for_dose()` (studies/fly64_dose_overlap/generation.py:220-224) still calls `validate_fn(dataset_path=..., design_params=..., expected_dose=dose)`.
2. `validate_dataset_contract()` now expects `data`, `view`, `gridsize`, `neighbor_count`, `design` and performs pure in-memory checks.
3. Existing Phase C pytest (`test_generate_dataset_pipeline_orchestration`) does not load real validator; it mocks and only asserts keyword names, so signature drift went unnoticed.

## Actions for Ralph
- Patch `generate_dataset_for_dose()` so Stage 5 loads each split NPZ via `np.load(split_path)` and calls `validate_dataset_contract` with `data=dict(np.load(...))`, `view=plan.view`, `gridsize=plan.model_config.gridsize`, `neighbor_count=design_params['neighbor_count']`, reusing cached `design`. Ensure files are closed (use context manager) and propagate `ValueError` for contract violations.
- Update Phase C test suite to cover the real validator:
  - Add regression test that injects the true validator (no Mock) and asserts Stage 5 passes without TypeError for a minimal synthetic dataset.
  - Keep orchestration test but assert call signature via spec-adhering stub (e.g., accepts `data`, `view`, `gridsize`, `neighbor_count`, `design`).
- Rerun targeted pytest selectors plus highlights preview test to confirm CLI parity.
- Re-execute dense pipeline (`bin/run_phase_g_dense.py --hub ... --clobber`) to regenerate Phase C outputs and continue Phase G evidence capture once validation passes.

## Exit Criteria
1. `pytest tests/study/test_dose_overlap_generation.py -k validate -vv` (or equivalent new selector) GREEN with real validator.
2. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv` GREEN.
3. Dense pipeline completes Phase C without TypeError; logs archived under this hub (cli/phase_c_generation.log, run_phase_g_dense_cli.log).
4. Summary updated with MS-SSIM/MAE deltas once pipeline finishes (may spill to subsequent loop if runtime-long, but validation fix must be confirmed.

## References
- Validator spec: `studies/fly64_dose_overlap/validation.py`
- Prior hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T130500Z/phase_g_dense_full_execution_real_run/`
- Findings: POLICY-001, CONFIG-001, DATA-001, TYPE-PATH-001, OVERSAMPLING-001
