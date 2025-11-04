# Phase C Regression — Loop Summary (2025-11-07T09:05:00Z)

## Context
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 (Phase G dense execution evidence)
- Previous loop (2025-11-07T07:05Z) delivered orchestration script but Phase C step failed with `TypeError: object of type 'float' has no len()` during simulation.

## Findings
- The regression originates in `build_simulation_plan`: the constructed `TrainingConfig` sets `n_groups` but leaves the legacy `n_images` field unset.
- `_generate_simulated_data_legacy_params` inside `ptycho/nongrid_simulation.py` still reads `config.n_images` to size coordinate arrays, so `None` yields scalar draws and the downstream `len(xcoords)` call raises.
- Existing Phase C tests only assert `n_groups` and miss the legacy attribute requirement.

## Plan of Record
1. Extend `tests/study/test_dose_overlap_generation.py::test_generate_dataset_config_construction` (or new test) to assert `TrainingConfig.n_images == plan.n_images` and capture regression message in RED.
2. Update `build_simulation_plan` to set both `n_groups` and `n_images` (explicit int conversion) while maintaining CONFIG-001 / TYPE-PATH-001 guardrails (no additional deps).
3. Re-run targeted pytest selector `pytest tests/study/test_dose_overlap_generation.py -vv` (GREEN expected after fix).
4. Re-run `bin/run_phase_g_dense.py` end-to-end to confirm Phase C completes; archive fresh CLI log + summary in this hub.

## Evidence Links
- Blocker log (previous loop): `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T070500Z/phase_g_execution_real_runs/cli/phase_c_generation.log`
- Error signature: `TypeError: object of type 'float' has no len()` at `ptycho/raw_data.py:227`

## Next Supervisor Actions
- Update `docs/fix_plan.md` with this regression note and artifact path.
- Prepare `input.md` directing Ralph to implement the fix + rerun Phase C.
