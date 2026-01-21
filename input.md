Summary: Extend D4f so the grid-mode dose_response_study workflow and the legacy data_preprocessing pipeline attach dataset_intensity_stats to their manual PtychoDataContainer instances.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D4f grid/preprocessing dataset stats (D4f.3)
Branch: paper
Mapped tests: pytest tests/scripts/test_dose_response_study.py::test_simulate_datasets_grid_mode_attaches_dataset_stats -v; pytest tests/test_data_preprocessing.py::TestCreatePtychoDataset::test_attaches_dataset_stats -v; pytest tests/test_train_pinn.py::TestIntensityScale::test_uses_dataset_stats -v; pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T020608Z/
Do Now (DEBUG-SIM-LINES-DOSE-001 / D4f.3 — see implementation.md §Phase D4f):
- Implement: Update `scripts/studies/dose_response_study.py::simulate_datasets_grid_mode`. Import `compute_dataset_intensity_stats`, compute raw stats for both `X_train` and `X_test` (grid-mode mk_simdata outputs) before building containers, and pass the dict through `dataset_intensity_stats=` so grid-mode training/inference never falls back to the 988.21 constant. Add concise comments referencing `specs/spec-ptycho-core.md §Normalization Invariants` explaining why stats must tag along whenever loader.load is bypassed.
- Implement: Update `ptycho/data_preprocessing.py::create_ptycho_dataset`. Use the helper to capture stats for train/test diffraction arrays (handle normalized-only callers by re-scaling via the recorded `intensity_scale` if needed), pass them into both `PtychoDataContainer` instances, and ensure the helper never touches `.X`. Refresh `docs/DATA_GENERATION_GUIDE.md §4.3` (direct container construction) so the sample code explicitly shows attaching `dataset_intensity_stats`.
- Implement: Regression tests. (a) Extend `tests/scripts/test_dose_response_study.py` with a monkeypatched grid-mode simulation (`test_simulate_datasets_grid_mode_attaches_dataset_stats`) that stubs `ptycho.diffsim.mk_simdata` to emit deterministic tensors and asserts that every returned container now exposes stats matching `compute_dataset_intensity_stats`. (b) Add `tests/test_data_preprocessing.py::TestCreatePtychoDataset::test_attaches_dataset_stats` to validate the preprocessing factory tags both train/test containers. Keep `tests/test_train_pinn.py::TestIntensityScale::test_uses_dataset_stats` green to guard priority ordering.
- Validate via: `pytest tests/scripts/test_dose_response_study.py::test_simulate_datasets_grid_mode_attaches_dataset_stats -v`; `pytest tests/test_data_preprocessing.py::TestCreatePtychoDataset::test_attaches_dataset_stats -v`; `pytest tests/test_train_pinn.py::TestIntensityScale::test_uses_dataset_stats -v`; `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`.
- Artifacts: Store each pytest log under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T020608Z/logs/pytest_<selector>.log`. If the new tests synthesize temporary NPZs or fixtures, drop them inside the same hub (git-ignored).
How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
2. Implement the dose_response_study changes, then run `pytest tests/scripts/test_dose_response_study.py::test_simulate_datasets_grid_mode_attaches_dataset_stats -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T020608Z/logs/pytest_dose_response_stats.log`.
3. Implement the data_preprocessing + doc updates, then run `pytest tests/test_data_preprocessing.py::TestCreatePtychoDataset::test_attaches_dataset_stats -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T020608Z/logs/pytest_data_preproc_stats.log`.
4. Run the guard `pytest tests/test_train_pinn.py::TestIntensityScale::test_uses_dataset_stats -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T020608Z/logs/pytest_train_pinn_dataset_stats.log`.
5. Finish with `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T020608Z/logs/pytest_cli_smoke.log`.
Pitfalls To Avoid:
- Never call `.X`/`.Y` inside the helper—stay on NumPy arrays to preserve PINN-CHUNKED-001 guarantees.
- Grid-mode mk_simdata outputs include nominal/true coords as stacked arrays; preserve their structure when computing stats and avoid mutating them in-place.
- When only normalized amplitudes are available, divide by the recorded intensity_scale before squaring; otherwise dataset_mean degenerates to `(N/2)^2`.
- Keep computations in float64 but cast stored values to Python floats for stable np.savez round trips.
- Ensure newly added tests monkeypatch mk_simdata/probe helpers deterministically so they do not require heavyweight simulations.
- Do not touch core physics files (`ptycho/model.py`, `ptycho/diffsim.py`) or alter params.cfg semantics; limit edits to the manual constructors + docs/tests.
- Capture CLI smoke logs; missing evidence violates TEST-CLI-001.
If Blocked: Record the failure signature (stack trace, bad stats) in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T020608Z/blocked.md`, update docs/fix_plan.md Attempts History with the blocker + repro command, and halt.
Findings Applied (Mandatory):
- PINN-CHUNKED-001 — dataset_intensity_stats must be preferred over `_X_np`, and `_tensor_cache` must remain untouched.
- NORMALIZATION-001 — normalize_data stays at the `(N/2)^2` target; dataset stats feed the subsequent intensity_scale stage.
Pointers:
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md (§Phase D4f.3 checklist)
- specs/spec-ptycho-core.md:80-110 (Normalization invariants / dataset-derived mode)
- docs/DATA_GENERATION_GUIDE.md §4.3 (Manual container recipe to update)
- docs/fix_plan.md (DEBUG-SIM-LINES-DOSE-001 attempts history, D4f entries)
Next Up (optional): If D4f.3 lands smoothly, start D5 instrumentation on loss wiring (normalized_to_prediction / prediction_to_truth ratios) using the refreshed telemetry.
Doc Sync Plan: After code/tests pass, run `pytest --collect-only tests/scripts/test_dose_response_study.py -q` and `pytest --collect-only tests/test_data_preprocessing.py -q`, archive the logs under the new hub, then update `docs/TESTING_GUIDE.md §2` and `docs/development/TEST_SUITE_INDEX.md` with the new selectors before finishing the loop.
