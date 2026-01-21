### Turn Summary
Implemented D4f.3: both simulate_datasets_grid_mode and create_ptycho_dataset now call compute_dataset_intensity_stats and attach the dict to every PtychoDataContainer, eliminating the 988.21 fallback for grid-mode/preprocessing flows.
Created two new regression tests (test_simulate_datasets_grid_mode_attaches_dataset_stats, TestCreatePtychoDataset::test_attaches_dataset_stats) plus updated docs/DATA_GENERATION_GUIDE.md with mandatory stats attachment guidance; all 4 mapped selectors pass.
Next: continue amplitude bias investigation (Phase D5) now that dataset-derived intensity_scale is propagated across all data paths.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T020608Z/ (pytest_all_selectors.log)
