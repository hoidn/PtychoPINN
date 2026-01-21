### Turn Summary
Implemented Phase D4f: loader.load() now captures raw diffraction statistics (E_batch[Σ_xy |X|²]) BEFORE L2 normalization and attaches them to PtychoDataContainer as dataset_intensity_stats.
Updated calculate_intensity_scale() to prefer these pre-normalization stats, fixing the spec-compliant dataset-derived intensity_scale formula instead of degenerating to the closed-form fallback.
Added 4 new tests covering stats attachment, train/test splits, and the priority logic; all 15 tests pass.
Next: Supervisor review - the rerun of gs1_ideal + gs2_ideal scenarios for bias_summary regeneration is deferred pending confirmation.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/ (pytest_full_test_suite.log)
