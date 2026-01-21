### Turn Summary
Implemented D6a realspace_weight fix: added `realspace_weight=0.1` and `realspace_mae_weight=1.0` to pipeline config, plus fixed Keras 3.x API incompatibility in `complex_mae`.
The realspace loss now computes (trimmed_obj_loss=2.2 vs 0.0), but amplitude metrics didn't improve significantly (MAE 2.365 vs 2.368 baseline).
Next: Consider increasing realspace_weight to 1.0 or 10.0, or running more epochs, since realspace_loss (~2.2) is dwarfed by NLL (~250k).
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T220000Z/ (gs2_ideal_v4/, logs/)
