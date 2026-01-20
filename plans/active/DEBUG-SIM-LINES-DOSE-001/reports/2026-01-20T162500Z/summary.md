### Turn Summary
Extended `analyze_intensity_bias.py` with loss-composition parsing per `specs/spec-ptycho-workflow.md §Loss and Optimization`, exposing individual loss components, dominance ratios, and inactive flags alongside stage-ratio citations to `specs/spec-ptycho-core.md §Normalization Invariants`.
Ran analyzer on gs2_base (5-epoch) vs gs2_ne60 (30-epoch); both show `pred_intensity_loss` dominating (~99%) with `trimmed_obj_loss=0`, confirming loss wiring is NOT the issue — the ~6.6× prediction→truth gap is upstream of the loss function.
Next: trace `IntensityScaler`/`IntensityScaler_inv` output scaling in `ptycho/model.py` to determine if the model architecture inherently outputs scaled-down amplitudes.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T162500Z/ (bias_summary.json, bias_summary.md, analyze_loss_wiring.log, pytest_cli_smoke.log)
