### Turn Summary
Implemented --backend CLI argument for inference.py (parse_arguments + setup_inference_configuration), matching training.py pattern.
Added POLICY-001 and CONFIG-001 help text, defaults to 'tensorflow' for backward compatibility.
Extended test suite with CLI parsing tests; all backend selectors GREEN (2 passed in 3.71s).
Ran PyTorch CLI smoke tests: training completed and saved bundle successfully; inference loaded bundle correctly but hit expected probe attribute error in legacy TF reconstruction path.
Artifacts: green/pytest_backend_selector_cli.log, cli/pytorch_cli_smoke/{train.log,inference.log}, analysis/artifact_inventory.txt

### Turn Summary
Implemented minimal persistence shim in ptycho_torch/api/base_api.py::save_pytorch() that emits Lightning checkpoint + JSON manifest with params.cfg snapshot per spec §4.6.
Verified config bridge wiring already complete: train.py and inference.py use config_factory pattern which internally calls update_legacy_dict before RawData access.
All 45 config bridge parity tests pass (100% green); missing defaults handled by adapter layer (no config_params.py changes needed).
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/ (green/pytest_config_bridge.log, analysis/artifact_inventory.txt, summary.md)
