Brief:
Phase A v3 evidence is in the hub, so start Phase B2 by capturing the real `intensity_scale` during `_train_with_lightning` (read the Lightning/PTycho scaler parameter when trainable, otherwise compute the fallback from `docs/specs/spec-ptycho-core.md:80-110`) and stash it in `train_results`.
Thread that scalar through `save_torch_bundle` so `params_snapshot['intensity_scale']` is no longer always `1.0`, make `load_inference_bundle_torch`/CLI inference prefer the stored value (the log line should report the new float), and document the behavior in `docs/workflows/pytorch.md`.
Add/extend pytest coverage (e.g., `tests/torch/test_model_manager.py` or a new workflow test) proving the scale survives a save→load round trip, then rerun the short baseline to refresh `cli/*_rerun_v3.log` so inference shows the persisted scale.
If CUDA/memory blocks any command, capture the minimal signature and drop `$HUB/red/blocked_<timestamp>.md` referencing POLICY-001 / CONFIG-001 immediately.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_model_manager.py::TestSaveTorchBundle::test_save_bundle_archives_config_snapshot
