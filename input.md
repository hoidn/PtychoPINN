Brief:
Set `AUTHORITATIVE_CMDS_DOC`, `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`, and `USE_XLA_TRANSLATE=0`, then wire `HUB/OUT_TORCH/OUT_TF/TF_BASE` per the plan (create `$TF_BASE/{cli,analysis,green}`) and record both env values at the top of each CLI log (Finding XLA-DYN-DOT-001, CONFIG-001).
Rerun `pytest tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle -vv | tee "$TF_BASE/green/pytest_tf_integration.log"` so we have a fresh green gate that shows both env exports.
With those env vars still set, rerun the Phase C1 CLI commands against `datasets/fly64/fly001_64_train_converted.npz` (non-identity) and the existing test split, tee logs into `$TF_BASE/cli/{train_tf_phase_c1,inference_tf_phase_c1}.log`, emit the debug dump and `stats.json` under `$TF_BASE/analysis/forward_parity_debug_tf`, and write `bundle_digest_tf_phase_c1.txt` plus `phase_c1_stats.txt` comparing against `scaling_alignment/phase_b3/analysis/forward_parity_debug_scaling/stats.json`.
Update `$HUB/analysis/artifact_inventory.txt` and `$HUB/summary.md` with a “Phase C1 — TF baseline” section that includes a Dataset note (path + whether we owe a PyTorch rerun) and the env capture; cite POLICY-001 if parity requires another PyTorch pass.
If TensorFlow still fails even with both env toggles, capture the RET_CHECK under `$TF_BASE/red/blocked_<timestamp>_tf_xla_disabled.md` quoting the env capture, cite XLA-DYN-DOT-001, and stop so we can decide whether to fall back to PyTorch-only Phase C evidence.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle
