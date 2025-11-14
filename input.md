Brief:
TF baseline is still missing because the identity dataset keeps crashing in `projective_warp_xla_jit`, so export AUTHORITATIVE_CMDS_DOC plus `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`, set HUB/OUT_TORCH/OUT_TF/TF_BASE per the plan, create `$TF_BASE/{cli,analysis,green}`, and rerun `pytest tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle -vv | tee "$TF_BASE/green/pytest_tf_integration.log"` with the env value captured in the log.
With TF_XLA_FLAGS still in scope, rerun the Phase C1 CLI commands but point `scripts/training/train.py` at `datasets/fly64_coord_variants/fly001_64_train_converted.npz` (non-identity) while keeping the existing test split, tee logs into `$TF_BASE/cli/{train_tf_phase_c1,inference_tf_phase_c1}.log`, and generate the debug dump plus `stats.json` under `$TF_BASE/analysis/forward_parity_debug_tf`.
Write `bundle_digest_tf_phase_c1.txt` and `phase_c1_stats.txt`, compare the TF stats to `scaling_alignment/phase_b3/analysis/forward_parity_debug_scaling/stats.json`, and update `$HUB/analysis/artifact_inventory.txt` + `$HUB/summary.md` with a “Phase C1 — TF baseline” entry that includes a Dataset note explaining whether we now owe a matching PyTorch rerun.
If TensorFlow still fails even on the non-identity dataset, stop after capturing the RET_CHECK signature under `$TF_BASE/red/blocked_<timestamp>_tf_xla_disabled.md`, cite Finding XLA-DYN-DOT-001, and document the proposed mitigation (e.g., params.cfg change) before another attempt.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle
