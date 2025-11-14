Brief:
Phase C1 is still blocked on the TensorFlow baseline because the identity dataset trips XLA; rerun it with `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` exported (log the value) while keeping the existing HUB/OUT/TF_BASE wiring so we can compare directly against the Phase B3 PyTorch stats.
After exporting AUTHORITATIVE_CMDS_DOC/HUB/OUT paths, create `$TF_BASE/{cli,analysis,green}` and run `pytest tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle -vv | tee "$TF_BASE/green/pytest_tf_integration.log"`, ensuring the log captures TF_XLA_FLAGS; file a blocker in `$HUB/red` if the selector fails.
Then rerun the documented TensorFlow training + inference commands (same datasets, 10 epochs, debug dump) with TF_XLA_FLAGS still exported, tee logs into `$TF_BASE/cli/*.log`, generate `bundle_digest_tf_phase_c1.txt`, and write a quick comparison of TF stats vs `scaling_alignment/phase_b3/analysis/forward_parity_debug_scaling/stats.json` into `$TF_BASE/analysis/phase_c1_stats.txt`.
Update `$HUB/analysis/artifact_inventory.txt` and `$HUB/summary.md` with a “Phase C1 — TF baseline” entry summarizing the pytest selector, CLI logs, bundle digest, and stat deltas; if the disabled-XLA run still fails, capture the RET_CHECK signature under `$TF_BASE/red/blocked_<timestamp>_tf_xla_disabled.md`, switch to `datasets/fly64_coord_variants/fly001_64_train_converted.npz`, and note in the summary whether a matching PyTorch rerun is required before Phase C2.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle
