Brief:
Phase B3 proved the persisted intensity_scale works (scaling_alignment/phase_b3), so Phase C1 now needs a matched TensorFlow baseline for comparison.
Export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md, set HUB="$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity", OUT_TORCH="$PWD/outputs/torch_forward_parity_baseline", OUT_TF="$PWD/outputs/tf_forward_parity_baseline", TF_BASE="$HUB/tf_baseline/phase_c1", mkdir -p "$TF_BASE"/{cli,analysis,green}.
Run `pytest tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle -vv | tee "$TF_BASE/green/pytest_tf_integration.log"` as the health gate (file `$HUB/red/blocked_<timestamp>.md` if POLICY-001/CONFIG-001 errors surface), then execute the documented TensorFlow training command (10 epochs, gridsize=2, same datasets as Torch) tee’d into `$TF_BASE/cli/train_tf_phase_c1.log`.
Follow with `python scripts/inference/inference.py --backend tensorflow ... --debug_dump "$TF_BASE/analysis/forward_parity_debug_tf"` to capture stats/offets PNGs (`|& tee "$TF_BASE/cli/inference_tf_phase_c1.log"`), record `shasum "$OUT_TF/wts.h5.zip"` into `$TF_BASE/analysis/bundle_digest_tf_phase_c1.txt`, and stash a quick JSON diff between TF stats and the existing `scaling_alignment/phase_b3/analysis/forward_parity_debug_scaling/stats.json`.
Update `$HUB/analysis/artifact_inventory.txt` and `$HUB/summary.md` with a “Phase C1 — TF baseline” entry summarizing the pytest selector, CLI logs, bundle digest, and the key stat deltas; drop blockers immediately if TensorFlow/CLI runs fail and cite POLICY-001/CONFIG-001.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle
