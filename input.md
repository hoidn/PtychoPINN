Brief:
Phase B2’s intensity_scale persistence landed in commit 9a09ece2, but the hub still shows `Loaded intensity_scale from bundle: 1.000000` (cli/inference_patch_stats_rerun_v3.log), so Phase B3 needs fresh scaling evidence.
Export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md, set HUB="$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity", OUT="$PWD/outputs/torch_forward_parity_baseline", SCALING="$HUB/scaling_alignment/phase_b3", mkdir -p "$SCALING"/{cli,analysis,green}, then run `pytest tests/torch/test_inference_reassembly_parity.py -vv | tee "$SCALING/green/pytest_inference_reassembly.log"` (emit `$HUB/red/blocked_<timestamp>.md` immediately if it fails with POLICY-001/CONFIG-001 issues).
Re-run the canonical 10-epoch short baseline with `--log-patch-stats --patch-stats-limit 2` using the commands in the plan, teeing to `$SCALING/cli/train_patch_stats_scaling.log` and `$SCALING/cli/inference_patch_stats_scaling.log`, and copy the debug dump into `$SCALING/analysis/forward_parity_debug_scaling`.
After inference, grep for `Loaded intensity_scale` to prove it now reports the stored scalar (not 1.000000), record a digest of `wts.h5.zip` and `diffraction_to_obj/params.dill` inside `$SCALING/analysis`, and update `$HUB/analysis/artifact_inventory.txt` plus `$HUB/summary.md` with a “Phase B3 scaling validation” section summarizing the pytest result, logs, observed scalar, and bundle digest paths.
Drop blockers immediately if CUDA/memory failures prevent any command from finishing.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/torch/test_inference_reassembly_parity.py
