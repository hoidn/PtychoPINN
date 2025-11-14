Brief:
Create the Tier-2 analysis helper `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/bin/phase_c2_compare_stats.py` (argparse + docstring) that ingests two hub stats JSON files and prints ratios/differences for patch/canvas means, stds, and var_zero_mean (cite POLICY-001/CONFIG-001 in the header comment).
Export `HUB="$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity"`, run the script for Phase B3 vs Phase C1 GS1 as documented in the plan (baseline stats = `scaling_alignment/phase_b3/analysis/forward_parity_debug_scaling/stats.json`, candidate = `scaling_alignment/phase_c1_gs1/analysis/forward_parity_debug_gs1/stats.json`), tee output to `cli/phase_c2_compare_stats.log`, and write metrics to `$HUB/analysis/phase_c2_pytorch_only_metrics.txt` plus `*.sha1`.
Update `$HUB/analysis/artifact_inventory.txt`, `$HUB/summary.md`, and `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md` with a “Phase C2 — PyTorch-only comparison” section citing the metrics, sha1, log, and existing TF blocker files; if either stats JSON is missing/corrupt, log `$HUB/scaling_alignment/phase_c1_gs1/red/blocked_<timestamp>_missing_stats.md` before editing anything else.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: none — evidence-only
