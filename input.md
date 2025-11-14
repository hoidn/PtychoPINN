Brief:
Use the §C1c python snippet to read `scaling_alignment/phase_b3/analysis/forward_parity_debug_scaling/stats.json` and `scaling_alignment/phase_c1_gs1/analysis/forward_parity_debug_gs1/stats.json`, write the delta text to `tf_baseline/phase_c1_gs1/analysis/phase_c1_gs1_stats.txt`, then copy it into `scaling_alignment/phase_c1_gs1/analysis/phase_c1_vs_phase_b3_stats.txt` and record both sha1s.
Append a “Phase C1 — GS1 fallback (PyTorch-only)” section to `$HUB/analysis/artifact_inventory.txt` that lists the GS1 train/infer logs, stats/digest files, the new stats-delta artifact, and the TF blocker files under `tf_baseline/phase_c1_gs1/red/`.
Update `$HUB/summary.md` and `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md` with a GS1 dataset note, the stats-delta path/sha1, and explicit references to the TF blocker filenames; if any prerequisite artifact is missing, log `$HUB/scaling_alignment/phase_c1_gs1/red/blocked_<timestamp>_missing_gs1_artifacts.md` before editing inventories.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: none — evidence-only
