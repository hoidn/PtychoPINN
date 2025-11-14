### Turn Summary
Confirmed the Run1084 exporter documentation already lives in `docs/DATA_MANAGEMENT_GUIDE.md:242-369` (full "Ptychodus Product Export" section) and `docs/index.md:125-133` (Data Management Guide entry), so the pending Do Now for EXPORT-PTYCHODUS-PRODUCT-001 is fully satisfied.
Verified the hub evidence (`analysis/data_guide_snippet.md`, `analysis/artifact_inventory.txt`, and both summary files under `$HUB/summary/`) references the published section path, so no repo edits were required this loop.
Marked the exporter initiative complete and pivoted the active focus to FIX-PYTORCH-FORWARD-PARITY-001 for Phase A instrumentation planning.
Next: execute the PyTorch parity Do Now (instrument per-patch stats + capture the short baselines) inside `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/`.
Artifacts: plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/2025-11-13T091500Z/hdf5_exporter_bootstrap/{analysis/artifact_inventory.txt,summary.md,summary/summary.md}

### Turn Summary
Forced Tier‑3 dwell block on STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 after re-checking `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dose_1000/dense/train_debug_v3/logs/logs/debug.log:347-365`, `.../analysis/dose_1000/dense/test/comparison_metrics.csv:2-13`, and `.../analysis/verification_report.json:1-40`, then updated the plan/summary and `analysis/dwell_escalation_report.md`.
Reset docs/fix_plan ordering (marking INTEGRATE-PYTORCH-PARITY-001B done) and pivoted the active focus to EXPORT-PTYCHODUS-PRODUCT-001.
Re-read docs/index.md, docs/findings.md (DATA-001), specs/data_contracts.md, docs/DATA_MANAGEMENT_GUIDE.md, and the Run1084 snippet at `plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/2025-11-13T091500Z/hdf5_exporter_bootstrap/analysis/data_guide_snippet.md`, then refreshed the plan/input so Ralph can publish the new “Ptychodus Product Export” section and cite it from the hub summaries/inventory.
Next: integrate the snippet into `docs/DATA_MANAGEMENT_GUIDE.md`, add the optional `docs/index.md` cross-link, and update `$HUB/{analysis/artifact_inventory.txt,summary.md,summary/summary.md}` once the doc lands.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/analysis/dwell_escalation_report.md; plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/2025-11-13T091500Z/hdf5_exporter_bootstrap/analysis/data_guide_snippet.md
