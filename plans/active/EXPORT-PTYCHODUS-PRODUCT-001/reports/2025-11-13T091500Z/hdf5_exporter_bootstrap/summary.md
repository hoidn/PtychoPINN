### Turn Summary
Reviewed the Run1084 exporter hub evidence (pytest log, CLI conversion, product verification) and confirmed the HDF5 + snippet deliverables are complete.
Updated the implementation plan, hub plan, docs/fix_plan.md, and input.md so the next Do Now focuses on merging the approved “Ptychodus Product Export” section into docs/DATA_MANAGEMENT_GUIDE.md (plus optional docs/index link).
Next: Ralph should integrate the snippet into the guide, refresh hub summaries/artifact inventory with the doc path, and report blockers under HUB/red if anything prevents the edit.
Artifacts: plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/2025-11-13T091500Z/hdf5_exporter_bootstrap/ (summary.md, plan/plan.md)

### Turn Summary
Forced the Phase G dense rerun focus into `blocked_escalation` after eight doc-only loops produced no new CLI/log/analysis evidence; updated `docs/fix_plan.md` and the hub dwell report so Ralph must deliver counted-run artifacts before we re-engage.
Pivoted the active focus to EXPORT-PTYCHODUS-PRODUCT-001, created the Run1084 exporter hub, and added a plan update that spells out the pytest selector, CLI conversion command, reader verification step, and DATA_MANAGEMENT_GUIDE snippet draft required for acceptance.
Next: run `pytest tests/io/test_ptychodus_product_io.py -vv`, convert `datasets/Run1084_recon3_postPC_shrunk_3.npz` via `scripts/tools/convert_to_ptychodus_product.py`, verify the resulting HDF5 with the Ptychodus reader, and capture logs/summary/doc draft under this hub (failures → `red/` with command + exit code).
Artifacts: plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/2025-11-13T091500Z/hdf5_exporter_bootstrap/ (plan/plan.md, summary.md)
