### Turn Summary
Advanced Phase C planning for the reassembly fix: mined the B4 telemetry, marked Phase B complete in the plan, and scoped C1 around updating `create_ptycho_data_container()` to expand `params.cfg['max_position_jitter']` based on actual grouped offsets (with pytest coverage) so `get_padded_size()` meets the spec requirement.
Captured the trimmed-down Do Now for Ralph (jitter updater + regression test) and refreshed docs/fix_plan.md/galph_memory with the new focus; no new artifacts this loop.
Next: Ralph updates the workflow helper, adds the targeted test, and reruns the existing selectors plus the SIM-LINES CLI guard to verify padded-size math is correct end-to-end.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T060156Z/ (planning notes only)
