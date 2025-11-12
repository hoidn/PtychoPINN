### Turn Summary
Implemented Phase 6A oversampling hardening with explicit opt-in flags (`enable_oversampling`, `neighbor_pool_size`) and OVERSAMPLING-001 guards in RawData.generate_grouped_data; all config/CLI/workflow plumbing complete with actionable ValueError messages referencing the finding.
Added test_enable_oversampling_flag_required and test_neighbor_pool_size_guard to verify guards work correctly; updated existing oversampling tests to set new flags; all 7 tests passing.
Next: Phase 6B would add example scripts/YAML configs demonstrating oversampling workflows, but Phase 6A acceptance (explicit controls + guards + tests + docs) is complete.
Artifacts: plans/active/independent-sampling-control/reports/2025-11-12T005637Z/phase6_hardening/ (green/pytest_*.log, summary.md)
