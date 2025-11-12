### Turn Summary
Verified the Phase 6A guardrails (enable_oversampling/neighbor_pool_size plumbing, RawData gating, docs/tests) via hub 2025-11-12T005637Z/phase6_hardening so the initiative can close.
Marked the implementation plan + docs/fix_plan.md as complete, rewrote input.md to pivot back to the Phase G dense rerun blocker, and reminded Ralph to rerun from `/home/ollie/Documents/PtychoPINN` with the scalar-mask regression + geometry acceptance tests as guards.
Next: Ralph must run the listed pytest selectors, execute the counted dense pipeline with `--clobber`, immediately run the fully parameterized `--post-verify-only`, and publish the SSIM/verification/highlights/metrics/preview bundle before touching Phase E/comparison work.
Artifacts: plans/active/independent-sampling-control/reports/2025-11-12T005637Z/phase6_hardening/ (summary.md, green/pytest_enable_flag.log, green/pytest_neighbor_pool_size_guard.log)

### Turn Summary
Implemented Phase 6A oversampling hardening with explicit opt-in flags (`enable_oversampling`, `neighbor_pool_size`) and OVERSAMPLING-001 guards in RawData.generate_grouped_data; all config/CLI/workflow plumbing complete with actionable ValueError messages referencing the finding.
Added test_enable_oversampling_flag_required and test_neighbor_pool_size_guard to verify guards work correctly; updated existing oversampling tests to set new flags; all 7 tests passing.
Next: Phase 6B would add example scripts/YAML configs demonstrating oversampling workflows, but Phase 6A acceptance (explicit controls + guards + tests + docs) is complete.
Artifacts: plans/active/independent-sampling-control/reports/2025-11-12T005637Z/phase6_hardening/ (green/pytest_*.log, summary.md)
