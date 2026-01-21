### Turn Summary
Aligned spec-bootstrap tooling with the canonical specs/ directory: updated SpecBootstrapConfig defaults, discover_shards() fallback logic, shell scripts (init_project.sh, init_spec_bootstrap.sh), README.md, and prompts/arch_reviewer.md.
Added test_spec_bootstrap_defaults pytest to verify specs_dir defaults to Path("specs") and legacy template fallback works correctly; all 9 router tests pass.
Next: commit changes and update docs/fix_plan.md with completion status.
Artifacts: plans/active/DOC-HYGIENE-20260120/reports/2026-01-21T000742Z/cli/ (pytest logs)
