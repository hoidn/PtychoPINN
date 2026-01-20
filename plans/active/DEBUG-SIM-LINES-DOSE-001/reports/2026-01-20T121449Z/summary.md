### Turn Summary
Reopened Phase D1 in the implementation plan and docs/fix_plan so the loss-weight capture tasks (D1a–D1c) remain visible, and issued a fresh Do Now targeting the updated CLI + artifact regeneration.
Documented the missing orchestration config by adding root-level `orchestration.yaml`, then logged the reviewer doc-hygiene update so prompts referencing `router.review_every_n`, `state_file`, and `logs_dir` point to a real file.
Next: Ralph extends `compare_sim_lines_params.py` to emit Markdown+JSON runtime loss snapshots, reruns the comparison CLI with the archived snapshot/dose config, and reruns the CLI pytest selector while updating the summary with the corrected evidence.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121449Z/
