Summary: Restore Phase C simulation so dense pipeline can progress by wiring `TrainingConfig.n_images` and proving the fix.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_dose_overlap_generation.py -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix/

Do Now (hard validity contract):
  - Implement: studies/fly64_dose_overlap/generation.py::build_simulation_plan — set `TrainingConfig.n_images` (int) alongside `n_groups` so legacy simulator sizes coordinate arrays correctly.
  - Implement: tests/study/test_dose_overlap_generation.py::test_generate_dataset_config_construction — extend expectations to fail when `TrainingConfig.n_images` is missing; confirm RED before code change.
  - Validate: pytest tests/study/test_dose_overlap_generation.py -vv
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix --dose 1000 --view dense --splits train test

How-To Map:
  1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  2. Add the new assertion in `tests/study/test_dose_overlap_generation.py::test_generate_dataset_config_construction`; run `pytest tests/study/test_dose_overlap_generation.py::test_generate_dataset_config_construction -vv` and capture the expected failure mentioning `None`/`object of type 'float' has no len()`.
  3. Update `build_simulation_plan` so the constructed `TrainingConfig` has both `n_groups` and `n_images` (use `int(plan.n_images)`); rerun the single-test selector to confirm GREEN, then run `pytest tests/study/test_dose_overlap_generation.py -vv` for the whole module.
  4. Dry-run the orchestrator to confirm command sequencing: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix --dose 1000 --view dense --splits train test --collect-only`.
  5. Execute the full pipeline: `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix --dose 1000 --view dense --splits train test`; archive updated `cli/phase_c_generation.log`, `analysis/blocker.log` (if any), and metric summaries under the same hub.

Pitfalls To Avoid:
  - Do not touch core stable modules (ptycho/model.py, ptycho/diffsim.py, ptycho/tf_helper.py).
  - Keep TYPE-PATH-001 path normalization intact; only adjust Phase C helper code.
  - Preserve CONFIG-001 by avoiding new global state mutations and leaving `update_legacy_dict` flow untouched.
  - Ensure added test remains pytest-native (no unittest mix) and documents the regression message.
  - Confirm `n_images` is an `int`; avoid passing numpy scalar or float.
  - Capture RED/GREEN logs in hub before rerunning to GREEN.
  - Leave orchestrator script behavior (collect-only, fail-fast) unchanged apart from rerun evidence.

If Blocked:
  - Capture the failure in `analysis/blocker.log` within the hub, note selector/command and error signature, update `docs/fix_plan.md` + `galph_memory.md`, and pause for supervisor guidance.

Findings Applied (Mandatory):
  - POLICY-001 — PyTorch dependency stays installed; Phase C still drives TensorFlow pipeline only.
  - CONFIG-001 — Maintain legacy bridge expectations when constructing configs.
  - DATA-001 — Keep dataset generation producing DATA-001 compliant artifacts after rerun.
  - OVERSAMPLING-001 — Do not alter dense/sparse spacing parameters while fixing Phase C.
  - TYPE-PATH-001 — Preserve Path normalization across CLI/script usage.

Pointers:
  - `studies/fly64_dose_overlap/generation.py:50`
  - `tests/study/test_dose_overlap_generation.py:70`
  - `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T090500Z/phase_c_generation_fix/plan/plan.md`
  - `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T070500Z/phase_g_execution_real_runs/cli/phase_c_generation.log`
  - `docs/findings.md#policy-001`

Next Up (optional): Re-run sparse view dense pipeline once Phase C succeeds and metrics are captured.
