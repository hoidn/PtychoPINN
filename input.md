Summary: Restore Phase C validation in the dense pipeline by updating `generate_dataset_for_dose` to use the refactored validator and extend Phase C tests before rerunning the highlights-aware orchestrator.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_dose_overlap_generation.py -k validator -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: studies/fly64_dose_overlap/generation.py::generate_dataset_for_dose (add NPZ-loading Stage 5 wrapper and new regression in tests/study/test_dose_overlap_generation.py::test_generate_dataset_validates_with_real_contract)
- Validate: pytest tests/study/test_dose_overlap_generation.py -k validator -vv
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`
2. Apply validator fix and add regression test (`tests/study/test_dose_overlap_generation.py::test_generate_dataset_validates_with_real_contract`) ensuring the stub-free path loads real NPZ data and asserts no unexpected kwargs.
3. `pytest tests/study/test_dose_overlap_generation.py -k validator -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run/green/pytest_phase_c_validator.log`
4. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run/green/pytest_highlights_preview_green.log`
5. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log`
6. After pipeline completes, archive Phase C/D/E/F/G logs + metrics under `analysis/` and update `summary/summary.md` with measured MS-SSIM/MAE deltas and highlights transcript; note any blockers if execution stops early.
7. Update docs/fix_plan.md Latest Attempt entry with execution results and reference the new artifacts.

Pitfalls To Avoid:
- Keep validator calls purely in-memory; do not reintroduce params.cfg mutations or skip DATA-001 checks.
- Ensure new pytest uses real validator imports; avoid silent mocks that hide signature drift.
- Guard against leaking artifacts outside the hub; delete any tmp files before concluding.
- Maintain Path objects when constructing split paths (TYPE-PATH-001) to avoid string math issues.
- Monitor the long-running pipeline; if any command exits non-zero, stop and record the failure instead of retrying automatically.
- Respect CONFIG-001: export AUTHORITATIVE_CMDS_DOC before every pytest/pipeline command.
- Do not touch stable core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).

If Blocked:
- Capture failing pytest or pipeline output under `red/` inside this hub, record the exact error signature, and update docs/fix_plan.md with `blocked` status plus unblock plan.
- Note validation failures (DATA-001) verbatim; include command + exit code and point to the specific log file.
- Escalate via galph_memory by switching focus only if validator fix requires upstream study design changes.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency remains required downstream; watch for import issues when rerunning pipeline.
- CONFIG-001 — Bridge legacy params and export AUTHORITATIVE_CMDS_DOC before invoking orchestrator/test commands.
- DATA-001 — Validation must enforce canonical NPZ keys/dtypes; failures are fatal and must be addressed immediately.
- TYPE-PATH-001 — Normalize filesystem interactions with `Path` to avoid `.exists` regressions.
- OVERSAMPLING-001 — Ensure neighbor_count ≥ gridsize² when validating dense view metadata.

Pointers:
- docs/fix_plan.md:35 — Current initiative status and attempts ledger.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1 — Validator recovery plan.
- studies/fly64_dose_overlap/generation.py:200 — Stage 5 validation call site requiring update.
- tests/study/test_dose_overlap_generation.py:90 — Existing orchestration test to adjust for new validator signature.
- docs/TESTING_GUIDE.md:268 — Phase G orchestrator workflow and highlights preview requirements.

Next Up (optional):
- Once dense run succeeds, schedule sparse-view rerun to complete the comparison matrix.

Doc Sync Plan (Conditional):
- After code/tests pass, run `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_generation.py -k validator --collect-only | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T150500Z/phase_g_dense_full_execution_real_run/collect/pytest_phase_c_validator_collect.log`
- Update `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` entry for Phase C validator tests with the new regression selector, referencing the collected log.

Mapped Tests Guardrail:
- Confirm `pytest tests/study/test_dose_overlap_generation.py -k validator -vv` collects the new regression before declaring victory; add the test first if collection fails.

Hard Gate:
- Do not consider the loop complete until Phase C validation passes with the new interface and the dense pipeline exits with code 0; otherwise document the block with precise logs.
