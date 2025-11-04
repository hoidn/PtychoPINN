Summary: Wire the Phase E training CLI to the real runner and prove a deterministic job executes via new tests and CLI evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5 — training runner integration
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv; pytest tests/study/test_dose_overlap_training.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094200Z/phase_e_training_e5/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5:
  - Test: Extend `tests/study/test_dose_overlap_training.py` with `test_training_cli_invokes_real_runner` (RED first) that monkeypatches the runner helper, executes CLI without `--dry-run`, and asserts the helper receives the resolved job + TrainingConfig; capture RED log at `.../red/pytest_training_cli_real_runner_red.log`.
  - Implement: studies/fly64_dose_overlap/training.py::execute_training_job — introduce a production runner helper that loads Phase C/D NPZ paths, constructs/bridges `TrainingConfig`, and delegates to the backend trainer (`train_cdi_model_torch` or `ptycho_train`), updating `main()` to default to this helper while remaining monkeypatchable for tests; emit logs/manifests under the passed artifact root.
  - Validate: Run `pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv` (tee → `.../green/pytest_training_cli_real_runner_green.log`), `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` (tee → `.../green/pytest_training_cli_suite_green.log`), and `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` (tee → `.../collect/pytest_collect.log`).
  - Run: If Phase C/D outputs are missing, regenerate via `python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly/fly001_transposed.npz --output-root tmp/phase_c_training_evidence` and `python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_training_evidence --output-root tmp/phase_d_training_evidence --artifact-root .../reports/2025-11-04T094200Z/phase_e_training_e5/overlap_cli`; then execute the training CLI without `--dry-run` for dose=1e3 baseline (`python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root .../reports/2025-11-04T094200Z/phase_e_training_e5/real_run --dose 1000 --view baseline --gridsize 1`) and archive stdout/log manifest in `real_run/`.
  - Doc: Mark plan row E5 `[x]` once evidence is captured, update `test_strategy.md` Phase E section with the new selector + execution proof, extend `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with the real-run selector/logs, and summarize Attempt #17 in `.../docs/summary.md` plus ledger entry.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:26-28 declares E5 pending to replace the stub runner and capture a deterministic run.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md Phase E future item 0 mandates a RED test for real-run wiring and evidence capture.
- docs/DEVELOPER_GUIDE.md:68-104 enforces CONFIG-001 bridging whenever training jobs are launched.
- docs/workflows/pytorch.md §12 and docs/pytorch_runtime_checklist.md outline the canonical PyTorch training invocation and artifact expectations.
- docs/findings.md#POLICY-001/#CONFIG-001/#DATA-001/#OVERSAMPLING-001 set the guardrails for backend selection, legacy bridging, dataset handling, and gridsize semantics.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094200Z/phase_e_training_e5/{red,green,collect,docs,real_run,overlap_cli}
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094200Z/phase_e_training_e5/red/pytest_training_cli_real_runner_red.log
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094200Z/phase_e_training_e5/green/pytest_training_cli_real_runner_green.log
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094200Z/phase_e_training_e5/green/pytest_training_cli_suite_green.log
- pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094200Z/phase_e_training_e5/collect/pytest_collect.log
- python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly/fly001_transposed.npz --output-root tmp/phase_c_training_evidence 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094200Z/phase_e_training_e5/real_run/phase_c_generation.log
- python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_training_evidence --output-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094200Z/phase_e_training_e5/overlap_cli 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094200Z/phase_e_training_e5/overlap_cli/phase_d_overlap.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094200Z/phase_e_training_e5/real_run --dose 1000 --view baseline --gridsize 1 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094200Z/phase_e_training_e5/real_run/training_cli_real_run.log

Pitfalls To Avoid:
- Do not regress CONFIG-001: ensure `update_legacy_dict` still executes exactly once per job before any backend loader.
- Keep the runner helper torch-optional friendly (import lazily or guard to satisfy POLICY-001 expectations).
- Avoid writing artifacts outside the provided `--artifact-root`; tests should confine side effects to tmp_path.
- Ensure CLI still supports `--dry-run` without executing the real runner (tests must cover both paths).
- Reset/restore `params.cfg` between tests to prevent state leakage after real-run assertions.
- Guard against missing Phase C/D datasets—fail fast with actionable errors rather than silent fallbacks.
- Do not hardcode absolute paths; derive from CLI arguments or fixtures for portability.
- Preserve ASCII-only manifest/log content for diffability.
- Keep test runtime under existing budgets; skip executing full training inside unit tests (use monkeypatch).
- Capture and archive stdout/stderr for the real-run CLI command under the artifact hub.

If Blocked:
- Capture failing command output in the artifact hub (e.g., `.../real_run/training_cli_real_run_failed.log`), annotate the blocker in summary.md, and log the issue as Attempt #17 in docs/fix_plan.md before pausing.

Findings Applied (Mandatory):
- CONFIG-001 — legacy bridge must precede any runner invocation.
- DATA-001 — reuse validator-backed datasets; refuse to train if NPZ contract fails.
- OVERSAMPLING-001 — preserve gridsize/view semantics (baseline=gs1, overlap=gs2) when selecting jobs.
- POLICY-001 — default to PyTorch backend; ensure imports respect torch-required policy.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:26
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:132
- docs/DEVELOPER_GUIDE.md:68
- docs/workflows/pytorch.md:245
- docs/pytorch_runtime_checklist.md:1

Next Up (optional):
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E6 — batch training across sparse/dense views once baseline run lands.

Doc Sync Plan:
- After GREEN, rerun `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` (already logged) and update `docs/TESTING_GUIDE.md` §Study Tests plus `docs/development/TEST_SUITE_INDEX.md` with the new real-run selector, referencing the artifact paths.
