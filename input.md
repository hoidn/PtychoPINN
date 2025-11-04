Summary: Replace the Phase E runner stub with a real PyTorch backend execution plus baseline CLI evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5 — training runner integration
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_delegates_to_pytorch_trainer -vv; pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv; pytest tests/study/test_dose_overlap_training.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5:
  - Test: Add `test_execute_training_job_delegates_to_pytorch_trainer` in `tests/study/test_dose_overlap_training.py` (RED first) that monkeypatches `ptycho_torch.workflows.components.train_cdi_model_torch`, calls `execute_training_job` with minimal Phase C/D NPZ fixtures, and asserts the spy receives grouped tensors plus the CONFIG-001 bridged TrainingConfig; capture RED log at `.../red/pytest_execute_training_job_red.log`.
  - Implement: studies/fly64_dose_overlap/training.py::execute_training_job — replace the marker stub with real backend wiring: load datasets through `ptycho_torch.memmap_bridge.MemmapDatasetBridge`, hydrate PyTorch containers, call `train_cdi_model_torch` with CLI execution knobs, persist returned metrics/checkpoints into `job.artifact_dir`, and ensure CLI manifests/logs reflect execution (no placeholder text).
  - Validate: Run `pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_delegates_to_pytorch_trainer -vv` → `.../green/pytest_execute_training_job_green.log`, `pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv` → `.../green/pytest_training_cli_real_runner_green.log`, `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` → `.../green/pytest_training_cli_suite_green.log`, and `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` → `.../collect/pytest_collect.log`.
  - Run: Regenerate inputs if absent with `python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly/fly001_transposed.npz --output-root tmp/phase_c_training_evidence` and `python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_training_evidence --output-root tmp/phase_d_training_evidence --artifact-root .../overlap_cli`; then execute the CLI baseline run (`python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root .../real_run --dose 1000 --view baseline --gridsize 1 --max_epochs 1 --n_images 32 --batch_size 4 --accelerator cpu --logger none --disable_checkpointing --num-workers 0`) and tee output to `.../real_run/training_cli_real_run.log`.
  - Doc: Update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/docs/summary.md`, flip plan/test_strategy E5 rows to `[x]`, refresh registry docs, and append Attempt #18 in docs/fix_plan.md with artifact links.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:19 keeps E5 `[P]` until runner integration + baseline evidence land.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163-166 requires a RED test plus deterministic CLI run before closing E5.
- docs/DEVELOPER_GUIDE.md:68-104 mandates CONFIG-001 bridging prior to any data loading or training calls.
- docs/workflows/pytorch.md:245-312 and docs/pytorch_runtime_checklist.md enumerate approved PyTorch training invocation knobs and artifact expectations.
- docs/findings.md (POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001) codify backend requirement, config bridge order, dataset contracts, and gridsize semantics that must remain intact.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/{red,green,collect,docs,real_run,overlap_cli,tmp}
- pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_delegates_to_pytorch_trainer -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/red/pytest_execute_training_job_red.log
- pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_delegates_to_pytorch_trainer -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/green/pytest_execute_training_job_green.log
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/green/pytest_training_cli_real_runner_green.log
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/green/pytest_training_cli_suite_green.log
- pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/collect/pytest_collect.log
- python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly/fly001_transposed.npz --output-root tmp/phase_c_training_evidence 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/real_run/phase_c_generation.log
- python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_training_evidence --output-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/overlap_cli 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/overlap_cli/phase_d_overlap.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/real_run --dose 1000 --view baseline --gridsize 1 --max_epochs 1 --n_images 32 --batch_size 4 --accelerator cpu --logger none --disable_checkpointing --num-workers 0 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/real_run/training_cli_real_run.log

Pitfalls To Avoid:
- Keep CONFIG-001 ordering: `update_legacy_dict` must execute once before any RawData/Lightning call.
- Preserve POLICY-001 compliance: import torch/lightning without optional guards; raise actionable error if missing.
- Maintain DATA-001 invariants when loading NPZs (diffraction amplitude, `Y` complex64).
- Reset `params.cfg` via fixtures when monkeypatching `update_legacy_dict` to avoid cross-test contamination.
- Keep tests lightweight—use spies instead of real training loops; rely on CLI run for execution proof.
- Respect `--dry-run` semantics so unit tests can short-circuit execution when requested.
- Write all execution logs/manifests under the artifact hub; avoid littering repo root or tmp files.
- Propagate OVERSAMPLING-001 assumptions (gs1 baseline vs gs2 overlap) without changing job matrix.
- Thread deterministic knobs (`deterministic`, `accelerator`, `num_workers`) through to `train_cdi_model_torch`.

If Blocked:
- Archive failing logs under the artifact hub (e.g., `.../real_run/training_cli_real_run_failed.log`), summarize the blocker in summary.md, and record the stalled status in docs/fix_plan.md before halting work.

Findings Applied (Mandatory):
- POLICY-001 — Ensure PyTorch backend is exercised (real trainer call, no optional fallbacks).
- CONFIG-001 — TrainingConfig bridge occurs before any memmap bridge or trainer invocation.
- DATA-001 — Fixture NPZs and runtime loaders must honor canonical keys/dtypes.
- OVERSAMPLING-001 — Preserve gridsize semantics across baseline and overlap runs.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:19
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163
- docs/DEVELOPER_GUIDE.md:68
- docs/workflows/pytorch.md:245
- docs/pytorch_runtime_checklist.md:1

Next Up (optional):
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E6 — batch training across dense/sparse views once baseline run lands.

Doc Sync Plan:
- After GREEN, rerun `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` (log already captured) and update `docs/TESTING_GUIDE.md` §2 plus `docs/development/TEST_SUITE_INDEX.md` with the new execute_training_job selector and real-run evidence paths.
