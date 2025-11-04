Summary: Replace the Phase E runner stub with a real PyTorch training execution, validated by new tests and a captured baseline CLI run.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5 — training runner integration
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_delegates_to_pytorch_trainer -vv; pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv; pytest tests/study/test_dose_overlap_training.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5:
  - Test: Add `test_execute_training_job_delegates_to_pytorch_trainer` in `tests/study/test_dose_overlap_training.py` (RED first) that monkeypatches `ptycho_torch.workflows.components.train_cdi_model_torch`, exercises `execute_training_job` with fixture NPZs, and asserts the spy receives normalized data + TrainingConfig; log to `.../red/pytest_execute_training_job_red.log`.
  - Implement: studies/fly64_dose_overlap/training.py::execute_training_job — load Phase C/D NPZs via `PyTorchMemmapBridge` (or RawDataTorch delegate), call `train_cdi_model_torch` with CONFIG-001-compliant TrainingConfig, persist Lightning outputs (marker replaced with trainer metadata), and ensure CLI `main()` emits updated manifest entries.
  - Validate: Run `pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_delegates_to_pytorch_trainer -vv` → `.../green/pytest_execute_training_job_green.log`, `pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv` → `.../green/pytest_training_cli_real_runner_green.log`, `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` → `.../green/pytest_training_cli_suite_green.log`, and `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` → `.../collect/pytest_collect.log`.
  - Run: Regenerate inputs if absent using `python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly/fly001_transposed.npz --output-root tmp/phase_c_training_evidence` and `python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_training_evidence --output-root tmp/phase_d_training_evidence --artifact-root .../overlap_cli`; then execute the CLI real run for dose=1e3 baseline with deterministic knobs (`python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root .../real_run --dose 1000 --view baseline --gridsize 1 --max_epochs 1 --n_images 32 --batch_size 4 --accelerator cpu --logger none --disable_checkpointing --num-workers 0`) and capture stdout/stderr to `.../real_run/training_cli_real_run.log`.
  - Doc: Update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T120500Z/phase_e_training_e5/docs/summary.md`, mark plan/test_strategy/doc registries for E5 as complete, and append Attempt #17 in docs/fix_plan.md with artifact links.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:19 keeps E5 `[P]` pending real runner integration and deterministic run evidence.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163-166 demands a RED test plus real-run artifacts before closing E5.
- docs/DEVELOPER_GUIDE.md:68-104 (CONFIG-001) mandates config bridging before loader/trainer dispatch.
- docs/workflows/pytorch.md:245-312 and docs/pytorch_runtime_checklist.md outline canonical PyTorch training invocation and artifact expectations.
- docs/findings.md (POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001) codify backend requirement, config bridge order, dataset contracts, and gridsize semantics we must honor.

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
- Maintain POLICY-001 imports: guard torch/lightning imports with actionable errors, no silent fallbacks.
- Preserve DATA-001 invariants when loading NPZs (diffraction amplitude, complex64 Y).
- Ensure tests reset `params.cfg` between runs to avoid global state pollution.
- Avoid embedding long-running training in unit tests—use spies/monkeypatch to keep RED/green fast.
- Continue supporting `--dry-run`; do not trigger training when flag set.
- Capture CLI artifacts strictly under the provided `--artifact-root` to satisfy doc requirements.
- Honor OVERSAMPLING-001 (baseline gs1, dense/sparse gs2) when building job filters.
- Ensure PyTorch execution knobs (deterministic, accelerator) propagate to training call; default to CPU.

If Blocked:
- Archive failing logs under the artifact hub (e.g., `.../real_run/training_cli_real_run_failed.log`), summarize the blocker in summary.md, and record the stalled status in docs/fix_plan.md before halting work.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch backend is mandatory; ensure torch/lightning imports remain guarded but present.
- CONFIG-001 — TrainingConfig must bridge params.cfg before data loaders/trainer usage.
- DATA-001 — NPZ loading must validate canonical keys/dtypes; fail fast on mismatch.
- OVERSAMPLING-001 — Retain gridsize/view semantics (gs1 baseline, gs2 dense/sparse) when executing jobs.

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
