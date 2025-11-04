Summary: Finish Phase E5 by switching the runner to MemmapDatasetBridge and capturing deterministic CLI evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5 — training runner integration
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_delegates_to_pytorch_trainer -vv; pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv; pytest tests/study/test_dose_overlap_training.py --collect-only -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.E5:
  - Test: Extend `tests/study/test_dose_overlap_training.py::test_execute_training_job_delegates_to_pytorch_trainer` (RED) to spy on `ptycho_torch.memmap_bridge.MemmapDatasetBridge`, asserting it is instantiated for both train/test datasets and that its RawDataTorch payload is passed to `train_cdi_model_torch`; log failure to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/red/pytest_execute_training_job_memmap_red.log`.
  - Implement: studies/fly64_dose_overlap/training.py::execute_training_job — replace direct `load_data` usage with an injectable MemmapDatasetBridge factory (defaulting to `MemmapDatasetBridge`), ensure bridge instances hydrate RawDataTorch once, forward the resulting containers to `train_cdi_model_torch`, and augment the manifest payload with returned metrics/checkpoints.
  - Validate: Run `pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_delegates_to_pytorch_trainer -vv` → `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/green/pytest_execute_training_job_memmap_green.log`, `pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv` → `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/green/pytest_training_cli_real_runner_green.log`, `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` → `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/green/pytest_training_cli_suite_green.log`, and `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` → `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/collect/pytest_collect.log`.
  - Run: Refresh datasets if missing (`python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly/fly001_transposed.npz --output-root tmp/phase_c_training_evidence`; `python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_training_evidence --output-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/real_run/overlap_cli`) and execute the real CLI baseline (`python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/real_run --dose 1000 --view baseline --gridsize 1 --max_epochs 1 --n_images 32 --batch_size 4 --accelerator cpu --deterministic --num-workers 0 --logger csv`) capturing stdout to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/real_run/training_cli_real_run.log`, manifest diff to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/real_run/training_manifest_after.json`, and lightning logs/checkpoints under the same hub.
  - Doc: Update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/docs/summary.md`, mark plan/test_strategy E5 row `[x]`, refresh docs/TESTING_GUIDE.md §2 and docs/development/TEST_SUITE_INDEX.md with the memmap-backed selector evidence, and record Attempt #20 in docs/fix_plan.md pointing to the new artifact hub.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:19 keeps E5 `[P]` until real runner evidence plus deterministic run land.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163-166 mandates RED coverage and deterministic CLI execution before closing Phase E5.
- docs/fix_plan.md:33-55 (Attempt #19) notes failing green logs and missing real-run outputs that must be replaced with passing evidence.
- docs/DEVELOPER_GUIDE.md:68-104 enforces CONFIG-001 sequencing when MemmapDatasetBridge instantiates RawDataTorch.
- docs/findings.md (POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001) remain binding during runner upgrades and baseline execution.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/{red,green,collect,docs,real_run}
- pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_delegates_to_pytorch_trainer -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/red/pytest_execute_training_job_memmap_red.log
- (after implementation) rerun the same selector teeing to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/green/pytest_execute_training_job_memmap_green.log
- pytest tests/study/test_dose_overlap_training.py::test_training_cli_invokes_real_runner -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/green/pytest_training_cli_real_runner_green.log
- pytest tests/study/test_dose_overlap_training.py -k training_cli -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/green/pytest_training_cli_suite_green.log
- pytest tests/study/test_dose_overlap_training.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/collect/pytest_collect.log
- python -m studies.fly64_dose_overlap.generation --base-npz datasets/fly/fly001_transposed.npz --output-root tmp/phase_c_training_evidence 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/real_run/phase_c_generation.log
- python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_training_evidence --output-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/real_run/overlap_cli 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/real_run/phase_d_overlap.log
- python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_training_evidence --phase-d-root tmp/phase_d_training_evidence --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/real_run --dose 1000 --view baseline --gridsize 1 --max_epochs 1 --n_images 32 --batch_size 4 --accelerator cpu --deterministic --num-workers 0 --logger csv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/real_run/training_cli_real_run.log
- Capture lightning logs + checkpoints under plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/real_run/lightning_logs/ and hash manifest changes in summary.md

Pitfalls To Avoid:
- Do not reuse the stale 2025-11-04T120500Z green logs showing failing assertions—replace them with new passing artifacts.
- Ensure MemmapDatasetBridge respects CONFIG-001 by passing the same TrainingConfig; avoid double-bridging that mutates params.cfg mid-run.
- Keep DATA-001 compliance when generating fixtures; no ad-hoc NPZ shortcuts beyond existing validators.
- Maintain monkeypatch cleanup in tests to prevent swapping bridge factory globally.
- Run CLI with deterministic knobs and '--logger csv' so output_dir contains metrics; avoid '--logger none'.
- Guard against accidental GPU selection by forcing `--accelerator cpu` and clearing CUDA env if necessary.
- Place all new evidence inside the 2025-11-04T133500Z hub; no stray tmp files at repo root.
- Leave existing Phase D artifacts untouched; regeneration should occur under tmp/ with cleanup after run.

If Blocked:
- Save failing pytest/CLI transcripts under the new artifact hub (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/red/` or `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T133500Z/phase_e_training_e5/real_run/training_cli_real_run_failed.log`), summarize the blocker in summary.md, and append a blocked Attempt entry to docs/fix_plan.md before stopping.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch backend must execute without optional fallbacks; confirmed via real runner + CLI baseline.
- CONFIG-001 — Preserve update_legacy_dict ordering when MemmapDatasetBridge instantiates RawDataTorch and when run_training_job bridges configs.
- DATA-001 — NPZ fixtures and runtime loaders must satisfy canonical keys/dtypes during memmap hydration.
- OVERSAMPLING-001 — Keep gridsize/neighbor_count semantics unchanged while wiring the runner and manifest.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T053500Z/phase_e_training_plan/plan.md:19
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:163
- docs/fix_plan.md:33
- studies/fly64_dose_overlap/training.py:292
- tests/study/test_dose_overlap_training.py:696

Doc Sync Plan: None (selectors unchanged; just refresh evidence and summaries once GREEN).
