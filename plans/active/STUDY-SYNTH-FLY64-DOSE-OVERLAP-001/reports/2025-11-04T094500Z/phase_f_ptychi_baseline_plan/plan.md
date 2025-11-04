# Phase F — PtyChi LSQML Baseline (Planning)

## Context
- Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 (Synthetic fly64 dose/overlap study)
- Phase Goal: Produce pty-chi LSQML reconstructions for Phase D/E overlap outputs so Phase G comparisons can quantify TensorFlow PINN vs PyTorch vs pty-chi parity.
- Dependencies: Phase D overlap bundles (`phase_d_overlap_filtering`, `phase_d_cli_validation`), Phase E training outputs (`phase_e_training_e5_real_run_baseline`), config bridge guardrails (CONFIG-001), DATA-001-compliant NPZ layout, PyTorch backend policy (POLICY-001), reconstruction script `scripts/reconstruction/ptychi_reconstruct_tike.py` (requires tike + cupy optional dependencies per docs/TESTING_GUIDE.md §4.3).

### F0 — Test Infrastructure Prep
Goal: Lock in Phase F acceptance criteria and RED test harness before implementation.
Prereqs: Phase E artifacts linked in plan, test strategy ready to describe new selectors.
Exit Criteria: Test strategy updated with Phase F sections; failing pytest selector exercising new pty-chi job builder committed with log captured under RED/.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| F0.1 | Update `test_strategy.md` Phase F section with planned selectors, execution proof rules, and artifact expectations | [x] | Completed during Phase F0 setup — see `reports/2025-11-04T094500Z/phase_f_ptychi_baseline/{red,collect}/` and refreshed strategy §Phase F noting RED selector + artifact policy. |
| F0.2 | Author minimal RED test `tests/study/test_dose_overlap_reconstruction.py::test_build_ptychi_jobs_manifest` asserting new builder raises `NotImplementedError` until implementation | [x] | RED scaffold merged in Phase F0 with evidence at `reports/2025-11-04T094500Z/phase_f_ptychi_baseline/red/pytest_phase_f_red.log`. |

### F1 — PtyChi Job Orchestrator
Goal: Expose programmatic API for enumerating and executing LSQML reconstruction jobs (6 jobs per dose, 18 total).
Prereqs: RED test from F0 failing, plan + fixtures ready.
Exit Criteria: Builder + CLI helpers emit manifest with LSQML jobs per dose/view; GREEN tests and collection proof stored; CLI dry-run path validated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| F1.1 | Implement `studies/fly64_dose_overlap/reconstruction.py::build_ptychi_jobs` returning dataclasses with CLI args + artifact destinations | [x] | Completed in Attempt #F1 — manifest now emits 18 jobs (3 doses × 3 views × 2 splits) with DATA-001 validation; GREEN log in `reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/green/pytest_phase_f_green.log`. |
| F1.2 | Extend RED test to GREEN by asserting manifest structure (3 doses × {dense,sparse} GS2 + gs1 baseline) and CLI arg correctness; add `test_run_ptychi_job_invokes_script` using stub subprocess runner | [x] | Tests updated in Attempt #F1 (`tests/study/test_dose_overlap_reconstruction.py`), subprocess runner covered via mocks; see `reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/{green,collect}/`. |
| F1.3 | Add CLI entry `studies.fly64_dose_overlap.reconstruction:main` mirroring training CLI filters (`--dose`, `--view`, `--gridsize`, `--dry-run`) and emitting manifest/summary to artifact root | [x] | Completed in Attempt #F1.3 — CLI implemented with deterministic filtering, manifest + skip summary emission, and dry-run evidence under `reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/{cli,docs,red,green,collect}/`. |

### F2 — Deterministic Baseline Execution
Goal: Run LSQML reconstructions (CPU mode by default) and archive outputs.
Prereqs: CLI + tests green, skip summary manifests ready, environment validated for pty-chi dependencies.
Exit Criteria: At least one deterministic LSQML run completes (dense gs2 recommended), logs + reconstructions stored under `reports/<timestamp>/phase_f_ptychi_baseline/real_run/`; summary.md updated with findings.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| F2.1 | Execute CLI dry-run (`--dry-run`) to validate manifest and skip reporting before expensive runs | [x] | Completed in Attempt #78 — synthetic Phase C/D datasets generated (rng=123), CLI dry-run captured at `reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/cli/{dry_run.log,reconstruction_manifest.json,skip_summary.json}` with expected skip telemetry (16 skipped). |
| F2.2 | Execute real LSQML run for at least one view per dose (start with dense view, 100 epochs) | [x] | Completed in Attempt #80 — argparse fix landed, dense/train LSQML run succeeded (return code 0), logs + visualization under `reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/real_run/dose_1000/dense/train/`, manifest + skip summary captured. |
| F2.3 | Update `summary.md` with run outcomes, deviations, and next-step recommendations for Phase G comparisons | [x] | `reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/docs/summary.md` documents RED→GREEN flow, CLI evidence, and findings alignment; summary marks F2 dense/train baseline COMPLETE. |
| F2.4 | Extend LSQML evidence to dense/test split and sync CLI selector documentation | [x] | Completed in Attempt #81 (dense/test run) and Attempt #82 (doc sync) — dense/test LSQML executed successfully with evidence at `reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/{cli,real_run}/` (CLI transcript, manifest, skip summary, reconstruction logs). Script portability fixed (repo-relative path in `tests/scripts/test_ptychi_reconstruct_tike.py`). Documentation sync completed at `reports/2025-11-04T233500Z/phase_f_ptychi_baseline_f2_doc_sync/`: added Phase F selector snippets, CLI commands, and evidence references to `docs/TESTING_GUIDE.md` (lines 146-208) and registered `test_dose_overlap_reconstruction.py` with selector details, artifact paths, and deterministic CLI command in `docs/development/TEST_SUITE_INDEX.md` (line 61). Collection proof: 4 tests collected. |

## Deliverables & Artifacts
- All evidence stored under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline/` (subdirs: plan/, docs/, red/, green/, collect/, cli/, real_run/).
- RED/ GREEN pytest logs capturing new selectors (`pytest tests/study/test_dose_overlap_reconstruction.py -k ptychi -vv`).
- CLI transcripts (`dry_run.log`, `lsqml_dense.log`, etc.) and manifests/skip summaries.
- `summary.md` narrating acceptance proof and linking downstream Phase G expectations.

## References
- `docs/TESTING_GUIDE.md` §§2,4 — authoritative pytest commands and reconstruction workflow prerequisites.
- `docs/development/TEST_SUITE_INDEX.md` — register new Phase F selectors after GREEN.
- `docs/findings.md` CONFIG-001, DATA-001, POLICY-001, OVERSAMPLING-001 — compliance checkpoints for reconstruction pipeline.
- `specs/data_contracts.md` §§4–6 — reconstruction input/output expectations (complex64 Y patches, amplitude diffraction data).
