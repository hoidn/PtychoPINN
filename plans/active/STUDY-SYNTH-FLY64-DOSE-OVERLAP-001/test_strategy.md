**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001

## 1. Framework & Style
- Use pytest-style tests and lightweight validation scripts (no unittest mix-in) to keep parity with project guidance.
- Tests focus on data contracts, selection invariants, and execution proofs; heavy training is validated via logs/CSVs captured as artifacts.

## 2. CI/CD & Optional Dependencies
- Pty-chi is optional and may be unavailable in CI; mark pty-chi validation as “evidence via artifact” (manual run) with justification in summary.md.
- Torch is required in project policy but not exercised directly in this TensorFlow-based study; ensure tests skip gracefully if torch-related paths are unneeded.

## 3. Test Tiers
- Unit: dataset contract checks (keys/dtypes), dose normalization ranges, overlap-metric calculations.
- Integration: end-to-end dataset creation + split + grouping + filtering (no training), ensuring artifacts produced.
- Smoke: minimal training invocation (short epochs) to verify plumbing, not performance.

## 4. Execution Proof
- For any test-related task, collect:
  - pytest log at `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/pytest.log`
  - results summary at `.../summary.md` with pass/fail counts and any skips justified
  - evidence CSVs from compare_models (MS-SSIM etc.)
- Acceptable SKIP reasons: pty-chi unavailable; GPU-specific long tests; long benchmarks.

## 5. Mock/Stub Strategy
- For pty-chi steps in CI, rely on recorded outputs (aligned NPZs) when available; otherwise skip with justification.
- For simulator inputs, use small synthetic objects/probes when sanity checking structure.

## 6. Specific Checks

### Phase A — Design Constants (COMPLETE)
**Test module:** `tests/study/test_dose_overlap_design.py`
**Status:** 3/3 PASSED

Validates:
- Dose list [1e3, 1e4, 1e5]
- Gridsizes {1, 2} with neighbor_count=7
- K ≥ C constraint (K=7 ≥ C=4 for gridsize=2)
- Subsampling fractions `s_img ∈ {1.0, 0.8, 0.6}` and requested group counts `n_groups ∈ {512, 768, 1024}`
- Disc-overlap metric defaults per `specs/overlap_metrics.md` (Metric 1 gs2-only, Metric 2 global, Metric 3 group COM)
- Probe diameter derived from patch size (N=128) → `probe_diameter_px = 0.6 * N` for Metric calculations
- Train/test split axis='y'
- RNG seeds: simulation=42, grouping=123, subsampling=456
- MS-SSIM config: sigma=1.0, emphasize_phase=True

**Execution proof:**
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T021500Z/green/pytest_green.log`
- All tests PASSED; no SKIPs

### Phase B — Dataset Contract Validation (COMPLETE)
**Status:** 11/11 tests PASSED — RED/GREEN/collect evidence captured
- Working Plan: `reports/2025-11-04T025541Z/phase_b_test_infra/plan.md`
- Validator: `studies/fly64_dose_overlap/validation.py::validate_dataset_contract` checks ✅
  - DATA-001 NPZ keys/dtypes and amplitude requirement (`diffraction` as amplitude float32, complex64 fields for object/probe) per `docs/specs/spec-ptycho-interfaces.md`.
  - Train/test split axis `'y'` using StudyDesign constants.
  - Geometry metadata (`geometry_acceptance_bound`, `effective_min_acceptance`) recorded for observability (no abort gate).
  - Oversampling preconditions (neighbor_count ≥ gridsize²) aligned with OVERSAMPLING-001.
- Tests: `tests/study/test_dose_overlap_dataset_contract.py` ✅
  - 11 tests covering happy path, missing keys, dtype violations, shape mismatches, overlap-metric metadata, oversampling constraints
  - All tests PASSED in GREEN phase (0.96s runtime)
  - RED phase: stub implementation → 1 FAILED (NotImplementedError)
  - GREEN phase: full implementation → 11 PASSED
  - Logs: RED and GREEN runs stored under `reports/2025-11-04T025541Z/phase_b_test_infra/{red,green}/pytest.log`; collect-only log under `collect/pytest_collect.log`.
- Documentation: Updated `summary.md`, `implementation.md`, and this strategy with validator coverage; recorded findings adherence (CONFIG-001, DATA-001, OVERSAMPLING-001). ✅

### Phase D — Overlap Metrics (SPEC ADOPTED; tests to update)
**Test module:** `tests/study/test_dose_overlap_overlap.py`
**Selectors (Planned):**
- Unit: disc-overlap function analytic cases (d=0 → 1.0; d=R; d≥D → 0.0).
- Unit: Metric 1 (gs=2 only) central-to-neighbors average per sample; global mean; skipped for gs=1.
- Unit: Metric 2 (global) with exact coord dedup; KNN average; global mean.
- Unit: Metric 3 (group↔group) COM computation; neighbor detection via `d < probe_diameter_px`; global mean.
- Integration: explicit `--gridsize`, `--s_img`, `--n_groups`, `--neighbor-count`, `--probe-diameter-px` produce per-split metrics JSON + aggregated bundle per `docs/specs/overlap_metrics.md`. No spacing/packing gates; non-degenerate runs do not abort on “insufficient spacing”.
**Bundle fields (Planned):**
- `gridsize`, `s_img`, `n_groups`, `neighbor_count` (default 6), `probe_diameter_px`, RNG seeds.
- `metric_1_group_based_avg` (gs=2 only), `metric_2_image_based_avg`, `metric_3_group_to_group_avg`.
**Notes:**
- Dense/sparse labels are deprecated for this study; tests must stop asserting spacing-acceptance fields.
**Execution Proof (to be collected post-implementation):**
- New RED/GREEN logs for metric unit tests and integration bundle generation under a dedicated `reports/.../phase_d_overlap_metrics/` hub.

### Phase E — Train PtychoPINN (In Progress)
**Test module:** `tests/study/test_dose_overlap_training.py` ✅
**Selectors (Active):**
- `pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_matrix -vv` (job enumeration: PASSED GREEN during Attempt #13; evidence in `reports/2025-11-04T060200Z/phase_e_training_e1/green/pytest_green.log`)
- `pytest tests/study/test_dose_overlap_training.py::test_run_training_job_invokes_runner -vv` (run helper invocation: RED→GREEN TDD cycle during Attempt #14; RED evidence in `reports/2025-11-04T070000Z/phase_e_training_e2/red/pytest_run_helper_invokes_runner_red.log`, GREEN in `.../green/pytest_run_helper_green.log`)
- `pytest tests/study/test_dose_overlap_training.py::test_run_training_job_dry_run -vv` (dry-run mode: RED→GREEN TDD cycle during Attempt #14; RED evidence in `reports/2025-11-04T070000Z/phase_e_training_e2/red/pytest_run_helper_dry_run_red.log`, GREEN in `.../green/pytest_run_helper_green.log`)
- `pytest tests/study/test_dose_overlap_training.py -k run_training_job -vv` (combined run helper selector: 2 PASSED in 3.19s; evidence in `reports/2025-11-04T070000Z/phase_e_training_e2/green/pytest_run_helper_green.log`)
- `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` (collection proof: 3 tests collected post-E3; log in `reports/2025-11-04T070000Z/phase_e_training_e2/collect/pytest_collect.log`)

**Coverage Delivered (E1):**
- Job matrix enumeration: Validate that `build_training_jobs()` produces exactly 9 jobs per dose (3 doses × 3 variants):
  - Baseline: gs1 using Phase C patched_{train,test}.npz
  - Dense overlap: gs2 using Phase D dense_{train,test}.npz
  - Sparse overlap: gs2 using Phase D sparse_{train,test}.npz
- Job metadata validation: Each `TrainingJob` dataclass must contain:
  - dose (float, from StudyDesign.dose_list)
  - view (str, one of {"baseline", "dense", "sparse"})
  - gridsize (int, 1 or 2)
  - train_data_path (str, validated existence)
  - test_data_path (str, validated existence)
  - artifact_dir (Path, deterministic from dose/view/gridsize)
  - log_path (Path, derived from artifact_dir)
- Dependency injection: Tests use `tmp_path` fixture to fabricate Phase C and Phase D NPZ trees with minimal valid datasets (DATA-001 keys only, small arrays).
- Failure handling: Test behavior when Phase C or Phase D artifacts are missing (expect FileNotFoundError with descriptive message).
- CONFIG-001 guard: Ensure `build_training_jobs()` does NOT call `update_legacy_dict`; that bridge remains in execution helper for E3.

**Coverage Delivered (E3):**
- CONFIG-001 compliance: `run_training_job()` directly updates `params.cfg` with essential fields (gridsize, N) before runner invocation to maintain legacy bridge ordering.
- Runner invocation: Tests spy on injected stub runner to verify it receives correct kwargs (`config`, `job`, `log_path`) and that the result is properly returned.
- Directory creation: Both artifact_dir and log_path parent directories are created with `mkdir(parents=True, exist_ok=True)` before any execution.
- Log file initialization: `log_path` is touched before runner invocation to ensure downstream tools find valid log files even if runner fails.
- Dry-run mode: When `dry_run=True`, runner invocation is skipped, a summary dict is returned, and a dry-run marker is written to the log file.
- Error handling: Runner exceptions are allowed to propagate after ensuring log directories are intact for post-mortem debugging.

**Execution Proof (E3):**
- RED logs: `reports/2025-11-04T070000Z/phase_e_training_e2/red/pytest_run_helper_invokes_runner_red.log` (ImportError: cannot import 'run_training_job'), `pytest_run_helper_dry_run_red.log` (same ImportError).
- GREEN logs: `reports/2025-11-04T070000Z/phase_e_training_e2/green/pytest_run_helper_green.log` (2 PASSED in 3.19s).
- Collection: `reports/2025-11-04T070000Z/phase_e_training_e2/collect/pytest_collect.log` (3 tests collected).
- Dry-run demonstration: `reports/2025-11-04T070000Z/phase_e_training_e2/dry_run/run_helper_dry_run_preview.txt` (summary dict with all required keys).
- Stub runner demonstration: `reports/2025-11-04T070000Z/phase_e_training_e2/runner/run_helper_stub.log` (runner received config, job, log_path kwargs and returned result).

**Selectors (Active — E4 CLI entrypoint):**
- `pytest tests/study/test_dose_overlap_training.py::test_run_training_job_invokes_runner -vv` (tightened in E4: now asserts `update_legacy_dict` receives `TrainingConfig`; RED evidence in `reports/2025-11-04T081500Z/phase_e_training_cli/red/pytest_run_job_bridging_red.log`, GREEN in `.../green/pytest_run_job_bridging_green.log`)
- `pytest tests/study/test_dose_overlap_training.py::test_training_cli_filters_jobs -vv` (CLI job filtering: RED→GREEN TDD cycle; RED in `reports/2025-11-04T081500Z/phase_e_training_cli/red/pytest_training_cli_filters_red.log`, GREEN in `.../green/pytest_training_cli_green.log`)
- `pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv` (manifest emission: RED→GREEN TDD cycle; RED in `reports/2025-11-04T081500Z/phase_e_training_cli/red/pytest_training_cli_manifest_red.log`, GREEN in `.../green/pytest_training_cli_green.log`)
- `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` (aggregated CLI selector: 2 PASSED in 3.21s; evidence in `reports/2025-11-04T081500Z/phase_e_training_cli/green/pytest_training_cli_green.log`)
- `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` (collection proof: 5 tests collected post-E4; log in `reports/2025-11-04T081500Z/phase_e_training_cli/collect/pytest_collect.log`)

**Coverage Delivered (E4 CLI entrypoint):**
- CLI parser: Accepts `--phase-c-root`, `--phase-d-root`, `--artifact-root` (required), optional `--dose`, `--view`, `--gridsize`, and `--dry-run`; emits informative messages for filtering results.
- Job selection: Calls `build_training_jobs()` once to enumerate full matrix (9 jobs), then filters per CLI selectors; handles gracefully when filters match nothing (prints warning and exits).
- CONFIG-001 bridge upgrade: `run_training_job()` now constructs `TrainingConfig` with `ModelConfig(gridsize=job.gridsize)` and calls `update_legacy_dict(params.cfg, config)` before runner invocation; tests spy on `update_legacy_dict` to assert `TrainingConfig` instance passed.
- Manifest emission: CLI emits `training_manifest.json` under `--artifact-root` with timestamp, Phase C/D roots, filters applied, dry-run flag, and per-job metadata (dose, view, gridsize, dataset paths, log paths, results).
- JSON serialization hygiene: Path objects converted to strings before JSON serialization to avoid `TypeError: Object of type PosixPath is not JSON serializable`.
- CLI dry-run demonstration: `python -m studies.fly64_dose_overlap.training --phase-c-root ... --dose 1000 --view baseline --dry-run` executed successfully; output captured in `reports/2025-11-04T081500Z/phase_e_training_cli/dry_run/training_cli_dry_run.txt`.

**Execution Proof (E4):**
- RED logs: `reports/2025-11-04T081500Z/phase_e_training_cli/red/pytest_training_cli_filters_red.log` (AttributeError: module has no attribute 'main'), `pytest_training_cli_manifest_red.log` (same), `pytest_run_job_bridging_red.log` (AssertionError: update_legacy_dict not called).
- GREEN logs: `reports/2025-11-04T081500Z/phase_e_training_cli/green/pytest_training_cli_green.log` (2 CLI tests PASSED in 3.21s), `pytest_run_job_bridging_green.log` (1 test PASSED in 3.01s).
- Collection: `reports/2025-11-04T081500Z/phase_e_training_cli/collect/pytest_collect.log` (5 tests collected: test_build_training_jobs_matrix, test_run_training_job_invokes_runner, test_run_training_job_dry_run, test_training_cli_filters_jobs, test_training_cli_manifest_and_bridging).
- Full suite: `reports/2025-11-04T081500Z/phase_e_training_cli/green/pytest_full_suite.log` (full regression evidence pending completion).
- CLI dry-run: `reports/2025-11-04T081500Z/phase_e_training_cli/dry_run/training_cli_dry_run.txt` (successfully enumerated 9 jobs, filtered to 1 baseline job for dose=1000, emitted manifest).
- Manifest artifact: `reports/2025-11-04T081500Z/phase_e_training_cli/artifacts/training_manifest.json` (valid JSON with timestamp, filters, and job metadata).
- Logging & artifacts: CLI writes per-job `train.log` (baseline + dense/sparse) under `<artifact_root>/phase_e_training/...`, captures stdout to `cli/training_cli.log`, and emits `training_manifest.json` summarizing executed jobs.
- Failure handling: Non-existent dose/view combinations should exit with status 1 and descriptive message (covered via stub runner raising exception surfaced by CLI).

**Execution Proof Expectations (E4):**
- RED logs for each new selector under `reports/2025-11-04T081500Z/phase_e_training_cli/red/`
- GREEN logs under `.../green/` (`pytest_training_cli_green.log`, `training_cli_stdout.log`)
- Collection proof stored at `.../collect/pytest_collect.log`
- CLI dry-run transcript under `.../dry_run/training_cli_dry_run.txt`
- Manifest snapshot saved to `.../artifacts/training_manifest.json` with hash recorded in summary.md

**Findings Alignment:**
- CONFIG-001: Job builder remains pure (no params.cfg mutation); legacy bridge deferred to `run_training_job()` in E3.
- DATA-001: Test fixtures must create NPZs with canonical keys (diffraction, objectGuess, probeGuess, Y, coords, filenames) per specs/data_contracts.md:190-260.
- OVERSAMPLING-001: Grouped jobs (gs2) assume neighbor_count=7 from Phase D outputs; tests validate gridsize field matches view expectation.

### Phase E5 — Training Runner Integration & Skip Summary (COMPLETE)

**Status:** COMPLETE — MemmapDatasetBridge wiring, skip-aware manifest, skip summary persistence, and deterministic CLI evidence delivered

**Selectors (Active — E5 skip summary persistence):**
- `pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv` (skip summary validation: RED→GREEN TDD cycle)
- `pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_delegates_to_pytorch_trainer -vv` (MemmapDatasetBridge instantiation)
- `pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_skips_missing_view -vv` (graceful skip with `allow_missing_phase_d=True`)
- `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` (all CLI tests: 3 PASSED)
- `pytest tests/study/test_dose_overlap_training.py --collect-only -vv` (collection proof: 8 tests)

**Coverage Delivered (E5):**
- **MemmapDatasetBridge Integration:** `execute_training_job()` (training.py:373-387) replaced `load_data` stub with `MemmapDatasetBridge` instantiation, extracting `raw_data_torch` payload for trainer. CONFIG-001 compliance maintained: `update_legacy_dict(params.cfg, config)` called before bridge construction.
- **Path Alignment:** Fixed Phase D layout mismatch (`dose_{dose}/{view}/{view}_{split}.npz` structure) and introduced `allow_missing_phase_d` parameter so CLI can skip absent overlap views while tests stay strict.
- **Skip Metadata Collection:** `build_training_jobs()` accepts optional `skip_events` list; when views missing and `allow_missing_phase_d=True`, appends `{dose, view, reason}` dicts (training.py:196-213).
- **Skip Summary Persistence:** CLI `main()` emits standalone `skip_summary.json` with schema `{timestamp, skipped_views: [{dose, view, reason}], skipped_count}`, records relative path in `training_manifest.json` under `skip_summary_path` field, and prints human-readable skip count (training.py:692-731).
- **Schema Validation:** `test_training_cli_manifest_and_bridging` asserts skip summary file exists, validates JSON structure, and confirms consistency between standalone skip summary and manifest inline fields.

**Execution Proof (E5):**
- **RED logs (Phase E5.5):** `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/red/pytest_training_cli_manifest_red.log` (AssertionError: skip_summary.json not found)
- **GREEN logs (Phase E5.5):**
  - `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/green/pytest_training_cli_manifest_green.log` (1 PASSED in 7.48s)
  - `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/green/pytest_training_cli_skips_green.log` (1 PASSED in 7.68s)
  - `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/green/pytest_training_cli_suite_green.log` (3 PASSED in 6.55s)
- **Collection proof:** `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/collect/pytest_collect.log` (8 tests collected)
- **Real CLI run (dry-run mode):** `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run/training_cli_real_run.log` (6 jobs enumerated, 3 sparse views skipped, 2 baseline/dense jobs executed, manifest + skip_summary.json written)
- **Artifacts:** `reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run/{training_manifest.json, skip_summary.json}` plus formatted copy at `docs/skip_summary_pretty.json`

**Deterministic CLI Baseline Command:**
```bash
python -m studies.fly64_dose_overlap.training \
  --phase-c-root tmp/phase_c_training_evidence \
  --phase-d-root tmp/phase_d_training_evidence \
  --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/real_run \
  --dose 1000 \
  --dry-run
```

**Findings Alignment:**
- CONFIG-001: Builder remains pure; `skip_events` accumulated client-side in CLI `main()`.
- DATA-001: Phase C/D regeneration reuses canonical NPZ contract; bridge validates keys/dtypes.
- POLICY-001: PyTorch backend required; `backend='pytorch'` set in training config.
- OVERSAMPLING-001: Skip reasons reference overlap metric emission and oversampling bounds from Phase D filtering.

**Documentation Registry Updates (Attempt #26):**
- `docs/TESTING_GUIDE.md:110-142` — Added Phase E5 skip summary narrative, updated selector snippets, and documented deterministic CLI dry-run command with artifact path.
- `docs/development/TEST_SUITE_INDEX.md:60` — Updated `test_dose_overlap_training.py` row with Phase E5 coverage (skip summary file, schema, manifest consistency) and evidence pointer.
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:138-184` — Expanded Phase E section from placeholder to full deliverables summary (E1-E5.5) with artifact hubs, test coverage, and CLI command.
- This file (test_strategy.md) — Replaced "Future Phases (Pending)" with Phase E5 COMPLETE section documenting selectors, coverage, execution proof, and findings alignment.

#### Phase E — Training Loss Guardrail (2025-11-12)
- Maintain a blessed manifest at `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/reference/training_manifest.json`. Populate it by copying the manifest from the first dense gs2 or baseline gs1 run that passes visual inspection; refresh it whenever a newer run becomes the quality baseline.
- Every real run must execute `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_training_loss.py --reference plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/reference/training_manifest.json --candidate <current_manifest> --dose <value> --view <value> --gridsize <value> [--tolerance 0.25]`.
- Archive the checker stdout/stderr under the same reports hub (e.g., `analysis/check_training_loss.log`) and cite it in summary.md. Treat any non-zero exit, missing `final_loss`, non-finite value, or tolerance breach as a BLOCKED state for the loop.
- How-To Map requirement: list the checker invocation as a mapped “selector” alongside pytest commands so Galph/Ralph can reference the artifact path and tolerance used.

### Phase F — PtyChi LSQML Baseline (COMPLETE)
**Test module:** `tests/study/test_dose_overlap_reconstruction.py`
**Selectors (Active):**
- `pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv` — manifest builder + runner coverage (GREEN Attempt #F1; logs in `reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/green/pytest_phase_f_green.log`)
- `pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv` — collection proof (GREEN Attempt #F1; log in `reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/collect/pytest_phase_f_collect.log`)
- `pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_filters_dry_run -vv` — Phase F1.3 CLI filter + manifest emission (GREEN Attempt #F1.3; log in `reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/green/pytest_phase_f_cli_green.log`)

**Selectors (Active — F2):**
- `pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv` — Phase F2 execution harness (non-dry-run path with mocked subprocess; RED→GREEN in Attempt #78; evidence in `reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/green/pytest_phase_f_cli_exec_green.log`)
- `pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv` — Full Phase F suite (all selectors GREEN)
- `pytest tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments -vv` — Script-level CLI parser coverage (Attempt #80 GREEN log at `reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/green/pytest_ptychi_cli_input_green.log`; TODO: convert absolute repo path to relative import)
- `pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv` — Sparse metadata assertions (Attempt #88 validation; GREEN log at `reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/green/pytest_phase_f_sparse_green.log`)

**Coverage Delivered (F0–F1.2):**
- Test strategy Phase F section documented RED/GREEN artifact policy and selector expectations (reports/2025-11-04T094500Z/phase_f_ptychi_baseline/).
- RED scaffold `test_build_ptychi_jobs_manifest` captured NotImplementedError evidence before implementation (`reports/2025-11-04T094500Z/phase_f_ptychi_baseline/red/pytest_phase_f_red.log`).
- GREEN implementation (Attempt #F1) now asserts 18 jobs (3 doses × 3 views × 2 splits), deterministic ordering, artifact root layout, and CLI payload (`--algorithm LSQML`, `--num-epochs 100`, DATA-001 NPZ paths).
- New `test_run_ptychi_job_invokes_script` exercises `run_ptychi_job` dry-run + mocked subprocess path, ensuring CONFIG-001 safety (builder remains pure) and argument propagation.
- Added script-level unit test `tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments` covering argparse defaults vs overrides; RED→GREEN evidence stored under `reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/{red,green}/`.
- CLI dry-run selector promoted to ACTIVE in Attempt #F1.3 validating filter combos, manifest + skip summary emission, and artifact logging under `reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/`.
- Sparse metadata assertions validated in Attempt #88 — no code changes required; tests confirmed GREEN and manifest telemetry present (`selection_strategy`, `acceptance_rate`, `geometry_acceptance_bound`, `n_accepted`, `n_rejected`).

**Execution Proof (F0–F3):**
- RED log: `reports/2025-11-04T094500Z/phase_f_ptychi_baseline/red/pytest_phase_f_red.log`
- GREEN log: `reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/green/pytest_phase_f_green.log`
- Collect-only proof: `reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/collect/pytest_phase_f_collect.log`
- CLI dry-run evidence: `reports/2025-11-04T130000Z/phase_f_ptychi_baseline_f1_cli/{red/pytest_phase_f_cli_red.log,green/pytest_phase_f_cli_green.log,collect/pytest_phase_f_cli_collect.log}`
- Sparse execution suite: `reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/{red/pytest_phase_f_sparse_red.log,green/pytest_phase_f_sparse_suite_green.log,collect/pytest_phase_f_sparse_collect.log}`
- Real-run logs: `reports/2025-11-04T133218Z/phase_f_ptychi_baseline_f3_metadata_recovery/cli/{sparse_train.log,sparse_test.log}` with manifests copied per split.

**Findings Alignment:**
- CONFIG-001: Job builder stays side-effect free; CLI/runner will call `update_legacy_dict` when executing scripts (Phase F2 gating).
- DATA-001: Fixtures and builder validation mirror Phase C/D layouts (`dose_{dose}/patched_{split}.npz`, `dose_{dose}/{view}/{view}_{split}.npz`).
- POLICY-001: PyTorch dependency acknowledged in summary.md for Phase F; pty-chi relies on torch>=2.2 runtime.
- OVERSAMPLING-001: Manifest covers gs2 overlap views generated with neighbor_count=7; tests assert dense/sparse splits remain present.

### Phase G — Comparison & Analysis (G1 complete, G2 pending)
**Test module:** `tests/study/test_dose_overlap_comparison.py`

**Active selectors (Attempt #90 evidence):**
- `pytest tests/study/test_dose_overlap_comparison.py::test_build_comparison_jobs_creates_all_conditions -vv` — verifies 12 jobs (3 doses × 2 views × 2 splits), deterministic ordering, metric config. Log: `reports/2025-11-05T140500Z/phase_g_comparison_plan/green/pytest_phase_g_target_green.log`.
- `pytest tests/study/test_dose_overlap_comparison.py -k "comparison" -vv` — module-level sweep; ensures no hidden selectors regress. Log: `.../green/pytest_phase_g_suite_green.log`.
- `pytest tests/study/test_dose_overlap_comparison.py --collect-only -k "comparison" -vv` — collection proof (1 test). Log: `.../collect/pytest_phase_g_collect.log`.

**CLI validation (dry-run only):**
- `python -m studies.fly64_dose_overlap.comparison --phase-c-root <phase C datasets> --phase-e-root <phase E checkpoints> --phase-f-root <Phase F manifests> --artifact-root .../cli --dose 1000 --view dense --split train --dry-run` — manifest + summary written; execution deferred. Log: `.../cli/phase_g_cli_dry_run.log`.

- **Absolute MS-SSIM sanity check (new):** After a real comparison run, execute  
  `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py --metrics <hub>/analysis/metrics_summary.json --ms-ssim-threshold 0.80 --output <hub>/analysis/aggregate_report.md --highlights <hub>/analysis/metrics_delta_highlights.txt`.  
  Archive the stdout/Markdown artifact and ensure the **MS-SSIM Sanity Check** table shows `OK` for every model; any `LOW (...)` status blocks the attempt until the reconstruction is re-run or the threshold rationale is documented. The downstream `analyze_dense_metrics.py` invocation inherits the same threshold and should be cited in How-To Maps alongside its log.

**Coverage status:** G1 (job builder + CLI scaffolding) satisfied. G0.1 inventory complete — see `reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md` for the authoritative dataset/checkpoint/manifests map and sparse acceptance stats. G0.2 test-strategy refresh landed with active selectors, collect-proof evidence, and CLI dry-run references.

**Remaining work for G2/G3:**
- Implement comparison execution (invoke `scripts/compare_models.py`) for the three ready conditions (dose_1000 dense/train, dense/test, sparse/train) and capture metrics/plots; register resulting selectors + CLI commands; archive logs under `reports/.../analysis/` and `.../cli/`.
- After GREEN evidence exists, append real-run selectors + CLI commands here and in `docs/TESTING_GUIDE.md` / `docs/development/TEST_SUITE_INDEX.md`, noting blocked dose/view/split combinations pending new Phase E/F outputs.

**Doc sync:** After G2 execution, update `docs/TESTING_GUIDE.md` §Phase G and `docs/development/TEST_SUITE_INDEX.md` with selectors + CLI commands, referencing GREEN/collect logs.

### Future Phases (Pending)
1) Phase E6 — Aggregated gs2 training evidence
   - Capture end-to-end training runs (non-dry-run) for dense/sparse overlap conditions with deterministic seeds.
   - Validate Lightning checkpoints, metrics logs, and final model artifacts.
2) Dose sanity per dataset
   - Confirm expected scaling behavior across doses (qualitative/log statistics)
3) Train/test separation
   - Validate y-axis split with non-overlapping regions

## 7. PASS Criteria
- All contract checks pass for generated datasets
- Filtering invariants satisfied; logs record overlap metric stats
- Comparison CSVs exist with non-empty records for each condition
- MS-SSIM sanity table reports `OK` for every model (or documented exception) at the configured threshold
- No unexpected SKIPs, and any expected SKIPs are justified in summary.md
