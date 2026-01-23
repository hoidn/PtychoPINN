# Fix Plan

## Status
`in_progress`

## Current Focus
**DEBUG-SIM-LINES-DOSE-001 — Legacy dose_experiments ground-truth bundle**

Maintainer <2> asked for a faithful simulate→train→infer run from the legacy `dose_experiments` pipeline because their TF/Keras 3.x environment crashes. We must package the photon_grid_study_20250826_152459 baseline (datasets, params, checkpoints, reconstructions, README with commands) under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/` using non-production tooling so Ralph can ship evidence without touching shipped modules.

- Working Plan: `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md`
- Request Source: `inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md`

## Completed Items
- [x] S1: Check `inbox/` for new requests — found `README_prepare_d0_response.md`
- [x] S2: Document response plan — created `inbox/response_prepare_d0_response.md`
- [x] S3: Ship D0 parity logger CLI + tests, produce PGRID-20250826-P1E5-T1024 evidence

## Artifacts Produced
- `plans/active/seed/bin/dose_baseline_snapshot.py` — reusable script for baseline snapshots
- `plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json` — machine-readable snapshot
- `plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.md` — human-readable summary
- `plans/active/seed/reports/2026-01-22T024002Z/pytest_seed.log` — test run log
- `inbox/response_prepare_d0_response.md` — maintainer reply covering sections 1–7
- `scripts/tools/d0_parity_logger.py` — shipping CLI for parity evidence capture
- `tests/tools/test_d0_parity_logger.py` — unit + integration tests for the CLI
- `plans/active/seed/reports/2026-01-22T030216Z/dose_parity_log.json` — structured parity evidence
- `plans/active/seed/reports/2026-01-22T030216Z/dose_parity_log.md` — human-readable parity summary
- `plans/active/seed/reports/2026-01-22T030216Z/probe_stats.csv` — probe amplitude/phase percentiles
- `plans/active/seed/reports/2026-01-22T030216Z/pytest_d0_parity_logger.log` — test execution log
- `plans/active/seed/reports/2026-01-22T030216Z/d0_parity_collect.log` — test collection verification
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/make_ground_truth_manifest.py` — CLI for generating dataset+baseline checksums
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.json` — full manifest with SHA256 + metadata
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.md` — human-readable manifest summary
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/files.csv` — flat file list with checksums
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py` — CLI for generating maintainer README
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/README.md` — maintainer-ready README with commands + provenance

## Attempts History

### 2026-01-22T02:40Z — seed S1–S2 (initial)
**Action:** Created dose_baseline_snapshot.py, ran it to generate JSON/MD summaries, drafted maintainer response.

**Metrics:**
- Scenario ID: PGRID-20250826-P1E5-T1024
- Baseline ms_ssim: 0.925 (train) / 0.921 (test)
- Baseline psnr: 71.3 dB (amplitude)
- 7 datasets captured (1e3 through 1e9 photons)
- All SHA256 checksums recorded

**Test:** `pytest tests/test_generic_loader.py::test_generic_loader -q` — PASSED (1 passed, 5 warnings)

**Note:** The mapped test `tests/test_generate_data.py::test_placeholder` failed collection due to `ptycho/data_preprocessing.py:13` assertion (`data_source == 'generic'`). Used alternative test to validate environment.

**Artifacts:**
- `plans/active/seed/reports/2026-01-22T024002Z/`

**Next Actions:**
- Await maintainer review of `inbox/response_prepare_d0_response.md`
- If time permits, capture gs2 variant for future parity logging

### 2026-01-22T04:26Z — seed S3 (blueprint)
**Action:** Authored `plans/active/seed/reports/2026-01-22T042640Z/d0_parity_logger_plan.md` to scope the shipping CLI + tests for D0 parity logging. Captures dataset/probe metrics, raw/grouped/normalized stats, and testing expectations before promoting the script under `scripts/tools/`.

**Artifacts:**
- `plans/active/seed/reports/2026-01-22T042640Z/d0_parity_logger_plan.md`

**Next Actions:**
- Promote the plan into implementation: add `scripts/tools/d0_parity_logger.py` plus `tests/tools/test_d0_parity_logger.py`, then run the CLI over photon_grid_study_20250826_152459 to produce JSON/MD artifacts under a fresh timestamped reports directory.

### 2026-01-22T03:06Z — seed S3 (shipped)
**Action:** Verified existing implementation of `scripts/tools/d0_parity_logger.py` and `tests/tools/test_d0_parity_logger.py`. Ran tests successfully, then executed CLI on photon_grid_study_20250826_152459 to produce parity evidence.

**Test:** `pytest tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q` — PASSED (1 passed in 0.11s)

**Collection:** `pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q` — 1 test collected

**CLI Run:**
```
python scripts/tools/d0_parity_logger.py \
  --dataset-root photon_grid_study_20250826_152459 \
  --baseline-params "photon_grid_study.../params.dill" \
  --scenario-id PGRID-20250826-P1E5-T1024 \
  --output plans/active/seed/reports/2026-01-22T030216Z
```

**Metrics captured:**
- Scenario ID: PGRID-20250826-P1E5-T1024
- 7 datasets processed (1e3 through 1e9 photons)
- Baseline ms_ssim: 0.9248 (train) / 0.9206 (test)
- Baseline psnr: 71.32 dB (train) / 158.06 dB (test)
- intensity_scale_value: 988.21
- Probe stats: identical across all datasets (same probe, different photon counts)
- Raw diffraction stats: min=0, max varies by dose, nonzero_fraction ~0.21
- All SHA256 checksums recorded for reproducibility

**Artifacts:**
- `plans/active/seed/reports/2026-01-22T030216Z/`

**Next Actions:**
- Await maintainer review of parity evidence
- Optionally capture gs2 scenario for multi-gridsize comparison

### 2026-01-22T03:11Z — seed S4 (doc sync + Markdown parity tables)
**Action:** Reviewed the emitted Markdown parity log and found it only lists stage-level stats for the first dataset, forcing maintainers to open the JSON blob to compare other photon doses. Also confirmed the testing docs still omit the new `tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs` selector that guards the CLI. Logged the gap and outlined the follow-up scope so the CLI deliverable fully matches the maintainer spec.

**Artifacts:**
- `plans/active/seed/reports/2026-01-22T031142Z/`

**Next Actions:**
- Extend `scripts/tools/d0_parity_logger.py::write_markdown` (and helpers if needed) to render stage-level stats tables for every processed dataset, including raw, normalized, and grouped values so reviewers can diff doses without JSON parsing.
- Update `tests/tools/test_d0_parity_logger.py` to assert the expanded Markdown content and add coverage for the `--limit-datasets` filter so maintainers can shorten runs.
- Sync `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with the new selector + CLI instructions, then re-run `pytest tests/tools/test_d0_parity_logger.py -q` with fresh logs.

### 2026-01-22T23:39Z — seed S4 (shipped)
**Action:** Implemented all S4 deliverables:
1. Extended `scripts/tools/d0_parity_logger.py::write_markdown` to emit "Stage-Level Stats by Dataset" section with per-dataset subsections and Markdown tables for raw/normalized/grouped stats (including `n_unique_scans`/`n_patterns`).
2. Updated `tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs` to use a multi-dataset fixture (two NPZ files), assert multi-dataset Markdown coverage with all three stage tables.
3. Added `tests/tools/test_d0_parity_logger.py::test_cli_limit_datasets_filters_inputs` to verify `--limit-datasets` filter excludes non-requested datasets from JSON/Markdown.
4. Updated `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with the parity logger CLI selector and usage reference.

**Test:** `pytest tests/tools/test_d0_parity_logger.py -q` — 17 passed in 0.16s

**Collection:**
- `pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_emits_outputs -q` — 1 test collected
- `pytest --collect-only tests/tools/test_d0_parity_logger.py::test_cli_limit_datasets_filters_inputs -q` — 1 test collected

**CLI Run:**
```
python scripts/tools/d0_parity_logger.py \
  --dataset-root photon_grid_study_20250826_152459 \
  --baseline-params "photon_grid_study_20250826_152459/results_p1e5/.../params.dill" \
  --scenario-id PGRID-20250826-P1E5-T1024 \
  --output plans/active/seed/reports/2026-01-22T233418Z
```

**Metrics captured:**
- 7 datasets processed (1e3 through 1e9 photons) — all with stage-level stats in Markdown
- Markdown now includes "Stage-Level Stats by Dataset" section with per-dataset raw/normalized/grouped tables
- Grouped stats include n_unique_scans and n_patterns counts

**Artifacts:**
- `plans/active/seed/reports/2026-01-22T233418Z/` (dose_parity_log.json, dose_parity_log.md, probe_stats.csv, pytest logs)

**Next Actions:**
- S4 complete; await maintainer review of parity evidence

### 2026-01-23T00:06Z — DEBUG-SIM-LINES-DOSE-001 (plan reboot + request triage)
**Action:** Switched focus from `seed` to the maintainer request in `inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md`, drafted a fresh implementation plan at `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md`, and documented the dependency chain + exit criteria for the legacy artifact bundle.

**Artifacts:**
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md`

**Next Actions:**
- Phase A: build `make_ground_truth_manifest.py`, gather SHA manifests for datasets + baseline outputs, and capture pytest loader logs under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/<ts>/`.

### 2026-01-23T00:15Z — DEBUG-SIM-LINES-DOSE-001.A1 (manifest CLI shipped)
**Action:** Created `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/make_ground_truth_manifest.py` CLI that:
1. Gathers SHA256 + size + modified timestamp for all `data_p1e*.npz` files in the dataset root
2. Validates NPZ keys against `specs/data_contracts.md` (requires `diff3d`, `probeGuess`, `scan_index`)
3. Extracts array shapes/dtypes from NPZ files
4. Parses baseline `params.dill` metadata (N, gridsize, nepochs)
5. Emits `ground_truth_manifest.json`, `ground_truth_manifest.md`, and `files.csv` to the output directory

**CLI Run:**
```
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/make_ground_truth_manifest.py \
  --dataset-root photon_grid_study_20250826_152459 \
  --baseline-params photon_grid_study_20250826_152459/results_p1e5/.../params.dill \
  --baseline-files baseline_model.h5 recon.dill \
  --pinn-weights wts.h5.zip \
  --scenario-id PGRID-20250826-P1E5-T1024 \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/
```

**Metrics:**
- 7 datasets processed (data_p1e3 through data_p1e9)
- All NPZ files validated: required keys present (`diff3d`, `probeGuess`, `scan_index`)
- Optional keys present: `objectGuess`, `ground_truth_patches`, `xcoords`, `ycoords`, `xcoords_start`, `ycoords_start`
- Total dataset size: ~192 MB
- Baseline params: N=64, gridsize=1, nepochs=50
- 12 files checksummed (7 datasets + params.dill + baseline_model.h5 + recon.dill + wts.h5.zip)

**Test:** `pytest tests/test_generic_loader.py::test_generic_loader -q` — PASSED (1 passed, 5 warnings)

**Artifacts:**
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.json`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.md`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/files.csv`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/manifest.log`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/pytest_loader.log`

**Next Actions:**
- Phase B: README documenting simulate→train→infer commands for Maintainer <2>

### 2026-01-23T00:24Z — DEBUG-SIM-LINES-DOSE-001.B1 (README generator shipped)
**Action:** Created `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py` CLI that:
1. Loads Phase-A manifest JSON + baseline summary JSON
2. Renders a maintainer-ready README.md with 7 sections:
   - Overview (scenario ID, key params, baseline metrics)
   - Environment Requirements (TF/Keras 2.x warning)
   - Simulation Commands (dose.py / dose_dependence.ipynb)
   - Training Commands (ptycho.train CLI)
   - Inference Commands (ptycho.inference CLI)
   - Artifact Provenance Table (datasets + baseline files with size/SHA256 from manifest)
   - NPZ Key Requirements (citing specs/data_contracts.md RawData NPZ)
3. Validates input files exist before proceeding (fail-fast)
4. Sorts provenance table by photon dose (1e3 → 1e9)

**CLI Run:**
```
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py \
  --manifest plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.json \
  --baseline-summary plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json \
  --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z
```

**Metrics:**
- 7 datasets in provenance table (sorted by photon dose)
- 4 baseline artifacts documented (params.dill, baseline_model.h5, recon.dill, wts.h5.zip)
- README includes TF/Keras 2.x environment warning per maintainer request
- Spec citations included (specs/data_contracts.md RawData NPZ)

**Test:** `pytest tests/test_generic_loader.py::test_generic_loader -q` — PASSED (1 passed, 5 warnings)

**Artifacts:**
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/README.md`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/generate_readme.log`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/pytest_loader.log`

**Next Actions:**
- B1 complete; B2 (provenance table polish) may be skipped since provenance table already included
- Proceed to C1: package datasets + baseline outputs into final bundle location

## TODOs
- [x] S4: Expand the D0 parity Markdown report to list stage-level stats for every dataset and document the new test selector (`tests/tools/test_d0_parity_logger.py`) inside `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md`.
- [x] S3: Promote D0 parity logger into `scripts/tools/` with stage-level stats + tests, then capture artifacts for photon_grid_study_20250826_152459
- [x] DEBUG-SIM-LINES-DOSE-001.B1: Produce the simulate→train→infer README plus command log for Maintainer <2> under a fresh `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/<ts>/README.md`.
- [x] DEBUG-SIM-LINES-DOSE-001.B2: Extend the README with a provenance table mapping every dataset/baseline/inference artifact to its SHA256 + source stage, referencing the Phase A manifest.
- [ ] DEBUG-SIM-LINES-DOSE-001.C1: Copy or package (tarball) the requested datasets + baseline outputs into `reports/2026-01-22T014445Z/dose_experiments_ground_truth/`, keeping the manifest + README in sync.
- [ ] DEBUG-SIM-LINES-DOSE-001.C2: Capture checksum verification logs for the final bundle (or tarball) and confirm size constraints / delivery instructions in `galph_memory.md` + maintainer inbox.
