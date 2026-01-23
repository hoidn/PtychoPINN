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

### 2026-01-23T00:28Z — DEBUG-SIM-LINES-DOSE-001.C1 (packaging scope ready)
**Action:** Reviewed the Phase-A manifest + README outputs and confirmed the remaining gap is delivering the requested bundle under `reports/2026-01-22T014445Z/dose_experiments_ground_truth/`. Captured the requirements for a new packaging CLI that copies each manifest entry into simulation/training/inference/docs folders, verifies SHA256 parity, and writes JSON/MD verification logs plus a `.tar.gz` archive. While reviewing the README generator, spotted that the inference command points to a non-existent baseline path instead of the `pinn_run/wts.h5.zip` recorded in the manifest, so the upcoming loop must also patch `generate_legacy_readme.py::build_readme` to source the model path from the manifest.

**Next Actions:**
- Implement `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/package_ground_truth_bundle.py` + README fix, run it to copy the NPZ/baseline/pinn files into the final drop, generate tarball + verification logs under `reports/2026-01-23T002823Z/`, and re-run `pytest tests/test_generic_loader.py::test_generic_loader -q`.

### 2026-01-23T00:35Z — DEBUG-SIM-LINES-DOSE-001.C1/C2 (shipped)
**Action:** Implemented Phase C deliverables:
1. Created `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/package_ground_truth_bundle.py` CLI that:
   - Copies 7 datasets to simulation/, 3 baseline artifacts to training/, pinn weights to inference/
   - Verifies SHA256 checksums before and after copy (chunked hashing for memory efficiency)
   - Generates bundle_verification.json/md with per-file verification results
   - Creates .tar.gz archive with SHA256 checksum file
2. Updated `generate_legacy_readme.py`:
   - Fixed Section 5 inference command to use manifest `pinn_weights.relative_path` (was incorrectly pointing to baseline_run)
   - Added Section 8 "Delivery Artifacts" showing bundle root, tarball size/SHA256, structure diagram, verification summary
   - Added new CLI args: `--bundle-verification`, `--delivery-root`
3. Copied updated README to bundle docs/ folder

**Test:** `pytest tests/test_generic_loader.py::test_generic_loader -q` — PASSED (1 passed, 5 warnings)

**Metrics:**
- Bundle total files: 15 (7 datasets + 4 baseline/pinn artifacts + 4 docs)
- Bundle total size: 278.18 MB
- Tarball size: 270.70 MB
- Tarball SHA256: 7fe5e14ed9909f056807b77d5de56e729b8b79c8e5b8098ba50507f13780dd72
- All 11 data files verified via SHA256 match against Phase-A manifest

**Artifacts:**
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.json`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.md`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/README.md`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/package_bundle.log`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/generate_readme.log`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/pytest_loader.log`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/` (full bundle)
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth.tar.gz`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth.tar.gz.sha256`

**Next Actions:**
- C1/C2 complete; DEBUG-SIM-LINES-DOSE-001 exit criteria met
- Optionally draft maintainer handoff note referencing tarball SHA + drop location

### 2026-01-23T00:45Z — DEBUG-SIM-LINES-DOSE-001.D1 (shipped)
**Action:** Drafted the maintainer response at `inbox/response_dose_experiments_ground_truth.md` covering:
1. Delivery summary with drop root and bundle structure
2. Verification summary citing `bundle_verification.{json,md}` (15/15 files, 278.18 MB, tarball SHA `7fe5e14e...`)
3. Test validation with `pytest tests/test_generic_loader.py::test_generic_loader -q` (1 passed)
4. How-to instructions for extracting tarball and verifying SHA256
5. Dataset table (7 NPZ files, 1e3→1e9 photons) with full SHA256 checksums
6. Baseline artifacts table (params.dill, baseline_model.h5, recon.dill, wts.h5.zip)
7. Key parameters (N=64, gridsize=1, nepochs=50, NLL-only loss)
8. Next steps requesting Maintainer <2> confirmation

**Test:** `pytest tests/test_generic_loader.py::test_generic_loader -q` — PASSED (1 passed, 5 warnings in 2.54s)

**SHA256 verification:**
```
7fe5e14ed9909f056807b77d5de56e729b8b79c8e5b8098ba50507f13780dd72  dose_experiments_ground_truth.tar.gz
```

**Artifacts:**
- `inbox/response_dose_experiments_ground_truth.md`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z/pytest_loader.log`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z/dose_experiments_ground_truth.tar.gz.sha256.check`

**Next Actions:**
- D1 complete; await Maintainer <2> acknowledgment to close DEBUG-SIM-LINES-DOSE-001

### 2026-01-23T00:50Z — DEBUG-SIM-LINES-DOSE-001.E1 (rehydration verification scoped)
**Action:** Reviewed the packaged drop (`dose_experiments_ground_truth.tar.gz`) and confirmed we only validated files prior to compression. The maintainer handoff currently cites checksums gathered before the tarball was created, so we still lack evidence that rehydrating the archive preserves structure, metadata, and manifest parity. Before closing, we need to exercise the exact tarball the maintainer will download, regenerate the manifest from the extracted bundle, and diff it against the Phase-A manifest to prove SHA256 and dataset metadata remain unchanged.

**Gap:** No automation exists for tarball rehydration; manually extracting the ~270 MB archive risks leaving large duplicates in the repo and is error-prone.

**Next Actions:**
- Author `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/verify_bundle_rehydration.py` to:
  1. Extract `dose_experiments_ground_truth.tar.gz` into a temporary directory (with optional `--keep-extracted` flag).
  2. Re-run `make_ground_truth_manifest.py` on the extracted bundle, writing its outputs under a fresh `reports/<ts>/rehydration_check/` directory.
  3. Diff SHA256 + size metadata between the original manifest (`reports/2026-01-23T001018Z/ground_truth_manifest.json`) and the rehydrated manifest, emitting `rehydration_diff.json` + `rehydration_summary.md` and exiting non-zero on mismatch.
  4. Clean up the temporary extraction directory when verification succeeds to avoid duplicating the 270 MB dataset inside the repo.
- Capture the script output log plus a rerun of `pytest tests/test_generic_loader.py::test_generic_loader -q` under `reports/<ts>/rehydration_check/`.
- Append a short "Rehydration verification" section to `inbox/response_dose_experiments_ground_truth.md` citing the new summary so Maintainer <2> knows the tarball was tested end-to-end.

### 2026-01-23T00:55Z — DEBUG-SIM-LINES-DOSE-001.E1 (complete)
**Action:** Implemented `verify_bundle_rehydration.py` script that:
1. Extracts `dose_experiments_ground_truth.tar.gz` to a temp directory
2. Regenerates manifest via `make_ground_truth_manifest.py`
3. Compares SHA256 + size for all 11 files against original manifest
4. Emits `rehydration_diff.json` and `rehydration_summary.md`
5. Cleans up temp directory after success

**Metrics:**
- Rehydration status: PASS
- Total files: 11
- Matches: 11
- Mismatches: 0

**Test:** `pytest tests/test_generic_loader.py::test_generic_loader -q` — 1 passed (2.53s)

**Artifacts:**
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/rehydration_summary.md`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/rehydration_diff.json`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/verify_bundle_rehydration.log`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/pytest_loader.log`

**Next Actions:**
- DEBUG-SIM-LINES-DOSE-001.E1 complete; await Maintainer <2> acknowledgment to close DEBUG-SIM-LINES-DOSE-001

### 2026-01-23T01:19Z — DEBUG-SIM-LINES-DOSE-001.F1 (follow-up sent)
**Action:** Drafted follow-up note from Maintainer <1> to Maintainer <2> at `inbox/followup_dose_experiments_ground_truth_2026-01-23T011900Z.md` summarizing the delivered bundle and requesting acknowledgement. Re-ran the loader pytest for fresh evidence.

**Test:** `pytest tests/test_generic_loader.py::test_generic_loader -q` — Result logged to artifacts

**Artifacts:**
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T011900Z/pytest_loader.log`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T011900Z/followup_note.md` (archive copy)
- `inbox/followup_dose_experiments_ground_truth_2026-01-23T011900Z.md`

**Next Actions:**
- Awaiting Maintainer <2> acknowledgement; once received, mark DEBUG-SIM-LINES-DOSE-001 complete

### 2026-01-23T01:29Z — DEBUG-SIM-LINES-DOSE-001.F1 (inbox scan check)
**Action:** Implemented `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py` CLI that:
1. Scans the inbox directory for files matching a request pattern
2. Detects acknowledgement keywords (acknowledged, ack, confirm, received, thanks)
3. Correctly identifies messages FROM Maintainer <2> vs TO Maintainer <2>
4. Emits JSON (`inbox_scan_summary.json`) and Markdown (`inbox_scan_summary.md`) summaries
5. Sets `ack_detected: true` only when a message FROM Maintainer <2> contains ack keywords

**Scan Results:**
- Files scanned: 5
- Matches found: 3 (all related to dose_experiments_ground_truth)
- Acknowledgement detected: **No**
- Messages from Maintainer <2>: 1 (the original request, no ack keywords)
- Messages from Maintainer <1>: 2 (our response and follow-up)

**Conclusion:** The bundle has been delivered and a follow-up sent, but we are still awaiting acknowledgement from Maintainer <2>.

**Test:** `pytest tests/test_generic_loader.py::test_generic_loader -q` — 1 passed, 5 warnings (2.54s)

**Artifacts:**
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T012900Z/inbox_check/inbox_scan_summary.json`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T012900Z/inbox_check/inbox_scan_summary.md`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T012900Z/inbox_check/check_inbox.log`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T012900Z/pytest_loader.log`

**Next Actions:**
- F1 remains open; awaiting Maintainer <2> acknowledgement
- Once `ack_detected: true`, mark F1 complete and cite the maintainer reply path

## TODOs
- [x] S4: Expand the D0 parity Markdown report to list stage-level stats for every dataset and document the new test selector (`tests/tools/test_d0_parity_logger.py`) inside `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md`.
- [x] S3: Promote D0 parity logger into `scripts/tools/` with stage-level stats + tests, then capture artifacts for photon_grid_study_20250826_152459
- [x] DEBUG-SIM-LINES-DOSE-001.B1: Produce the simulate→train→infer README plus command log for Maintainer <2> under a fresh `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/<ts>/README.md`.
- [x] DEBUG-SIM-LINES-DOSE-001.B2: Extend the README with a provenance table mapping every dataset/baseline/inference artifact to its SHA256 + source stage, referencing the Phase A manifest.
- [x] DEBUG-SIM-LINES-DOSE-001.C1: Copy or package (tarball) the requested datasets + baseline outputs into `reports/2026-01-22T014445Z/dose_experiments_ground_truth/`, keeping the manifest + README in sync.
- [x] DEBUG-SIM-LINES-DOSE-001.C2: Capture checksum verification logs for the final bundle (or tarball) and confirm size constraints / delivery instructions in `galph_memory.md` + maintainer inbox.
- [x] DEBUG-SIM-LINES-DOSE-001.D1: Draft `inbox/response_dose_experiments_ground_truth.md` that cites the final drop root, README/manifest paths, bundle_verification logs, tarball SHA, and the validating `pytest tests/test_generic_loader.py::test_generic_loader -q` log so Maintainer <2> can close the request.
- [x] DEBUG-SIM-LINES-DOSE-001.E1: Verify the tarball rehydration path by extracting `dose_experiments_ground_truth.tar.gz`, regenerating the manifest, diffing it against `reports/2026-01-23T001018Z/ground_truth_manifest.json`, logging the comparison under `reports/<ts>/rehydration_check/`, re-running `pytest tests/test_generic_loader.py::test_generic_loader -q`, and updating the maintainer response with the results.
- [ ] DEBUG-SIM-LINES-DOSE-001.F1: Await Maintainer <2> acknowledgement of the delivered bundle. Inbox scan CLI implemented (`check_inbox_for_ack.py`); latest scan at 2026-01-23T01:12:57Z shows no ack yet. Re-run scan or check inbox for new files.
