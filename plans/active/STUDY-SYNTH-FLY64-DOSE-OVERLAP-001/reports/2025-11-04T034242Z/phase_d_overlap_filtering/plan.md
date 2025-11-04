# Phase D Plan — Group-Level Overlap Filtering

## Context
- **Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- **Phase Goal:** Transform Phase C dose datasets into grouped dense vs sparse overlap views (gridsize 2) while enforcing spacing thresholds and oversampling policies so downstream training uses reproducible, contract-compliant NPZs.
- **Dependencies:** Phase C outputs (`studies/fly64_dose_overlap/generation.py`), Phase B validator (`validation.py::validate_dataset_contract`), design constants (`design.py`), spacing/oversampling guidance (`docs/GRIDSIZE_N_GROUPS_GUIDE.md:141-178`, `docs/SAMPLING_USER_GUIDE.md:112-140`), DATA-001 contract (`specs/data_contracts.md:200-240`), and findings CONFIG-001/DATA-001/OVERSAMPLING-001.

### Phase D — Group-Level Overlap Views
Goal: Produce per-dose dense/sparse NPZ views by filtering grouped scan positions to satisfy StudyDesign spacing thresholds (S ≈ (1 − f_overlap) × N) and K-choose-C invariants (neighbor_count ≥ gridsize²).
Prereqs: Phase C dataset manifests exist for the targeted doses; validator callable without params.cfg; StudyDesign spacing thresholds validated.
Exit Criteria: New module exposes filtering/build pipeline, pytest suite encodes RED→GREEN spacing enforcement, CLI materializes dense/sparse NPZ pairs per dose with dataset manifests, artifacts/logs stored under this phase directory, documentation + ledger updated.

| ID | Task Description | State | How/Why & Guidance (API/doc/artifact/source refs) |
| --- | --- | --- | --- |
| D1 | Implement `studies/fly64_dose_overlap/overlap.py::compute_spacing_matrix` + helpers to derive pairwise distances and acceptance masks for a set of scan coordinates using StudyDesign thresholds. | [x] | Implemented in commit `d9521b95` (Attempt #7 artifact set) — pairwise spacing + acceptance mask utilities now live in overlap.py with CONFIG-001 compliance. |
| D2 | Implement `overlap.py::generate_overlap_views` that loads Phase C train/test NPZs, forms gridsize=2 groups (KNN-based) using neighbor_count=7, filters to dense/sparse selections via D1 mask, runs validator with `view` hints, and writes `{dense,sparse}_{train,test}.npz` + spacing stats JSON. | [P] | Core pipeline implemented (same commit) with validator hook + metadata. Outstanding: align spacing metrics output with plan requirement (`metrics/<dose>/<view>.json` under reports hub) instead of single aggregated `spacing_metrics.json`. New test + code change needed. |
| D3 | Author pytest module `tests/study/test_dose_overlap_overlap.py` covering: (a) `compute_spacing_matrix` dense vs sparse thresholds on synthetic coords, (b) `generate_overlap_views` orchestration via monkeypatched loaders/validator ensuring correct files + metadata, (c) failure case when spacing threshold violated (expect ValueError). Capture RED (spacing failure) → GREEN logs to this phase directory and collect-only proof. | [x] | Module landed with 9 tests + RED/green logs at `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/{red,green,collect}/` (selector: `pytest tests/study/test_dose_overlap_overlap.py -k spacing_filter -vv`). |
| D4 | CLI + docs: add `python -m studies.fly64_dose_overlap.overlap --phase-c-root ... --output-root ...` to iterate doses, produce dense/sparse NPZs, write manifest + spacing metrics. Update `implementation.md` Phase D, `test_strategy.md` sections, `summary.md`, and `docs/fix_plan.md` Attempt log with artifact paths & findings alignment. | [ ] | Follow `docs/TESTING_GUIDE.md` commands (export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`). Capture CLI log + manifest, and copy per-dose/view metrics JSON into `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/metrics/<dose>/<view>.json` alongside summary updates and ledger attempt. |

## Notes & Guardrails
- Maintain CONFIG-001 boundaries: grouping module loads NPZ data via `np.load` only; legacy bridges reside in downstream training, not here.
- Dense view target: overlap fraction 0.7 → threshold ≈ 38.4 px; Sparse view target: 0.2 → threshold ≈ 102.4 px. Reject groups violating thresholds and log rejection counts.
- Preserve deterministic ordering (seeded RNG) so test expectations stay stable.
- Do not materialize large datasets inside pytest; rely on small synthetic arrays with patch size from design constants.
- Artifact hub root: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T034242Z/phase_d_overlap_filtering/`.
