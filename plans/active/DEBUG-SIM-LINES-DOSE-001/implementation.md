# DEBUG-SIM-LINES-DOSE-001 — Legacy dose_experiments Ground-Truth Bundle

## Metadata
- **ID:** DEBUG-SIM-LINES-DOSE-001
- **Title:** Package photon_grid_study_20250826 baseline artifacts for Maintainer <2>
- **Owner:** Galph (supervisor) / Ralph (implementation)
- **Status:** Active — Phase C
- **Linked Request:** `inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md`
- **Artifacts Hub:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/`

## Exit Criteria
1. Deliver a reproducible bundle at `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/` containing:
   - The seven photon_grid_study_20250826_152459 datasets (`data_p1e3.npz` … `data_p1e9.npz`) with SHA256 manifest per `specs/data_contracts.md §RawData NPZ`.
   - Baseline training outputs (params.dill, baseline_model.h5, recon.dill, history) and inference weights (`wts.h5.zip`) validated via checksum + size metadata.
   - README describing legacy simulate→train→infer commands, config overrides, and environment notes.
2. Capture an automation script under `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/` that regenerates the manifest/README from the on-disk artifacts without touching shipped modules.
3. Provide a verification log (pytest selector collecting >0) showing the TensorFlow data loader still ingests the datasets referenced in the manifest.

## Spec Alignment
- `specs/data_contracts.md §RawData NPZ` — enumerates required NPZ keys (`diff3d`, `probeGuess`, `scan_index`, etc.) that each dataset must satisfy.
- `docs/DATA_MANAGEMENT_GUIDE.md §Checksum Manifests` — mandates SHA tracking for shared datasets.
- `docs/WORKFLOW_GUIDE.md §Dose Experiments` — describes the simulate→train→infer order to be documented in the README.

## Phases & Checklists

### Phase A — Manifest + Verification
- [x] A1: Implemented `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/make_ground_truth_manifest.py` and captured JSON/MD/CSV manifests with SHA256 + dataset metadata (see `reports/2026-01-23T001018Z/ground_truth_manifest.*`).
- [x] A2: Re-ran `pytest tests/test_generic_loader.py::test_generic_loader -q` with logs under `reports/2026-01-23T001018Z/pytest_loader.log` to prove the manifest's NPZ references still load.

### Phase B — README + Command Blueprint
- [x] B1: Authored `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py` and emitted `reports/2026-01-23T001931Z/README.md` covering simulate→train→infer commands plus environment prerequisites (TF/Keras 2.x guardrails).
- [x] B2: README includes provenance tables for datasets/baseline artifacts sourced from `ground_truth_manifest.json` with SHA256 + size metadata sorted by photon dose.

### Phase C — Artifact Drop
- [ ] C1: Add `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/package_ground_truth_bundle.py` that loads the Phase-A manifest + README, copies every dataset/baseline/pinn artifact into `reports/2026-01-22T014445Z/dose_experiments_ground_truth/` under simulation/training/inference/docs subfolders, and emits JSON/MD verification logs with manifest-vs-copy SHA matches.
- [ ] C2: Use the packaging CLI to produce a `.tar.gz` of the final drop alongside checksum metadata so we can deliver a single archive and cite its size + SHA256 to the maintainer.

## Dependency Analysis
- Uses existing photon_grid_study_20250826_152459 datasets and baseline outputs already present under the repo root; no production code changes or new dependencies required.
- Relies on `plans/active/seed/bin/dose_baseline_snapshot.py` for reference formatting but replicates logic locally to avoid shared coupling.
- Risk: Copying 200+ MB of NPZ files into `plans/active/...` may blow repo size if committed; coordinate with maintainer if Git LFS or external storage is preferable before running C1/C2.

## Artifacts
- Reports path per loop: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/<ISO8601Z>/`
- Final drop: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/`
- Scripts: `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/make_ground_truth_manifest.py`

## Abort / Escalation Triggers
- If any dataset or baseline file listed in the request is missing or corrupted (checksum mismatch), halt and open a maintainer request documenting the loss with evidence.
- If copying artifacts risks exceeding repo storage constraints, pause Phase C and negotiate an alternate delivery channel via maintainer inbox before resuming.
