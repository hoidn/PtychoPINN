Summary: Build the legacy photon_grid dataset/weights manifest so Maintainer <2> can fetch a verified bundle without rerunning dose_experiments.
Focus: DEBUG-SIM-LINES-DOSE-001 — Legacy dose_experiments ground-truth bundle
Branch: dose_experiments
Mapped tests: tests/test_generic_loader.py::test_generic_loader
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.A1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/make_ground_truth_manifest.py::main — new CLI that records SHA256 + size + stage metadata for every photon_grid_study_20250826 dataset (`data_p1e*.npz`), baseline outputs (`params.dill`, `baseline_model.h5`, `recon.dill`), and inference weights (`wts.h5.zip`), validating `specs/data_contracts.md` keys before emitting `{ground_truth_manifest.json, ground_truth_manifest.md, files.csv}` under the artifacts path.
- Pytest: pytest tests/test_generic_loader.py::test_generic_loader -q
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/

How-To Map
1. `mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/bin plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/`
2. Write `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/make_ground_truth_manifest.py` with args `--dataset-root`, `--baseline-params`, `--baseline-files` (list or glob), `--pinn-weights`, `--scenario-id`, `--output`; ensure it loads each NPZ with `allow_pickle=True`, asserts keys `diff3d`, `probeGuess`, `objectGuess`, `scan_index`, `ground_truth_patches`, and gathers SHA256 + size info for every file while extracting key metrics from `params.dill`.
3. Run `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/make_ground_truth_manifest.py --dataset-root photon_grid_study_20250826_152459 --baseline-params photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/params.dill --baseline-files photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/baseline_model.h5 photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/recon.dill --pinn-weights photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/pinn_run/wts.h5.zip --scenario-id PGRID-20250826-P1E5-T1024 --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/manifest.log`
4. `pytest tests/test_generic_loader.py::test_generic_loader -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/pytest_loader.log`

Pitfalls To Avoid
- Do not edit shipped modules; confine new code to `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/`.
- No dataset copies yet—manifest only; avoid `cp` of 200MB files until Phase C.
- Ensure SHA256 hashes cover every artifact referenced in the README (dataset NPZs + baseline/inference outputs).
- Validate NPZ keys exactly as `specs/data_contracts.md` states; fail fast if any are missing so Maintainer <2> gets a clear signal.
- Keep CLI output deterministic (sorted file lists) to ease diffing between reruns.
- Treat `allow_pickle=True` carefully; never write back mutated arrays.
- Capture stdout/stderr into artifacts; no ad-hoc logs scattered in repo root.
- Honor `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` when describing pytest usage.
- Do not assume GPU/TF availability; this loop only inspects files.
- Preserve relative paths in manifests so Maintainer <2> can locate originals if they need bigger files than we can bundle.

If Blocked
- If any required file is missing or unreadable, stop immediately, write the error plus `ls -R` snippet to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/blocker.md`, update `docs/fix_plan.md` Attempts History with the failure signature, and notify Galph before attempting fallbacks.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- specs/data_contracts.md:1 — defines NPZ schema the manifest must validate.
- docs/DATA_MANAGEMENT_GUIDE.md:1 — checksum and provenance guidance for shared datasets.
- inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md:1 — maintainer requirements for the artifact bundle.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:1 — phase checklist and exit criteria for this initiative.

Next Up (optional)
- DEBUG-SIM-LINES-DOSE-001.B1 — README detailing simulate→train→infer commands once manifest lands.
