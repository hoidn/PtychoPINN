Summary: Finish DEBUG-SIM-LINES-DOSE-001 Phase C by fixing the README generator's inference section and packaging the photon_grid_study_20250826_152459 datasets + baseline outputs into the maintainer drop with verification logs and a tarball checksum.
Focus: DEBUG-SIM-LINES-DOSE-001.C1/C2 — Legacy dose_experiments ground-truth bundle
Branch: dose_experiments
Mapped tests: pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.C1/C2
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/package_ground_truth_bundle.py::main — new CLI that ingests the Phase-A manifest + README, copies every dataset/baseline/pinn artifact into `reports/2026-01-22T014445Z/dose_experiments_ground_truth/` (simulation/training/inference/docs layout), re-hashes sources vs manifest entries, writes `bundle_verification.json`/`.md`, and emits a `.tar.gz` plus SHA256 file; plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py::build_readme — load the manifest + new `bundle_verification.json` so the inference command uses the manifest `pinn_weights` path and a new "Delivery Artifacts" section lists the drop root + tarball size/SHA256.
- Pytest: pytest tests/test_generic_loader.py::test_generic_loader -q
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/

How-To Map
1. Export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md for the session. Remove any stale bundle to avoid mixed files: `rm -rf plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth` and recreate via the packaging CLI (it will build simulation/training/inference/docs itself).
2. `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/package_ground_truth_bundle.py --manifest-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.json --manifest-md plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.md --baseline-summary plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json --readme plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/README.md --drop-root plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth --reports-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z --tarball plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth.tar.gz | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/package_bundle.log` — script must: create simulation/training/inference/docs dirs, copy all seven NPZs plus params.dill/baseline_model.h5/recon.dill/wts.h5.zip, compute SHA256 for sources + copies, assert they match the manifest hashes, emit `bundle_verification.json` + `.md` under the reports dir, and write the tarball + `.sha256` next to the drop.
3. `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py --manifest plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.json --baseline-summary plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json --bundle-verification plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.json --delivery-root plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/generate_readme.log` — verify the inference command path now matches the manifest `pinn_weights` entry and the new Section 8 reports tarball size/SHA256 from the verification JSON.
4. `cp plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/README.md plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/docs/README.md` so the final bundle ships with the refreshed README alongside the manifest MD/JSON and baseline summary.
5. `pytest tests/test_generic_loader.py::test_generic_loader -q | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/pytest_loader.log` — confirms the datasets referenced by the manifest (and now copied into simulation/) still satisfy specs/data_contracts.md §RawData NPZ.

Pitfalls To Avoid
- Keep all new scripts and logs inside `plans/active/DEBUG-SIM-LINES-DOSE-001/`; do not touch shipped modules or global tooling.
- Do not hard-code tarball checksums or dataset hashes; always recompute from the manifest entries and copied files before writing verification logs.
- Ensure the packaging CLI preserves device-neutral behavior (no TF imports) and streams file copies with `shutil.copy2` so metadata is retained.
- Copy datasets/baseline artifacts exactly once—no lossy conversions or `.npz` extraction.
- The README's inference command must use `manifest['pinn_weights']['relative_path']`; pointing at the baseline directory is incorrect.
- When generating the tarball use `shutil.make_archive` or `tarfile` so the entire `dose_experiments_ground_truth/` folder (including docs) is captured; do not tar the reports directory itself.
- Capture every CLI stdout/stderr via `tee` into the artifacts path; maintainer review depends on those logs.
- Avoid `sudo`, `pip install`, or environment mutations; Environment Freeze applies.
- Large dataset copies will consume ~200 MB—ensure disk space exists before starting and prefer chunked hashing to avoid loading NPZs into memory.
- If the drop path already exists, remove it before copying so no stale files sneak into the tarball.

If Blocked
- If hashing or copy verification fails, write the failing path + traceback into `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/blocker.md`, keep the partial bundle untouched for inspection, and update `docs/fix_plan.md` Attempts History plus `galph_memory.md` with the error signature while awaiting guidance.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:6 — DEBUG-SIM-LINES-DOSE-001 scope and remaining C1/C2 checklist.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:34 — Phase C requirements for the packaging + tarball verification.
- inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md:1 — Maintainer drop path and artifact expectations.
- specs/data_contracts.md:3 — Required NPZ keys to double-check after copying datasets.
- docs/TESTING_GUIDE.md:1 — Pytest selector reference for `tests/test_generic_loader.py::test_generic_loader`.

Next Up (optional)
- If time remains, draft the maintainer handoff note referencing the tarball SHA and drop location so DEBUG-SIM-LINES-DOSE-001 can close once they acknowledge receipt.
