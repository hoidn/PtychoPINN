### Turn Summary
Shipped `make_ground_truth_manifest.py` CLI that validates NPZ keys against `specs/data_contracts.md` and emits JSON/MD/CSV manifests with SHA256 checksums for 7 datasets + baseline outputs + PINN weights.
All required keys (`diff3d`, `probeGuess`, `scan_index`) present; pytest `test_generic_loader` passed.
Next: Phase B README documenting simulate→train→infer commands for Maintainer <2>.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ (ground_truth_manifest.json, files.csv)

---

### Prior Turn Summary (Galph)
Shifted the fix plan focus from seed to DEBUG-SIM-LINES-DOSE-001 and documented the maintainer's ground-truth bundle ask across fix_plan + galph memory.
Authored the new implementation plan plus Phase A1 manifest scope and rewrote input.md so Ralph builds make_ground_truth_manifest.py and verifies the datasets with pytest.
Next: Ralph implements the manifest CLI, runs the loader smoke test, and archives outputs in the 2026-01-23T001018Z report dir.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ (implementation.md, input.md)
