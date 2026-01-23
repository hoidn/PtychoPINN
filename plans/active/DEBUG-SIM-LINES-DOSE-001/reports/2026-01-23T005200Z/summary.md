### Turn Summary
Implemented `verify_bundle_rehydration.py` that extracts the dose_experiments tarball, regenerates the manifest, and compares SHA256 checksums for all 11 files.
Rehydration verification passed with 11/11 files matching; pytest loader test also passed (1 passed, 2.53s).
Next: await Maintainer <2> acknowledgment to close DEBUG-SIM-LINES-DOSE-001.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/ (rehydration_check/, pytest_loader.log)
