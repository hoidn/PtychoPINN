### Turn Summary (2026-01-23T00:24Z)
Created generate_legacy_readme.py CLI that loads manifest + baseline summary and emits a maintainer-ready README with simulate/train/infer commands and provenance tables.
The README documents TF/Keras 2.x environment requirements and cites specs/data_contracts.md for NPZ key validation; all artifact sizes/SHA256 sourced from Phase-A manifest.
Next: proceed to Phase C (package datasets + baseline outputs into final bundle location for Maintainer <2>).
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/ (README.md, generate_readme.log, pytest_loader.log)

### Turn Summary (2026-01-23T00:19Z - supervisor)
Marked Phase A (manifest + pytest) as complete in the implementation plan and added explicit Phase B/C TODO rows in docs/fix_plan.md so the maintainer bundle scope is visible.
Authored a new DEBUG-SIM-LINES-DOSE-001.B1 Do Now that has Ralph build generate_legacy_readme.py, reuse the manifest/baseline summary, and run pytest test_generic_loader to validate the datasets referenced by the README.
Next: Ralph implements the README generator CLI, captures README/log artifacts under 2026-01-23T001931Z, and we can then expand provenance (B2) before copying data (Phase C).
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z/ (summary.md)
