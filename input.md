Summary: Build a rehydration verification script for the legacy dose_experiments bundle so we can prove the .tar.gz extracts cleanly, regenerates the manifest, and still passes the loader gate.
Focus: DEBUG-SIM-LINES-DOSE-001.E1 — Legacy tarball rehydration verification
Branch: dose_experiments
Mapped tests: pytest tests/test_generic_loader.py::test_generic_loader -q
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/

Do Now (hard validity contract)
- Focus ID: DEBUG-SIM-LINES-DOSE-001.E1
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/verify_bundle_rehydration.py::main — add a CLI that (1) extracts `dose_experiments_ground_truth.tar.gz` into a temporary directory, (2) re-runs `make_ground_truth_manifest.py` against the extracted bundle, (3) compares SHA256 + size metadata with the original manifest at `reports/2026-01-23T001018Z/ground_truth_manifest.json`, emitting `rehydration_diff.json` and `rehydration_summary.md`, and (4) optionally cleans up the extraction directory unless `--keep-extracted` is provided; exit non-zero if any mismatch.
- Update: inbox/response_dose_experiments_ground_truth.md::RehydrationVerification — append a brief section that cites the new `rehydration_summary.md`, highlights the tarball rehydration result, and links to the verifying script/log so Maintainer <2> knows the archive itself was exercised.
- Pytest: pytest tests/test_generic_loader.py::test_generic_loader -q
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/

How-To Map
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && ARTIFACT_DIR=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z && CHECK_DIR="$ARTIFACT_DIR/rehydration_check" && mkdir -p "$CHECK_DIR"
2. Implement `verify_bundle_rehydration.py`: use argparse to collect `--tarball`, `--manifest`, `--output`, and optional `--keep-extracted`; extract the tarball (default to deleting the extraction directory on success), call `make_ground_truth_manifest.py` via `sys.executable` so the regenerated manifest lands in `$CHECK_DIR/manifest`, load both manifests, compare SHA256 + size for datasets/baseline artifacts/pinn weights keyed by filename, and write `rehydration_diff.json` plus `rehydration_summary.md` that states counts, mismatches, and scenario ID status.
3. Run the new script and capture its log:
   python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/verify_bundle_rehydration.py \
     --tarball plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth.tar.gz \
     --manifest plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.json \
     --output "$CHECK_DIR" \
     | tee "$CHECK_DIR/verify_bundle_rehydration.log"
4. Re-run the loader test and capture the log: pytest tests/test_generic_loader.py::test_generic_loader -q | tee "$ARTIFACT_DIR/pytest_loader.log"
5. Update `inbox/response_dose_experiments_ground_truth.md` with a new "Rehydration verification" section that references `$CHECK_DIR/rehydration_summary.md`, the script path, and the new pytest log so the maintainer hears about the tarball-level check.

Pitfalls To Avoid
- Do not leave the extracted ~270 MB bundle inside the repo; make the script clean up the temporary directory unless `--keep-extracted` is passed.
- Compare manifest entries by filename and file_type (sha/size), not by absolute path, because extraction paths will differ.
- Use `sys.executable` when shelling out to `make_ground_truth_manifest.py` to avoid relying on PATH or virtualenv assumptions.
- Preserve the original scenario ID from `ground_truth_manifest.json`; the regenerated manifest should inherit it automatically.
- Keep script outputs (rehydration manifest, diff, summary, log) under `$CHECK_DIR` so we don't scatter artifacts across the repo.
- Don't delete or modify the original bundle/tarball; verification must be read-only aside from temporary extraction.
- Capture the pytest log with `tee` and store it under the same artifact directory for traceability.
- When editing the maintainer response, append a concise section instead of rewriting prior sections so earlier audit trails remain intact.
- If the script detects a mismatch, exit non-zero and stop—do not overwrite manifests to make them "match".
- Avoid environment mutations or dataset symlinks; everything should run against local files only.

If Blocked
- If extraction or manifest regeneration fails, capture the stack trace plus any partial logs into `$CHECK_DIR/blocker.md`, update the Attempt History in docs/fix_plan.md + galph_memory with the failure signature, and halt further steps until the issue is triaged.

Findings Applied (Mandatory)
- No relevant findings in the knowledge base.

Pointers
- docs/fix_plan.md:304 — Scope + checklist for DEBUG-SIM-LINES-DOSE-001.E1 rehydration verification.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/make_ground_truth_manifest.py:1 — Existing CLI whose output/structure the new rehydration script should reuse when regenerating manifests.
- specs/data_contracts.md:5 — RawData NPZ schema requirements enforced by the manifest CLI during rehydration checks.
- docs/TESTING_GUIDE.md:5 — Reference for the `pytest tests/test_generic_loader.py::test_generic_loader -q` guard.
- inbox/response_dose_experiments_ground_truth.md:1 — Maintainer response template that needs the new rehydration verification section.

Next Up (optional)
- If the rehydration evidence looks good and Maintainer <2> acknowledges, mark DEBUG-SIM-LINES-DOSE-001 complete in docs/fix_plan.md and pivot back to the next roadmap initiative.
