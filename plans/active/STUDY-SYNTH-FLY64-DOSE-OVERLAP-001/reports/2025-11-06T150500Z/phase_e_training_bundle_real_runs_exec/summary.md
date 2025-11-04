# Phase E6 Dense/Baseline Evidence — Loop 2025-11-06T15:05:00Z

**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison prep (Phase E real bundle evidence)

## Current Gaps
- Dense (gs2) and baseline (gs1) CLI runs for dose=1000 still pending with manifest/bundle/SHA artifacts.
- `bundle_sha256` equality between CLI stdout and manifest is not yet enforced by test coverage.
- `analysis/bundle_checksums.txt` needs fresh digest proof tied to this hub.

## Loop Targets
1. Extend `test_training_cli_records_bundle_path` to compare stdout SHA256 lines against manifest `bundle_sha256` entries (RED→GREEN evidence).
2. Execute deterministic dense + baseline CLI runs with logs captured under `cli/` and bundles/manifests archived via helper script.
3. Populate `analysis/bundle_checksums.txt` and update this summary with observations (view/dose context, SHA digest, regeneration steps).

## Artifact Checklist
- red/pytest_training_cli_sha_red.log (expected failure before CLI/test alignment).
- green/pytest_training_cli_sha_green.log, green/pytest_training_cli_suite_green.log.
- collect/pytest_training_cli_collect.log.
- prep/phase_c_generation.log, prep/phase_d_generation.log (only if regeneration triggered).
- cli/dose1000_dense_gs2.log, cli/dose1000_baseline_gs1.log.
- data/{dose1000_dense_manifest.json,dose1000_baseline_manifest.json,skip_summary.json,wts_dense_gs2.h5.zip,wts_baseline_gs1.h5.zip}.
- analysis/{bundle_checksums.txt,training_manifest_pretty.json}.

## Notes
- Honor POLICY-001 / CONFIG-001 / DATA-001 / OVERSAMPLING-001 guardrails.
- Use `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` for every pytest/CLI command.
- Archive helper script already promoted at `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py`.

## Next Focus Candidate
- After dense/baseline proof goes green, schedule sparse view run or Phase F evidence refresh before Phase G comparisons.
