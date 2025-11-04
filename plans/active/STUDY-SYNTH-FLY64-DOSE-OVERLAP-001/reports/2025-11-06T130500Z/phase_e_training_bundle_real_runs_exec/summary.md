# Phase E6 Dense/Baseline Evidence — Planning Snapshot

**Timestamp:** 2025-11-06T13:05:00Z  
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison prep (Phase E real bundle evidence)

## Current Status
- CLI stdout now carries bundle/SHA lines with view/dose context; regression test coverage landed in Attempt #103 (b7aea51e).
- Deterministic real runs for dose=1000 dense (gs2) and baseline (gs1) remain outstanding; no fresh bundles or SHA proofs exist under the active hub.
- `tmp/phase_c_f2_cli` / `tmp/phase_d_f2_cli` / `tmp/phase_e_training_gs2` are absent in the workspace; regeneration is required before execution.

## Loop Goals (Ralph)
1. Tighten regression by asserting CLI stdout emits artifact-relative bundle paths (not absolute) and that SHA digests printed in stdout match manifest entries.
2. Execute deterministic training runs for dose=1000 dense & baseline views, capturing logs, manifests, bundles, and SHA256 proofs in this hub.
3. Archive artifacts (manifests, skip summaries, bundles, checksum manifests) and summarize outcomes plus residual gaps (sparse run, higher-dose backlog).

## Key References
- `docs/findings.md`: POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001.
- `specs/ptychodus_api_spec.md` §4.6 — bundle persistence + SHA requirements.
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md` §268 — Phase E6 evidence expectations.
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md` — authoritative dataset/checkpoint map.

## Artifacts To Produce
- `red/pytest_training_cli_relative_red.log` (expected failure before CLI stdout change).
- `green/pytest_training_cli_relative_green.log`, `green/pytest_training_cli_suite_green.log`, `collect/pytest_training_cli_collect.log`.
- `cli/dose1000_dense_gs2.log`, `cli/dose1000_baseline_gs1.log`.
- `data/{dose1000_dense_manifest.json,dose1000_baseline_manifest.json,skip_summary.json,wts_dense_gs2.h5.zip,wts_baseline_gs1.h5.zip}`.
- `analysis/{bundle_checksums.txt,training_manifest_pretty.json}` plus updated `summary.md`.

## Risks / Watchouts
- CLI currently prints absolute bundle paths; test will fail until stdout is normalized.
- Regeneration of Phase C/D assets is mandatory if tmp roots missing; ensure artifact logs captured per policy.
- pty-chi artifacts unchanged this loop; avoid touching Phase F directories.
- Maintain `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` for every pytest/CLI invocation.

