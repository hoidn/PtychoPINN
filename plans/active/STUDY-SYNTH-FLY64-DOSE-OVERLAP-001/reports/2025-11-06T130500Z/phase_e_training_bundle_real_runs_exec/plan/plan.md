# Phase E6 Dense/Baseline Evidence — Plan (2025-11-06T13:05Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Loop Type:** Supervisor planning (TDD implementation handoff)

## Objectives
1. Normalize CLI stdout bundle/SHA lines to use artifact-relative paths and assert manifest parity.
2. Regenerate Phase C/D assets iff missing; execute deterministic Phase E training for dose=1000 dense (gs2) and baseline (gs1).
3. Archive manifests, bundles, checksum proofs, and loop summary under this hub.

## Tasks & Evidence

| ID | Description | Evidence Target |
|----|-------------|-----------------|
| T1 | Extend `tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path` to parse stdout, assert bundle paths are relative, and confirm SHA digests match manifest entries. Capture RED log before code change. | `red/pytest_training_cli_relative_red.log` |
| T2 | Update `studies/fly64_dose_overlap/training.py::main` to emit relative bundle paths in stdout (matching manifest normalization). Re-run targeted selector until GREEN. | `green/pytest_training_cli_relative_green.log` |
| T3 | Run regression sweep: `pytest tests/study/test_dose_overlap_training.py -k training_cli -vv`; capture GREEN log + collect-only proof. | `green/pytest_training_cli_suite_green.log`, `collect/pytest_training_cli_collect.log` |
| T4 | Ensure prerequisites exist: regenerate Phase C (`python -m studies.fly64_dose_overlap.generation ...`) and Phase D overlap (`python -m studies.fly64_dose_overlap.overlap ...`) if tmp roots missing; archive stdout in `prep/`. | `prep/phase_c_generation.log`, `prep/phase_d_generation.log` |
| T5 | Execute deterministic CLI runs for dense gs2 + baseline gs1 (dose 1000) with CPU/deterministic knobs; capture stdout per job. | `cli/dose1000_dense_gs2.log`, `cli/dose1000_baseline_gs1.log` |
| T6 | Copy manifests, skip summaries, and bundles into `data/`; rename bundles (`wts_dense_gs2.h5.zip`, `wts_baseline_gs1.h5.zip`); compute SHA256 digest report and pretty-print manifest for audit. | `data/`, `analysis/bundle_checksums.txt`, `analysis/training_manifest_pretty.json` |
| T7 | Summarize outcomes (success metrics, SHA parity, outstanding sparse run) in `analysis/summary.md`; ensure ledger + galph_memory link artifacts. | `analysis/summary.md` |

## Guardrails
- Findings: POLICY-001 (PyTorch dependency), CONFIG-001 (legacy bridge), DATA-001 (dataset contract), OVERSAMPLING-001 (K ≥ C).
- All CLI/test commands must set `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
- Preserve tmp roots between commands; do not delete Phase F outputs.
- If CLI run fails, stop after archiving logs + update ledger with BLOCKED rationale.

