# Phase E6 Dense/Baseline Real Runs — Loop Summary (2025-11-06T190500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Phase:** E6 — Deterministic training evidence (dense gs2 + baseline gs1)
**Branch:** feature/torchapi-newprompt
**Mode:** TDD (supervisor scoped)

---

## Objective

Capture deterministic Phase E6 training runs for dose=1000 (dense gs2 and baseline gs1) with full manifest + checksum proof, now that stdout/manifest SHA parity is enforced. Extend manifest schema to include bundle size bytes for integrity tracking.

## Pending Actions

- [ ] Update training CLI manifest emission to record `bundle_size_bytes` alongside `bundle_sha256`.
- [ ] Strengthen `test_training_cli_records_bundle_path` to assert `bundle_size_bytes` presence and format.
- [ ] Execute deterministic CLI runs (dense gs2 and baseline gs1) with logs under `cli/`.
- [ ] Archive outputs via `bin/archive_phase_e_outputs.py`, capturing checksums and manifest snapshots under `analysis/` + `data/`.
- [ ] Summarize evidence and ledger updates within this hub.

## Notes

- Reuse `tmp/phase_c_f2_cli` and `tmp/phase_d_f2_cli` if intact; regenerate via How-To prep commands otherwise.
- Ensure `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` is exported for all pytest/CLI invocations.
- Findings to enforce: POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001, TYPE-PATH-001.

