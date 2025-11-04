# Phase E6 Dense/Baseline Evidence — Summary (2025-11-06T210500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis — Phase E real bundle evidence  
**Mode:** TDD (execution loop pending)  
**Status:** ☐ In Progress

## Objectives
- Run deterministic Phase E6 training jobs for dense gs2 and baseline gs1 at dose=1000.
- Extend archive helper to validate and report bundle sizes alongside SHA256 checksums.
- Consolidate manifest, checksum, and size evidence inside this hub.

## Pending Tasks
- [ ] Update archive helper to compare manifest vs filesystem size.
- [ ] Execute dense gs2 CLI job (dose=1000) and capture stdout (`cli/dose1000_dense_gs2.log`).
- [ ] Execute baseline gs1 CLI job (dose=1000) and capture stdout (`cli/dose1000_baseline_gs1.log`).
- [ ] Run archive helper; store outputs under `analysis/` and `data/`.
- [ ] Summarize SHA+size table and CLI command list here.

## Notes
- Set `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before running pytest/CLI commands.
- If manifest `bundle_size_bytes` diverges from filesystem size, abort and capture failure output in `analysis/archive_phase_e.log`.
- Sparse view runs remain deferred; document status once dense/baseline evidence is captured.
