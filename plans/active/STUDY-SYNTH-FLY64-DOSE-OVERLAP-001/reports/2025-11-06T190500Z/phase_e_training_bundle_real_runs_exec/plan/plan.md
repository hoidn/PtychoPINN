# Phase E6 Dense/Baseline Evidence — Loop Plan (2025-11-06T19:05:00Z)

## Objective
Deliver deterministic Phase E6 runs for dose=1000 dense (gs2) and baseline (gs1) with archival proof, while augmenting the CLI manifest schema to record bundle sizes for ongoing integrity tracking.

## Context Refresh
- SHA parity enforcement landed in Attempt 2025-11-06T170500Z; manifest + stdout alignment proven in tests.
- Real-run artifacts for dense/baseline were deferred; `bin/archive_phase_e_outputs.py` remains the canonical archiver.
- Bundle size metadata is absent, making it hard to detect truncated archives when comparing runs over time.

## Deliverables This Loop
1. Modify `studies/fly64_dose_overlap/training.py` so `run_training_job` captures bundle file sizes and persists them (manifest + stdout if applicable).
2. Extend `test_training_cli_records_bundle_path` to assert `bundle_size_bytes` presence and format.
3. RED→GREEN evidence for the selector above (logs under this hub) plus `--collect-only` proof.
4. Deterministic CLI executions for dense gs2 and baseline gs1 with logs under `cli/`.
5. Archive outputs (manifest, skip summary, bundles) via `bin/archive_phase_e_outputs.py`, generating checksum + size report under `analysis/`.
6. Update `summary.md` and `docs/fix_plan.md` with results; ensure findings ledger references TYPE-PATH-001 remains accurate.

## Step-by-Step
1. **Prep (idempotent):** Verify `tmp/phase_c_f2_cli` and `tmp/phase_d_f2_cli`; regenerate via How-To prep commands if missing.
2. **Implementation + Tests:**
   - Update `run_training_job` (and associated serialization) to include `bundle_size_bytes` when a bundle exists.
   - Adjust CLI stdout emission to display the size for transparency.
   - Strengthen `test_training_cli_records_bundle_path` to assert the new field and stdout snippet.
   - Execute RED/GREEN cycle for `pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv` with tees to `red/` and `green/`.
   - Run `pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv` and the focused subset once GREEN, capturing logs in `collect/` and `green/`.
3. **Deterministic Runs:**
   - Execute dense gs2 and baseline gs1 CLI runs (non dry-run) with deterministic flags; tee logs to `cli/`.
   - Verify CLI logs include bundle path, sha, and size lines.
4. **Archive + Proof:**
   - Invoke `bin/archive_phase_e_outputs.py` for dose=1000 (views: dense baseline).
   - Ensure `analysis/bundle_checksums.txt` includes size remarks (update script output if needed).
5. **Documentation:**
   - Update `summary.md` with run metadata (command log references, manifest path, checksum + size table).
   - Append Latest Attempt entry in `docs/fix_plan.md` with artifact path + findings applied.
   - Record knowledge in `galph_memory.md`.

## References
- specs/ptychodus_api_spec.md §4.6 (bundle persistence contract)
- docs/TESTING_GUIDE.md §3.2 (Phase E selectors)
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md §268 (Phase E6 evidence)
- docs/findings.md (POLICY-001 / CONFIG-001 / DATA-001 / OVERSAMPLING-001 / TYPE-PATH-001)

## Exit Criteria
- Manifest includes `bundle_size_bytes` for each job; stdout echoes size info.
- Targeted test RED→GREEN with collect evidence recorded in this hub.
- Dense gs2 + baseline gs1 deterministic runs produce manifest/bundles archived with checksum + size proof.
- Summary + ledger updated with artifact pointers and findings compliance.
