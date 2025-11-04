# Phase E6 Dense/Baseline Evidence — Loop Plan (2025-11-06T21:05:00Z)

## Objective
Capture deterministic Phase E6 training runs for the dense (gs2) and baseline (gs1) views at dose=1000, enrich the archival helper to report bundle sizes, and surface checksum+size proof inside this hub.

## Context Refresh
- Bundle size tracking logic and CLI test hardening landed in Attempt 2025-11-06T190500Z+exec.
- Real CLI executions remain outstanding; previous loops deferred them to avoid blocking the size feature.
- `bin/archive_phase_e_outputs.py` currently emits only SHA lines; we now require size parity to accompany checksum evidence.

## Deliverables This Loop
1. Deterministic Phase E6 CLI runs for dense gs2 and baseline gs1 at dose=1000 with logs captured under `cli/`.
2. Updated archive helper (`plans/.../bin/archive_phase_e_outputs.py`) to ingest `bundle_size_bytes`, validate parity with filesystem stat, and append size info to `analysis/bundle_checksums.txt`.
3. RED→GREEN proof (if script unit harness updated) or command transcript verifying size validation behaviour.
4. Archived manifest, skip summary, and bundle payloads within this hub’s `data/` directory.
5. Updated `summary.md` documenting commands, SHA+size table, and any anomalies; ledger + galph_memory refresh.

## Step-by-Step
1. **Prep:** Confirm Phase C/D roots exist (`tmp/phase_c_f2_cli`, `tmp/phase_d_f2_cli`). Regenerate with How-To prep commands if missing.
2. **Implementation (Archive Helper):**
   - Extend `archive_phase_e_outputs.py::archive_bundles` to fetch `bundle_size_bytes` for each job and compare against on-disk bytes.
   - Emit combined `sha256  filename  size_bytes` rows to `analysis/bundle_checksums.txt`.
   - Optionally add a lightweight pytest in `tests/study/test_dose_overlap_training.py` (or dedicated helper) covering new behaviour with fixture data.
3. **Deterministic Runs:**
   - Execute dense gs2 CLI run with deterministic flags; tee stdout to `cli/dose1000_dense_gs2.log`.
   - Execute baseline gs1 CLI run; tee to `cli/dose1000_baseline_gs1.log`.
4. **Archive + Proof:**
   - Run `archive_phase_e_outputs.py` pointing to this hub; verify failure if manifest size mismatch occurs.
   - Capture script stdout/stderr in `analysis/archive_phase_e.log`.
5. **Documentation:**
   - Update `summary.md` with command list, bundle table (view, sha, size), and validation notes.
   - Append Attempt entry to `docs/fix_plan.md`; log loop in `galph_memory.md`.

## References
- specs/ptychodus_api_spec.md §4.6 (bundle persistence contract)
- docs/TESTING_GUIDE.md §3.2 (Phase E selectors)
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md §268 (Phase E6 evidence)
- docs/findings.md POLICY-001 / CONFIG-001 / DATA-001 / OVERSAMPLING-001 / TYPE-PATH-001

## Exit Criteria
- CLI logs show successful deterministic runs (dense & baseline) with bundle SHA/size lines.
- `analysis/bundle_checksums.txt` records SHA + filesystem size values matching manifest `bundle_size_bytes`.
- `data/` directory contains manifest + skip summary copies and both bundle archives.
- `summary.md`, `docs/fix_plan.md`, and `galph_memory.md` updated with evidence references.
