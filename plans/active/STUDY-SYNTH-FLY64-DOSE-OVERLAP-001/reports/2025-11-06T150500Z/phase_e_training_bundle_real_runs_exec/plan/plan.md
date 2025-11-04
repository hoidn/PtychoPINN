# Phase E6 Dense/Baseline Evidence — Loop Plan (2025-11-06T15:05:00Z)

## Objective
Land the remaining Phase E6 evidence for dose=1000 dense (gs2) and baseline (gs1) runs by tightening stdout/manifest regression coverage and capturing deterministic CLI execution artifacts (logs, manifests, bundles, SHA proofs) under the new hub.

## Scope
- Focus item: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E real bundle evidence)
- Deliverables this loop:
  1. Strengthen `test_training_cli_records_bundle_path` so stdout SHA256 lines must match manifest entries.
  2. RED→GREEN log pair for the updated test (`tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path`).
  3. Deterministic Phase E CLI runs for dense gs2 + baseline gs1 with logs, manifests, bundles, and checksum manifest archived under this hub.
  4. Updated `analysis/summary.md` noting digest proofs and any regeneration steps.

## Key References
- docs/findings.md — POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001.
- specs/ptychodus_api_spec.md §4.6 — Bundle persistence + SHA requirements.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md §268 — Phase E6 evidence definition.
- plans/.../reports/2025-11-05T162500Z/phase_g_inventory/analysis/inventory.md — authoritative dataset checklist.
- docs/TESTING_GUIDE.md §3.2 — Phase E CLI selectors.

## Step-by-Step
1. **Prep** (idempotent regeneration)
   - If `tmp/phase_c_f2_cli/` missing → run `python -m studies.fly64_dose_overlap.generation ...` (base NPZ: `tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz`).
   - If `tmp/phase_d_f2_cli/` missing → run `python -m studies.fly64_dose_overlap.overlap ... --doses 1000 --views dense sparse` to rebuild dense/sparse Phase D assets within audit hub (`prep/`).
2. **TDD**
   - Edit `tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path` to gather stdout SHA lines and compare against manifest `bundle_sha256` values.
   - Run RED: selector above (expected failure until assertion matches CLI behavior). Capture log under `red/`.
   - If CLI already compliant, document neutral red (test fails before code change by temporarily stashing / or verifying?).
   - Run GREEN after any necessary CLI tweak; capture log under `green/` and collect log as guardrail.
3. **Deterministic CLI Runs**
   - Dense: `python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0` → log to `cli/dose1000_dense_gs2.log`.
   - Baseline: same command with `--view baseline --gridsize 1` → `cli/dose1000_baseline_gs1.log`.
4. **Archive + Proof**
   - `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py --phase-e-root tmp/phase_e_training_gs2 --hub <this hub> --dose 1000 --views dense baseline` to copy manifests/bundles/skip summaries and compute `analysis/bundle_checksums.txt` (verifies CLI SHA vs recomputed digest).
5. **Documentation**
   - Update `analysis/summary.md` with CLI log references, SHA proof, regeneration notes, and open next steps (sparse run backlog).
   - Ensure Attempt entry + galph_memory note reference this hub and enumerated artifacts.

## Artifacts
- Hub root: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/`
- RED/GREEN logs under `red/` and `green/`.
- CLI logs under `cli/`.
- Bundles/manifests/checksums under `data/` + `analysis/`.
- Summary updates captured in `summary.md` (this directory root) and appended to ledger.

## Exit Criteria for the Loop
- Strengthened test passes with GREEN log recorded.
- Dense + baseline CLI logs show SHA lines with [view/dose] context and manifest path/sha equality (confirmed by archive script).
- `analysis/bundle_checksums.txt` demonstrates computed digest matches manifest/CLI output for each view.
- `summary.md` updated with concrete observations and next-step pointer (sparse view).
