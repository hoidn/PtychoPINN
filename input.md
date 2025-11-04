Summary: Produce Phase E dense/baseline training bundles with SHA256 evidence now that Memmap fallback is in place.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E real bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/

Do Now:
- Implement: tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle — after the bundle file is written, compute its SHA256 on disk and assert it matches result['bundle_sha256']; keep the dummy bundle content so the assertion exercises the real digest.
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/green/pytest_bundle_sha_green.log
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/green/pytest_training_cli_suite_green.log
- Collect: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/collect/pytest_training_cli_collect.log
- Data prep: if [ ! -d tmp/phase_c_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --output-root tmp/phase_c_f2_cli --base-npz tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/prep/phase_c_generation.log; fi
- Data prep: if [ ! -d tmp/phase_d_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_f2_cli --output-root tmp/phase_d_f2_cli --doses 1000 --views dense --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/prep | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/prep/phase_d_generation.log; fi
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/cli/dose1000_dense_gs2.log
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/cli/dose1000_baseline_gs1.log
- Archive: mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/data && cp tmp/phase_e_training_gs2/training_manifest.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/data/ && cp tmp/phase_e_training_gs2/skip_summary.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/data/ && find tmp/phase_e_training_gs2/dose_1000 -name 'wts.h5.zip' -exec cp {} plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/data/ \;
- Verify: (cd plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/data && sha256sum wts.h5.zip* ) | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/analysis/bundle_checksums.txt
- Verify: python - <<'PY' | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/analysis/manifest_sha_verification.txt
import json, hashlib, pathlib, sys
hub = pathlib.Path('plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry')
data_dir = hub / 'data'
manifest = json.loads((data_dir / 'training_manifest.json').read_text())
missing = []
for job in manifest['jobs']:
    result = job['result']
    bundle_rel = result.get('bundle_path')
    sha_manifest = result.get('bundle_sha256')
    if bundle_rel and sha_manifest:
        bundle_path = data_dir / bundle_rel
        if not bundle_path.exists():
            missing.append((job['view'], bundle_rel, 'missing file'))
            continue
        digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        if digest != sha_manifest:
            missing.append((job['view'], bundle_rel, f'mismatch: {digest}'))
print('manifest_sha_verification: OK' if not missing else f'FAIL: {missing}')
PY
- Summarize: python - <<'PY' > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/analysis/training_manifest_pretty.json
import json, pathlib
manifest = json.loads(pathlib.Path('plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/data/training_manifest.json').read_text())
json.dump(manifest, open('/dev/stdout', 'w'), indent=2, sort_keys=True)
PY
- Summarize: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/analysis/summary.md with CLI outcomes, bundle locations, checksum verification results, and any remaining sparse/doc follow-ups before touching docs or the ledger.
- Document: Once evidence is in place, record Attempt details in docs/fix_plan.md and append real-run notes + artifact links.

Priorities & Rationale:
- Phase E6 acceptance criteria (plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268) require dense/baseline bundles plus SHA256 proof before Phase G comparisons.
- specs/ptychodus_api_spec.md §4.6 mandates manifests expose bundle_path and bundle_sha256 to guarantee reproducible checkpoints.
- DATA-001 (specs/data_contracts.md:207) obligates us to re-run training after Memmap fallback validation to ensure legacy `diffraction` NPZ inputs stay compliant.
- Attempt #99 (docs/fix_plan.md:99) left real CLI execution outstanding; completing it unblocks Phase G comparisons and Study summary updates.
- Running targeted training_cli tests after tightening assertions protects against regressions while CLI commands produce artifacts.

How-To Map:
- Set AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md on every pytest/CLI invocation to respect the command contract.
- Use the existing dummy bundle content in test_execute_training_job_persists_bundle so SHA256 instrumentation remains deterministic; calculate digest with hashlib.sha256().
- For data prep commands, pass --doses 1000 and --views dense to minimize runtime while covering the jobs required for Phase G analysis.
- After CLI runs, inspect tmp/phase_e_training_gs2/training_manifest.json to ensure each job status is "success" with populated bundle fields before copying.
- When verifying manifest SHA, run the provided Python snippet so any mismatch is captured in manifest_sha_verification.txt (non-empty output should halt work).
- If tmp/phase_c_f2_cli or tmp/phase_d_f2_cli already exist, leave them untouched and document reuse in summary.md.

Pitfalls To Avoid:
- Do not skip the SHA equality assertion in the test—this loop must modify the test file to satisfy the implementation floor.
- Avoid regenerating Phase C/D datasets inside the artifact hub; keep tmp/ directories as the mutable workspace.
- Do not run training with GPU accelerators or non-deterministic worker counts; stay with CPU/deterministic knobs.
- If CLI fails, do not rerun blindly; capture the failing log and stop to document the blocker.
- Do not modify protected physics or TensorFlow core modules (ptycho/model.py, ptycho/diffsim.py, ptycho/tf_helper.py).
- Keep bundle copies only in the artifact hub; do not move originals out of tmp/phase_e_training_gs2/.
- Ensure sha256sum command runs inside the data directory so relative names in manifest remain valid.
- Do not update docs/TESTING_GUIDE.md or TEST_SUITE_INDEX.md unless selectors change; if unchanged, note that in summary.md.
- Preserve existing artifact files (e.g., prior green logs); append new evidence instead of overwriting without backups.

If Blocked:
- Capture failing CLI or pytest output in the respective log file, summarize the error in analysis/summary.md, and mark Attempt blocked in docs/fix_plan.md referencing the log.
- If dataset generation fails, archive the failing command output under prep/ and halt for follow-up—do not delete partial tmp outputs.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch backend is required; CLI and tests assume torch>=2.2 is available.
- CONFIG-001 — Maintain update_legacy_dict ordering via execute_training_job/CLI helpers.
- DATA-001 — Regenerated datasets must respect canonical NPZ schema and legacy fallback behavior.
- OVERSAMPLING-001 — gridsize choices (gs1 baseline, gs2 dense) must keep K vs C invariants intact.

Pointers:
- specs/ptychodus_api_spec.md:239 — Bundle persistence and checksum contract driving manifest expectations.
- specs/data_contracts.md:207 — Canonical vs legacy diffraction keys for MemmapDatasetBridge fallback validation.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 evidence checklist for real runs.
- docs/fix_plan.md:99 — Attempt #99 notes on pending CLI execution.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/analysis/summary.md — Reference for fallback implementation.

Next Up (optional): Once dense/baseline bundles are archived, proceed to Phase G dense comparison CLI with the new bundle artifacts.
