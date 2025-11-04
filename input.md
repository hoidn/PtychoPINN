Summary: Capture Phase E dense/baseline training bundles with SHA256 evidence now that MemmapDatasetBridge fallback landed.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E real bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv; pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/

Do Now:
- Implement: tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path — add manifest assertions requiring bundle_sha256 alongside bundle_path; leave the mock runner unchanged initially so RED captures the missing field before wiring in the checksum.
- Validate: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/red/pytest_manifest_red.log (expect failure while mock lacks bundle_sha256).
- Update mock: tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path — extend mock_execute_training_job to return a 64-character hex bundle_sha256 so CLI manifests propagate the checksum.
- Validate: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/green/pytest_manifest_green.log (confirm manifest now contains bundle_sha256).
- Validate: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/green/pytest_bundle_sha_green.log.
- Validate: pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/green/pytest_training_cli_suite_green.log.
- Collect: pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/collect/pytest_training_cli_collect.log.
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/cli/dose1000_dense_gs2.log.
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/cli/dose1000_baseline_gs1.log.
- Archive: mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/data && cp tmp/phase_e_training_gs2/training_manifest.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/data/ && cp tmp/phase_e_training_gs2/skip_summary.json plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/data/ && find tmp/phase_e_training_gs2/dose_1000 -name 'wts.h5.zip' -exec cp {} plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/data/ \;.
- Summarize: (cd plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/data && sha256sum wts.h5.zip* ) | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/analysis/bundle_checksums.txt; python - <<'PY' > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/analysis/training_manifest_pretty.json
import json, pathlib
manifest = json.loads(pathlib.Path("tmp/phase_e_training_gs2/training_manifest.json").read_text())
json.dump(manifest, open("/dev/stdout", "w"), indent=2, sort_keys=True)
PY
- Summarize: Update plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T050500Z/phase_e_training_bundle_real_runs_retry/analysis/summary.md with CLI outcomes, bundle locations, checksum verification, and remaining sparse/doc tasks before touching docs/registries.

Priorities & Rationale:
- Phase E6 evidence gate (`plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268`) demands real bundle artifacts plus SHA256 proof before Phase G comparisons can proceed.
- specs/ptychodus_api_spec.md §4.6 mandates bundle persistence integrity (bundle_path + bundle_sha256) captured in manifests for reproducibility.
- specs/data_contracts.md:207 + finding DATA-001 require we validate real CLI runs after Memmap fallback to ensure dataset readers behave under canonical/legacy keys.
- Attempt #96 (docs/fix_plan.md:54) documented KeyError blocker; this loop confirms the fallback resolved it and preserves manifest schema.
- Maintaining POLICY-001 / CONFIG-001 compliance ensures PyTorch training path remains deterministic and reusable for runtime parity analysis.

How-To Map:
- Ensure `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` for every pytest/CLI command; reuse tmp Phase C/D directories unless missing.
- Before CLI execution confirm tmp/phase_c_f2_cli and tmp/phase_d_f2_cli exist; regenerate via docs/TESTING_GUIDE.md commands only if absent.
- After CLI runs, inspect `training_manifest.json` to verify each job has `status: "success"`, `bundle_path`, and `bundle_sha256`; if not, stop and log blocker.
- Use `python -m json.tool` as fallback if manifest pretty-print fails.
- When copying bundles, keep original directory structure intact in tmp; use artifact hub for immutable evidence only.
- Avoid running sparse view jobs this loop; focus on dense/baseline to unblock Phase G comparisons.

Pitfalls To Avoid:
- Do not skip RED phase for the updated manifest test; capture failure log first.
- Avoid modifying Phase E generators or overlap filters—only training/test/assertion updates allowed.
- Do not remove prior artifact hubs or overwrite Attempt #98 summaries; new evidence must live under the fresh timestamp.
- Keep deterministic CLI flags; do not introduce GPU execution or nondeterministic worker counts.
- Do not touch protected modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Ensure SHA256 sums are computed on copied bundles, not original tmp files (integrity capture lives in artifact hub).
- If CLI fails, refrain from rerunning blindly; document failure signature immediately.
- Do not update docs/TESTING_GUIDE.md or TEST_SUITE_INDEX.md until GREEN evidence confirmed.
- Preserve skip_summary JSON structure when copying (no jq edits inline).

If Blocked:
- Capture failing CLI command output to the relevant `cli/*.log`, annotate the error in `analysis/summary.md`, and mark Attempt #99 blocked in docs/fix_plan.md + galph_memory.md.
- If tests still fail after fallback, retain RED log, collect stack trace, and halt before CLI execution.

Findings Applied (Mandatory):
- POLICY-001 — Torch backend required; CLI remains torch-backed (docs/findings.md:8).
- CONFIG-001 — Maintain legacy params bridge sequencing before data/model calls (docs/findings.md:10).
- DATA-001 — Validate canonical NPZ schema + legacy tolerance by exercising real datasets (docs/findings.md:14).
- OVERSAMPLING-001 — Keep gridsize/K invariants during CLI reruns (docs/findings.md:17).

Pointers:
- specs/data_contracts.md:207 — Diffraction vs diff3d schema requirements.
- specs/ptychodus_api_spec.md:239 — Bundle persistence and checksum contract.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 evidence checklist.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T030500Z/phase_e_memmap_diffraction_fallback/analysis/summary.md — Fallback implementation evidence for reference.
- docs/fix_plan.md:54-55 — Attempts #96-#98 context and remaining exit criteria.

Next Up (optional): Re-run Phase G dense comparison CLI once bundles archived and summary updated.
