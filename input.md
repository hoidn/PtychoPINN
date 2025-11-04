Summary: Capture deterministic Phase E dense/baseline training bundles with SHA256 proof now that tests enforce on-disk integrity.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E real bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/

Do Now:
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase E6 dense/baseline real-run evidence
- Implement: studies/fly64_dose_overlap/training.py::main — emit each job's bundle_path and bundle_sha256 to stdout right after run_training_job returns so CLI logs capture the digest alongside manifest pointers.
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/green/pytest_bundle_sha_green.log
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_green.log
- Collect: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/collect/pytest_training_cli_collect.log
- Prep: if [ ! -d tmp/phase_c_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --output-root tmp/phase_c_f2_cli --base-npz tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/prep/phase_c_generation.log; fi
- Prep: if [ ! -d tmp/phase_d_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_f2_cli --output-root tmp/phase_d_f2_cli --doses 1000 --views dense baseline --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/prep | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/prep/phase_d_generation.log; fi
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_dense_gs2.log
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_baseline_gs1.log
- Archive: python - <<'PY'\nfrom pathlib import Path\nimport shutil\nhub = Path('plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec')\nroot_data = hub / 'data'\nroot_data.mkdir(parents=True, exist_ok=True)\nmanifest_src = Path('tmp/phase_e_training_gs2/training_manifest.json')\nskip_src = Path('tmp/phase_e_training_gs2/skip_summary.json')\nif not manifest_src.exists():\n    raise SystemExit('training_manifest.json missing — CLI run failed or wrong path')\nshutil.copy2(manifest_src, root_data / 'training_manifest.json')\nif skip_src.exists():\n    shutil.copy2(skip_src, root_data / 'skip_summary.json')\nfor bundle_path in Path('tmp/phase_e_training_gs2').glob('dose_1000/*/wts.h5.zip'):\n    view = bundle_path.parent.name\n    dest = root_data / f'wts_{view}.h5.zip'\n    shutil.copy2(bundle_path, dest)\nPY
- Verify: (cd plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/data && sha256sum wts_*.h5.zip) | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/analysis/bundle_checksums.txt
- Verify: python - <<'PY' | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/analysis/training_manifest_pretty.json
import json
from pathlib import Path
manifest = json.loads(Path('tmp/phase_e_training_gs2/training_manifest.json').read_text())
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
- Document: python - <<'PY'
import json
from pathlib import Path
hub = Path('plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec')
summary = hub / 'analysis' / 'summary.md'
summary.parent.mkdir(parents=True, exist_ok=True)
manifest = json.loads((hub / 'data' / 'training_manifest.json').read_text())
checksums_path = hub / 'analysis' / 'bundle_checksums.txt'
checksums = checksums_path.read_text().strip() if checksums_path.exists() else 'SHA256 data missing'
lines = ['# Phase E6 Dense/Baseline CLI Evidence — RED→GREEN', '', '## CLI Runs', '- Dense gs2 CLI log: cli/dose1000_dense_gs2.log', '- Baseline gs1 CLI log: cli/dose1000_baseline_gs1.log', '', '## Bundle Artifacts', f'- training_manifest.json ({len(manifest.get(\"jobs\", []))} jobs captured)', '- skip_summary.json (present if sparse views skipped)', '']
for job in manifest.get('jobs', []):
    result = job.get('result', {})
    lines.append(f\"- Job view={job.get('view')} dose={job.get('dose')} gridsize={job.get('gridsize')} → bundle={result.get('bundle_path')} sha256={result.get('bundle_sha256')}\")
lines.extend(['', '## SHA256 Proof', '```', checksums, '```', '', 'Next actions: extend coverage to sparse view and feed bundles into Phase G comparisons.'])
summary.write_text('\\n'.join(lines))
PY
- Ledger: note attempt + artifact paths in docs/fix_plan.md once runs complete.

Priorities & Rationale:
- specs/ptychodus_api_spec.md:239 — Phase E bundles must persist as `wts.h5.zip` with verifiable SHA256 digests; CLI evidence proves contract compliance.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 exit criteria demand real dense/baseline training runs with deterministic settings and archived outputs.
- docs/TESTING_GUIDE.md:101-119 — Authoritative pytest selectors and CLI command patterns for Phase E validation; reuse ensures reproducible harness behavior.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/plan/plan.md — Task breakdown (E1–E5) aligns this Do Now with staged evidence capture.

How-To Map:
- Set env: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- Run targeted tests & collect proof (commands listed under Validate/Collect).
- Regenerate Phase C/D data only if directories missing (Prep commands).
- Execute dense/baseline CLI runs with deterministic flags (Execute commands).
- Copy manifests/bundles into artifact hub and compute SHA256 (Archive/Verify commands).
- Pretty-print manifest and update analysis/summary.md via provided Python snippets.
- Update docs/fix_plan.md Attempt entry referencing new artifacts after evidence captured.

Pitfalls To Avoid:
- Do not delete existing tmp datasets; deterministic runs rely on consistent RNG state.
- Keep `AUTHORITATIVE_CMDS_DOC` exported for every pytest/CLI invocation (policy guardrail).
- Avoid GPU accelerators; stick to CPU + deterministic flags per reproducibility rules.
- Confirm the archive script produces `wts_dense.h5.zip` and `wts_baseline.h5.zip` before running SHA256; rerun if either bundle missing.
- Do not overwrite existing artifact hubs; all outputs must land in the 2025-11-06T090500Z directory.
- Capture stderr/stdout via `tee`; missing logs invalidate evidence.
- If Phase C/D regeneration reruns, archive logs under `prep/` and note rerun in summary.
- Do not modify Phase D sparse logic this loop; focus strictly on dense/baseline evidence.
- Stop if SHA256 mismatch occurs; record discrepancy instead of forcing success.
- Avoid editing core trainer logic beyond stdout summary emission mandated above.

If Blocked:
- Capture failing CLI stderr to `cli/*.log`, summarize the error in analysis/summary.md, and mark attempt BLOCKED in docs/fix_plan.md with exit criteria not met.
- If dataset generation scripts fail, save full command + traceback under `prep/` and halt further steps until supervisor guidance.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency satisfied; runs stay within torch-enabled environment.
- CONFIG-001 — Maintain `update_legacy_dict` sequencing via existing CLI (no edits to ordering).
- DATA-001 — Regenerated datasets (if needed) must obey canonical NPZ schema; validator already enforces.
- OVERSAMPLING-001 — Respect gridsize rules by using gs1 baseline and gs2 dense with K≥C.

Pointers:
- specs/ptychodus_api_spec.md:239 (bundle persistence contract)
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 (Phase E6 requirements)
- docs/TESTING_GUIDE.md:101-119 (Phase E pytest/CLI commands)
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T090500Z/phase_e_training_bundle_real_runs_exec/plan/plan.md (current loop tasks)

Next Up (optional):
- Extend real-run coverage to sparse view and capture SHA256 evidence after dense/baseline succeed.
