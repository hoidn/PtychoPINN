Summary: Capture deterministic Phase E dense/baseline training bundles with SHA256 proof and codify CLI stdout digest checks before archiving evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E real bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv; pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/

Do Now:
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase E6 dense/baseline real-run evidence
- Implement: studies/fly64_dose_overlap/training.py::main — include view/dose context in bundle/SHA stdout lines and extend tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path to assert the new format (capsys capture saved to artifact hub).
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/red/pytest_training_cli_stdout_red.log
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_stdout_green.log
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/green/pytest_bundle_sha_green.log
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_green.log
- Collect: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/collect/pytest_training_cli_collect.log
- Prep: if [ ! -d tmp/phase_c_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --output-root tmp/phase_c_f2_cli --base-npz tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/prep/phase_c_generation.log; fi
- Prep: if [ ! -d tmp/phase_d_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_f2_cli --output-root tmp/phase_d_f2_cli --doses 1000 --views dense baseline --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/prep | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/prep/phase_d_generation.log; fi
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_dense_gs2.log
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_baseline_gs1.log
- Archive: python - <<'PY'
from pathlib import Path
import json
import shutil
hub = Path('plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec')
root_data = hub / 'data'
analysis_dir = hub / 'analysis'
root_data.mkdir(parents=True, exist_ok=True)
analysis_dir.mkdir(parents=True, exist_ok=True)
manifest_src = Path('tmp/phase_e_training_gs2/training_manifest.json')
skip_src = Path('tmp/phase_e_training_gs2/skip_summary.json')
if not manifest_src.exists():
    raise SystemExit('training_manifest.json missing — CLI run failed or wrong path')
shutil.copy2(manifest_src, root_data / 'training_manifest.json')
if skip_src.exists():
    shutil.copy2(skip_src, root_data / 'skip_summary.json')
for bundle_path in Path('tmp/phase_e_training_gs2').glob('dose_1000/*/wts.h5.zip'):
    view = bundle_path.parent.name
    dest = root_data / f'wts_{view}.h5.zip'
    shutil.copy2(bundle_path, dest)
manifest_pretty = json.loads(manifest_src.read_text())
(analysis_dir / 'training_manifest_pretty.json').write_text(json.dumps(manifest_pretty, indent=2))
PY
- Verify: (cd plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/data && sha256sum wts_*.h5.zip) | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/analysis/bundle_checksums.txt
- Summarize: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python - <<'PY'
from datetime import datetime
from pathlib import Path
hub = Path('plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec')
summary = hub / 'analysis' / 'summary.md'
ts = datetime.utcnow().isoformat() + 'Z'
lines = [
    f"# Phase E6 Dense/Baseline Training Evidence — {ts}",
    "",
    "## Outcomes",
    "- Dense gs2 and baseline gs1 CLI runs executed with deterministic knobs; stdout captured in cli/ logs.",
    "- Bundle paths and SHA256 digests verified; checksums recorded in analysis/bundle_checksums.txt.",
    "- training_manifest.json + skip_summary.json copied under data/ with pretty-print companion.",
    "",
    "## Next Steps",
    "- Execute sparse view once dense/baseline validated (pending skip threshold review).",
    "- Update docs/TESTING_GUIDE.md §2 and docs/development/TEST_SUITE_INDEX.md once CLI selectors remain green after sparse evidence.",
]
summary.write_text('\n'.join(lines))
PY

Priorities & Rationale:
- specs/ptychodus_api_spec.md:239 — CLI must produce `wts.h5.zip` bundles with integrity proof before Phase G comparisons; stdout digest ensures traceability.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268 — Phase E6 exit criteria demand deterministic training runs with SHA256 archival and manifest validation.
- docs/findings.md#L8 — POLICY-001 requires torch>=2.2 and justifies enforcing CLI execution on environments with PyTorch installed.
- docs/findings.md#L10 — CONFIG-001 mandates updating legacy params before data/model construction; CLI workflow must respect bridge ordering.
- docs/findings.md#L14 — DATA-001 keeps regenerated NPZ assets compliant; regenerate only via sanctioned generators to avoid contract drift.

How-To Map:
- Export env var before commands: `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
- Run RED test: `pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv` (expect fail until bundle/SHA lines include view/dose context) → log RED.
- Flip test GREEN after assertion update; rerun targeted selectors and CLI suite with `tee` to `green/` logs.
- Dataset prep (conditional) via provided generation/overlap commands; only run if tmp dirs missing.
- Execute dense/baseline CLI runs with deterministic flags; capture stdout using `tee` into `cli/` logs.
- Archive outputs using provided Python snippet; confirm manifest + skip summary exist.
- Compute bundle SHA digest with `sha256sum`; store in `analysis/bundle_checksums.txt`.
- Summarize outcomes using provided Python helper; review summary.md before final ledger update.

Pitfalls To Avoid:
- Do not delete or overwrite previous artifact hubs; use the new timestamped directory only.
- Keep CLI runs deterministic (`--deterministic`, `--num-workers 0`) to avoid checksum drift.
- Ensure params bridge (`update_legacy_dict`) occurs implicitly via CLI helpers; avoid importing training modules before configs load.
- Do not regenerate Phase C/D assets if tmp directories already present; preserve reproducibility.
- Avoid rerunning CLI with `--dry-run` — real bundles required for SHA proof.
- Do not stash bundles outside artifact hub; copy/rename via archive step only.
- Ensure SHA check uses copied bundles, not tmp originals, to detect corruption during transfer.
- Capture any CLI failure stdout/stderr before retrying; rename logs if multiple attempts occur.
- Leave sparse view untouched this loop to maintain one-thing-per-loop discipline.

If Blocked:
- If targeted test fails due to logging capture, keep RED log, halt implementation, and record issue + stdout snippet in docs/fix_plan.md Attempt entry.
- If CLI run errors (non-zero exit), stop after archiving failing log under `cli/`; update summary with error signature and mark attempt BLOCKED in ledger/memory.
- If SHA mismatch occurs, preserve both manifest and checksum outputs, skip doc sync, and note discrepancy for follow-up before ending loop.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency enforced; CLI execution assumes torch>=2.2 available.
- CONFIG-001 — Training CLI relies on config bridge; no direct params.cfg mutation allowed.
- DATA-001 — Generated NPZ datasets must remain spec-compliant; regeneration only via sanctioned commands.
- OVERSAMPLING-001 — Dense (gs2) and baseline (gs1) grid settings must match oversampling constraints.

Pointers:
- specs/ptychodus_api_spec.md:239
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:268
- docs/findings.md#L8
- docs/findings.md#L10
- docs/findings.md#L14
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T110500Z/phase_e_training_bundle_real_runs_exec/plan/plan.md

Next Up (optional):
- Phase E6 sparse view execution once dense/baseline evidence captured.
