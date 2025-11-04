Summary: Enforce Phase E CLI stdout/manifest SHA parity while capturing dense/baseline deterministic evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (Phase E real bundle evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv; pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv; pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/

Do Now:
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase E6 dense/baseline real-run evidence (stdout SHA parity + deterministic runs)
- Implement: tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path — compare stdout SHA256 lines to manifest `bundle_sha256` entries; adjust studies/fly64_dose_overlap/training.py::main only if the RED run exposes a mismatch.
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/red/pytest_training_cli_sha_red.log
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_sha_green.log
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/green/pytest_bundle_sha_green.log
- Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/green/pytest_training_cli_suite_green.log
- Collect: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py --collect-only -k training_cli -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/collect/pytest_training_cli_collect.log
- Prep: if [ ! -d tmp/phase_c_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --base-npz tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz --output-root tmp/phase_c_f2_cli | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/prep/phase_c_generation.log; fi
- Prep: if [ ! -d tmp/phase_d_f2_cli ]; then mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/prep && AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root tmp/phase_c_f2_cli --output-root tmp/phase_d_f2_cli --doses 1000 --views dense sparse --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/prep | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/prep/phase_d_generation.log; fi
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view dense --gridsize 2 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_dense_gs2.log
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/phase_e_training_gs2 --dose 1000 --view baseline --gridsize 1 --accelerator cpu --deterministic --num-workers 0 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/cli/dose1000_baseline_gs1.log
- Archive: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py --phase-e-root tmp/phase_e_training_gs2 --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec --dose 1000 --views dense baseline
- Summarize: python - <<'PY'
from pathlib import Path
import json

hub = Path("plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec")
summary_path = hub / "summary.md"
checksums_path = hub / "analysis" / "bundle_checksums.txt"
manifest_path = hub / "data" / "training_manifest_dose1000.json"
skip_summary_path = hub / "data" / "skip_summary_dose1000.json"

checksums = checksums_path.read_text().strip().splitlines() if checksums_path.exists() else []
manifest = {}
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    jobs = {(job.get("view"), job.get("dose")): job for job in manifest.get("jobs", [])}
else:
    jobs = {}

cli_logs = [
    ("cli/dose1000_dense_gs2.log", hub / "cli" / "dose1000_dense_gs2.log"),
    ("cli/dose1000_baseline_gs1.log", hub / "cli" / "dose1000_baseline_gs1.log"),
]
prep_logs = [
    ("prep/phase_c_generation.log", hub / "prep" / "phase_c_generation.log"),
    ("prep/phase_d_generation.log", hub / "prep" / "phase_d_generation.log"),
]

def format_entry(name, path):
    status = "present" if path.exists() else "missing"
    return f"- {name}: {status}"

def manifest_digest(view):
    job = jobs.get((view, 1000))
    if not job:
        return f"- {view}: manifest entry missing"
    result = job.get("result", {})
    bundle = result.get("bundle_path", "<missing>")
    sha = result.get("bundle_sha256", "<missing>")
    return f"- {view}: bundle={bundle}, sha256={sha}"

lines = [
    "# Phase E6 Dense/Baseline Evidence — Results",
    "",
    "## CLI Outputs",
]
lines.extend(format_entry(name, path) for name, path in cli_logs)
lines.append("")
lines.append("## Prep Logs")
lines.extend(format_entry(name, path) for name, path in prep_logs)
lines.append("")
lines.append("## Manifest & SHA Parity")
lines.extend(manifest_digest(view) for view in ("dense", "baseline"))
lines.append("")
lines.append("## Archive Checksums")
if checksums:
    lines.extend(f"- {row}" for row in checksums)
else:
    lines.append("- bundle_checksums.txt missing")
lines.append("")
if skip_summary_path.exists():
    lines.append("## Skip Summary")
    lines.append(f"- skip_summary_dose1000.json present: {skip_summary_path.exists()}")
    lines.append("")
lines.append("## Next Steps")
lines.append("- [ ] Sparse view deterministic run")
lines.append("- [ ] Update Phase G inventory after sparse evidence")

summary_path.write_text("\n".join(lines))
PY

How-To Map:
- `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_training_cli_records_bundle_path -vv` (run RED then GREEN logs under this hub).
- `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py::test_execute_training_job_persists_bundle -vv` to confirm bundle persistence contract unchanged.
- `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_training.py -k training_cli -vv` plus collect-only selector for guardrail.
- Regenerate Phase C/D assets only if tmp roots absent (commands above capture logs into `prep/`).
- Execute dense/baseline CLI runs with deterministic knobs, capturing stdout via `tee` into `cli/` logs.
- Archive bundles/manifests with `archive_phase_e_outputs.py` to compute SHA parity and populate `analysis/` outputs.
- Run the provided Python snippet to refresh `summary.md` with digest results and outstanding tasks.

Pitfalls To Avoid:
- Do not skip the RED log; capture the failing output even if mismatch is neutral (document in summary if already GREEN).
- Preserve CONFIG-001 ordering inside training helpers; avoid touching legacy bridge logic.
- Ensure CLI commands run with `--deterministic --num-workers 0` to keep evidence reproducible.
- Record every CLI/test invocation with `tee`; missing logs invalidate audit trail.
- Keep artifact-relative paths when archiving; never move bundles outside the hub.
- Recreate tmp datasets when absent rather than pointing to prior report directories.
- Leave Phase F artifacts untouched; this loop handles Phase E only.
- Maintain PyTorch dependency assumption (POLICY-001); no torch-optional fallbacks.
- Avoid editing prompt/plan docs outside this focus without documenting rationale.

If Blocked:
- Capture failing test/CLI logs under `red/` or `cli/failed_*.log`, update `summary.md` with the error signature, and mark the attempt as blocked in docs/fix_plan.md.
- If datasets cannot regenerate (missing base NPZ), stop after logging the error, keep tmp roots unchanged, and document the dependency gap in summary + ledger.
- On SHA mismatches, retain both manifest and checksum outputs, note the mismatch in `analysis/summary.md`, and halt before attempting Phase G comparisons.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch runtime required for training helpers; do not gate dependencies (docs/findings.md:8).
- CONFIG-001 — Maintain legacy bridge ordering before backend-specific imports (docs/findings.md:10).
- DATA-001 — Regenerated NPZ datasets must satisfy canonical contract keys/dtypes (docs/findings.md:14).
- OVERSAMPLING-001 — Dense view expects K ≥ C; document sparse skips if triggered (docs/findings.md:17).

Pointers:
- docs/fix_plan.md:17 — Current initiative status and latest attempt summary.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/plan/plan.md:1 — Loop plan with step-by-step deliverables.
- tests/study/test_dose_overlap_training.py:1450 — Target test covering CLI stdout bundle/SHA assertions.
- docs/TESTING_GUIDE.md:101 — Phase E training CLI selectors and guardrail commands.
- specs/ptychodus_api_spec.md:239 — Bundle persistence contract and SHA requirements.

Next Up (optional):
- Sparse view Phase E deterministic run after dense/baseline evidence goes green.
