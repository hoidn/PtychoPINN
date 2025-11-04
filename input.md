Summary: Extend sparse-view skip metadata regression and capture fresh Phase F dry-run evidence for missing Phase D NPZs.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2 — Phase F pty-chi baseline execution
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T020500Z/phase_f_ptychi_baseline_f2_sparse_skip_assertions/

Do Now:
- Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2): tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d — assert `manifest["missing_jobs"]` mirrors skip summary (length==6, sparse-only) and record `skip_summary["missing_phase_d_count"] == 6` with docstring schema note.
- Validate: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T020500Z/phase_f_ptychi_baseline_f2_sparse_skip_assertions/green/pytest_sparse_skip_green.log
- Capture CLI: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_sparse_missing --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T020500Z/phase_f_ptychi_baseline_f2_sparse_skip_assertions/cli --view sparse --dry-run --allow-missing-phase-d 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T020500Z/phase_f_ptychi_baseline_f2_sparse_skip_assertions/cli/dry_run_sparse.log
- Artifacts: tee pytest collect-only output to plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T020500Z/phase_f_ptychi_baseline_f2_sparse_skip_assertions/collect/pytest_sparse_skip_collect.log and summarize outcomes + skip field values in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T020500Z/phase_f_ptychi_baseline_f2_sparse_skip_assertions/docs/summary.md; log Attempt #84 results in docs/fix_plan.md with new artifact links.

Priorities & Rationale:
- docs/fix_plan.md:4 — Active focus requires sparse skip validation before Phase G comparisons; ledger now records instrumentation results but needs follow-through evidence.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/docs/summary.md:12 — Summary calls out `missing_jobs`/`missing_phase_d_count`; tests must lock those fields down.
- docs/TESTING_GUIDE.md:146 — Phase F section mandates manifest + skip summary proof for reconstruction workflows.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:228 — Phase F checklist expects regression for skip telemetry, preventing regressions when sparse NPZs are absent.

How-To Map:
- Prep tmp dataset (mirrors test fixture): \
python - <<'PY' \
from pathlib import Path \
import numpy as np \
from studies.fly64_dose_overlap.design import get_study_design \
phase_d_root = Path(\"tmp/phase_d_sparse_missing\") \
phase_d_root.mkdir(parents=True, exist_ok=True) \
minimal = { \
    \"diffraction\": np.random.rand(10, 64, 64).astype(np.float32), \
    \"objectGuess\": (np.random.rand(128, 128) + 1j * np.random.rand(128, 128)).astype(np.complex64), \
    \"probeGuess\": (np.random.rand(64, 64) + 1j * np.random.rand(64, 64)).astype(np.complex64), \
    \"Y\": (np.random.rand(10, 128, 128) + 1j * np.random.rand(10, 128, 128)).astype(np.complex64), \
    \"xcoords\": np.random.rand(10).astype(np.float32), \
    \"ycoords\": np.random.rand(10).astype(np.float32), \
    \"filenames\": np.array([f\"img_{i:04d}\" for i in range(10)]) \
} \
design = get_study_design() \
for dose in design.dose_list: \
    dense_dir = phase_d_root / f\"dose_{int(dose)}\" / \"dense\" \
    dense_dir.mkdir(parents=True, exist_ok=True) \
    for split in (\"train\", \"test\"): \
        np.savez_compressed(dense_dir / f\"dense_{split}.npz\", **minimal) \
PY
- Export env: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- RED (optional sanity): pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d -vv --maxfail=1
- Implement assertions, then rerun GREEN command (tee log to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T020500Z/phase_f_ptychi_baseline_f2_sparse_skip_assertions/green/pytest_sparse_skip_green.log`).
- Collect proof: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_skips_missing_phase_d --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T020500Z/phase_f_ptychi_baseline_f2_sparse_skip_assertions/collect/pytest_sparse_skip_collect.log
- CLI dry-run command (above) writes manifest/skip_summary into artifact hub; verify `missing_phase_d_count`.
- Update summary: record skip counts + command string in `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T020500Z/phase_f_ptychi_baseline_f2_sparse_skip_assertions/docs/summary.md` and cross-link in docs/fix_plan.md.

Pitfalls To Avoid:
- Keep reconstruction builder pure; pass skip_events list from caller only (CONFIG-001).
- Do not regenerate dense/test evidence; scope is sparse dry-run coverage.
- Avoid absolute paths in tests; rely on `tmp_path` or repo-relative resolution.
- Ensure CLI command uses dedicated `tmp/phase_d_sparse_missing` to avoid corrupting prior artifacts.
- Store every log under the reserved artifact hub; no root-level files.
- Do not relax DATA-001 assertions when fabricating NPZ content; use amplitude + complex64.
- Prevent flaky assertions by sorting skip entries before comparing if needed.
- Keep pytest exits non-zero on failure; no blanket `pytest -k` without explicit selector.
- Leave Phase G tasks untouched; focus solely on F2 skip telemetry.
- No doc/test registry edits unless selector changes demand it.

If Blocked:
- If CLI dry-run still finds sparse NPZs (count=0), document in summary, attach manifest/skip_summary, note dataset prep script output, and mark Attempt blocked in docs/fix_plan.md with follow-up to regenerate sparse NPZ data.

Findings Applied (Mandatory):
- CONFIG-001 — Builder must stay pure; skip metadata collected via injected list.
- DATA-001 — Temporary NPZs must honor canonical keys/dtypes even when sparse files omitted.
- POLICY-001 — Phase F CLI assumes torch>=2.2 present; ensure commands respect PyTorch requirement.
- OVERSAMPLING-001 — Skip reasons should cite overlap spacing guard; validate text remains clear.

Pointers:
- docs/fix_plan.md:4
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T003000Z/phase_f_ptychi_baseline_f2_sparse_skip/docs/summary.md:12
- docs/TESTING_GUIDE.md:146
- tests/study/test_dose_overlap_reconstruction.py:519

Next Up:
- Real sparse/train and sparse/test LSQML runs once skip metadata is locked down.
