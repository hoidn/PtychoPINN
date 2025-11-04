Summary: Capture Phase F dry-run plus first LSQML execution evidence and sync plan/test docs for F2.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2 — Phase F pty-chi baseline execution
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv; pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2:
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md — promote `test_cli_executes_selected_jobs` to Active, capture execution-proof references (RED/GREEN/collect/CLI logs), and align Phase F coverage notes with execution telemetry requirements.
  - Data Prep: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python - <<'PY'
from pathlib import Path
import shutil
import numpy as np
from studies.fly64_dose_overlap.design import get_study_design

phase_c = Path("tmp/phase_c_f2_cli")
phase_d = Path("tmp/phase_d_f2_cli")
for root in (phase_c, phase_d):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

design = get_study_design()
rng = np.random.default_rng(123)

def make_payload(num_images: int = 8):
    diffraction = rng.random((num_images, 64, 64), dtype=np.float32)
    object_guess = (rng.random((128, 128), dtype=np.float32) + 1j * rng.random((128, 128), dtype=np.float32)).astype(np.complex64)
    probe_guess = (rng.random((64, 64), dtype=np.float32) + 1j * rng.random((64, 64), dtype=np.float32)).astype(np.complex64)
    patches = (rng.random((num_images, 128, 128), dtype=np.float32) + 1j * rng.random((num_images, 128, 128), dtype=np.float32)).astype(np.complex64)
    filenames = np.array([f"img_{i:04d}" for i in range(num_images)])
    coords_x = rng.random(num_images, dtype=np.float32)
    coords_y = rng.random(num_images, dtype=np.float32)
    return {
        "diffraction": diffraction.astype(np.float32),
        "objectGuess": object_guess,
        "probeGuess": probe_guess,
        "Y": patches,
        "xcoords": coords_x,
        "ycoords": coords_y,
        "filenames": filenames,
    }

payload = make_payload()
for dose in design.dose_list:
    dose_key = f"dose_{int(dose)}"
    c_dir = phase_c / dose_key
    c_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        np.savez_compressed(c_dir / f"patched_{split}.npz", **payload)

    d_root = phase_d / dose_key
    for view in ("dense", "sparse"):
        view_dir = d_root / view
        view_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            np.savez_compressed(view_dir / f"{view}_{split}.npz", **payload)
PY
  - Dry Run: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/cli && python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/cli --dose 1000 --view dense --allow-missing-phase-d --dry-run 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/cli/dry_run.log
  - Validate: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/green/pytest_phase_f_cli_exec_green.log
  - Suite: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/green/pytest_phase_f_cli_suite_green.log
  - Collect: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/collect/pytest_phase_f_cli_collect.log
  - Real Run: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/real_run/dose_1000/dense/train && python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/real_run --dose 1000 --view dense --split train --allow-missing-phase-d 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/real_run/dose_1000/dense/train/run.log
  - Docs: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/docs/summary.md — append dry-run + real-run telemetry, hardware/runtime notes, execution_results manifest excerpt, and reference the new logs; update `phase_f_ptychi_baseline_plan/plan.md` F2.1 row (mark complete once evidence lands) and link artifacts in summary/test_strategy.
  - Ledger: Update docs/fix_plan.md Attempts History (Attempt #78) with outcomes, blockers (if any), and artifact paths.

Priorities & Rationale:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:24-38 — F2.1/F2.2/F2.3 remain open; this loop must supply dry-run + real execution proof.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212-244 — `test_cli_executes_selected_jobs` currently listed as Planned; promoting it to Active requires execution logs and CLI artifacts.
- docs/TESTING_GUIDE.md:100-142 — authoritative commands demand `AUTHORITATIVE_CMDS_DOC` export and log capture for CLI/test runs.
- docs/findings.md:8-17 (CONFIG-001, CONFIG-002, DATA-001, POLICY-001, OVERSAMPLING-001) — govern params.cfg isolation, data contract compliance, PyTorch dependency, and overlap spacing documentation.
- specs/data_contracts.md:120-214 — enforce NPZ key/dtype expectations for the synthetic payloads generated in Data Prep.

How-To Map:
- Prepare deterministic synthetic datasets via the inline Python script (uses StudyDesign constants and complex64 payloads) before invoking the CLI.
- Run dry-run + real-run commands with `AUTHORITATIVE_CMDS_DOC` exported; capture stdout/stderr using `tee` directly into the initiative hub (`cli/dry_run.log`, `real_run/.../run.log`).
- After CLI commands, inspect `reconstruction_manifest.json`, `skip_summary.json`, and per-job `ptychi.log` files to summarize execution telemetry.
- Re-run targeted pytest selectors to confirm `test_cli_executes_selected_jobs` passes with the instrumentation, then re-collect to prove selector availability.
- Update test_strategy and summary docs once evidence is in place; include file:line anchors and artifact pointers per documentation standards.

Pitfalls To Avoid:
- Do not leave generated NPZs outside `tmp/`; clean them only after documentation references are captured.
- Ensure synthetic payloads use amplitude diffraction (float32) and complex64 `Y` arrays to satisfy DATA-001; avoid accidental float64 promotion.
- Keep CLI artifact roots within the initiative reports tree; no residual logs in repo root.
- Preserve deterministic RNG seed when generating payloads so reruns stay stable.
- If the real LSQML run fails (missing ptychi, CUDA issues), capture the non-zero return code and stderr in both CLI stdout and `ptychi.log` rather than re-running blindly.
- Do not modify `studies/fly64_dose_overlap/reconstruction.py` unless a blocker is discovered; this loop focuses on evidence + docs.
- Avoid rerouting `params.cfg`; CONFIG-001 bridge occurs inside downstream scripts.
- Keep `AUTHORITATIVE_CMDS_DOC` exported for every pytest/CLI command to stay compliant with testing guide.
- When updating docs, note exact artifact paths and execution timestamps; no vague references.
- Ensure manifest edits remain JSON-serializable (convert Paths to str).

If Blocked:
- Capture failing command output in the designated artifact directory (`cli/dry_run.log` or `real_run/.../run.log`), preserve the generated manifest/log (even if partial), and record the error signature plus unblocker hypothesis in docs/fix_plan.md Attempt #78 and summary.md before pausing.

Findings Applied (Mandatory):
- CONFIG-001 — Keep reconstruction orchestration pure; note downstream bridge responsibility in summary.
- CONFIG-002 — Treat execution config isolation as immutable; no `params.cfg` writes from CLI runs.
- DATA-001 — Synthetic NPZs must honor amplitude + complex64 contracts; validate shapes before running CLI.
- POLICY-001 — Assume torch>=2.2 present; a missing PyTorch/pty-chi dependency constitutes a blocker to log.
- OVERSAMPLING-001 — Document any skipped sparse views and tie them back to overlap spacing rationale.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:24-38
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212-244
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T180000Z/phase_f_ptychi_baseline_f2/docs/summary.md
- docs/TESTING_GUIDE.md:100-142
- specs/data_contracts.md:120-214

Next Up (optional):
- If real LSQML execution succeeds quickly, extend Phase F2.2 to process the matching `dense/test` split and capture comparative logs for Phase G planning.

Doc Sync Plan:
- After GREEN tests, rerun the collect-only selector (command above already included) and update `docs/TESTING_GUIDE.md` §2 plus `docs/development/TEST_SUITE_INDEX.md` with the ACTIVE status + artifact pointers for `test_cli_executes_selected_jobs`.
