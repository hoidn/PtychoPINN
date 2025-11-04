Summary: Fix Phase F2 real-run blocker by making `ptychi_reconstruct_tike.py` honor orchestrator CLI arguments and rerunning the dense/train LSQML baseline.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2 — Phase F pty-chi baseline execution
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments -vv; pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv; pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/

Do Now — STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2:
  - Setup: mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/{red,green,collect,cli,real_run,docs} && export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md.
  - Test: tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments — add pytest that stubs pty-chi modules, calls `main(["--input-npz", ...])`, and asserts the stubbed helpers receive the CLI overrides; run RED `pytest tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments -vv || true` and tee output to `.../red/pytest_ptychi_cli_input_red.log` capturing the expected failure signature.
  - Implement: scripts/reconstruction/ptychi_reconstruct_tike.py::main — introduce argparse parsing for `--input-npz`, `--output-dir`, `--algorithm`, `--num-epochs`, `--n-images`; thread parsed Paths into `load_and_convert_tike_data`, ensure output directories exist, preserve defaults for manual runs, and keep non-zero return codes surfaced without raising.
  - Validate: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/green/pytest_ptychi_cli_input_green.log.
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
  - Validate: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/green/pytest_phase_f_cli_exec_green.log.
  - Suite: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/green/pytest_phase_f_cli_suite_green.log.
  - Collect: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/collect/pytest_phase_f_cli_collect.log.
  - Real Run: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/real_run --dose 1000 --view dense --split train --allow-missing-phase-d 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/cli/real_run_dense_train.log.
  - Docs: Summarize results in `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/docs/summary.md`, mark F2.2 progress in `phase_f_ptychi_baseline_plan/plan.md`, update `plans/active/.../test_strategy.md` and `docs/TESTING_GUIDE.md`/`docs/development/TEST_SUITE_INDEX.md` with the new script-level selector, and append Attempt #79 outcome to docs/fix_plan.md.

Priorities & Rationale:
- scripts/reconstruction/ptychi_reconstruct_tike.py:296 hardcodes dataset/output paths; respecting CLI overrides is prerequisite for F2.2 real runs.
- studies/fly64_dose_overlap/reconstruction.py:73 assembles `--input-npz`/`--output-dir` arguments, so downstream script must accept them to keep manifest telemetry truthful.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:34 marks F2.2 blocked on the CLI parsing fix; clearing this unlocks Phase G comparisons.
- docs/findings.md:8 (POLICY-001) obligates PyTorch availability, allowing us to rely on pty-chi without adding optional guards.
- docs/findings.md:10 (CONFIG-001) requires we keep reconstruction orchestration side-effect free—CLI bridge must not mutate `params.cfg` while honoring new inputs.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/{red,green,collect,cli,real_run,docs}
- pytest tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments -vv 2>&1 | tee .../red/pytest_ptychi_cli_input_red.log || true
- Implement argparse refactor in scripts/reconstruction/ptychi_reconstruct_tike.py (respect defaults, create directories, propagate return codes)
- pytest tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments -vv 2>&1 | tee .../green/pytest_ptychi_cli_input_green.log
- python - <<'PY'
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
- pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv 2>&1 | tee .../green/pytest_phase_f_cli_exec_green.log
- pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv 2>&1 | tee .../green/pytest_phase_f_cli_suite_green.log
- pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv 2>&1 | tee .../collect/pytest_phase_f_cli_collect.log
- python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root .../real_run --dose 1000 --view dense --split train --allow-missing-phase-d 2>&1 | tee .../cli/real_run_dense_train.log
- Update summary/test docs/plan/ledger with new evidence and artifact references

Pitfalls To Avoid:
- Do not tweak `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`; fix stays within CLI script and tests.
- Keep new pytest stubs lightweight—stub `ptychi` modules instead of importing heavy dependencies during tests.
- Preserve existing defaults for manual CLI usage; only override when arguments are provided.
- Ensure new argparse parsing returns `Path` objects and creates directories before writing logs.
- Capture RED and GREEN logs before/after implementation; do not overwrite prior evidence.
- Maintain DATA-001 compliance in synthetic fixtures (complex64 patches, amplitude diffraction) to avoid false positives.
- Keep environment frozen—no package installs or pip tweaks.
- Verify CLI command uses same artifact root pattern so manifest/logs land under the new timestamped hub.
- Avoid swallowing non-zero return codes; orchestrator expects return codes surfaced in manifest telemetry.

If Blocked:
- If `ptychi` imports fail or subprocess exits with missing dependency errors, capture the exact traceback into `.../docs/summary.md`, log the failure in docs/fix_plan.md Attempts History, and halt implementation until dependency availability is clarified.
- If argparse refactor causes regression in existing CLI tests, revert local changes, restore RED log evidence, and document the regression instead of force-passing tests.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency required; no optional gating when calling pty-chi.
- CONFIG-001 — Reconstruction CLI remains pure; no params.cfg writes while parsing new arguments.
- CONFIG-002 — Execution config responsibilities stay separate from CLI parsing.
- DATA-001 — Synthetic NPZ fixtures stay compliant (dtype/keys) before rerunning jobs.
- OVERSAMPLING-001 — Retain dense view (K≥C) to keep overlap constraints satisfied during real run.

Pointers:
- scripts/reconstruction/ptychi_reconstruct_tike.py:296
- studies/fly64_dose_overlap/reconstruction.py:73
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:34
- docs/fix_plan.md:52
- docs/findings.md:8

Next Up (optional):
- 1. After dense/train succeeds, extend real run to dense/test to validate multi-split coverage.

Doc Sync Plan (Conditional):
- After tests pass, update `docs/TESTING_GUIDE.md` §Phase F selectors and `docs/development/TEST_SUITE_INDEX.md` with `tests/scripts/test_ptychi_reconstruct_tike.py::test_main_uses_cli_arguments`, referencing new GREEN logs.
