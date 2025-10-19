Summary: Turn the PyTorch integration pytest from RED to GREEN by wiring `_run_pytorch_workflow` to the real train→infer subprocess pipeline and validating artifacts.
Mode: TDD
Focus: TEST-PYTORCH-001 — Phase C pytest modernization
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T122449Z/phase_c_modernization/{summary.md,pytest_modernization_green.log}

Do Now:
1. TEST-PYTORCH-001 C2.A @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md — Implement `_run_pytorch_workflow` to launch the documented train/infer subprocesses and return a SimpleNamespace with artifact paths (tests: none).
2. TEST-PYTORCH-001 C2.B+C2.C @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md — Update pytest assertions for success, refresh module docstring/status, and retire the skipped unittest scaffolding (tests: none).
3. TEST-PYTORCH-001 C2.D @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md + docs/fix_plan.md — Run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T122449Z/phase_c_modernization/pytest_modernization_green.log`, capture any `train_debug.log` into the same directory, summarize results, and log Attempt #6 in docs/fix_plan.md before exiting.

If Blocked: If either subprocess returns non-zero, keep the helper raising a descriptive `RuntimeError`, capture the failing log to the new timestamped directory, and document the exact command/output in docs/fix_plan.md Attempt history instead of proceeding.

Priorities & Rationale:
- tests/torch/test_integration_workflow_torch.py:65 — Helper still raises `NotImplementedError`, blocking GREEN run.
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md:36 — Phase C2 checklist defines the implementation/validation contract to satisfy this loop.
- docs/workflows/pytorch.md:128 — Checkpoint layout and deterministic trainer settings must remain intact when driving subprocesses.
- specs/ptychodus_api_spec.md:193 — Persistence contract requires checkpoint + recon artifacts after train/infer.
- docs/findings.md:8 — POLICY-001 mandates PyTorch availability; selector must assume torch is installed.

How-To Map:
- Create the artifact directory: `mkdir -p plans/active/TEST-PYTORCH-001/reports/2025-10-19T122449Z/phase_c_modernization`.
- In `_run_pytorch_workflow`, derive `training_output_dir = tmp_path / "training_outputs"` and `inference_output_dir = tmp_path / "pytorch_output"`; use `env = dict(cuda_cpu_env)` so CUDA stays disabled.
- Reproduce the subprocess commands from `git show 77f793c^:tests/torch/test_integration_workflow_torch.py` (train command with `--train_data_file`, `--test_data_file`, `--output_dir`, `--max_epochs 2`, `--n_images 64`, `--batch_size 4`, `--device cpu`, `--disable_mlflow`; inference command using `ptycho_torch.inference` and `--model_path`, `--test_data`, `--output_dir`, `--n_images 32`, `--device cpu`).
- Call `subprocess.run(..., capture_output=True, text=True, env=env, check=False)` for each command; if `returncode != 0`, raise `RuntimeError` with stdout/stderr in the message.
- After successful runs, compute canonical paths (`checkpoint_path = training_output_dir / "checkpoints" / "last.ckpt"`, `recon_amp_path = inference_output_dir / "reconstructed_amplitude.png"`, etc.) and return `SimpleNamespace` populated with these paths.
- Update `test_run_pytorch_train_save_load_infer` to consume the namespace, assert on `returncode`-derived behavior (no exception), then verify artifact existence and file sizes > 1 KB.
- Remove the skipped unittest class (or convert to thin delegators) and refresh the module docstring to reflect Phase C2 GREEN once the test passes.
- Run the mapped pytest selector with `CUDA_VISIBLE_DEVICES=""` and `tee` the output to `pytest_modernization_green.log`; move any generated `train_debug.log` into the same timestamped directory.
- Append a short narrative to `plans/active/TEST-PYTORCH-001/reports/2025-10-19T122449Z/phase_c_modernization/summary.md` (runtime, artifact paths, follow-ups), update the Phase C2 rows in `plan.md` / `implementation.md`, and add Attempt #6 to `docs/fix_plan.md` linking the new artifacts.

Pitfalls To Avoid:
- Do not leave `train_debug.log` at repo root—relocate it to the new artifact directory.
- Keep environment edits scoped; always start from `cuda_cpu_env` instead of mutating `os.environ`.
- Avoid changing PyTorch workflow scripts (`ptycho_torch/train.py`, `inference.py`) in this loop.
- Ensure subprocess commands run inside the pytest helper; do not shell out in the test body itself.
- Return a namespace/object with path attributes; do not rely on globals or implicit working directories.
- Preserve deterministic CPU execution (no accidental GPU usage or random seeds removed).
- Keep assertions specific—check both existence and minimum size so regressions surface early.
- Update documentation artifacts in the same loop; leaving plan checkboxes stale will block the next supervisor review.

Pointers:
- tests/torch/test_integration_workflow_torch.py:65 — Helper + pytest test to implement.
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md:34 — Detailed C2 guidance.
- docs/workflows/pytorch.md:128 — Checkpoint/determinism rules for Lightning workflows.
- specs/ptychodus_api_spec.md:193 — Persistence expectations after training/inference.
- docs/findings.md:8 — PyTorch dependency policy to respect.

Next Up: Phase C3 (artifact audit + documentation alignment) once the GREEN run and Attempt #6 are captured.
