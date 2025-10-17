Summary: Implement PyTorch CLI wiring so Phase E2 integration tests go green
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 – Phase E2 Integration Regression & Parity Harness (E2.C)
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_backend_selection.py -vv; pytest tests/torch/test_integration_workflow_torch.py -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T215500Z/{phase_e2_green.md,phase_e_backend_green.log,phase_e_integration_green.log}

Do Now:
1. INTEGRATE-PYTORCH-001 E2.C1+C2+C3+C4 @ plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md — implement train/inference CLI, MLflow disable flag, and lightning dependency updates; log decisions in phase_e2_green.md (tests: none)
2. INTEGRATE-PYTORCH-001 E2.C5 @ plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md — run pytest tests/torch/test_backend_selection.py -vv && pytest tests/torch/test_integration_workflow_torch.py -vv, store logs as phase_e_backend_green.log and phase_e_integration_green.log (tests: targeted)

If Blocked: Capture partial progress and errors in phase_e2_green.md, archive any failing pytest output under the timestamped reports directory, and note blockers in docs/fix_plan.md Attempt history.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md §Phase C — freshly authored execution plan for E2.C wiring.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T213500Z/red_phase.md §2 — documents the lightning import failure and CLI gaps you must close.
- specs/ptychodus_api_spec.md §4.5 — defines reconstructor CLI contract that PyTorch path must honour.
- docs/workflows/pytorch.md §2–5 — baseline expectations for PyTorch prerequisites and runtime flags.
- tests/torch/test_integration_workflow_torch.py — red test encoding the acceptance criteria for this loop.

How-To Map:
- Create working directory: `timestamp=2025-10-17T215500Z; mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp`.
- C1: In `ptycho_torch/train.py`, add argparse entry point matching TensorFlow CLI; hydrate dataclasses via config_bridge, call `update_legacy_dict(params.cfg, config)` before delegating to `run_cdi_example_torch`. Record summary + CLI synopsis in `reports/$timestamp/phase_e2_green.md`.
- C2: Extend `ptycho_torch/inference.py` CLI to load checkpoints emitted by C1; write reconstructed amplitude/phase PNGs into provided output dir (names must match test assertions). Document artifact names in phase_e2_green.md.
- C3: Wire `--disable_mlflow` flag to skip `mlflow.pytorch.autolog` and related logging; confirm default behaviour unchanged. Note toggle behaviour in the summary file.
- C4: Update `setup.py` (and pyproject, if applicable) `[torch]` extras to include `lightning`; guard CLI startup with a try/except that raises `RuntimeError` pointing to `pip install .[torch]`. Update docs if the prerequisite message changes materially.
- C5: Run tests sequentially inside repo root:
  - `pytest tests/torch/test_backend_selection.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/phase_e_backend_green.log`
  - `pytest tests/torch/test_integration_workflow_torch.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/phase_e_integration_green.log`
  Ensure both logs are referenced in phase_e2_green.md and attach exit codes.
- Update `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` checklist states (E2.C1–E2.C2) and docs/fix_plan.md Attempts history once work completes.

Pitfalls To Avoid:
- Do not bypass CONFIG-001; call `update_legacy_dict` before any PyTorch data access.
- Keep CLI imports torch-optional; raise RuntimeError with guidance instead of silently falling back.
- Avoid writing artifacts outside the timestamped reports directory.
- Don’t mix unittest patterns into new pytest helpers; tests must remain pytest-native.
- Ensure MLflow disable flag defaults to current behaviour; no regression for existing workflows.
- Keep subprocess commands deterministic — pin dataset paths and CPU execution for consistency.
- Preserve existing TensorFlow paths when editing shared modules (no behavioural drift for backend='tensorflow').
- Handle missing Lightning gracefully (RuntimeError), not generic stack traces.
- Avoid editing protected physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Verify reconstructed image filenames match test expectations before sealing the loop.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T213500Z/red_phase.md
- specs/ptychodus_api_spec.md:120-210
- docs/workflows/pytorch.md:12-86
- tests/torch/test_integration_workflow_torch.py:1-190

Next Up: Execute Phase D parity evidence (plan tasks D1–D3) once the green tests and CLI wiring are complete.
