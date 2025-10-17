# Phase E2 Implementation & Parity Plan (INTEGRATE-PYTORCH-001)

## Context
- Initiative: INTEGRATE-PYTORCH-001
- Phase Goal: Close Phase E2 by wiring the PyTorch train/infer workflow so the new integration tests pass, then capture TensorFlow ↔ PyTorch parity evidence.
- Dependencies: `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` (§E2), backend selection blueprint (`reports/2025-10-17T180500Z/phase_e_backend_design.md`), red-phase evidence (`reports/2025-10-17T213500Z/{phase_e_fixture_sync.md,red_phase.md,phase_e_red_integration.log}`), config bridge adapters (`ptycho_torch/config_bridge.py`), CLI contract in `specs/ptychodus_api_spec.md` §4.5, workflow guidance in `docs/workflows/pytorch.md`.
- Artifact Discipline: Store new execution artifacts under ISO directories (e.g., `plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/phase_e2_green.md`, `.../phase_e_tf_baseline.log`). Reference each artifact from docs/fix_plan.md Attempts history.

---

### Phase C — PyTorch Workflow Wiring (E2.C)
Goal: Implement the CLI + dispatcher wiring so `pytest tests/torch/test_integration_workflow_torch.py -vv` passes and CONFIG-001 gates are honoured.
Prereqs: Phase E2 red evidence captured (`reports/2025-10-17T213500Z/`), backend selector plan (Phase E1) understood, PyTorch extras installed locally (`pip install -e .[torch]`).
Exit Criteria: Green logs for backend selection + integration tests stored under a fresh timestamp; CLI wrappers support required flags, fail-fast message present, MLflow suppression available, and persistence artifacts created in the expected location.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Implement training CLI wrapper | [ ] | Add argparse entry point to `ptycho_torch/train.py` that matches TensorFlow CLI (`--train_data_file`, `--test_data_file`, `--output_dir`, `--max_epochs`, `--n_images`, `--gridsize`, `--batch_size`, `--device`, `--disable_mlflow`). Bridge parsed args into the existing `main()` flow, call `update_legacy_dict(params.cfg, config)` before delegating to `ptycho_torch.workflows.components.run_cdi_example_torch`. Document behavior in module docstring. Guidance: mirror `tests/torch/test_integration_workflow_torch.py` reproduction commands; reuse config bridge to hydrate dataclasses. |
| C2 | Provide inference CLI + artifact checks | [ ] | Extend `ptycho_torch/inference.py` with CLI that accepts `--model_path`, `--test_data_file`, `--output_dir`, `--n_images`, `--device`, `--quiet`. Load checkpoints via `load_inference_bundle_torch` once Phase D3 bundle exists; until then fall back to Lightning `Trainer.predict` with saved checkpoint path produced by C1. Emit amplitude/phase PNGs named per test expectations. Capture outputs under `<output_dir>/reconstructed_amplitude.png` etc. |
| C3 | Lightning/MLflow runtime controls | [ ] | Ensure `--disable_mlflow` toggles autologging in `train.py`; default keeps existing behavior. When disabled, skip `mlflow.pytorch.autolog` and guard `print_auto_logged_info`. Confirm tests can run without MLflow server. |
| C4 | Dependency + fail-fast policy updates | [ ] | Update `setup.py`/`pyproject.toml` extras (`[torch]`) to include `lightning` ≥ documented minimum. Add actionable error handling in CLI entry point: catching `ImportError` for `lightning` or other PyTorch runtime libs should raise `RuntimeError("PyTorch backend requires lightning; install via pip install .[torch]")`. Update `docs/workflows/pytorch.md` prerequisites if gap remains (Phase E3 will polish wording). |
| C5 | Regression + validation tests | [ ] | After implementing C1–C4, run `pytest tests/torch/test_backend_selection.py -vv` and `pytest tests/torch/test_integration_workflow_torch.py -vv`. Store logs as `reports/<ISO8601>/phase_e_backend_green.log` and `.../phase_e_integration_green.log`. Update `phase_e_integration.md` rows E2.C1/E2.C2 with timestamps + checklist states. |

---

### Phase D — Parity Evidence Capture (E2.D)
Goal: Demonstrate TensorFlow ↔ PyTorch workflow parity after wiring is complete.
Prereqs: Phase C complete (tests green, CLI available), fixture inventory validated (`phase_e_fixture_sync.md`).
Exit Criteria: TensorFlow and PyTorch integration logs archived, summary document published with metrics + qualitative comparison, docs/fix_plan.md attempt logged.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Capture TensorFlow baseline log | [ ] | Run `pytest tests/test_integration_workflow.py -k full_cycle -vv` using canonical dataset. Save output to `reports/<ISO8601>/phase_e_tf_baseline.log`. Note runtime, key artifacts, and any warnings for parity context. |
| D2 | Capture PyTorch integration log | [ ] | Re-run `pytest tests/torch/test_integration_workflow_torch.py -vv` with green implementation. Save output to `reports/<ISO8601>/phase_e_torch_run.log`. Confirm CLI output directories contain expected artifacts (checkpoint + reconstructed images). |
| D3 | Publish parity summary | [ ] | Author `reports/<ISO8601>/phase_e_parity_summary.md` comparing metrics (if available), artifact inventories, backend selection logs, and residual risks. Reference POLICY-001 + CONFIG-001 compliance evidence. Update `phase_e_integration.md` E2.D rows and docs/fix_plan.md Attempt with links. |

---

## Verification & Reporting Checklist
- [ ] All C-phase tasks reference the new CLI contract and log storage path.
- [ ] PyTorch dependency guidance synchronized with POLICY-001 messaging.
- [ ] Each D-phase task records command, output paths, and conclusions in the summary artifact.
- [ ] docs/fix_plan.md Attempts history updated in same loop as execution.
- [ ] `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` cross-references this plan for E2.C/E2.D guidance.

