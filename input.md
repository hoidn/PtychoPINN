Summary: Prep PyTorch persistence shim by making D3.B archive writer test-first and cleaning stray artifacts
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 / Phase D3.B persistence shim
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_model_manager.py::TestSaveTorchBundle::test_archive_structure -vv; pytest tests/torch/test_model_manager.py::TestSaveTorchBundle::test_params_snapshot -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T110500Z/{phase_d3_writer.md,archive_tree.txt,pytest_red.log,pytest_green.log,bundle_manifest.json}
Do Now:
1. INTEGRATE-PYTORCH-001 — D2.C cleanup @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md: move `train_debug.log` under `reports/2025-10-17T101500Z/` and drop the repo-root copy (tests: none).
2. INTEGRATE-PYTORCH-001 — D3.B red test @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md: author torch-optional pytest (`tests/torch/test_model_manager.py`) that fails by asserting new `save_torch_bundle` outputs manifest + params snapshot; run `pytest tests/torch/test_model_manager.py::TestSaveTorchBundle::test_archive_structure -vv` (expect fail) and capture log to artifact dir.
3. INTEGRATE-PYTORCH-001 — D3.B implementation @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md: implement torch-optional persistence shim (`ptycho_torch/model_manager.py::save_torch_bundle`) and integrate hook in `ptycho_torch/workflows/components.py`/`train.py`, ensuring dual-model archive + params.dill snapshot; store sample archive tree + manifest under artifact dir and turn the new tests green with `pytest tests/torch/test_model_manager.py -k bundle -vv`.
If Blocked: If Lightning checkpoints cannot be produced without torch runtime, pivot to serializing dummy `nn.Module` instances and log blocker in phase_d3_writer.md; update Fix Plan attempt with failure details.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:139,194 — Reconstructor contract requires `wts.h5.zip` dual-model archives and load support, so persistence shim must match schema.
- ptycho/model_manager.py:119 — TensorFlow loader restores `params.cfg`, highlighting CONFIG-001 obligation for PyTorch bundle loader.
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md:44-63 — Phase D3 checklist defines save/load deliverables and artifact expectations.
- docs/fix_plan.md:57 — INTEGRATE-PYTORCH-001 remains active gate for downstream PyTorch parity, so persistence work unblocks TEST-PYTORCH-001.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T104700Z/phase_d3_callchain/summary.md — Callchain findings enumerate exact gaps (no params snapshot, MLflow coupling) the shim must close.
How-To Map:
- Cleanup: `mv train_debug.log plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T101500Z/train_debug.log` then `git rm train_debug.log` if tracked; note move in artifact summary.
- Red test: create `tests/torch/test_model_manager.py` using pytest fixtures + `tmp_path`, guard imports via `pytest.importorskip("torch", reason=...)` wrapped in helper from `tests/conftest.py`; expect failure because function missing.
- Implementation: add `ptycho_torch/model_manager.py` with torch-optional save routine using `dataclass_to_legacy_dict` from `ptycho.config.config`; integrate call in workflow after training; ensure packaging uses `zipfile.ZipFile` to create manifest + per-model dirs; write manifest JSON snapshot to artifact folder.
- Validation: after implementation run `pytest tests/torch/test_model_manager.py -k bundle -vv` capturing red/green logs to artifact dir; dump `zipfile.ZipFile.namelist()` output into `archive_tree.txt`; record params snapshot (e.g., JSON) as `bundle_manifest.json`.
Pitfalls To Avoid:
- Do not require torch at import time for new modules/tests; follow skip rules in tests/conftest.py.
- Keep MLflow optional—persistence shim must run even when MLflow disabled.
- Ensure params snapshot uses Phase B bridge (`dataclass_to_legacy_dict`) rather than copying `params.cfg` directly.
- Avoid writing artifacts outside the initiative report directory (no more repo-root logs).
- Do not modify TensorFlow persistence code while implementing PyTorch shim.
- Keep archive write path configurable and avoid hard-coded /tmp paths.
- Make sure tests clean up temporary files to prevent git noise.
- Maintain CONFIG-001 order: update `params.cfg` before constructing inference modules in loader step.
- Ensure zip archive uses binary mode; don’t rely on Lightning callbacks that pull in heavy dependencies for test.
- Keep new tests torch-optional; skip cleanly when torch unavailable.
Pointers:
- specs/ptychodus_api_spec.md:139-206 — Archive structure + lifecycle contract.
- ptycho/model_manager.py:69-150 — TensorFlow save/load reference for params.dill handling.
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md:44-71 — D3 task definitions and artifact expectations.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T104700Z/phase_d3_callchain/summary.md — Persistence gaps + next steps for D3.B.
- tests/torch/test_workflows_components.py:20-160 — Torch-optional test patterns and params.cfg snapshot fixture.
Next Up: If persistence shim finishes early, begin D3.C loader implementation with corresponding failing test per plan.
