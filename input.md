Summary: Implement PyTorch persistence loader and prove params restoration parity
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 / Phase D3.C persistence loader
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_load_round_trip_updates_params_cfg -vv; pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_missing_params_raises_value_error -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T113200Z/{phase_d3_loader.md,pytest_red.log,pytest_green.log,params_snapshot_before.json,params_snapshot_after.json,archive_tree.txt}
Do Now:
1. INTEGRATE-PYTORCH-001 — D3.C red tests @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md: add pytest class `TestLoadTorchBundle` covering round-trip params restoration and missing-params failure; run `pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_load_round_trip_updates_params_cfg -vv` (expect fail) and `pytest tests/torch/test_model_manager.py::TestLoadTorchBundle::test_missing_params_raises_value_error -vv` (expect fail).
2. INTEGRATE-PYTORCH-001 — D3.C loader implementation @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md: implement `ptycho_torch/model_manager.py::load_torch_bundle` + wire `load_inference_bundle_torch` to use it; ensure model reconstruction helpers reuse Phase D2 assets and persist params snapshots; turn tests green with `pytest tests/torch/test_model_manager.py -k load_torch_bundle -vv`.
3. INTEGRATE-PYTORCH-001 — D3.C evidence capture @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md: record params.cfg before/after JSON dumps, archive tree listing, and write `phase_d3_loader.md` summary with open questions (tests: none).
If Blocked: If torch runtime unavailable, create sentinel loader path that hydrates dummy dicts and log blocker in `phase_d3_loader.md` while leaving tests xfailed; update Attempt history accordingly.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:180-216 — Defines dual-model archive and loader contract PyTorch must honor.
- ptycho/model_manager.py:112-189 — TensorFlow load path showing CONFIG-001 restoration sequence to mirror.
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md#L40 — D3 checklist now points to loader parity deliverables.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T104700Z/phase_d3_callchain/summary.md — Highlights required params snapshot taps for D3.C.
- tests/torch/test_model_manager.py:1-220 — Existing save_torch_bundle tests to extend without breaking torch-optional pattern.
How-To Map:
- Red tests: extend `tests/torch/test_model_manager.py` with pytest fixtures reusing `dummy_torch_models`; add `params_before = params.cfg.copy()` + helper to dump JSON into artifact dir; expect NotImplementedError until loader implemented.
- Implementation: add helper (e.g., `_create_torch_model_from_params`) leveraging `ptycho_torch.model.PtychoPINN`; call `update_legacy_dict(params.cfg, restored_config)` after unpacking archive; load weights via `torch.load`; update `ptycho_torch/workflows/components.py::load_inference_bundle_torch` to delegate to new loader; ensure torch-optional imports guarded.
- Validation: capture pytest output with `pytest ... > plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T113200Z/pytest_green.log`; for params snapshots run `python scripts/save_params_snapshot.py` equivalent inline (`json.dump(params.cfg, open(...))`); list archive contents with `python - <<'PY'` using `zipfile.ZipFile.namelist()`.
- Documentation: summarize findings + open issues in `phase_d3_loader.md` including tap point status; attach JSON diff between before/after snapshots.
Pitfalls To Avoid:
- Don’t import torch at module top level outside guarded block.
- Always call `update_legacy_dict` before touching legacy modules; verify CONFIG-001 compliance.
- Keep archive extraction within TemporaryDirectory to prevent repository litter.
- Avoid hard-coded filesystem paths; respect provided `base_path`.
- Maintain dual-model requirement; error clearly if archive missing required keys.
- Ensure tests clean generated archives from tmp_path to avoid git noise.
- Preserve manifest version tagging (`2.0-pytorch`).
- Skip tests gracefully when torch unavailable (use existing conftest guard).
- Keep loader deterministic; no random seed drift.
- Do not modify TensorFlow persistence logic.
Pointers:
- specs/ptychodus_api_spec.md:192-206
- ptycho/model_manager.py:346-410
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md:40-58
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T110500Z/phase_d3b_summary.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T104700Z/phase_d3_callchain/tap_points.md
Next Up: Begin D3.D regression hooks (tests linking into TEST-PYTORCH-001) once loader parity is green.
