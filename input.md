Summary: Remove remaining TORCH_AVAILABLE guards now that PyTorch is mandatory.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 / Phase F3.2 Guard Removal
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_config_bridge.py -q; pytest tests/torch/test_data_pipeline.py -q; pytest tests/torch/test_model_manager.py -q; pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_not_implemented -q
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T193753Z/{guard_removal_summary.md,pytest_guard_removal.log}
Do Now:
- INTEGRATE-PYTORCH-001 Phase F3 — F3.2 @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md (tests: pytest tests/torch/test_config_bridge.py -q && pytest tests/torch/test_data_pipeline.py -q && pytest tests/torch/test_model_manager.py -q && pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_not_implemented -q): implement guard removal per f3_2_guard_removal_brief.md across the six listed modules, then rerun the targeted torch selectors.
- INTEGRATE-PYTORCH-001 Phase F3 — F3.2 docs @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md (tests: none): capture guard_removal_summary.md + pytest_guard_removal.log, update phase_f_torch_mandatory.md state, and log Attempt #70 in docs/fix_plan.md.
If Blocked: Stop before partial guard removal; document failing module, stack trace, and pytest output in guard_removal_summary.md, note the blocker in docs/fix_plan.md, and leave code untouched for escalation.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:41 — F3.2 instructions and parity test list.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T193753Z/f3_2_guard_removal_brief.md — module-by-module removal guidance.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T192500Z/migration_plan.md#phase-f32-—-guard-removal — authoritative sequence and rollback notes.
- docs/fix_plan.md:133-150 — Latest attempts show F3.1 complete; next gate is guard removal.
- specs/ptychodus_api_spec.md:192-275 — Persistence + workflow requirements that must stay intact after guard removal.
How-To Map:
- For each module listed in the brief remove `try/except ImportError` patterns, delete `TORCH_AVAILABLE` flags, and make torch imports unconditional (`ptycho_torch/config_params.py:1-25`, `ptycho_torch/config_bridge.py:70-165`, `ptycho_torch/data_container_bridge.py:1-260`, `ptycho_torch/memmap_bridge.py:1-70`, `ptycho_torch/model_manager.py:1-240`, `ptycho_torch/workflows/components.py:1-220`). Replace any fallback branches with direct torch usage or explicit RuntimeError messaging that instructs users to install torch.
- Ensure `ptycho_torch/data_container_bridge.PtychoDataContainerTorch` always returns torch tensors; drop NumPy fallback branch and adjust `__repr__` to inspect dtypes without `TORCH_AVAILABLE`.
- Simplify `ptycho_torch/model_manager.save_torch_bundle` and `load_torch_bundle` to require `nn.Module` inputs, removing sentinel dict logic and ImportError fallback.
- Update `ptycho_torch/workflows/components` imports to pull RawDataTorch/DataContainerTorch/save_torch_bundle/load_torch_bundle at module scope; add explicit RuntimeError in entry points if those dependencies are missing unexpectedly.
- After editing, reinstall the package (`pip install -e .`) if dependency resolution demands it, then run the four targeted pytest selectors in the order listed; tee combined output to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T193753Z/pytest_guard_removal.log`.
- Summarize code touchpoints, command transcript, and test outcomes in `guard_removal_summary.md`; update `phase_f_torch_mandatory.md` (mark F3.2 [x]) and append Attempt #70 in `docs/fix_plan.md` with artifact links.
Pitfalls To Avoid:
- Do not leave partial NumPy fallback paths—guard removal must be all-or-nothing for each module.
- Avoid introducing circular imports when moving torch imports to module scope; verify modules still import cleanly.
- Keep RuntimeError messaging precise (`pip install torch>=2.2`) instead of silent failure.
- Maintain TensorType alias compatibility in config_params without relying on Any.
- Do not adjust F3.3 skip logic yet; that work remains for the following loop.
- Ensure pytest selectors run after code edits; no skipped validation allowed.
- Preserve existing docstrings/comments unless they only describe guard behavior being removed.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:41
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T193753Z/f3_2_guard_removal_brief.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T192500Z/migration_plan.md#phase-f32-—-guard-removal
- ptycho_torch/config_params.py:1-35
- ptycho_torch/data_container_bridge.py:1-260
- ptycho_torch/model_manager.py:1-240
Next Up: F3.3 pytest skip rewrite once guard removal lands cleanly.
