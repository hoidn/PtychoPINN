Summary: Restore PyTorch dataloader DATA-001 compliance and document the green integration run
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-DATALOADER — Restore PyTorch dataloader DATA-001 compliance
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_dataloader.py::test_loads_canonical_diffraction -vv; pytest tests/torch/test_dataloader.py -vv; pytest tests/torch/test_integration_workflow_torch.py -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/{pytest_dataloader_red.log,pytest_dataloader_green.log,pytest_integration_green.log,parity_summary.md}

Do Now:
1. INTEGRATE-PYTORCH-001-DATALOADER @ docs/fix_plan.md — Author pytest regression `tests/torch/test_dataloader.py::test_loads_canonical_diffraction` (canonical+legacy fixture) per plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T223200Z/dataloader_triage.md; run `pytest tests/torch/test_dataloader.py::test_loads_canonical_diffraction -vv` (expect fail) | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/pytest_dataloader_red.log (tests: targeted)
2. INTEGRATE-PYTORCH-001-DATALOADER @ docs/fix_plan.md — Update `ptycho_torch/dataloader.py` to prefer DATA-001 `diffraction` key, fall back to `diff3d`, and share helper across calculate_length/memory_map paths; keep dtype conversions identical (tests: none)
3. INTEGRATE-PYTORCH-001-DATALOADER @ docs/fix_plan.md — Re-run unit coverage `pytest tests/torch/test_dataloader.py -vv` | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/pytest_dataloader_green.log; capture assertions for canonical + legacy keys (tests: targeted)
4. INTEGRATE-PYTORCH-001-DATALOADER @ docs/fix_plan.md — Execute parity log `pytest tests/torch/test_integration_workflow_torch.py -vv` | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/pytest_integration_green.log; update parity summary + plan checkboxes + docs/fix_plan Attempts with new artifact (tests: targeted)
5. INTEGRATE-PYTORCH-001-DATALOADER @ docs/fix_plan.md — Refresh `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/parity_summary.md`, flip relevant plan rows to [x], and sync docs/fix_plan.md Attempts history noting green run + DATA-001 fix (tests: none)

If Blocked: Preserve red logs under the artifact directory, document failure + hypothesis in parity summary, leave plan checkbox unchecked, and note blocker in docs/fix_plan.md Attempts history.

Priorities & Rationale:
- docs/fix_plan.md#L23 — New [INTEGRATE-PYTORCH-001-DATALOADER] entry requires canonical key support before broader parity work continues.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T223200Z/dataloader_triage.md — Confirms root cause and lays out fallback strategy.
- specs/data_contracts.md§1 — Canonical NPZ schema mandates `diffraction` key; loader must comply.
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md — E2.D2 evidence depends on passing PyTorch integration log.
- specs/ptychodus_api_spec.md§4.5 — ptychodus reconstructor users expect canonical dataset compatibility.

How-To Map:
- export timestamp=2025-10-17T224500Z; mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp
- pytest tests/torch/test_dataloader.py::test_loads_canonical_diffraction -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/pytest_dataloader_red.log  # expect failure pre-fix
- After implementation, pytest tests/torch/test_dataloader.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/pytest_dataloader_green.log
- pytest tests/torch/test_integration_workflow_torch.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/pytest_integration_green.log
- Summarize key metrics + DATA-001 compliance in plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/parity_summary.md; cite CONFIG-001 + DATA-001 findings
- Update plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md (D2 row commentary), phase_e_integration.md, implementation.md, and docs/fix_plan.md Attempts referencing the new timestamp

Pitfalls To Avoid:
- Do not remove legacy `diff3d` support; provide fallback with clear error when neither key exists.
- Preserve dtype conversions (float32 amplitude) and rounding behavior to avoid regressions.
- Keep new tests pure pytest (no unittest.TestCase mix-ins) and isolate temporary NPZ fixtures via tmp_path.
- Ensure logs live inside the timestamped reports directory; no stray files at repo root.
- Run targeted selectors from repo root to use editable install and torch deps.
- Avoid touching stable core TensorFlow modules (model.py, diffsim.py, tf_helper.py).
- Re-run integration test only after unit tests pass to save time.
- Capture red test output before implementing fix to satisfy TDD evidence.
- If pytest fails after fix, do not mark plan rows complete; document blocker and leave entry open.
- Check docs/findings.md for DATA-001 references when writing parity summary.

Pointers:
- docs/fix_plan.md#L12
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T223200Z/dataloader_triage.md
- specs/data_contracts.md#L1
- specs/ptychodus_api_spec.md#L1
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md#L46

Next Up: Revisit [INTEGRATE-PYTORCH-001-STUBS] once integration parity is green.
