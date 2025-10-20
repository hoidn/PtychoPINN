Summary: Capture Phase D baseline evidence and design decisions before refactoring the PyTorch CLIs.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase D.A Baseline
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/{baseline.md,pytest_cli_train_baseline.log,pytest_cli_inference_baseline.log,design_notes.md}

Do Now:
1. ADR-003-BACKEND-API A1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:13 — Inventory the current training/inference CLI call graph and save findings to baseline.md; tests: none
2. ADR-003-BACKEND-API A2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:14 — Run the documented CLI pytest selectors and archive logs to the artifact hub; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv
3. ADR-003-BACKEND-API A3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:15 — Author design_notes.md covering legacy flag handling decisions and deprecation strategy; tests: none

If Blocked: Record the blocker in plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/blocker.md, revert any partial plan checklist updates, and note the issue in docs/fix_plan.md before stopping.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:6 — Phase D requires baseline inventory before refactors can start.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123500Z/phase_c4f_closeout/summary.md:214 — Phase D prerequisites call for a call-graph snapshot and legacy flag decisions.
- specs/ptychodus_api_spec.md:190 — Backend routing/CLI contracts must remain intact during refactor.
- docs/workflows/pytorch.md:420 — Current CLI usage expectations provide the comparison target for thin wrappers.

How-To Map:
- `mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline`
- For A1, inspect `ptycho_torch/train.py` and `ptycho_torch/inference.py`, optionally run `python -m scripts.tools.print_import_tree ptycho_torch.train` and `python -m scripts.tools.print_import_tree ptycho_torch.inference`; capture module/function flow with file:line anchors in `baseline.md`.
- For A2, run the selectors with logging:  
  `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/pytest_cli_train_baseline.log`  
  `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/pytest_cli_inference_baseline.log`
- Summarise assumptions discovered during A2 (mock usage, fixtures, skip markers) in `baseline.md`.
- For A3, outline how `--device`, `--accelerator`, and `--disable_mlflow` should behave post-refactor; tag decisions that require Phase E governance in `design_notes.md`.
- Update the plan checklist states (A1–A3 → `[x]`) once artifacts are stored and notes complete.

Pitfalls To Avoid:
- No production code changes in this loop.
- Keep all logs and notes inside the specified artifact directory.
- Do not modify or re-run integration fixture generators; baseline only.
- Preserve existing pytest skip/xfail markers when running selectors.
- Note any unexpected test failures immediately instead of masking them.
- Do not delete or relocate legacy CLI descriptions before decisions are logged.
- Avoid introducing new TODO headings; log open questions in design_notes.md instead.
- Maintain CONFIG-001 mindset when considering future refactors (update_legacy_dict ordering).
- Run commands from repository root to keep relative paths valid.
- Clean up temporary directories created during inspection (e.g., `__pycache__`).

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md
- plans/active/ADR-003-BACKEND-API/implementation.md:42
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123500Z/phase_c4f_closeout/summary.md:182
- specs/ptychodus_api_spec.md:178
- docs/workflows/pytorch.md:398

Next Up: Begin Phase B (training CLI thin wrapper) once baseline evidence and design decisions are locked.
