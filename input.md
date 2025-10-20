Summary: Deliver Phase C4 design docs and RED CLI test scaffolds (ADR-003-BACKEND-API)
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4 CLI integration
Branch: feature/torchapi
Mapped tests: targeted red selectors — expected fail (see How-To Map)
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/{cli_flag_inventory.md,flag_selection_rationale.md,flag_naming_decisions.md,argparse_schema.md,pytest_cli_train_red.log,pytest_cli_inference_red.log,red_baseline.md}

Do Now:
1. ADR-003-BACKEND-API C4.A design docs @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — complete C4.A1+C4.A2+C4.A3+C4.A4 deliverables (inventory, selection rationale, naming decisions, argparse schema); tests: none.
2. ADR-003-BACKEND-API C4.B RED scaffolds @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — implement C4.B1+C4.B2+C4.B3+C4.B4 by authoring pytest CLI roundtrip tests, running selectors to capture RED logs, and writing red_baseline.md; tests: targeted red (expected fail).
3. ADR-003-BACKEND-API C4 progress sync @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — mark C4.A/C4.B rows, update phase summary, and log new attempt in docs/fix_plan.md (reference design/testing artifacts); tests: none.

If Blocked: If CLI flag inventory cannot be reconciled (missing spec precedent or conflicting defaults), suspend implementation, document blockers in summary.md, flag affected flags in red_baseline.md, and update docs/fix_plan.md Attempt entry with the obstruction.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — authoritative task checklist for C4.A–C4.B.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md — execution knob source of truth for flag selection/defaults.
- ptycho_torch/train.py:360 and ptycho_torch/inference.py:300 — current argparse surfaces to refactor; needed for gap analysis.
- specs/ptychodus_api_spec.md:210 and docs/workflows/pytorch.md:260 — contract references for backend routing and CLI behavior.
- docs/TESTING_GUIDE.md:80 — TDD guidance for authoring new pytest modules and handling RED states.

How-To Map:
- Create/append files under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/`: `cli_flag_inventory.md`, `flag_selection_rationale.md`, `flag_naming_decisions.md`, `argparse_schema.md`, `red_baseline.md`. Each should include file:line citations (train.py, inference.py, override_matrix.md § refs) and note deferred knobs per plan §Deferred.
- For `cli_flag_inventory.md`, tabulate existing + new flags with columns: Flag, Default, Type, Destination (TrainingPayload/InferncePayload field), TF equivalent, Notes. Source data from plan §C4.A1, override_matrix.md §§2–5, and CLI source files.
- `flag_selection_rationale.md` must justify inclusion of `--accelerator`, `--deterministic`, `--num-workers`, `--learning-rate`, `--inference-batch-size` (training) and inference equivalents; cite override precedence (plan §C4.A2) and POLICY-001 for accelerator defaults.
- `flag_naming_decisions.md` should compare PyTorch vs TensorFlow CLI naming (see `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T225905Z/phase_a_inventory/cli_inventory.md` for references) and record final string choices.
- `argparse_schema.md` must outline argparse arguments: option string(s), type, default, help text, validation, and notes on mutual exclusivity. Include TODO markers for help text updates in `docs/workflows/pytorch.md` §13 and spec CLI table citations.
- Author tests in `tests/torch/test_cli_train_torch.py` and `tests/torch/test_cli_inference_torch.py` using native pytest. Follow plan §C4.B1–B2: fixtures for temporary CLI invocation, patch factories (`ptycho_torch.config_factory.create_training_payload`) with `monkeypatch`, and assert payload `execution_config` fields reflect CLI overrides.
- Run RED selectors and capture logs (expected failures until C4.C lands):
  - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/pytest_cli_train_red.log || true`
  - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/pytest_cli_inference_red.log || true`
- Summarize failure signatures (e.g., AttributeError for missing flags, NotImplementedError placeholders) in `red_baseline.md` along with acceptance criteria for GREEN.
- Update `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md` marking C4.A and C4.B rows `[x]`, and refresh `phase_c_execution/summary.md` with brief notes + links to new artifacts.
- Append Attempt #17 (or next index) to `docs/fix_plan.md` documenting design deliverables, RED logs, and outstanding C4.C tasks.

Pitfalls To Avoid:
- Do not modify production CLI code or factories yet; stay in design/RED scope.
- Keep all new artifacts ASCII and stored in the specified plan directory—no root-level logs.
- Avoid inventing pytest command names; ensure new modules live under `tests/torch/` with snake_case filenames.
- Ensure new tests import via project modules (no relative path hacks); respect pytest fixture patterns (use `tmp_path`, `monkeypatch`).
- Document every sourced default with file:line citations to prevent drift from override_matrix.md.
- Capture RED logs even if pytest exits non-zero; use `|| true` to avoid aborting the loop.
- Respect CONFIG-001 ordering in pseudocode/examples (update_legacy_dict before workflow imports).
- Mark plan checklists accurately; do not set `[x]` without artefacts saved.
- Reference POLICY-001 when justifying accelerator/deterministic defaults; no silent CPU-only assumptions.
- Do not delete or overwrite prior plan artifacts; append new docs/summaries.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:40
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/summary.md:20
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md:1
- ptycho_torch/train.py:360
- ptycho_torch/inference.py:320
- specs/ptychodus_api_spec.md:210
- docs/workflows/pytorch.md:260
- docs/TESTING_GUIDE.md:80

Next Up:
1. C4.C implementation (refactor CLI to factories + execution config wiring) once RED scaffolds pass review.
