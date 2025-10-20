Summary: Capture Phase D CLI smoke evidence and sync plan/ledger for ADR-003 thin wrappers.
Mode: Parity
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase D (CLI thin wrappers, D1–D3)
Branch: feature/torchapi
Mapped tests: none — smoke evidence
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/{train_cli_smoke.log,infer_cli_smoke.log,train_cli_tree.txt,infer_cli_tree.txt,smoke_summary.md,handoff_summary.md}

Do Now:
1. ADR-003-BACKEND-API D1 (training CLI smoke) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:58 — run the documented training command, capture `/usr/bin/time` output with `tee`, and stage logs under the new artifact hub; tests: none.
2. ADR-003-BACKEND-API D1 (inference CLI smoke) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:58 — run the documented inference command against the smoke model, archive outputs (PNG + stdout) in the same artifact hub; tests: none.
3. ADR-003-BACKEND-API D2 (plan + ledger sync) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:59 — update plan row statuses, add C/D checkpoint notes, and append docs/fix_plan Attempt #53 with artifact references; tests: none.
4. ADR-003-BACKEND-API D3 (hygiene + handoff summary) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:60 — write `handoff_summary.md` (hygiene + next-phase inputs), record cleanup commands, and ensure tmp outputs removed after copying artifacts; tests: none.

If Blocked: Capture the failing command output in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/blocker.md`, keep plan row state `[P]`, and log the blocker (with command + exit code) in docs/fix_plan.md before stopping.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:55 — Phase D requires smoke evidence before moving to governance.
- specs/ptychodus_api_spec.md:312 — CLI workflows must honour CONFIG-001 and emit actionable errors; smoke run verifies behaviour on minimal dataset.
- docs/workflows/pytorch.md:332 — Inference CLI section documents expected flags and outputs; smoke evidence proves docs match reality.
- docs/fix_plan.md:121 — Active initiative gates on completing Phase D rows (D1–D3) with artifacts logged.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T122425Z/phase_d_cli_wrappers_inference_followup/summary.md — Previous loop confirmed GREEN tests; now need runtime evidence to close the loop.

How-To Map:
- Prep: `mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke`; `rm -rf tmp/cli_train_smoke tmp/cli_infer_smoke`.
- Training command (CPU-only): `CUDA_VISIBLE_DEVICES="" /usr/bin/time -p python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir tmp/cli_train_smoke --max_epochs 1 --n_images 16 --accelerator cpu --deterministic --num-workers 0 --learning-rate 5e-4 --disable_mlflow 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/train_cli_smoke.log`.
- Capture training artifacts: `ls -R tmp/cli_train_smoke > plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/train_cli_tree.txt`; copy key outputs (e.g., `cp tmp/cli_train_smoke/train_debug.log plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/train_debug.log` if present).
- Inference command: `CUDA_VISIBLE_DEVICES="" /usr/bin/time -p python -m ptycho_torch.inference --model_path tmp/cli_train_smoke --test_data tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir tmp/cli_infer_smoke --n_images 16 --accelerator cpu --quiet 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/infer_cli_smoke.log`.
- Archive inference outputs: `ls -R tmp/cli_infer_smoke > plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/infer_cli_tree.txt`; copy amplitude/phase PNGs into the artifact hub (e.g., `cp tmp/cli_infer_smoke/reconstructed_*.png plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/`).
- Summaries: author `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T125500Z/phase_d_cli_wrappers_smoke/smoke_summary.md` covering commands, runtimes, warnings, outputs, and CLI flag observations.
- Plan & ledger: mark D1–D3 rows `[x]` with notes + artifact pointer, update `summary.md` with Phase D checkpoint, append Attempt #53 to `docs/fix_plan.md` referencing the smoke artifacts and noting Mode (Parity) / tests (none).
- Handoff: create `handoff_summary.md` noting hygiene commands (git status, tmp cleanup), residual knobs deferred to Phase E, and confirm removal of `tmp/cli_*` directories (document commands used). After copying artifacts, run `rm -rf tmp/cli_train_smoke tmp/cli_infer_smoke`.

Pitfalls To Avoid:
- Don’t leave `tmp/cli_*` directories or `train_debug.log` at repo root—copy artifacts first, then clean.
- Keep commands CPU-only (`CUDA_VISIBLE_DEVICES=""`) to avoid GPU variance.
- Ensure artifact files stay under the timestamped reports directory; no logs at repository root.
- Do not modify production code or tests—this loop is evidence + documentation only.
- Maintain CONFIG-001 order by letting CLI run naturally; avoid manual params.cfg tweaks.
- Capture full stdout/stderr via `tee`; don’t truncate logs.
- Confirm inference command uses the freshly trained bundle (`tmp/cli_train_smoke`) before cleanup.
- When editing docs/fix_plan/plan files, preserve checklist tables and formatting.
- Avoid running full pytest—smoke commands only.
- Include runtime metrics from `/usr/bin/time` in summaries; don’t omit them.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:55
- docs/workflows/pytorch.md:330
- specs/ptychodus_api_spec.md:304
- docs/fix_plan.md:118
- tests/torch/test_cli_inference_torch.py:250

Next Up: Phase E governance prep (ADR acceptance + legacy API deprecation) once Phase D smoke + handoff close out.
