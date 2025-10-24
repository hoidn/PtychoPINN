Summary: Capture EB4 runtime smoke evidence for PyTorch execution knobs (accelerator auto + checkpoint/early-stop settings).
Mode: Perf
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003
Branch: feature/torchapi
Mapped tests: none — CLI smoke evidence
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/runtime_smoke/2025-10-24T061500Z/{train_cli_runtime_smoke.log,metrics.csv,lightning_tree.txt,checkpoints_ls.txt,summary.md}
Do Now:
- [ADR-003-BACKEND-API] EB4.A @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — run the Phase EB4 runtime smoke command (gridsize=3, accelerator auto, checkpoint top-k=2, early stop patience=5), archive logs + metrics under 2025-10-24T061500Z, remove tmp/runtime_smoke when finished; tests: none — CLI smoke.
- [ADR-003-BACKEND-API] EB4.B @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — author summary.md capturing runtime, accelerator resolution, checkpoint file count, and callback wiring; update docs/fix_plan.md Attempts + mark plan rows `[x]`; tests: none — docs.
If Blocked: Capture stdout/stderr to `runtime_smoke/2025-10-24T061500Z/blocker.log`, note command + exit code in summary.md, register the blocker in docs/fix_plan.md Attempts, then stop.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:274-286 requires checkpoint + early-stop knobs to behave per contract; smoke validates CLI wiring.
- docs/workflows/pytorch.md:324-338 documents the user-facing flags we just exposed; evidence keeps guide trustworthy.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md rows EB4.A-B gate Phase E completion.
- docs/findings.md:12 (CONFIG-LOGGER-001) promises CSV metrics persistence; smoke confirms it under accelerator auto.
How-To Map:
- `artifact_dir=plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/runtime_smoke/2025-10-24T061500Z`
- `rm -rf tmp/runtime_smoke && mkdir -p "$artifact_dir"`
- `CUDA_VISIBLE_DEVICES="" /usr/bin/time -p python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir tmp/runtime_smoke --n_images 64 --gridsize 3 --max_epochs 6 --batch_size 4 --accelerator auto --deterministic --num-workers 0 --logger csv --checkpoint-save-top-k 2 --early-stop-patience 5 | tee "$artifact_dir/train_cli_runtime_smoke.log"`
- `cp tmp/runtime_smoke/lightning_logs/version_0/metrics.csv "$artifact_dir/metrics.csv"`
- `tree tmp/runtime_smoke/lightning_logs > "$artifact_dir/lightning_tree.txt"`
- `ls -l tmp/runtime_smoke/checkpoints > "$artifact_dir/checkpoints_ls.txt"`
- `rm -rf tmp/runtime_smoke`
- Draft `$artifact_dir/summary.md` noting runtime, GPU detection message (auto→cpu), checkpoint files present, early-stop settings (patience 5, not triggered over 6 epochs), and cross-links to spec/workflow/finding. Update docs/fix_plan.md with Attempt #71 and flip EB4 rows to `[x]`.
Pitfalls To Avoid:
- Do not change accelerator back to `cpu`; evidence must demonstrate `auto` fallback behaviour.
- Keep `gridsize 3` and `--checkpoint-save-top-k 2`; other values undermine EB4 coverage.
- Copy metrics.csv before deleting tmp/runtime_smoke; otherwise CSV evidence is lost.
- Avoid leaving checkpoint artifacts in the repo root; ensure cleanup after packaging evidence.
- Reference artifact paths relative to repo root (no absolute `/home/...`).
- Summaries must cite spec/workflow lines and CONFIG-LOGGER-001; avoid vague prose.
- Skip pytest and full-suite runs; this loop is evidence-only.
- Capture tree/ls output after training, not before.
- Watch for DeprecationWarnings (device/disable_mlflow) and record them in summary if they appear.
- Ensure docs/fix_plan.md Attempts History reflects this smoke run with artifact links.
Pointers:
- specs/ptychodus_api_spec.md:274
- docs/workflows/pytorch.md:324
- docs/findings.md:12
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md
- docs/ci_logger_notes.md:1
Next Up: Phase E.C deprecation tasks (template available in phase_e_governance/plan.md) once EB4 evidence lands.
