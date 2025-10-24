Summary: Capture logger smoke evidence and publish CI usage notes before EB3 governance wrap-up.
Mode: Perf
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003
Branch: feature/torchapi
Mapped tests: none — CLI smoke evidence
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/2025-10-24T050500Z/{train_cli_logger_csv.log,metrics.csv,logger_tree.txt,summary.md}
Do Now:
- [ADR-003-BACKEND-API] EB3.D1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — run CSV logger smoke command, archive metrics/tree/summary under the new smoke/2025-10-24T050500Z/ hub; tests: none — CLI smoke.
- [ADR-003-BACKEND-API] EB3.D2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — draft docs/ci_logger_notes.md with CI routing/cleanup guidance referencing fresh smoke artifacts; tests: none — docs.
If Blocked: Capture stdout/stderr to the same smoke directory, note blocker + command + exit code in summary.md, and register the issue in docs/fix_plan.md Attempts History before stopping.
Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md rows D1–D2 control EB3.D exit criteria; both must be `[x]` before Phase EB3 close-out.
- specs/ptychodus_api_spec.md:281 documents logger_backend default `'csv'`; smoke evidence confirms spec reality.
- docs/workflows/pytorch.md:329 lists `--logger` usage; CI note keeps guide + practice aligned.
- docs/findings.md:12 (CONFIG-LOGGER-001) promises CSV metrics are persisted—smoke run validates the claim.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md expects CI guidance before governance sign-off.
How-To Map:
- Prep dirs: `mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/2025-10-24T050500Z` and ensure `tmp/logger_smoke/` is clear (`rm -rf tmp/logger_smoke`).
- Run smoke (CPU-only):
  ```bash
  CUDA_VISIBLE_DEVICES="" /usr/bin/time -p python -m ptycho_torch.train \
    --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
    --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
    --output_dir tmp/logger_smoke \
    --n_images 64 \
    --max_epochs 1 \
    --gridsize 2 \
    --batch_size 4 \
    --accelerator cpu \
    --deterministic \
    --num-workers 0 \
    --logger csv \
  | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/2025-10-24T050500Z/train_cli_logger_csv.log
  ```
- Post-run packaging: copy `tmp/logger_smoke/lightning_logs/version_0/metrics.csv` to the smoke directory, run `tree tmp/logger_smoke/lightning_logs > plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/2025-10-24T050500Z/logger_tree.txt`, author `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/smoke/2025-10-24T050500Z/summary.md` (runtime, logger backend, warnings, follow-ups), then `rm -rf tmp/logger_smoke`.
- CI note: create `docs/ci_logger_notes.md` describing CSV artifact handling, disabling logs (`--logger none --quiet`), cleanup expectations, optional TensorBoard/MLflow requirements, and link back to CONFIG-LOGGER-001 and the smoke evidence path.
Pitfalls To Avoid:
- Do not leave `tmp/logger_smoke/` or `lightning_logs/` directories untracked—archive then delete.
- Keep artifact paths relative; avoid absolute `/home/...` references in summary.md or ci_logger_notes.md.
- Preserve DeprecationWarning wording for `--disable_mlflow`; mirror CLI output precisely.
- No production code edits this loop—CLI command + docs only.
- Avoid storing large binary checkpoints in the repository; reference their location via tree output instead.
- Ensure summary.md includes command + runtime so future loops can spot regressions.
Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md
- specs/ptychodus_api_spec.md:281
- docs/workflows/pytorch.md:329
- docs/findings.md:12
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/docs/2025-10-24T041500Z/summary.md:1
Next Up: Optional TensorBoard logger smoke or Phase EB3 governance dossier once CSV evidence + CI notes land.
