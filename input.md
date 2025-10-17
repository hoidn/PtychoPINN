Summary: Align PyTorch CLI config with dataset probe dimensions to unblock integration workflow
Mode: TDD
Focus: INTEGRATE-PYTORCH-001-PROBE-SIZE — Resolve PyTorch probe size mismatch in integration test
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_train_probe_size.py::test_cli_infers_probe_size -vv; pytest tests/torch/test_train_probe_size.py -vv; pytest tests/torch/test_integration_workflow_torch.py -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/{pytest_probe_red.log,pytest_probe_green.log,pytest_integration_green.log,parity_summary.md}

Do Now:
1. INTEGRATE-PYTORCH-001-PROBE-SIZE @ docs/fix_plan.md — Add failing pytest `tests/torch/test_train_probe_size.py::test_cli_infers_probe_size` capturing expected DataConfig.N from canonical NPZ (mock `PtychoDataModule` to record `data_config.N`); run `pytest tests/torch/test_train_probe_size.py::test_cli_infers_probe_size -vv` | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/pytest_probe_red.log (tests: targeted)
2. INTEGRATE-PYTORCH-001-PROBE-SIZE @ docs/fix_plan.md — Update `ptycho_torch/train.py` (cli_main path) to derive `DataConfig.N` and related geometry from NPZ metadata before bridge (reuse `ptycho_torch.dataloader.npz_headers` or load `probeGuess`); ensure legacy defaults preserved when metadata absent (tests: none)
3. INTEGRATE-PYTORCH-001-PROBE-SIZE @ docs/fix_plan.md — Re-run TDD coverage `pytest tests/torch/test_train_probe_size.py -vv` | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/pytest_probe_green.log; confirm new assertions pass (tests: targeted)
4. INTEGRATE-PYTORCH-001-PROBE-SIZE @ docs/fix_plan.md — Execute parity check `pytest tests/torch/test_integration_workflow_torch.py -vv` | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/pytest_integration_green.log`; verify train→infer completes without probe mismatch (tests: targeted)
5. INTEGRATE-PYTORCH-001-PROBE-SIZE @ docs/fix_plan.md — Refresh parity summary under 2025-10-17T231500Z (note probe fix + new logs), flip plan checklist notes, and log docs/fix_plan.md Attempt #1 with artifact paths (tests: none)

If Blocked: Archive failing selector output under the timestamped reports directory, summarize the blocker + hypothesis in parity_summary.md, leave plan rows unchecked, and record the status in docs/fix_plan.md Attempts history.

Priorities & Rationale:
- docs/fix_plan.md:55-68 — New probe size item defines acceptance (integration must finish, plans updated).
- ptycho_torch/train.py:420-499 — CLI currently hardcodes `DataConfig(N=128)` causing probe tensor mismatch.
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md:31-35 — Phase E2.D2 guidance now flags probe mismatch and expects follow-up resolution.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/parity_summary.md:149-158 — Immediate next step is fixing probe mismatch and documenting results.
- specs/data_contracts.md:1-52 — Canonical NPZ contract guarantees probe/object dimensions required for config harmonization.

How-To Map:
- export ts=2025-10-17T231500Z; mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/$ts
- Write pytest module using pure pytest style; monkeypatch `ptycho_torch.train.PtychoDataModule` to capture `data_config` and stub out trainer work
- pytest tests/torch/test_train_probe_size.py::test_cli_infers_probe_size -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$ts/pytest_probe_red.log
- Implement helper (e.g., `_infer_probe_size(train_data_file)`) in `ptycho_torch/train.py` that reads NPZ headers or `probeGuess` and sets both `data_config.N` and any dependent fields before calling the bridge; handle rectangular diffraction by trusting probe dims
- pytest tests/torch/test_train_probe_size.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$ts/pytest_probe_green.log
- pytest tests/torch/test_integration_workflow_torch.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$ts/pytest_integration_green.log
- Update `plans/active/INTEGRATE-PYTORCH-001/reports/$ts/parity_summary.md` summarizing commands, dataset shapes, config changes, and confirming alignment with DATA-001/CONFIG-001
- Mark D2 guidance in `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md` with new evidence link; append docs/fix_plan.md Attempt #1 citing artifact paths

Pitfalls To Avoid:
- Do not hardcode dataset-specific constants; derive sizes from NPZ metadata to keep workflow general.
- Keep new pytest module free of unittest.TestCase to respect project testing guidance.
- Avoid loading entire diffraction stack when probing metadata; rely on headers or specific keys to limit IO.
- Preserve backward compatibility for legacy CLI path (`--ptycho_dir/--config`); guard new helper usage accordingly.
- Maintain CONFIG-001 order: update params.cfg only after adjusting `data_config` so legacy modules see correct `N`.
- Ensure logs and summaries stay within the timestamped reports directory; no artifacts at repo root.
- If integration still fails, do not mark ledger items complete; capture failure and rationale in parity summary.
- Re-run targeted tests from repo root using editable install to keep import paths consistent.

Pointers:
- specs/data_contracts.md:1 — Canonical NPZ dimensions (diffraction/object/probe)
- specs/ptychodus_api_spec.md:70 — ModelConfig.N requirements for reconstructor contract
- ptycho_torch/train.py:420 — Hardcoded DataConfig defaults that must adapt to NPZ metadata
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md:33 — D2 parity guidance now referencing probe blocker
- docs/fix_plan.md:55 — Ledger entry tracking probe mismatch scope and exit criteria

Next Up: Once probe sizing is green, revisit [INTEGRATE-PYTORCH-001-STUBS] to finish Phase D2 workflow stubs.
