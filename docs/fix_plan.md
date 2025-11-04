# PtychoPINN Fix Plan Ledger (Condensed)

**Last Updated:** 2025-11-06
**Active Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis [in_progress]

---

Housekeeping: This ledger is intentionally brief. Detailed “Attempts History” entries were moved to archives to keep decision‑making crisp.

- Current archive snapshot: `docs/archive/2025-11-06_fix_plan_archive.md`
- Earlier snapshots: `docs/archive/2025-10-17_fix_plan_archive.md`, `docs/archive/2025-10-20_fix_plan_archive.md`

Use the “Working Plan” and “reports/” under each initiative for day‑to‑day artifacts.

---

## [STUDY-SYNTH-FLY64-DOSE-OVERLAP-001] Synthetic fly64 dose/overlap study
- Depends on: —
- Priority: High
- Status: Phase E6 complete (CLI stdout normalization with artifact-relative bundle paths + view/dose context); Phase G comparison ready pending Phase E/F real runs
- Owner/Date: Codex Agent/2025-11-04
- Working Plan: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md`
- Test Strategy: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md`
- Constraints: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/constraint_analysis.md`
- Notes: Use existing fly64 object/probe; enforce y‑axis split; group‑level overlap control via min center spacing; emphasize phase MS‑SSIM; PINN backend: TensorFlow; pty‑chi LSQML baseline (100 epochs) uses PyTorch internally.
- Attempts History: See `docs/archive/2025-11-06_fix_plan_archive.md` (section for this initiative) and the initiative's `reports/` directories for run logs and metrics.
- Latest Attempt (2025-11-06T130500Z): Implementation — Phase E6 CLI stdout normalization GREEN. Enhanced test with stdout relative-path assertions (tests/study/test_dose_overlap_training.py:1645-1652); updated training CLI to normalize bundle_path before stdout emission (studies/fly64_dose_overlap/training.py:736-742). RED confirmed absolute-path bug; GREEN validates `wts.h5.zip` relative format. Full suite: 397 passed, 1 pre-existing fail (test_interop_h5_reader), 17 skipped in 249s. Artifacts: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/` (RED/GREEN/collect/full logs + summary.md). Commit: [pending].
- Latest Attempt (2025-11-06T150500Z): Bug Fix — PyTorch workflows Path type violations. While attempting Phase E6 evidence capture, discovered **critical production bugs** in `ptycho_torch/workflows/components.py:650,682` where string paths from `TrainingConfig` were passed to functions expecting `Path` objects. Symptoms: `AttributeError: 'str' object has no attribute 'exists'` and `TypeError: unsupported operand type(s) for /`. Fixed by wrapping `config.train_data_file` and `config.output_dir` with `Path()` at call sites (ptycho_torch/workflows/components.py:650-651, 682). Impact: Unblocks all PyTorch backend training workflows. Tests: RED passed (mocked code bypassed bug), GREEN passed after fix, full suite 397 passed/1 pre-existing fail/17 skipped in 249s. New finding: TYPE-PATH-001 (to document). Artifacts: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/` (summary.md, test logs). Commit: [next].

## [EXPORT-PTYCHODUS-PRODUCT-001] TF-side Ptychodus product exporter/importer + Run1084 conversion
- Depends on: —
- Priority: Medium
- Status: planning
- Owner/Date: Codex Agent/2025-10-28
- Working Plan: `plans/active/EXPORT-PTYCHODUS-PRODUCT-001/implementation_plan.md`
- Test Strategy: `plans/active/EXPORT-PTYCHODUS-PRODUCT-001/test_strategy.md`
- Attempts History: Initial plan/test strategy drafted — see plan files for details.

## [INTEGRATE-PYTORCH-001-STUBS] Finish PyTorch workflow stubs deferred from Phase D2
- Status: archived 2025-10-20 — see `docs/archive/2025-10-20_fix_plan_archive.md#integrate-pytorch-001-stubs-finish-pytorch-workflow-stubs-deferred-from-phase-d2`.

## [INTEGRATE-PYTORCH-001-DATALOADER] Restore PyTorch dataloader DATA-001 compliance
- Status: archived 2025-10-20 — see `docs/archive/2025-10-20_fix_plan_archive.md#integrate-pytorch-001-dataloader-restore-pytorch-dataloader-data-001-compliance`.

---

Process references:
- Supervisor prompt: `prompts/supervisor.md`
- Engineer prompt: `prompts/main.md`
- Findings ledger: `docs/findings.md`
- Data contracts: `specs/data_contracts.md`
