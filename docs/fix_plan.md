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
- Status: in_progress — Phase E6 bundle size tracking implementation
- Owner/Date: Codex Agent/2025-11-04
- Working Plan: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md`
- Test Strategy: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md`
- Constraints: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/constraint_analysis.md`
- Notes: Use existing fly64 object/probe; enforce y‑axis split; group‑level overlap control via min center spacing; emphasize phase MS‑SSIM; PINN backend: TensorFlow; pty‑chi LSQML baseline (100 epochs) uses PyTorch internally.
- Attempts History: See `docs/archive/2025-11-06_fix_plan_archive.md` (section for this initiative) and the initiative's `reports/` directories for run logs and metrics.
- Latest Attempt (2025-11-06T130500Z): Implementation — Phase E6 CLI stdout normalization GREEN. Enhanced test with stdout relative-path assertions (tests/study/test_dose_overlap_training.py:1645-1652); updated training CLI to normalize bundle_path before stdout emission (studies/fly64_dose_overlap/training.py:736-742). RED confirmed absolute-path bug; GREEN validates `wts.h5.zip` relative format. Full suite: 397 passed, 1 pre-existing fail (test_interop_h5_reader), 17 skipped in 249s. Artifacts: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T130500Z/phase_e_training_bundle_real_runs_exec/` (RED/GREEN/collect/full logs + summary.md). Commit: [pending].
- Latest Attempt (2025-11-06T150500Z): Bug Fix — PyTorch workflows Path type violations. While attempting Phase E6 evidence capture, discovered **critical production bugs** in `ptycho_torch/workflows/components.py:650,682` where string paths from `TrainingConfig` were passed to functions expecting `Path` objects. Symptoms: `AttributeError: 'str' object has no attribute 'exists'` and `TypeError: unsupported operand type(s) for /`. Fixed by wrapping `config.train_data_file` and `config.output_dir` with `Path()` at call sites (ptycho_torch/workflows/components.py:650-651, 682). Impact: Unblocks all PyTorch backend training workflows. Tests: RED passed (mocked code bypassed bug), GREEN passed after fix, full suite 397 passed/1 pre-existing fail/17 skipped in 249s. New finding: TYPE-PATH-001 (to document). Artifacts: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/` (summary.md, test logs). Commit: [next].
- Latest Attempt (2025-11-06T170500Z): Planning — Phase E6 dense/baseline SHA parity staging. Reviewed prior RED/GREEN logs, confirmed Path fix unblock, and reserved new hub `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/` with refreshed plan/summary. Do Now directs strengthening `test_training_cli_records_bundle_path`, rerunning deterministic dense+baseline CLI jobs, archiving via `bin/archive_phase_e_outputs.py`, and documenting TYPE-PATH-001 in findings. Findings applied: POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001, TYPE-PATH-001.
- Latest Attempt (2025-11-06T170500Z+exec): Implementation — Phase E6 SHA parity enforcement GREEN. Enhanced test with stdout→manifest SHA cross-validation (tests/study/test_dose_overlap_training.py:1678-1702). Build (view,dose)→SHA map from manifest, assert each stdout checksum matches. RED: PASSED 7.11s (baseline intact). GREEN: PASSED 7.45s (parity enforced). Selector: 4 collected. Full suite: 397 passed/1 pre-existing fail (test_interop_h5_reader)/17 skipped in 284s. Documented TYPE-PATH-001 in docs/findings.md:21 (PyTorch Path normalization bug from 2025-11-06T150500Z). Nucleus complete: test enhancement + TYPE-PATH-001 doc. Deferred dense/baseline real runs + archival to follow-up loop (evidence collection vs core acceptance). Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T170500Z/phase_e_training_bundle_real_runs_exec/ (RED/GREEN/collect/full logs + summary.md). Commit: [next].
- Latest Attempt (2025-11-06T190500Z): Planning — Phase E6 bundle size metadata + dense/baseline real-run evidence. Reviewed 2025-11-06T170500Z execution summary and verified stdout/manifest SHA parity landed; identified gap: manifest lacks bundle size metrics for integrity tracking. Reserved new hub `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/` with refreshed plan/summary. Do Now directs augmenting `studies/fly64_dose_overlap/training.py` to emit `bundle_size_bytes`, updating the Phase E6 CLI test, rerunning deterministic dense (gs2) + baseline (gs1) jobs, archiving outputs via `bin/archive_phase_e_outputs.py`, and logging checksum + size evidence. Findings applied: POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001, TYPE-PATH-001.
- Latest Attempt (2025-11-06T190500Z+exec): Implementation — Phase E6 bundle size tracking GREEN. Enhanced test with manifest bundle_size_bytes validation (tests/study/test_dose_overlap_training.py:1604-1618) and stdout Size line format/parity checks (1634-1641, 1721-1784). RED: FAILED on missing Size lines (0/2 expected). GREEN: PASSED 6.82s after implementing size computation (studies/fly64_dose_overlap/training.py:515 via .stat().st_size), result dict propagation (546), stdout emission (755-756). Full suite: 397 passed/1 pre-existing fail/17 skipped in 332.53s. Nucleus complete: bundle_size_bytes field computed, serialized, emitted with view/dose context, cross-validated. Dense/baseline real runs deferred to follow-up loop. Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T190500Z/phase_e_training_bundle_real_runs_exec/ (RED/GREEN/collect/full logs + summary.md). Commit: [next].
- Latest Attempt (2025-11-06T210500Z): Planning — Phase E6 dense/baseline deterministic evidence & archive parity. Confirmed bundle size feature landed (Attempt 2025-11-06T190500Z+exec) and identified remaining gap: real CLI runs + archive helper still lack size validation output. Reserved hub `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/` with new plan/summary. Do Now instructs updating `bin/archive_phase_e_outputs.py` to compare manifest `bundle_size_bytes` against filesystem sizes, running deterministic dose=1000 dense/gs2 and baseline/gs1 jobs, re-archiving bundles with SHA+size proof, and refreshing summary ledger notes. Findings applied: POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001, TYPE-PATH-001.

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
