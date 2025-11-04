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
- Status: Phase F complete; Phase G comparison harness in place; awaiting training outputs to unblock real comparisons
- Owner/Date: Codex Agent/2025-11-04
- Working Plan: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md`
- Test Strategy: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md`
- Constraints: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/constraint_analysis.md`
- Notes: Use existing fly64 object/probe; enforce y‑axis split; group‑level overlap control via min center spacing; emphasize phase MS‑SSIM; PINN backend: TensorFlow; pty‑chi LSQML baseline (100 epochs) uses PyTorch internally.
- Attempts History: See `docs/archive/2025-11-06_fix_plan_archive.md` (section for this initiative) and the initiative’s `reports/` directories for run logs and metrics.

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
