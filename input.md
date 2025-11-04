Summary: Kick off Phase A for STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 by finalizing design parameters (doses, gridsize set, neighbor K) and documenting inter‑group spacing rules + seeds in the plan.

Mode: Docs

Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Synthetic fly64 dose/overlap study

Branch: feature/torchapi-newprompt

Mapped tests: none — evidence-only

Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T003530Z/

Do Now:
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md — add Phase A specifics:
      • Dose list (e.g., 1e3, 1e4, 1e5)
      • Gridsize set {1, 2} and neighbor_count=7 for gs2
      • Inter‑group spacing targets S per view (dense/sparse) with rule S ≈ (1 − f_group) × N
      • Fixed seeds for reproducibility (simulation + grouping)
    Also update test_strategy.md PASS criteria with explicit spacing validation + CSV presence checks.
  - Validating selector: none — evidence-only
  - Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T003530Z/

Priorities & Rationale:
  - specs/data_contracts.md — dataset keys/dtypes and amplitude requirement
  - docs/GRIDSIZE_N_GROUPS_GUIDE.md — unified n_groups and new inter‑group overlap guidance
  - docs/SAMPLING_USER_GUIDE.md — oversampling now documented; use K≥C and n_groups>n_subsample when needed
  - docs/findings.md (CONFIG-001, DATA-001) — config bridge + contract adherence

How‑To Map:
  - Edit plan docs (implementation.md, test_strategy.md) to include the concrete dose list, gridsize set, K, spacing targets, and seeds
  - Save changes; record the artifact path in the plan and in docs/fix_plan.md Attempts

Pitfalls To Avoid:
  - Do not mix intensity vs amplitude in docs or examples
  - Avoid ambiguous overlap definitions; always specify S in pixels relative to N
  - Do not reference uninstalled dependencies for validation steps
  - Respect one‑loop docs rule; next loop must include a code/run task
  - Keep artifacts under the reports/ timestamped directory

If Blocked:
  - Log the block in docs/fix_plan.md Attempts with brief reason and path to partial draft

Findings Applied (Mandatory):
  - CONFIG-001 — update_legacy_dict must precede legacy calls (acknowledged in simulator usage)
  - DATA-001 — amplitude datasets; contract checks added to PASS criteria

Pointers:
  - specs/data_contracts.md:1
  - docs/GRIDSIZE_N_GROUPS_GUIDE.md:1
  - docs/SAMPLING_USER_GUIDE.md:1
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md:1
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:1

Next Up (optional):
  - Phase C pilot: generate one dose (1e4) dataset via simulate_and_save.py and split by y‑axis

