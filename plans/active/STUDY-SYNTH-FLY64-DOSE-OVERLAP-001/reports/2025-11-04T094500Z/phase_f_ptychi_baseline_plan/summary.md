# Phase F PtyChi Baseline Planning Summary — 2025-11-04T094500Z

## Focus
- Initiative: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase F (pty-chi LSQML baseline)
- Objective: Stage TDD + execution plan so Ralph can implement a reconstruction orchestrator that reuses Phase D/E outputs and drives pty-chi LSQML runs with verifiable artifacts.

## Key Decisions
- Treat Phase F as a new CLI + orchestrator layer (`studies/fly64_dose_overlap/reconstruction.py`) mirroring the Phase E training CLI for consistency.
- Enforce TDD: first add RED manifest test (`test_build_ptychi_jobs_manifest`), then implement builder/CLI to turn Phase E manifest into LSQML job definitions.
- Artifact hub reserved at `reports/2025-11-04T094500Z/phase_f_ptychi_baseline/` with subdirectories `{plan,docs,red,green,collect,cli,real_run}` to keep RED/GREEN logs, manifests, and real reconstruction outputs aligned with governance rules.
- Deterministic LSQML runs will initially target dense gs2 view per dose; additional jobs (sparse, gs1 baseline) queued once infrastructure proves out.

## References Consulted
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md` (Phase F placeholders, Phase E artifact pointers).
- `docs/TESTING_GUIDE.md` §§2,4 for reconstruction CLI expectations and authoritative pytest commands.
- `docs/findings.md` entries CONFIG-001, DATA-001, POLICY-001, OVERSAMPLING-001 to keep reconstruction steps compliant.
- `specs/data_contracts.md` §§4–6 for NPZ field/dtype requirements feeding pty-chi.

## Next Actions for Engineer
1. Update `test_strategy.md` with Phase F section and add RED test for pty-chi job manifest (`tests/study/test_dose_overlap_reconstruction.py`).
2. Implement reconstruction builder/CLI to satisfy GREEN criteria and capture pytest logs under the new artifact hub.
3. Run CLI in dry-run mode, then execute first LSQML job once dependencies confirmed, archiving outputs + summary updates in the same hub.

## Artifact Map
- Plan: `plan.md`
- Summary: this file
- RED logs: `../phase_f_ptychi_baseline/red/pytest_phase_f_red.log` (expected once engineer runs RED)
- GREEN logs: `../phase_f_ptychi_baseline/green/pytest_phase_f_green.log`
- CLI transcripts: `../phase_f_ptychi_baseline/cli/`
- Real reconstructions: `../phase_f_ptychi_baseline/real_run/`

## Outstanding Risks
- pty-chi script GPU dependency (cupy) may not be available; if blocked, require fallback documentation + skip markers referencing hardware constraint.
- Dataset paths may need regeneration (Phase D/E). Include fallback step to rerun CLI generators if manifests missing.
- Long runtime: enforce deterministic seeds and limit initial scope to one job to stay within loop time budget.
