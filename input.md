## Summary
Add normalization-invariant diagnostics to the Phase D analyzer so we can prove exactly where the symmetry in §Normalization Invariants breaks, then regenerate the gs1_ideal + dose_legacy_gs2 evidence under the new hub.

## Focus
DEBUG-SIM-LINES-DOSE-001 — Phase D2 normalization parity instrumentation

## Branch
paper

## Mapped tests
- pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v

## Artifacts
plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T122937Z/

## Do Now — DEBUG-SIM-LINES-DOSE-001.D2 normalization invariants
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py::{build_stage_ratio_summary,gather_scenario_data,render_markdown} — add a normalization-invariant section that multiplies the raw→grouped→normalized→prediction→truth ratios, flags deviations vs a 5% tolerance, and surfaces the results (JSON + Markdown) with an explicit citation to `specs/spec-ptycho-core.md §Normalization Invariants`.
- Implement: rerun the analyzer with the updated script for the existing D2 scenarios so the new JSON/Markdown land in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T122937Z/` (include analyzer stdout in `analyzer.log`).
- Test: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T122937Z/pytest_cli_smoke.log

## How-To Map
- Analyzer rerun:
  ```bash
  AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py \
    --scenario gs1_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/gs1_ideal \
    --scenario dose_legacy_gs2=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/dose_legacy_gs2 \
    --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T122937Z \
    > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T122937Z/analyze_intensity_bias.log
  ```
- Pytest guard already listed under Do Now.

## Pitfalls To Avoid
- Keep the normalization-invariant math tolerant to missing stage ratios—fallback gracefully instead of raising.
- Do not hardcode scenario names; analyzer changes must work for any directories produced by the runner.
- Preserve deterministic JSON ordering (e.g., `sort_keys=True`) so future diffs stay reviewable.
- Reference the spec verbatim; do not paraphrase the normalization equations inside the code comments.
- Leave production modules untouched; all changes stay under `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/`.

## If Blocked
Capture the failing command + stack trace into `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T122937Z/blocker.log`, note the exact scenario path, and update docs/fix_plan.md Attempts History with the blocker details before stopping.

## Findings Applied (Mandatory)
- NORMALIZATION-001 — analyzer output must explicitly show whether the loader obeys the three normalization systems.
- CONFIG-001 — keep the analyzer read-only; do not touch params.cfg or legacy modules.

## Pointers
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:338 — Phase D2 checklist and evidence requirements.
- docs/fix_plan.md:200 — D2 Attempts History + new 2026-01-20T122937Z entry.
- specs/spec-ptycho-core.md §Normalization Invariants — normative symmetry definition.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py — target script for the new diagnostics.

## Next Up (optional)
If the analyzer work finishes early, start designing the plan-local CLI that replays the dose_experiments normalization stack (D2b) so we no longer rely on sim_lines overrides.
