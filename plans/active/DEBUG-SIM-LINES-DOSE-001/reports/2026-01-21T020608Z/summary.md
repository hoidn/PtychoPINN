### Turn Summary
Re-scoped D4f so grid-mode dose_response_study and the legacy data_preprocessing pipeline now have a concrete plan to attach dataset_intensity_stats instead of silently reverting to the 988.21 fallback.
Recorded the outstanding gaps in docs/fix_plan.md and implementation.md, opened artifacts hub 2026-01-21T020608Z, and rewrote input.md with the new Do Now (code/test/doc steps + pytest selectors + doc-sync plan).
Next: Ralph wires the helper into those manual constructors, refreshes the docs/tests, and archives the mapped pytest logs plus collect-only evidence.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T020608Z/
