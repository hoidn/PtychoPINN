### Turn Summary

Implemented normalization-invariant diagnostics in `bin/analyze_intensity_bias.py` that multiply the raw→grouped→normalized→prediction→truth stage ratios, flag deviations vs 5% tolerance, and surface results with explicit spec citations.
Reran the analyzer for gs1_ideal + dose_legacy_gs2 scenarios; dose_legacy_gs2 shows full chain product=1.986 (deviation 98.6%), confirming symmetry violation per `specs/spec-ptycho-core.md §Normalization Invariants`.
Next: trace `normalize_data` gain formula in loader/synthetic_helpers to determine whether intensity_scale is being double-applied or if model outputs require inverse-scaling.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T122937Z/ (bias_summary.json, bias_summary.md, pytest_cli_smoke.log)
