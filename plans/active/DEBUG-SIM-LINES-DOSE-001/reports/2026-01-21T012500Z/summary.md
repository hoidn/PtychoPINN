### Turn Summary
Verified that D4f implementation is working: `loader.load()` attaches raw diffraction stats to containers, and `calculate_intensity_scale()` correctly uses them to compute the spec-compliant dataset-derived intensity scale.
The key evidence shows bundles now save ~558/~273 (training data scale) instead of the 988.21 fallback; the analyzer's different 576/577 value is from TEST data (expected mismatch).
Next: Phase D5 forward-pass instrumentation to trace the remaining ~2.3-2.7x amplitude bias, which now appears to originate in model/loss wiring rather than normalization.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T012500Z/ (bias_summary.md, gs*_ideal/, logs/)
