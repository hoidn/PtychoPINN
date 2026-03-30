# Hybrid ResNet Baseline Docs Plan

Goal: make the recommended Hybrid ResNet training baseline discoverable from `docs/index.md` so study agents naturally read it before launching experiments.

Planned changes:
- add `docs/model_baselines.md` as the canonical recommended-baseline doc
- add an index entry in `docs/index.md`
- update `docs/workflows/pytorch.md` to distinguish parameter definitions from recommended baselines
- update `docs/studies/lines_256_arch_improvement_loop.md` to inherit the baseline schedule from `docs/model_baselines.md`
- update `docs/studies/index.md` to surface that inheritance rule

Authority intent:
- `docs/model_baselines.md` = recommended project baselines
- `docs/CONFIGURATION.md` = parameter catalog and raw defaults
- study docs = explicit overrides to the recommended baseline
