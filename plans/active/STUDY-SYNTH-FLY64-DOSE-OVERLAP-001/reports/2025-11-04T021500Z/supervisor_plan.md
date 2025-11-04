# Supervisor Plan — Phase A Constants

- Mode: TDD planning
- Goal: lock canonical dose list, gridsize combinations, neighbor count, spacing heuristic, and seeds for the synthetic fly64 study.
- Next engineer loop: implement `studies/fly64_dose_overlap/design.py::get_study_design`, author RED test `tests/study/test_dose_overlap_design.py::test_study_design_constants`, then document the concrete values in phase docs.
- Artifact expectations: pytest red/green logs, collect-only proof, updated implementation/test_strategy docs, summary.md capturing spacing derivation.
