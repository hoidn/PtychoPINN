# Phase 2 PDEBench SWE Longer Execution Report

## Completed In This Pass

- Extended the existing `scripts/studies/pdebench_swe/` harness with longer-run profiles, run-budget validation, full/run split manifests, sample-based normalization provenance, per-profile CUDA peak resets, freshness validation, comparison collation, and a thin CLI entrypoint.
- Ran the official longer SWE one-step benchmark in tmux with selected run ID `20260420T115509.961336393Z`, tracked child PID `3052378`, and selected run root `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/`.
- Wrote the durable Phase 2 summary at `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md` and updated `docs/studies/index.md` and `docs/index.md` for discoverability.

## Completed Plan Tasks

- Phase A: prerequisite gate checks, ignored artifact root, disk/GPU preflight, package provenance, license/access note, and target run budget.
- Phase B: red tests for split manifests, normalization semantics, model profiles, longer CLI/freshness behavior, and reporting collation.
- Phase C/D: implementation of split/data/metric/profile/manifest extensions, longer runner, and reporting.
- Phase E/F: focused verification, longer help check, official inspect-only pass, tmux launch, exact PID tracking, and selected-run freshness validation.
- Phase G/H: PDE execution summary, discoverability updates, final focused pytest, plan pointer verification, selected run verification, and artifact hygiene check.

## Remaining Required Plan Tasks

- No remaining tasks in this tranche.
- Because SWE primary was noncompetitive, the next roadmap action is a separate OpenFWI FlatVel-A fallback execution tranche; that fallback was not executed here by plan scope.

## Verification

- `python -m pytest tests/studies/test_pdebench_swe_manifest.py tests/studies/test_pdebench_swe_splits_data.py tests/studies/test_pdebench_swe_metrics.py tests/studies/test_pdebench_swe_models.py tests/studies/test_pdebench_swe_run_config.py tests/studies/test_pdebench_swe_longer_cli.py tests/studies/test_pdebench_swe_reporting.py tests/studies/test_pdebench_swe_smoke_cli.py -v`
  - Result: `46 passed in 16.14s`
  - Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/final_pytest.log`
- Structural summary/discoverability check:
  - Result: `PDE execution summary and discoverability checks passed`
- Selected longer-run validation:
  - Result: `selected longer run verified: run_id=20260420T115509.961336393Z pid=3052378 root=.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z`
- Official freshness validation:
  - Result: `official longer SWE artifacts are fresh for run_id=20260420T115509.961336393Z pid=3052378`
- Plan pointer check:
  - Result: `plan_path pointer is valid`

## Residual Risks

- The SWE longer run is one seed and one one-step budget; it is sufficient for the Phase 2 gate but not a broad robustness claim.
- Hybrid ResNet failed the operational 10% competitiveness threshold against the best local baseline (`relative_gap_vs_best_baseline=0.8900414694`), so this tranche records `Decision: pivot to OpenFWI FlatVel-A`.
- The OpenFWI fallback still needs its own approved implementation tranche and storage/access preflight.
