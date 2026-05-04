---
priority: 19
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-pdebench-darcy-full-training-benchmark/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md"),
        Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing Darcy benchmark inputs: {missing}")
    print("darcy benchmark inputs present")
    PY
  - pytest -q tests/studies/test_pdebench_darcy_data.py tests/studies/test_pdebench_darcy_metrics.py tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
related_roadmap_phases:
  - phase-2-pdebench-darcy-static-operator-benchmark
signals_for_selection:
  - Darcy readiness is implemented, but the full available training-split benchmark is still missing from the active queue.
  - The Phase 2 suite plan explicitly allows task-local Darcy full-benchmark scheduling while Phase 2 PDEBench remains the preferred evidence lane.
  - This item advances required PDEBench evidence without consuming the queue on optional Phase 3 or candidate work.
---

# Backlog Item: Run Darcy Full-Training Benchmark

## Objective

- Execute the authorized PDEBench Darcy beta `1.0` full-training benchmark on
  the fixed `8000 / 1000 / 1000` split for `hybrid_resnet_base`, `fno_base`,
  and `unet_strong`, then publish the resulting Phase 2 evidence under the
  existing Darcy contract.

## Scope

- Reuse the existing Darcy adapter, split, normalization, reporting, and
  run-budget contract already implemented under the image-suite plan.
- Launch the three required benchmark profiles on the full available training
  split with the fixed Darcy training recipe and benchmark-mode provenance.
- Collate same-contract metrics and comparison outputs for the three required
  profiles, including the recorded literature-context caveats.
- Update the durable Darcy Phase 2 summary and the required NeurIPS evidence
  indexes for any new benchmark rows or generated artifacts.

## Notes for Reviewer

- Do not treat readiness or capped pilot roots as benchmark-performance
  evidence for this item.
- Do not reopen CNS, CDI, Phase 4, or Phase 5 scope from this Darcy benchmark
  item.
- If any required profile cannot complete honestly under the fixed full-split
  contract, record the blocker and keep the item benchmark-incomplete rather
  than downgrading to a capped substitute.
