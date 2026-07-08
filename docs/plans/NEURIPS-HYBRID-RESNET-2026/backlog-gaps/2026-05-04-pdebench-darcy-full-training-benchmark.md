# Seed Plan: PDEBench Darcy Full-Training Benchmark

## Objective

Execute the already-authorized PDEBench Darcy beta `1.0` benchmark on the full
`8000 / 1000 / 1000` split for `hybrid_resnet_base`, `fno_base`, and
`unet_strong`, then close the current Phase 2 Darcy evidence gap with durable
same-contract metrics and provenance.

## Scope

- Consume the existing authority in:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md`
- Reuse the written run budget and fixed Darcy contract: official beta `1.0`
  file, deterministic `8000 / 1000 / 1000` split, separate train-only
  input/target normalization, relative-L2 primary loss, and required benchmark
  profiles `hybrid_resnet_base`, `fno_base`, and `unet_strong`.
- Run the benchmark profiles under benchmark mode, one at a time if needed for
  the RTX 3090 budget, with tmux plus `ptycho311`.
- Write durable closeout updates only for the Darcy summary and required
  evidence indexes after the benchmark rows exist.

## Non-Goals

- No CDI, Phase 3, candidate, Phase 4, or Phase 5 work.
- No roadmap, steering, progress-ledger, or backlog-queue rewrites beyond this
  drafted item.
- No capped-pilot relabeling as benchmark evidence.
- No new adapter/model/test implementation unless the executor finds an honest
  benchmark blocker that must be documented separately.

## Expected Artifacts

- Benchmark run roots under:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/`
- Updated comparison payloads for the three required benchmark profiles:
  `comparison_summary.json` and `comparison_summary.csv`
- Per-profile metrics and provenance artifacts already defined by the Darcy
  execution plan
- Updated durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md`
- Required evidence-index updates for any new benchmark rows or artifacts

## Verification Commands

```bash
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
pytest -q tests/studies/test_pdebench_darcy_data.py tests/studies/test_pdebench_darcy_metrics.py tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
```
