# Execution Report

## Completed In This Pass

- Fixed complete-table promoted-row recovery so row-local launcher completion proof can be rebuilt from the promoted source root's durable wrapper logs, with fallback to `live_*.log` and `tmux.log` when `launcher_stdout.log` / `launcher_stderr.log` are absent.
- Aligned paper-bundle validation with the approved complete-table contract so recovered PyTorch rows in `complete_table` mode require `launcher_completion.json`; missing proof now downgrades the bundle to `benchmark_incomplete` instead of silently leaving `paper_complete`.
- Added mixed-source complete-table regressions in `tests/studies/test_lines128_paper_benchmark.py` covering both:
  - successful FFNO proof recovery from promoted-source logs
  - enforced downgrade when recovered completion proof is still unavailable
- Rebuilt the authoritative complete-table root through tmux using the checked-in repair execution manifest:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`

## Completed Current-Scope Work

- The blocking review issue is resolved in both code and artifact state. The new authoritative root reports `paper_complete` across `metrics.json`, `model_manifest.json`, and `paper_benchmark_manifest.json`, and all promoted recovered PyTorch rows now carry row-local `launcher_completion.json`, including `pinn_ffno`.
- The approved execution contract remains unchanged. No design, roster, comparator, seed, sample, or location boundary changed in this pass; the repair stayed within the existing plan and design authority.
- The earlier repaired roots remain on disk for provenance, but they are superseded and must not be used for paper-facing claims:
  - `complete_table_20260430T140954Z_repair`
  - `complete_table_20260430T141325Z_repair`
  - `complete_table_20260430T150642Z_repair`

## Follow-Up Work

- The FFNO prerequisite root still lacks dedicated `launcher_stdout.log` / `launcher_stderr.log` artifacts, so completion proof currently depends on the durable `tmux.log` fallback. A later hygiene pass could backfill explicit wrapper launcher logs in the prerequisite root to reduce provenance indirection.
- Superseded repaired roots remain in the artifact tree for auditability. If storage or operator clarity becomes a concern, a later artifact-curation pass can mark them more explicitly as non-authoritative.

## Verification

- Focused regression:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_paper_provenance.py tests/test_grid_lines_compare_wrapper.py`
  - result: `82 passed, 23 warnings in 18.41s`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/pytest_focused_20260430T150921Z.log`
- Required deterministic gate:
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - result: `173 passed, 47 warnings in 302.06s (0:05:02)`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/pytest_required_20260430T150921Z.log`
- Compile gate:
  - `python -m compileall -q ptycho_torch scripts/studies`
  - result: exit `0`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/compileall_required_20260430T150921Z.log`
- Deterministic repair rebuild:
  - `python scripts/studies/lines128_paper_benchmark.py --mode complete_table ... --execution-manifest .../complete_execution_manifest_repair.json --output-dir .../complete_table_20260430T150757Z_repair_tmux`
  - result: exit `0`; rebuilt root is `paper_complete`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/lines128_complete_table_repair_20260430T150757Z.log`

## Residual Risks

- The authoritative root still inherits recovered FFNO row provenance from the earlier prerequisite repair pass; this pass repaired the missing row-local completion proof, but it does not change the underlying prerequisite provenance caveat.
- The complete-table repair logic now correctly refuses to overclaim when launcher proof is missing. Any future promoted-source root that lacks usable wrapper logs will remain `benchmark_incomplete` until explicit completion evidence is available.
