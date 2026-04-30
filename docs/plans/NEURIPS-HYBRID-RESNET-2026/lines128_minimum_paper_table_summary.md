# NeurIPS Lines128 Minimum Paper Table Summary

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-minimum-paper-table`
- State: `paper_complete`
- Chosen execution path: `fresh_rerun_then_same_root_bundle_regeneration`
- Authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`

## Completed In This Pass

- fixed the review-blocking wrapper-finalization gap in
  `scripts/studies/lines128_paper_benchmark.py`: once the wrapper invocation is
  `completed`, the finalizer now reattaches recovered-Torch
  `launcher_completion.json` evidence into the row `outputs` payloads, rewrites
  `metrics.json` / `model_manifest.json` from those refreshed rows, and
  refreshes `paper_benchmark_manifest.json` row artifacts to match
- added a regression in
  `tests/studies/test_lines128_paper_benchmark.py` that reproduces the exact
  retained-root failure mode under `--reuse-existing-recons`
- reran same-root minimum-subset bundle regeneration under a dedicated tmux
  launcher contract and refreshed the authoritative
  `minimum_subset_20260430T084339Z` root so `metrics.json`,
  `model_manifest.json`, and `paper_benchmark_manifest.json` now all agree on
  `paper_complete`
- reran the focused suite, required backlog verification gate, and
  `compileall` with fresh passing evidence
- accepted `minimum_subset_20260430T084339Z` as the authoritative minimum-table
  root: its metrics, model manifest, and paper benchmark manifest report
  `paper_complete`, empty `missing_fields_by_row`, and no missing bundle
  artifacts, and the recovered-Torch launcher-completion evidence is now
  attached directly inside the row outputs/manifests
- retained the earlier `minimum_subset_20260430T035104Z` rerun and the stopped
  `minimum_subset_20260430T051928Z` follow-up as historical, non-authoritative
  roots under the current launcher/row invocation contract

## Current Contract State

- fixed runtime roster:
  `baseline`, `pinn`, `pinn_hybrid_resnet`, `pinn_fno_vanilla`
- paper-facing row labels:
  `CDI CNN + supervised`, `CDI CNN + PINN`, `Hybrid ResNet + PINN`,
  `FNO Vanilla + PINN`
- selected FNO comparator:
  `fno_vanilla`
- fixed seed policy:
  `seed=3` for all four rows under the frozen minimum-subset contract; no
  row-specific reseeding or post-metric seed changes were allowed
- claim boundary:
  `minimum_draftable_cdi_subset`
- status:
  `paper_complete`
- bundle completeness:
  `minimum_subset_20260430T084339Z` has all required rows and no missing
  required fields or bundle artifacts
- fixed sample ids:
  `0`, `1`
- shared visual-scale policy:
  stitched numeric arrays for amplitude/phase and derived shared absolute-error
  scales
- baseline-family labeling note:
  the CDI `cnn` rows (`baseline`, `pinn`) fill the same paper-facing
  local-baseline role here that `unet_strong` fills in the bounded PDEBench CNS
  package, but they remain task-local implementations and are not identical or
  interchangeable model families

## Root Status

- authoritative root:
  `runs/minimum_subset_20260430T084339Z`
  - status: `paper_complete`
  - reason: structured row provenance, metrics, source arrays, visuals,
    dataset/split identity, git/environment metadata, process-completion
    evidence, wrapper launcher provenance, and the recovered-Torch
    `launcher_completion.json` artifacts now satisfy the current evidence
    contract. Shared or duplicated diagnostic logs are not a paper-grade
    blocker by themselves.
- superseded earlier fresh rerun:
  `runs/minimum_subset_20260430T035104Z`
  - reason: this root predates the final wrapper/torch invocation contract
    repair and is no longer the promoted paper-grade minimum-table evidence
- interrupted follow-up rerun:
  `runs/minimum_subset_20260430T051928Z`
  - reason: the fresh post-fix benchmark rerun was started successfully but
    stopped before completion because the full execution extends beyond a normal
    review-fix pass

## Verification

- focused review-fix regression suite:
  `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/test_grid_lines_compare_wrapper.py tests/studies/test_paper_provenance.py`
  -> `76 passed, 23 warnings in 19.15s`
- required deterministic gates:
  `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `173 passed, 47 warnings in 301.72s (0:05:01)`
- required compile gate:
  `python -m compileall -q ptycho_torch scripts/studies`
  -> exit `0`
- archived logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_focused_20260430T122925Z.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_required_20260430T122925Z.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_required_20260430T122925Z.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/lines128_same_root_bundle_regen_20260430T122925Z.log`

## Boundary And Remaining Scope

- this item restores a paper-grade minimum CDI subset root under the current
  evidence-sufficient contract
- `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno` remain later
  out-of-scope rows
- the harness preflight note remains readiness-only authority; the required
  minimum-subset execution note plus its machine-readable execution manifest
  remain the launch-controlling surfaces for this item
