---
priority: 19
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
        Path("docs/backlog/done/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md"),
        Path("scripts/studies/grid_lines_compare_wrapper.py"),
        Path("scripts/studies/grid_lines_torch_runner.py"),
        Path("ptycho_torch/generators/hybrid_resnet.py"),
        Path("ptycho_torch/generators/ffno.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing probe-channel conditioning inputs: {missing}")
    print("probe-channel conditioning inputs present")
    PY
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno or probe"
  - pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno or probe"
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
  - 2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The current CDI input hides the known probe from the learned generator even though the physics forward model uses it.
  - A same-contract probe-channel ablation tests whether explicit probe conditioning improves SRU-Net or pure FFNO on Lines128.
  - This is append-only Phase 3 CDI mechanism evidence and should not rerun completed baseline rows.
---

# Backlog Item: Lines128 CDI Probe-Channel Conditioning Ablation

## Objective

- Add a controlled Lines128 CDI input-conditioning ablation that concatenates
  the fixed probe to the learned-model input channels.
- Test whether explicit probe access helps the two most relevant learned CDI
  generator families:
  - `pinn_hybrid_resnet` / SRU-Net + PINN;
  - corrected pure `pinn_ffno` with `fno_blocks=4` and `fno_cnn_blocks=0`.
- Compare only against completed same-contract rows by lineage.

## Scope

- Preserve the locked Lines128 contract:
  - `N=128`, `gridsize=1`, synthetic grid-lines, `set_phi=True`;
  - Run1084 fixed probe with `probe_scale_mode=pad_extrapolate` and
    `probe_smoothing_sigma=0.5`;
  - fixed train/test split, seed, epoch budget, scheduler, loss, output mode,
    metric schema, and fixed visual sample policy.
- Add a config-controlled input path that appends two probe channels to each
  CDI model input:
  - probe real part;
  - probe imaginary part.
- The appended probe must be the same preprocessed probe used by the forward
  model, broadcast per sample, and recorded in invocation/config artifacts.
- Do not change the physics loss probe, target reconstruction, dataset, visual
  scaling, or baseline row definitions.
- Run only the fresh probe-conditioned rows. Do not rerun the unconditioned
  SRU-Net or FFNO baselines unless their existing lineage audit fails.

## Outputs

- Item-local artifacts under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/`.
- Row-local invocation/config/history/metrics/checkpoint/reconstruction/visual
  artifacts for:
  - `pinn_hybrid_resnet_probe_channels`;
  - `pinn_ffno_probe_channels`.
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_probe_channel_conditioning_ablation_summary.md`.
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`;
  - `docs/studies/index.md`.

## Completion Gate

- The fresh row configs must prove that probe channels were concatenated to the
  learned input and that the same Run1084 probe preprocessing was used.
- The summary must report metric deltas against unconditioned SRU-Net and
  corrected pure FFNO baselines under the same Lines128 contract.
- Any conclusion must frame this as an input-conditioning ablation, not a new
  default model family or a replacement for the completed Lines128 authority.

## Notes For Reviewer

- Reject implementations that expose the probe through hidden global state
  rather than explicit dataset/runner/model configuration.
- Reject rows that change both probe conditioning and unrelated architecture,
  loss, scheduler, probe preprocessing, seed, or sample policy.
- Reject table refreshes that overwrite completed baseline rows instead of
  appending the probe-conditioned variants by lineage.
