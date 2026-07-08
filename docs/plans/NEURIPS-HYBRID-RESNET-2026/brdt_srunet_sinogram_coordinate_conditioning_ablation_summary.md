# BRDT SRU-Net Sinogram Coordinate-Conditioning Ablation Summary

- Date: `2026-05-08`
- Backlog item: `2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation/execution_plan.md`
- State: `decision_support_append_only`
- Claim boundary: `decision_support_append_only_coordgrid_diagnostic`
- Governing design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Item root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation/`

## Objective

- Add exactly one append-only BRDT sinogram-input SRU-Net row,
  `sru_net_coordgrid`, that keeps the completed
  `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` contract fixed while
  appending deterministic normalized object-grid `x/y` channels after the
  sinogram-to-grid bilinear resize.
- Compare that fresh row only against the unconditioned learned SRU-Net
  authority from
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_40ep_paper_evidence_summary.md`.
- Preserve `ffno` and `classical_born_backprop` as lineage context only, not as
  widened required scope.

## Claim Boundary

- This item is append-only representational diagnostic evidence only.
- It does not replace the current BRDT sinogram-input additive-secondary
  authority.
- It does not replace CDI `lines128` or PDEBench CNS.
- It is not a physically principled inverse operator path.

## Fixed Contract And Conditioning Proof

- Comparison standard: exact equality on the inherited dataset/operator/split/
  optimizer/loss/scheduler/sample policy fields recorded in
  `preflight_manifest.json`; only the learned input composition changed.
- Frozen inherited contract:
  - dataset id `brdt128_decision_support_preflight`
  - split `2048 / 256 / 256`
  - operator `BornRytovForward2D`, Born mode, `N=D=128`, `A=64`,
    `wavelength_px=8.0`, `medium_ri=1.333`, `normalize=unitary_fft`
  - loss weights `image=1.0`, `physics=0.1`, `relative_physics=0.1`,
    `tv=1e-5`, `positivity=1e-4`
  - optimizer Adam, `lr=2e-4`, batch size `16`, seed `42`, `40` epochs,
    `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
  - fixed paper sample `255`
- Allowed change only:
  - `input_mode="sinogram"` remains fixed
  - `sinogram_to_grid="bilinear_resize"` remains fixed
  - learned input channels change `2 -> 4`
  - `coordinate_channels="object_xy"` are appended deterministically on the
    `128 x 128` object grid after resize
- Row-local contract proof:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation/rows/sru_net_coordgrid/adapter_contract.json`

## Row Outcome

Fresh row root:
`.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation/rows/sru_net_coordgrid/`

Baseline authority root:
`.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/rows/sru_net/`

| Metric | `sru_net` baseline | `sru_net_coordgrid` | Delta |
|---|---:|---:|---:|
| image relative L2 | 0.716527 | 0.999938 | +0.283411 |
| image RMSE | 0.004064 | 0.005671 | +0.001607 |
| image MAE | 0.001408 | 0.001951 | +0.000542 |
| PSNR proxy | 22.711670 | 19.816859 | -2.894811 |
| SSIM phys | 0.735932 | 0.026206 | -0.709726 |
| measurement relative L2 | 0.466556 | 0.999775 | +0.533220 |
| measurement RMSE | 0.003951 | 0.008468 | +0.004516 |
| measurement MAE | 0.002028 | 0.003626 | +0.001598 |
| final train loss | 0.302442 | 0.453399 | +0.150957 |
| best observed train loss | 0.302442 | 0.453394 | +0.150953 |

- Parameter count: `142,450` versus baseline `142,162` (`+288`)
- Eval throughput: `378.8693` samples/s versus baseline `404.6978`
  (`-25.8286`)
- Train wall time: `165.4470 s` versus baseline `164.2903 s` (`+1.1568 s`)

## Convergence Read

- Tracked live run completed under PID `3468206` with `exit_code=0`.
- The row emitted the planned `40` history records.
- Final learning rate reached `1e-5` after `5` reductions.
- `materially_improving_at_stop=false`.
- The correct read is therefore not “undertrained and obviously still moving in
  the right direction”; the appended coordinate channels converged to a clearly
  worse same-contract solution within the fixed `40`-epoch budget.

## Lineage Context

- `ffno` from the immutable sinogram-input authority remained weak on this
  contract and stays lineage context only:
  `image_relative_l2_phys=0.999968`, `meas_relative_l2=0.999908`,
  `PSNR_phys=19.816599`, `SSIM_phys=0.050775`.
- `classical_born_backprop` remains a non-learned reference only:
  `image_relative_l2_phys=3.468618`, `meas_relative_l2=7.693149`,
  `PSNR_phys=9.013189`, `SSIM_phys=0.624561`.

## Interpretation

- Appending deterministic object-grid `x/y` channels after the sinogram resize
  does not improve the learned BRDT sinogram-input SRU-Net on the locked
  additive-secondary contract.
- The coordgrid row materially regresses every required same-contract image and
  measurement metric versus the unconditioned `sru_net` authority.
- The row is therefore not promotable and should be read only as negative
  representational-diagnostic evidence for this BRDT lane.

## Verification

Verification logs and outputs are owned by:
`.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation/`

- Deterministic code gates:
  - `pytest -q tests/studies/test_born_rytov_dt_adapters.py -k "sinogram or coordgrid or coordinate"`
  - `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "sinogram_input_40ep or coordgrid or coordinate"`
  - `python -m compileall -q scripts/studies/born_rytov_dt`
  - `python -m scripts.studies.born_rytov_dt.run_srunet_coordgrid_extension --dry-run`
- Live run:
  - `python -m scripts.studies.born_rytov_dt.run_srunet_coordgrid_extension`
  - tracked completion proof:
    `run_exit_status.json`
- Bundle audit:
  - `combined_metrics.{json,csv}`
  - `comparison_summary.md`
  - `convergence_audit.{json,csv}`
  - `rows/sru_net_coordgrid/{history.json,history.csv,model_profile.json,row_summary.json,adapter_contract.json}`
  - `visuals/sample_0255_{compare_q,error_q,sinogram_residual}.png`
  - `figures/source_arrays/sample_0255_*`

## Residual Risks

- This remains a single-contract, single-seed append-only BRDT diagnostic; it
  is useful for directionality, not for broader statistical claims.
- The coordgrid row converges to a near-collapsed output dynamic range
  (`physical_q_std ~= 2.12e-06`), so future follow-up work should treat the
  degradation as a real representation/problem-setup issue rather than a small
  metric wobble.
- Because the learned input composition changed while everything else stayed
  fixed, future comparisons must not blur this diagnostic row into the current
  manuscript-facing BRDT sinogram-input authority.
