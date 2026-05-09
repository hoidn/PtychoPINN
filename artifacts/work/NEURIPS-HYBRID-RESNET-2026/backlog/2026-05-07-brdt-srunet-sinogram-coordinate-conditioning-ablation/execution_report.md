# Execution Report

## Completed In This Pass

- Implemented the append-only BRDT coordgrid lane in code:
  `scripts/studies/born_rytov_dt/run_srunet_coordgrid_extension.py`,
  coordgrid-aware sinogram adapter support in
  `scripts/studies/born_rytov_dt/models.py`,
  explicit row config in `run_config.py`, runner wiring in `run_preflight.py`,
  and required comparison/audit surfacing in `convergence.py` and
  `preflight_visuals.py`.
- Added targeted regression coverage for the coordgrid adapter and runner in
  `tests/studies/test_born_rytov_dt_adapters.py` and
  `tests/studies/test_born_rytov_dt_preflight.py`.
- Executed the dedicated 40-epoch coordgrid run under tmux with exact PID
  tracking, validated the output bundle, and refreshed the derived summary
  surfaces in the new artifact root.
- Published the durable summary and refreshed the BRDT discoverability surfaces
  in `evidence_matrix.md`, `model_variant_index.json`,
  `ablation_index.json`, and `paper_evidence_index.md`.

## Completed Plan Tasks

- Task 1: prerequisite authority presence and immutable-lineage audit passed.
- Task 2: coordinate-conditioned sinogram adapter implemented with deterministic
  normalized object-grid `x/y` channels appended after bilinear resize; saved
  contract surfaces record `input_mode="sinogram"`,
  `sinogram_to_grid="bilinear_resize"`, `coordinate_channels="object_xy"`, and
  `in_channels=4`.
- Task 3: dedicated append-only coordgrid runner implemented, tested, compiled,
  and dry-run validated.
- Task 4: live 40-epoch `sru_net_coordgrid` run completed successfully; tracked
  PID `3468206`, `exit_code=0`; required bundle artifacts present.
- Task 5: durable summary and discoverability surfaces published. `docs/index.md`
  was intentionally left unchanged because this item remains discoverable
  through `evidence_matrix.md` and `paper_evidence_index.md`, matching the plan
  allowance.

## Remaining Required Plan Tasks

- None.

## Verification

- `python -c "from pathlib import Path; ... print('brdt sinogram-input authority present')"`:
  passed.
- Immutable lineage audit for baseline row summaries and sample-255 arrays:
  passed.
- `pytest -q tests/studies/test_born_rytov_dt_adapters.py -k "sinogram or coordgrid or coordinate"`:
  passed (`12 passed, 33 deselected`).
- `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "sinogram_input_40ep or coordgrid or coordinate"`:
  passed (`6 passed, 97 deselected`).
- `python -m compileall -q scripts/studies/born_rytov_dt`:
  passed.
- `python -m scripts.studies.born_rytov_dt.run_srunet_coordgrid_extension --dry-run`:
  passed.
- Live run:
  `python -m scripts.studies.born_rytov_dt.run_srunet_coordgrid_extension`
  under `ptycho311` tmux session; `run_exit_status.json` records
  `tracked_pid=3468206`, `exit_code=0`.
- Post-run artifact audit:
  passed.
- Discoverability audit from Task 5:
  passed (`coordgrid discoverability surfaces updated`).
- `pytest -q tests/studies/test_paper_evidence_audit.py -k brdt`:
  passed (`5 passed, 17 deselected`).

Key same-contract result versus the unconditioned sinogram-input `sru_net`
authority:

- `image_relative_l2_phys`: `0.7165268569981865 -> 0.9999378558677322`
  (`+0.2834109988695457`)
- `meas_relative_l2`: `0.46655587527377046 -> 0.9997754813674038`
  (`+0.5332196060936334`)
- `psnr_phys`: `22.711670060252654 -> 19.816859321063482`
  (`-2.8948107391891718`)
- `ssim_phys`: `0.7359318624719449 -> 0.026205697384189968`
  (`-0.7097261650877549`)
- `parameter_count`: `142162 -> 142450`
- `eval_samples_per_second`: `404.69782004336173 -> 378.8692637001242`
- history-backed final train loss: `0.30244183412287384 -> 0.45339918648824096`
- history-backed best train loss: `0.30244183412287384 -> 0.45339433546178043`
- Comparison standard for inherited contract fields: exact equality.

## Residual Risks

- The coordgrid row is materially worse than the unconditioned sinogram-input
  SRU-Net on every required same-contract image and measurement metric, so this
  stays negative diagnostic evidence only.
- `materially_improving_at_stop=false`, which argues against a simple
  “needs more epochs” explanation within the fixed 40-epoch contract.
- The row collapses to a very narrow physical output dynamic range
  (`physical_q_std ~= 2.12e-06`), so follow-up work should treat this as a real
  representation/setup failure mode.
- This remains a single-seed append-only BRDT diagnostic and must not be
  promoted into manuscript-facing authority over the current sinogram-input
  `sru_net` bundle.
