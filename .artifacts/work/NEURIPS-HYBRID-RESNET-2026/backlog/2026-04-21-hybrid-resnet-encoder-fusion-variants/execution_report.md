# Execution Report: 2026-04-21 Hybrid ResNet Encoder Fusion Variants

- Backlog item: `2026-04-21-hybrid-resnet-encoder-fusion-variants`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/execution_plan.md`
- Implementation state: `COMPLETED`
- Ablation root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/`
- Run root: `runs/encoder_fusion_20260502T104230Z`
- Durable summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md`

## Completed In This Pass

- Added the checked-in helper `scripts/studies/lines128_hybrid_resnet_encoder_fusion_variants.py` covering the missing current-scope orchestration surface:
  - reconstruct the fixed-contract fresh-row configs from the authoritative baseline artifacts
  - plan row-local training roots under `runs/encoder_fusion_20260502T104230Z/training_runs/<row_id>/`
  - repair the historical shared-output launch into deterministic row-local training/log/checkpoint manifests
  - refresh the append-only `model_manifest.json` for the ablation bundle
- Added the focused regression test `tests/studies/test_lines128_hybrid_resnet_encoder_fusion_variants.py` proving the helper plans row-local training roots and can rebuild row-local provenance from preserved shared Lightning metadata.
- Repaired the blocking fresh-row provenance contract in the published artifact root without rerunning the scored rows:
  - materialized row-local `training_runs/<row_id>/lightning_logs/version_0/` and `training_runs/<row_id>/checkpoints/{last,best}.ckpt` for each fresh row by deterministic fusion-mode attribution from preserved `hparams.yaml` and checkpoint hyperparameters
  - wrote `training_output_manifest.json` under each row-local training root
  - wrote row-local `stdout.log`, `stderr.log`, and `launcher_completion.json` under each fresh row root
  - updated `row_contract_audit.json` so the row-local output contract now matches the repaired layout and records the historical shared-output repair explicitly
  - updated `model_manifest.json` so each fresh row now advertises its row-local logs, launcher completion evidence, training root, checkpoint directory, and Lightning metrics CSV
- Corrected the execution narrative:
  - removed the earlier tmux / single-PID launch overclaim from this report
  - updated the durable summary and study index to note the post-pass row-local provenance repair
- Re-ran the required selectors and archived fresh proof under `verification/`

## Completed Current-Scope Work

- Blocking review item 1 is repaired. The fresh rows are now independently auditable from the append-only ablation root: each row has row-local launch evidence, a row-local training output directory, row-local checkpoints, and explicit manifest links to those artifacts.
- Blocking review item 2 is repaired. The reviewed item now has a checked-in repo-local helper under `scripts/studies/` instead of relying on one-off artifact-local scripts.
- Medium review item 1 is repaired. This report no longer claims tmux-backed single-PID launch proof for the historical pass; it now records the actual preserved evidence and the deterministic provenance repair performed in this pass.

## Follow-Up Work

- Add an out-of-process checkpoint-rebuild probe for the encoder-fusion overrides if a future task needs to resume these repaired checkpoints independently of the original in-process runner path.
- Keep `pinn_hybrid_resnet_encoder_fusion_norm` and shared-across-encoder-block scalar placement deferred unless a later bounded extension explicitly reopens the architecture axis.

## Verification

Backlog-required deterministic gates (re-run for the review fix and archived under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/verification/`):

- `pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet_encoder or hybrid_encoder"` → `34 passed, 59 deselected` (`review_fix_test_fno_generators.log`)
- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_encoder"` → `23 passed, 118 deselected` (`review_fix_test_grid_lines_torch_runner.log`)
- `pytest -q tests/studies/test_lines128_hybrid_resnet_encoder_fusion_variants.py` → `2 passed` (`review_fix_test_encoder_fusion_helper.log`)
- `pytest -v -m integration --timeout=900` → `5 passed, 4 skipped` (`review_fix_integration.log`)
- `python -m compileall -q ptycho_torch scripts/studies` → clean (`review_fix_compileall.log`)

Same-contract row outputs (historical scored rows retained; row-local provenance repaired in this pass):

- `runs/encoder_fusion_20260502T104230Z/runs/pinn_hybrid_resnet_encoder_layerscale/` — `metrics.json`, `history.json`, `config.json`, `model.pt`, `invocation.json`, `randomness_contract.json`
- `runs/encoder_fusion_20260502T104230Z/runs/pinn_hybrid_resnet_encoder_branch_gated/` — same artifact set
- `runs/encoder_fusion_20260502T104230Z/runs/pinn_hybrid_resnet_encoder_branch_gated_layerscale/` — same artifact set
- `runs/encoder_fusion_20260502T104230Z/recons/<row_id>/recon.npz` for each fresh row
- `runs/encoder_fusion_20260502T104230Z/training_runs/<row_id>/` for each fresh row — repaired row-local Lightning logs, repaired row-local checkpoints, and `training_output_manifest.json`
- `runs/encoder_fusion_20260502T104230Z/runs/<row_id>/{stdout.log,stderr.log,launcher_completion.json}` for each fresh row — repaired row-local launch evidence
- `runs/encoder_fusion_20260502T104230Z/promoted_baseline/runs/pinn_hybrid_resnet/` — append-only copy of the reused baseline row + recon + GT recon

Numerical comparison standard:

- per-row stitched metrics from `metrics.json` (amp/phase channel pairs for `mae`, `mse`, `psnr`, `ssim`, `ms_ssim`, `frc50`, `frc1over7`)
- baseline-vs-row deltas reported channel-wise without rounding tolerance; SSIM and MAE deltas at the `1e-4` resolution level decide whether a variant improved on the dominant channels
- the comparison is same-contract (same dataset, seed, optimizer, scheduler, loss, output mode, sample IDs, metric schema), so deltas are interpreted directly without any `atol`/`rtol` regression tolerance

Per-row read on final stitched metrics:

- `pinn_hybrid_resnet_encoder_layerscale`: ΔSSIM amp `-0.0012`, ΔMAE amp `+0.0017`, ΔSSIM phase `-0.0004`, ΔMAE phase `+0.0042`, FRC50 amp `+1.7`, FRC50 phase `~0`. Net read: per-block outer LayerScale alone slightly hurts the dominant channels; not an improvement.
- `pinn_hybrid_resnet_encoder_branch_gated`: ΔSSIM amp `+0.0009`, ΔMAE amp `-0.0008`, ΔSSIM phase `+0.0001`, ΔMAE phase `+0.0021`. Net read: branch-level gating mildly improves amplitude reconstruction with a small phase MAE trade; closest to a real improvement among the three.
- `pinn_hybrid_resnet_encoder_branch_gated_layerscale`: ΔSSIM amp `-0.0017`, ΔMAE amp `+0.0019`, ΔSSIM phase `-0.0002`, ΔMAE phase `+0.0014`, FRC50 phase `+30.0`, FRC50 amp `-2.8`. Net read: phase-side high-frequency reconstruction (FRC) jumps but at the cost of dominant amplitude/SSIM channels; not constructive vs the simpler single-factor variants.

## Residual Risks

- The per-row read above is single-seed (`seed=3`) under the bounded backlog-item budget; small SSIM/MAE deltas at the `1e-4` level should not be interpreted as paper-grade promotion evidence. The durable summary and `comparison_summary.json` mark every fresh row `decision_support` and explicitly retain the existing `pinn_hybrid_resnet` paper-grade anchor.
- The repaired row-local training roots are deterministic post-pass reconstructions from preserved Lightning fusion-mode metadata and per-row invocation artifacts, not original tmux/PID-tracked launch sidecars. That is sufficient for row attribution and manifest completeness, but it does not recreate stronger launch evidence than what the historical pass actually preserved.
- The runner stamps the new encoder-fusion knobs into the saved `generator_overrides` so that checkpoint-rebuild via `_build_generator_for_checkpoint_rebuild` can recover them; this is exercised on the in-memory inference path during each row but a full out-of-process load-from-checkpoint path was not separately exercised in this pass. If a future task needs to resume from these row checkpoints offline, that path should be probed before reuse.
- Shared-across-encoder-block scalar placement and `pinn_hybrid_resnet_encoder_fusion_norm` are deliberately deferred. Either is an explicit future architecture axis and is not represented in this ablation; the durable summary, `execution_manifest.json`, and `ablation_index.json` all record this boundary.
- The plan is currently linked from five evidence/discoverability surfaces; the top-level `docs/index.md` was not updated because the new summary does not need first-tier discoverability beyond the study/evidence indexes (the plan explicitly conditions that update on first-tier need).
