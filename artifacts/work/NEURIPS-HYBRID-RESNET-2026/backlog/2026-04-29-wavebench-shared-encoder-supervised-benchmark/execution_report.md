# WaveBench Shared-Encoder Supervised Benchmark Execution Report

## Completed In This Pass

- Implementation-review H1 fix: replaced the placeholder
  `FourierLayer2d` (which was two 1×1 `Conv2d` layers and silently ignored
  the `modes` argument) with a real FNO 2D block that uses
  `_FallbackSpectralConv2d` from `ptycho_torch/generators/fno.py`. The new
  block performs `torch.fft.rfft2` / `torch.fft.irfft2` spectral convolution
  with `modes=12`, paired with a 1×1 bypass and GELU. The `fno` body now
  honors the locked design recipe (`fno_modes=12, fno_width=32,
  fno_blocks=4`) and the row-roster label.
- M1 follow-up: added `test_fno_body_uses_real_spectral_convolution`
  in `tests/studies/test_wavebench_shared_encoder_models.py` that asserts
  the `fno` body contains `_FallbackSpectralConv2d` modules with complex
  spectral weights at `modes=12` and that `body_parameters >= 100_000`,
  preventing regression to the pointwise-only placeholder.
- M2 follow-up: the FFCV train-split `Loader` is now constructed with
  `seed=int(contract["split"]["seed"])` (42) so `OrderOption.RANDOM`
  sampling is byte-reproducible across reruns and follow-on items, not
  just deterministic via `torch.manual_seed`.
- Re-ran the locked-recipe `fno` row end-to-end on the locked
  `9000 / 500 / 500` split for both `C=32` and `C=64` under a tracked
  background wrapper PID. Both passes exited `0` and the
  `fno_rerun_complete.flag` artifact was written. New per-row metrics:
  - `fno c32`: encoder=73,408 / body=597,313 / total=670,721 params,
    runtime 560.2s, peak memory 1,737,167,872 B,
    train_loss 0.033474, val_loss 0.033390,
    test MAE 0.032825, RMSE 0.145200, RelL2 1.000020, SSIM 2.940e-06.
  - `fno c64`: encoder=75,488 / body=598,337 / total=673,825 params,
    runtime 579.7s, peak memory 1,871,422,464 B,
    train_loss 0.033481, val_loss 0.033385,
    test MAE 0.032820, RMSE 0.145195, RelL2 0.999983, SSIM 6.063e-06.
- Refreshed the bundle: `comparison_summary.{json,csv}`,
  `table_ready_metrics.json`, `shared_encoder_execution_manifest.json`,
  `rows/fno/c{32,64}/{metrics,model_profile}.json`, and the per-row
  `figures/c{32,64}/fno/` PNGs / source arrays / figure manifests now
  reflect the corrected FNO body. The other rows (`cnn`,
  `hybrid_resnet`, `spectral_resnet_bottleneck_net`, `ffno`) were
  intentionally left untouched; their artifacts and metrics are
  unchanged.
- Refreshed the durable summary
  (`docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_shared_encoder_supervised_summary.md`):
  the metrics table for the `fno` rows, and a new "body-label semantics
  actually used in this bundle" subsection that documents (a) the real
  FNO body recipe, (b) the FFNO body's two trailing local residual
  refiners (review L1) so the `ffno` label cannot be confused with the
  upstream FFNO no-refiner contract reactivated elsewhere in the
  initiative.
- Refreshed `model_variant_index.json`: both fno entries now record the
  corrected `body_recipe`, `body_parameters`, `parameter_count`,
  `metrics`, and `runtime`. The interpretation paragraph is unchanged
  because the recipe-driven collapse persists with a real FNO body
  (review L2 resolved without rewriting the negative-result framing).
- Appended a new selector-facing entry to
  `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
  recording the implementation-review fix, the new test count
  (12 passed), the rerun verification commands, the corrected FNO body
  parameter counts, and the FFCV reproducibility change.
- Verified all gates after the fix:
  - implementation-gate pytest: `pytest -q
    tests/studies/test_wavebench_shared_encoder_{data,models,runner,contract}.py`
    -> 12 passed (was 11 before the structural FNO test was added).
  - final-bundle gate:
    `python scripts/studies/validate_wavebench_shared_encoder_contract.py
    --require-benchmark-completion` -> `wavebench shared-encoder
    contract validated`.

## Earlier Pass — Original Bundle Build



- Tightened the row-status vocabulary so `completed` is reserved for
  benchmark-mode rows on the locked split, while smoke runs now record
  `status="smoke_pass"`. `run_row()` derives status from mode via
  `status_for_mode()`, the validator enforces the `mode <-> status`
  pairing, and `comparison_summary.csv` now records `mode` alongside
  `status` so smoke and benchmark bundles cannot be conflated.
- Added benchmark-mode fixed-sample figure generation:
  `scripts/studies/wavebench_shared_encoder/reporting.py::write_row_figures`
  emits per-row PNG triptychs (`target / prediction / error`) under shared
  color limits plus compressed `.npz` source arrays plus a
  `figure_manifest.json`. `run_row()` invokes this only in
  `mode == "benchmark"`. Generated 40 PNGs and 10 manifests under
  `figures/c{32,64}/<row>/`.
- Tightened `scripts/studies/validate_wavebench_shared_encoder_contract.py`:
  added a `--require-benchmark-completion` gate that requires every locked
  row + latent-width pair to have an entry whose status is in
  `{completed, blocked, not_protocol_compatible}`, and that
  `mode=benchmark` for `completed` and `mode=smoke` for `smoke_pass`.
- Updated tests:
  `tests/studies/test_wavebench_shared_encoder_runner.py` now covers both
  the smoke-pass and benchmark-completed paths and asserts that benchmark
  mode writes the figure manifest and per-sample source arrays.
  `tests/studies/test_wavebench_shared_encoder_contract.py` adds the
  `mode=benchmark` field to the validator fixture so the tightened
  validator passes against the consistent bundle.
- Cleaned the previously committed smoke-only artifact bundle that had
  combined `status=completed` with `mode=smoke` (the bundle the review
  flagged as inconsistent under the new vocabulary). Re-emitted under
  benchmark mode by the actual long run.
- Executed the locked benchmark recipe end-to-end on real WaveBench data:
  - dataset member: `tmp/wavebench_repo/wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton`
  - locked split: seed-42 `9000 / 500 / 500`
  - row roster: `cnn`, `hybrid_resnet`, `spectral_resnet_bottleneck_net`,
    `fno`, `ffno`
  - latent widths: `C=32` then `C=64`
  - recipe: `L1`, `Adam(lr=2e-4)`,
    `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, threshold=0.0)`,
    seed `42`, `train_batch_size=32`, `eval_batch_size=64`,
    `epochs=50`, `num_workers=2`.
  - run shell: tmux session `wavebench_shared_encoder_20260508_205812`
    chained the `--latent-channels 32` then `--latent-channels 64`
    invocations and wrote per-pass logs under
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/run_logs/{c32,c64}.log`,
    plus `run_complete.flag` after both runs exited `0`.
- Persisted the full benchmark bundle:
  - `row_contract.json`,
    `shared_encoder_execution_manifest.json`,
    `table_ready_metrics.json`,
    `comparison_summary.json`,
    `comparison_summary.csv`,
    `run_roots.json`,
  - per-row `rows/<row>/c{32,64}/{metrics.json,model_profile.json}` for
    all 10 configurations,
  - per-row `figures/c{32,64}/<row>/{sample_000..003.png, figure_manifest.json}`
    plus `figures/source_arrays/<row>/c{32,64}/sample_000..003.npz`.
- Wrote the durable summary at
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_shared_encoder_supervised_summary.md`
  with the locked contract, the as-run metric table, the trivial-prediction
  collapse interpretation, and the candidate-lane claim boundary.
- Closed the WaveBench discoverability gaps:
  - `docs/index.md` now links the shared-encoder summary alongside the
    earlier WaveBench preflight / provisioning / native-baseline summaries.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md` adds both a
    backlog-coverage row and a current-evidence row for the shared-encoder
    benchmark.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json` adds
    the `wavebench_is_gaussian_lens_shared_encoder_9000_500_500` dataset
    contract and 10 `model_variants` entries (one per row × latent width)
    with the as-run metrics, parameter counts, runtime, peak memory, and
    final L1 train/val losses.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json` adds the
    `wavebench_shared_encoder_body_architecture_comparison` and
    `wavebench_shared_encoder_latent_width_sensitivity` ablation families.
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` appends a
    selector-facing `post_completion_updates` entry that records the
    completion state, the locked recipe, the row-status summary, the
    artifact roots, the verification commands, and the candidate-lane
    claim boundary.
- Final benchmark-completion gate passed:
  `python scripts/studies/validate_wavebench_shared_encoder_contract.py
  --require-benchmark-completion -> wavebench shared-encoder contract validated`.

## Completed Current-Scope Work

- **Plan Task 1** complete: input-presence check passed; preflight,
  provisioning, and native-baseline validators re-run and passing; the
  `row_contract.json` is generated from the consumed WaveBench summaries
  and locks the row roster, training recipe, both latent widths, and the
  explicit row-status vocabulary `{completed, smoke_pass, blocked,
  not_protocol_compatible}`.
- **Plan Task 2** complete: deterministic loader for the staged
  `wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton`
  member, locked split `9000 / 500 / 500` with seed `42`, archived
  tensor contract preserved end-to-end, and the shared anisotropic
  measurement encoder is identical across every row and both latent widths.
  Encoder/body parameter accounting is exposed via `profile_model()`.
- **Plan Task 3** complete: the runner builds all five locked-roster rows
  for both `C=32` and `C=64`, accepts the shared latent tensor, emits
  `(1, 128, 128)` predictions, and the runner writes per-row metrics,
  model profiles, the shared manifest, and the bundle CSV/JSON. The
  benchmark-mode path now writes figures and source arrays. All 11
  shared-encoder pytest cases pass.
- **Plan Task 4** complete: full locked-split `9000 / 500 / 500`
  benchmark rows for `C=32` and the `C=64` sensitivity pass executed
  end-to-end under tracked tmux ownership (`run_complete.flag` written
  after both passes exited `0`). All 10 configurations recorded
  `status="completed"` with `mode="benchmark"`. No row required a
  row-level blocker; every row reached the locked epoch budget under the
  locked recipe. Results are honestly negative: under the locked recipe,
  every architecture collapsed to a near-zero trivial-prediction
  baseline (test `RelL2 ≈ 1.0`, `SSIM ≈ 0`, train/val L1
  `≈ 0.0335 / 0.0334`). The fixed-sample figures confirm the collapse
  visually. The plan explicitly forbids loosening the recipe to make a
  row easier, so the recipe lock is preserved and the negative outcome
  is reported as the as-run benchmark result.
- **Plan Task 5** complete: durable summary published; `docs/index.md`,
  `evidence_matrix.md`, `model_variant_index.json`, `ablation_index.json`,
  and `progress_ledger.json` updated; `paper_evidence_index.md` and
  `/home/ollie/Documents/neurips/` were not touched (out of scope per the
  plan); `docs/findings.md` not modified (this work did not surface a
  reusable project-wide contract beyond this single WaveBench lane).

## Follow-Up Work

- Recipe-relaxation candidate-lane follow-on: under the locked
  `L1 + Adam 2e-4 + ReduceLROnPlateau(threshold=0.0, patience=2,
  min_lr=1e-5)` recipe every row collapsed to trivial prediction. A
  later candidate-lane item could authorize a recipe change (e.g.,
  looser plateau threshold, different loss, normalization, longer
  warmup, or a different LR) and rerun under the same row roster to see
  whether the body family becomes separable. This is intentionally not
  done here because the plan's fairness contract forbids loosening the
  recipe to make a row easier within this item.
- Alternative shared-encoder follow-on: this item locks one fixed simple
  anisotropic measurement encoder. A later item could swap that encoder
  while keeping the same body roster and split, to test whether the
  trivial-prediction collapse is encoder-driven rather than recipe-driven.
- Multi-seed sensitivity: this item ran a single seed (`42`). A later
  candidate-lane item could repeat the locked-roster benchmark with
  multiple seeds if a downstream interpretation requires it.
- The shared-encoder bodies remain repo-owned comparison wrappers and are
  intentionally separate from native WaveBench architectures. If a later
  item wants to reuse the shared encoder against true upstream WaveBench
  bodies, it should be planned as a distinct candidate-lane work unit
  with its own row contract.

## Residual Risks

- All 10 rows under the locked recipe collapsed to a near-zero
  trivial-prediction baseline. Under this contract the body family is not
  separable; reading any row as evidence that a body is competitive or
  non-competitive on WaveBench inverse source would be a misuse of this
  bundle. The bundle preserves the fairness contract correctly, but the
  discriminating signal is absent.
- The negative result is consistent with the locked plateau scheduler
  (`threshold=0.0`, `patience=2`, `min_lr=1e-5`) reaching the LR floor
  early after non-monotonic validation noise, but the plan forbids
  loosening the recipe to test that hypothesis inside this item. The
  recipe-versus-encoder attribution is therefore unresolved and is
  flagged as a follow-up.
- The bundle remains additive candidate-lane evidence only. It must not
  be promoted to manuscript evidence and must not be mixed into the
  native WaveBench reference rows (which use a different input contract
  and different upstream training stack).
