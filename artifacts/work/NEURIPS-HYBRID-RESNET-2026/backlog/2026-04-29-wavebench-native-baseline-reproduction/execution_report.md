## Completed In This Pass

- Added the repo-owned native WaveBench baseline wrapper
  `scripts/studies/run_wavebench_native_baselines.py`, the contract validator
  `scripts/studies/validate_wavebench_native_baseline_contract.py`, and the
  regression test `tests/studies/test_wavebench_native_baseline_contract.py`.
- Re-ran the required input/contract checks for the completed WaveBench
  preflight and provisioning decision before new execution work began.
- Repaired the active `ptycho311` runtime narrowly enough to execute supervised
  WaveBench rows:
  added `opencv`, `pkg-config`, and `libjpeg-turbo` via conda; added `ffcv`
  and `ml-collections` via pip; restored `numpy==1.26.4` after the initial
  `ffcv` install upgraded NumPy past the repo-compatible TensorFlow bound.
- Ran the native U-Net smoke and then the full locked-test evaluation through
  the reusable official checkpoint route, producing
  `native_unet_eval.json` on the full `500`-sample test split.
- Discovered and fixed an eval-contract bug where FFCV dropped the remainder
  batch when the eval batch size did not divide `500`; the wrapper now chooses
  a divisor batch size for full locked-split evaluation.
- Ran the native FNO smoke, then launched the full official retraining/eval
  route under tracked tmux PID ownership. The wrapper exited `0` and produced
  `native_fno_result.json` on the full `500`-sample test split.
- Published the durable WaveBench summary, docs-index entry, evidence-matrix
  updates, model-variant index entries, and machine-readable metric/manifest
  bundle for later candidate-lane work.

## Completed Plan Tasks

- Task 1 complete: authoritative inputs were revalidated, the locked WaveBench
  contract was loaded into a repo-owned wrapper, and the execution manifest /
  metric bundle surfaces were established.
- Task 2 complete: the supervised-loader runtime was made runnable, a real
  native U-Net smoke passed, and the representative native U-Net checkpoint was
  evaluated on the full locked `500`-sample test split.
- Task 3 complete: the representative native FNO pre-run smoke passed, the
  official `train_fno_is.py --medium_type gaussian_lens --num_layers 4` route
  completed under tracked ownership, and the full locked-test evaluation row
  was written durably.
- Task 4 complete: the concise native-baseline summary, docs/index update,
  evidence-matrix update, model-variant-index additions, validator, and tests
  were finished against the actual produced bundle.

## Remaining Required Plan Tasks

- None. The scoped native-baseline reproduction item is complete.

## Verification

- Required input checks passed:
  `python - <<'PY' ...`
  `python scripts/studies/validate_wavebench_preflight_contract.py`
  `python scripts/studies/validate_wavebench_provisioning_decision.py`
- Prior contract tests passed:
  `pytest -q tests/studies/test_wavebench_preflight_contract.py tests/studies/test_wavebench_provisioning_decision_contract.py`
  with `6 passed in 0.99s`.
- Native bundle validator passed:
  `python scripts/studies/validate_wavebench_native_baseline_contract.py`
- Native bundle test passed:
  `pytest -q tests/studies/test_wavebench_native_baseline_contract.py`
  with `2 passed in 0.97s`.
- JSON structure checks passed for
  `native_baseline_execution_manifest.json` and `table_ready_metrics.json`
  via `python -m json.tool`.
- U-Net full-row metrics on the locked `500`-sample test split:
  `MAE=0.01440997221507132`, `RMSE=0.06833207067102194`,
  `RelL2=0.44603234976530076`, `SSIM=0.8819505550504577`.
- FNO full-row metrics on the locked `500`-sample test split:
  `MAE=0.027049200143665075`, `RMSE=0.08206634037196636`,
  `RelL2=0.5503136277198791`, `SSIM=0.7268695075416097`.
- Long-run ownership proof:
  full FNO wrapper launched in tmux under tracked shell PID `3531756`;
  wrapper reported `__EXIT_CODE__=0`; the produced best checkpoint was
  `tmp/wavebench_repo/saved_models/is_gaussian_lens/yk51w0ou/checkpoints/epoch=8-step=2529.ckpt`.

## Residual Risks

- The executed runtime is a repaired `ptycho311` environment, not a fresh
  standalone `wavebench` env. The exact package surface is now durable in the
  summary and artifacts, but later WaveBench items may still prefer a cleaner
  dedicated environment for reproducibility.
- The public native FNO checkpoint remains incompatible with the current
  upstream code, so future exact-checkpoint reuse is still unavailable unless a
  loadable older upstream surface is recovered. This item intentionally used
  the official retraining route instead.
- WaveBench remains an additive candidate lane only. These native rows are
  external-reference context, not shared-encoder fairness evidence and not a
  manuscript-evidence promotion.
