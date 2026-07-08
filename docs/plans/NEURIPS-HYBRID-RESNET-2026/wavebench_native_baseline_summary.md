# NeurIPS WaveBench Native Baseline Summary

## Decision

- Selected variant: `time_varying/is/thick_lines_gaussian_lens`
- Locked split: seed-42 `9000 / 500 / 500` (`train / val / test`)
- Stable dataset target: `<wavebench repo>/wavebench_dataset/time_varying/is/`
- Observed checkout path: `tmp/wavebench_repo` at revision
  `2bea258d9f05ec7182741293be11be1e545576ae`
- Environment actually used: PATH `python`
  `/home/ollie/miniconda3/envs/ptycho311/bin/python` (`Python 3.11.13`)
  after adding WaveBench supervised-loader requirements
  (`opencv`, `pkg-config`, `libjpeg-turbo`, `ffcv`, `ml-collections`) and
  restoring `numpy==1.26.4`
- Claim boundary: candidate-lane external references only. These native
  WaveBench rows remain separate from later shared-encoder rows and do not
  promote WaveBench into manuscript evidence.

This pass completed the native reference-baseline question for the selected
WaveBench inverse-source lane. The representative official U-Net checkpoint was
evaluated on the full locked test split through the native loader/input
contract, and the representative native FNO row was reproduced via the
official upstream retraining route because the public checkpoint remained
incompatible with the current upstream code.

## Native Rows

| Row | Route | Status | Test samples | MAE | RMSE | RelL2 | SSIM | Params | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `wavebench_unet_ch32_native` | reusable official checkpoint `17FXa31RSMA-7vwRB_492Ex8AY-2YgSdl` | `completed` | `500` | `0.014410` | `0.068332` | `0.446032` | `0.881951` | `7,762,465` | native checkpoint-eval route |
| `wavebench_fno_depth4_native` | official retraining route (`train_fno_is.py --medium_type gaussian_lens --num_layers 4`) | `completed` | `500` | `0.027049` | `0.082066` | `0.550314` | `0.726870` | `16,811,073` | best checkpoint after the full `50`-epoch rerun: `epoch=8-step=2529.ckpt` |

Observed read:

- On the locked native contract, the reusable WaveBench U-Net row is stronger
  than the reproduced native FNO row on all four packaged metrics.
- The native FNO route is now durable despite the public checkpoint/code
  mismatch, but the best checkpoint from the official rerun arrived early in
  training (`epoch=8-step=2529`) and later epochs did not improve the monitored
  validation metric enough to replace it.
- These rows answer the external-reference question only: “what do the native
  WaveBench baselines do on the selected inverse-source lane?” They are not a
  fair repo-local architecture comparison against later shared-encoder rows.

## Durable Outputs

- Summary authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_native_baseline_summary.md`
- Native artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-native-baseline-reproduction/`
- Machine-readable manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-native-baseline-reproduction/native_baseline_execution_manifest.json`
- Table-ready metric bundle:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-native-baseline-reproduction/table_ready_metrics.json`
- Native U-Net row:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-native-baseline-reproduction/native_unet_eval.json`
- Native FNO row:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-native-baseline-reproduction/native_fno_result.json`
- Run-root pointers:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-native-baseline-reproduction/run_roots.json`
- CSV projection:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-native-baseline-reproduction/wavebench_native_rows.csv`

## Verification

- Input and prior-contract checks:
  `python scripts/studies/validate_wavebench_preflight_contract.py`
  and
  `python scripts/studies/validate_wavebench_provisioning_decision.py`
- Prior contract tests:
  `pytest -q tests/studies/test_wavebench_preflight_contract.py tests/studies/test_wavebench_provisioning_decision_contract.py`
- U-Net smoke:
  `python scripts/studies/run_wavebench_native_baselines.py --row unet --mode smoke --max-test-samples 4 --batch-size 2 --num-workers 0`
- U-Net full evaluation:
  `python scripts/studies/run_wavebench_native_baselines.py --row unet --mode checkpoint_eval --batch-size 32 --num-workers 0`
- FNO smoke:
  `python scripts/studies/run_wavebench_native_baselines.py --row fno --mode smoke --max-test-samples 4 --batch-size 2 --num-workers 0`
- FNO full rerun/eval:
  `python scripts/studies/run_wavebench_native_baselines.py --row fno --mode train_and_eval --batch-size 32 --num-workers 0 --train-num-epochs 50`
  under tracked tmux PID ownership; wrapper exit code `0`
- Final bundle validation:
  `python scripts/studies/validate_wavebench_native_baseline_contract.py`
  and
  `pytest -q tests/studies/test_wavebench_native_baseline_contract.py`

## Residual Risks

- The active runnable environment is a repaired `ptycho311` env rather than a
  fresh standalone `wavebench` env. The executed package surface is durable in
  the artifacts, but later WaveBench items may still prefer a cleaner dedicated
  environment.
- The public native FNO checkpoint remains incompatible with the current
  upstream code; this item intentionally preserved the official retraining path
  instead of patching checkpoint tensors or mutating upstream FNO code.
- WaveBench physics validation, shared-encoder rows, and any manuscript-facing
  promotion remain out of scope. This summary is candidate-lane context only.
