# NeurIPS Hybrid ResNet CDI Anchor Regeneration Plan

## Scope and Trigger

This note schedules, but does not launch, the fresh `128x128` grid-lines Hybrid ResNet CDI anchor required for the NeurIPS Hybrid ResNet campaign.

Phase 0 inspected local N=128 grid-lines artifacts and did not recover a complete paper-grade `pinn_hybrid_resnet` anchor. The raw gate evidence is recorded in `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_128_hybrid_candidates.json`.

## Lost-Run Evidence

The local inventory found seven N=128 candidate roots. Six contain legacy `pinn_hybrid` or related historical metrics, and one wrapper root has no Hybrid/Hybrid ResNet metric entry. None has the complete paper-grade bundle required by the design: current `pinn_hybrid_resnet` identity, invocation/config, git commit or explicit commit gap, seed, epoch count, scheduler, dataset split, metrics, and qualitative-output provenance.

The usable status for Phase 0 is therefore:

- paper-grade anchor: none recovered
- decision-support artifacts: historical legacy hybrid/FNO context only
- required later action: regenerate the `128x128` Hybrid ResNet anchor in Roadmap Phase 3

## Regeneration Command / Runbook Source

Use the existing grid-lines wrapper path:

- study/runbook index: `docs/studies/index.md`
- runtime scaffold: `tests/torch/test_grid_lines_hybrid_resnet_integration.py`
- parent wrapper: `scripts/studies/grid_lines_compare_wrapper.py`
- Torch child runner: `scripts/studies/grid_lines_torch_runner.py`
- current PyTorch workflow reference: `docs/workflows/pytorch.md`
- recommended baseline reference: `docs/model_baselines.md`

The Phase 3 command and runtime budget should be derived from the study index plus the Hybrid ResNet integration test scaffold before any full anchor launch. The integration test is not paper evidence by itself, but it is the local runtime guidance for dataset generation, runner invocation shape, expected artifact layout, and minimum preflight checks.

Phase 3 should derive the final launch from this template after a runtime preflight:

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 128 \
  --gridsize 1 \
  --output-dir outputs/neurips_hybrid_resnet_n128_anchor_seed3_<timestamp> \
  --architectures hybrid_resnet \
  --set-phi \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nphotons 1e9 \
  --nepochs 50 \
  --batch-size 16 \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --probe-smoothing-sigma 0.5 \
  --seed 3 \
  --torch-epochs 50 \
  --torch-learning-rate 2e-4 \
  --torch-scheduler ReduceLROnPlateau \
  --torch-plateau-factor 0.5 \
  --torch-plateau-patience 2 \
  --torch-plateau-min-lr 1e-4 \
  --torch-plateau-threshold 0.0 \
  --torch-loss-mode mae \
  --torch-output-mode real_imag \
  --fno-modes 12 \
  --fno-width 32 \
  --fno-blocks 4 \
  --fno-cnn-blocks 2 \
  --torch-mae-pred-l2-match-target
```

This is intentionally a wrapper-level command. `grid_lines_compare_wrapper.py` does not expose every child `grid_lines_torch_runner.py` Hybrid ResNet flag. The omitted runner-level Hybrid ResNet fields should be verified in the emitted child invocation/config artifacts, where the current runner defaults match the recommended baseline: `hybrid_skip_connections=off`, `hybrid_downsample_steps=2`, `hybrid_downsample_op=stride_conv`, `hybrid_encoder_conv_hidden_scale=2.0`, `hybrid_encoder_spectral_hidden_scale=1.0`, `hybrid_resnet_blocks=6`, `hybrid_skip_style=add`, and `probe_mask=off`. If Phase 3 needs to override any of those fields explicitly, it must either add wrapper support in a separately reviewed implementation plan or use a direct `grid_lines_torch_runner.py` launch after a separately recorded dataset-generation step.

If Phase 3 changes the epoch budget, probe scaling, dataset size, or any baseline field, the launch manifest must label the change as an intentional Phase 3 override.

## Baseline Configuration

The Phase 3 anchor should inherit the current recommended Hybrid ResNet baseline unless that phase explicitly overrides it before launch:

- architecture: `hybrid_resnet`
- `fno_modes=12`
- `fno_width=32`
- `fno_blocks=4`
- `hybrid_skip_connections=off`
- `hybrid_downsample_steps=2`
- `hybrid_downsample_op=stride_conv`
- `hybrid_encoder_conv_hidden_scale=2.0`
- `hybrid_encoder_spectral_hidden_scale=1.0`
- `hybrid_resnet_blocks=6`
- `hybrid_skip_style=add`
- optimizer: `adam`
- learning rate: `2e-4`
- scheduler: `ReduceLROnPlateau`
- plateau factor: `0.5`
- plateau patience: `2`
- plateau minimum learning rate: `1e-4`
- plateau threshold: `0.0`
- weight decay: `0.0`
- loss mode: `mae`
- `torch_mae_pred_l2_match_target=on`
- `probe_mask=off`

The PyTorch grid-lines runner must preserve the active findings relevant to this run: `POLICY-001`, `CONFIG-001`, `PROBE-MASK-DEFAULT-001`, `GRIDLINES-OBJECT-BIG-001`, `GRIDLINES-PROBE-BIG-001`, `FORWARD-SIG-001`, and `OUTPUT-COMPLEX-001`.

## Dataset and Split Identity

The intended anchor is the grid-lines `N=128`, `gridsize=1`, `set_phi` synthetic-lines condition with the custom Run1084 probe path above. Phase 3 must persist:

- generated train/test dataset paths
- split identity and sample counts
- probe source NPZ path and probe preprocessing flags
- `N`, `gridsize`, `nphotons`, `nimgs_train`, `nimgs_test`, `seed`, and `set_phi`
- any dataset manifest/checksum files emitted by the wrapper

If the wrapper regenerates datasets under the run root, the generated files are part of the paper-grade provenance and must not be deleted before metric review.

## Seed, Config, and Provenance Capture

The Phase 3 run must capture:

- wrapper `invocation.json` and `invocation.sh`
- child runner `runs/pinn_hybrid_resnet/invocation.json` and `invocation.sh`
- git commit SHA and dirty-state note
- Python, PyTorch, Lightning, CUDA, GPU name, and driver
- exact model config and execution config
- training seed and any sampling/randomness contract emitted by the runner
- stdout/stderr log path
- final run root path and file mtimes

Use PATH `python` per `PYTHON-ENV-001`. Because this will be a long run, launch it in tmux with the `ptycho311` environment, track the exact launched PID, and wait on that PID. Do not launch a duplicate run writing to the same `--output-dir`.

## Metric Contract

The regenerated anchor must write and preserve:

- wrapper `metrics.json`
- child `runs/pinn_hybrid_resnet/metrics.json`
- `metrics_table.tex` and `metrics_table_best.tex` when emitted
- amplitude and phase SSIM, MS-SSIM, PSNR, MSE, MAE, and FRC metrics as available
- any metric caveat if a metric is absent or has changed schema

Baseline rows may be rerun in Phase 3 only after their comparator contract is named and tied to the same dataset/split.

## Qualitative Output Plan

The run should preserve the wrapper visual outputs, at minimum:

- `visuals/compare_amp_phase.png` when emitted
- model-specific reconstruction outputs under `runs/pinn_hybrid_resnet/`
- any reconstruction NPZ or stitched amplitude/phase arrays needed to regenerate paper figures

Qualitative outputs must be linked from the Phase 3 CDI summary before any paper-facing figure manifest is assembled.

## Runtime and Resource Budget

Phase 0 did not launch runtime measurement. Phase 3 must run a bounded preflight before the full anchor if runtime is uncertain. The expected compute target is one local RTX 3090 with 24 GB VRAM; the full anchor should use a unique output root and should stop rather than overwrite or merge into an existing root.

If a `50` epoch run exceeds the available deadline or GPU budget, Phase 3 must record the preflight evidence and choose a revised epoch budget before launch, not after inspecting headline metrics.

## Freshness and Output-Root Guardrails

- Use a fresh output root such as `outputs/neurips_hybrid_resnet_n128_anchor_seed3_<timestamp>`.
- Before launch, check that no process is already writing to that root.
- Treat the run as complete only when the tracked PID exits `0` and required metrics, invocation artifacts, and qualitative outputs are freshly written.
- Store bulky logs and derived scratch outside tracked docs or under ignored artifact roots.

## Later-Phase Boundary

Phase 0 does not launch regeneration, compact baselines, ablations, N=256 variants, PDE runs, or `/home/ollie/Documents/neurips/` artifact assembly. Those actions belong to later roadmap phases after the Phase 0 inventory and Phase 1 PDE screen gates.

## Verification

For this note, Phase 0 verification is structural:

- the lost-run trigger is explicit
- the command/runbook source is named
- the wrapper-level command parses with `grid_lines_compare_wrapper.py`; runner-only Hybrid ResNet fields are documented as defaults to verify from child invocation/config artifacts
- seed/config/provenance capture is specified
- metric and qualitative output contracts are specified
- runtime and output-root guardrails are specified
- no run is launched in Phase 0
