## Completed In This Pass

- fixed the blocking implementation-review checkpoint defect in
  `ptycho_torch/model.py` by rebuilding generator-backed Lightning modules
  from saved config state during `load_from_checkpoint()`, including the
  supervised FFNO path that previously fell back to the legacy CNN
  `Autoencoder`
- extended the checkpoint persistence contract tests in
  `tests/torch/test_lightning_checkpoint.py` with:
  - a supervised FFNO checkpoint round-trip that asserts the restored module
    rebuilds `FfnoGeneratorModule`
  - broader generator-backed checkpoint round-trips for `ffno`, `fno`,
    `hybrid`, `stable_hybrid`, `fno_vanilla`, `hybrid_resnet`, and
    `spectral_resnet_bottleneck_net`
- verified the real reviewed artifact root and found the narrow remaining
  artifact defect: the canonical
  `.../supervised_ffno_extension_20260430T180217Z/checkpoints/last.ckpt`
  still pointed at the stale pre-fix checkpoint while the rerun had already
  emitted a loadable `last-v1.ckpt`
- refreshed the canonical artifact checkpoint by replacing `last.ckpt` with the
  current rerun checkpoint content from `last-v1.ckpt`, then revalidated the
  exact review-cited path through `PtychoPINN_Lightning.load_from_checkpoint()`
- reran the focused checkpoint selector, the mandatory backlog gate, the
  compile gate, and the repo `integration` marker with fresh archived logs

## Completed Current-Scope Work

- the authoritative adjacent extension root remains
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z`
  and its canonical checkpoint path is now truthful again
- the exact review-cited checkpoint path
  `.../supervised_ffno_extension_20260430T180217Z/checkpoints/last.ckpt`
  now reloads successfully as:
  - `architecture= ffno`
  - `mode= Supervised`
  - `autoencoder_type= FfnoGeneratorModule`
  - `generator_output= real_imag`
  proof log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/checkpoint_reload_real_artifact_20260430.log`
- current verification evidence for approval is archived at:
  - checkpoint reload regression suite:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_checkpoint_reload_20260430.log`
    Note: command result was `11 passed, 11 warnings in 7.06s`
  - required deterministic gate:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_20260430_supervised_equivalent_rows_checkpoint_fix.log`
    Note: tmux-tracked PID `1532857` exited `0`; command result was
    `182 passed, 47 warnings in 303.23s`
  - compile gate:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/compileall_20260430_supervised_equivalent_rows_checkpoint_fix.log`
  - repo integration marker:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_integration_20260430_supervised_equivalent_rows_checkpoint_fix.log`
    Note: tmux-tracked PID `1534793` exited `0`; command result was
    `5 passed, 4 skipped, 1748 deselected, 2 warnings in 302.70s`

## Follow-Up Work

- none is required for current-scope backlog approval
- if paper-facing interpretation needs a stronger supervised FFNO control row,
  that is new experimental scope rather than remaining implementation debt on
  this item

## Residual Risks

- the extension still reuses the accepted `pinn_ffno` comparator by promotion
  instead of rerunning it in this pass
- the checkpoint directory still preserves older non-canonical files
  (`epoch=33...ckpt`, `epoch=37...ckpt`, `last-v1.ckpt`) alongside the repaired
  canonical `last.ckpt`; consumers should continue to treat `last.ckpt` as the
  authoritative reload path
- verification logs still contain the known non-fatal `tight_layout`,
  `skimage` SSIM, FRC, and TensorFlow Addons compatibility warnings already
  present on related Lines128 and repo integration surfaces
