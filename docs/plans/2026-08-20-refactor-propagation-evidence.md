# Refactor-branch propagation evidence

Companion to `docs/plans/2026-08-20-refactor-consolidation-propagation.md`.

## Entry state

- `refactor` tip: `2948e52e1ce52bb7e5d9ca6dcc55b66347caa983` (pull --ff-only: already up to date, 2026-08-20)
- Baseline suite (`bash ci/run_ci_tests.sh`, ptycho311 PATH python, 2m17s):
  `1438 passed, 6 skipped, 56 deselected, 11 xfailed, 1 xpassed` — **zero failed**.
  Note: the plan's pre-measured baseline (`1 failed, 1482 passed, 4 skipped,
  13 deselected`) was taken under different collection conditions; this
  session's measured line above is the comparison baseline for every task.
- Known-red baseline fact:
  `tests/torch/test_workflows_components.py::TestLightningCheckpointCallbacks::test_model_checkpoint_callback_configured`
  PASSES inside the CI suite run but FAILS when run in isolation
  (test-ordering dependence). Task 3 re-checks it at the facade split.
- Caller inventory (git grep counts at tip; listings in /tmp/refactor-*.txt):
  - TF facade importers (`ptycho.workflows.components`): **29**
  - torch facade importers (`ptycho_torch.workflows.components`): **134**
  - dill/weights_only sites (`manifest.dill|params.dill|weights_only`): **113**
  - old-name sites (`n_groups|n_subsample|.K|C_model|C_forward|grid_size`
    in `ptycho_torch/*.py` + `ptycho/config/*.py`): **255**

## Phase closeouts

| Phase | Commit | Suite line | Notes |
| --- | --- | --- | --- |
| 0 | `9cc9a811a` | n/a — no full-suite CI (per-task constraint); targeted tests green | Verified-dead barycentric route deleted; loud oversampling guard placed in the previously-silent `else` branch |
| 1 | `20b09fab6` | 1441 passed / 0 failed | Behavior change: TF training honors `subsample_seed`; torch oversampling request now reaches the Phase-0 guard |
| 2 | `0baf9eeb9` | 1443 passed / 0 failed | Facades pure; 63+16 names re-exported; moves AST-verified byte-identical; known-red baseline unchanged |
| 3 | `af354ba44` | 1441 passed / 0 failed (baseline 1438/0) | dill/weights_only load flips + JSON manifest; artifact v1/v2 frozen literals |
| 4 | `845670972` + `e09e08712` | 1481 passed / 0 failed | `torch-artifact-portable-v3` + `torch-model-spec-portable-v3`; `gridsize`/`n_raw_frames_selected` honest fields; C-twins deleted |
| closeout | `78fb2d263` + `4157cf5da` + `c5831c9b2` | 1488 passed / 0 failed | `torch-artifact-portable-v4` (v1–v3 frozen); one name per quantity; omitted ratchets: module-size gates, coordinator size gate, D2 tuple seam (task-6-report.md); model-spec era stays portable-v3 (no model-spec wire change) |

## Spec-drift finding (cross-branch)

`fno-stable`'s `specs/ptychodus_api_spec.md` §4.6 declares, for newly written
PyTorch archives, `artifact_schema_version='torch-artifact-v2'` and a nested
`torch-model-spec-v2` (lines 303–305), while
`ptycho_torch/artifact_schema.py:40` stamps
`CURRENT_ARTIFACT_SCHEMA_VERSION = ARTIFACT_SCHEMA_V4_VERSION`
(`torch-artifact-v4`; v1–v4 literals at lines 36–39) and
`ptycho_torch/model_spec.py:32` stamps `torch-model-spec-v3`. The spec lags the
code by two archive eras and one model-spec era; Phase 4 and the closeout each
changed the archive contract without the same-commit spec amendment that
Decision 1 requires. Back-fix belongs on `fno-stable` (its canonical
`docs/findings.md`); out of scope for this propagation, which fixes the pattern
on `refactor` rather than inheriting it.

## C4 quality-gate propagation (2026-08-21)

### Applicability

- Reviewed all 15 C4 commits from `927f9a662` through `3f4a6d9bb`, inclusive.
  Its Hybrid ResNet runtime, fixtures, quality helpers, contract test, and
  source-tree module-size gate are source-only under propagation rule 3.
- The destination overlay excludes those surfaces explicitly. The retained
  `tests/torch/test_synthetic_cnn_c4_ci_quality_integration.py` remains the
  branch-native CNN C4 smoke; the public FFNO selector remains the destination
  quality gate. No Hybrid ResNet substitute or mixed-family harness was added.
- The source changes that are architecture-independent were propagated:
  serving-checkpoint identity is now persisted in `manifest.json`; the
  validation score is read from the selected checkpoint, final-epoch train loss
  remains a health check, and raw reconstruction metrics are recomputed.
- Overlay proof against source tip `3f4a6d9bb`: 104 paths excluded, 19 patches
  applied, grep/dangling-import/import/closure gates all passed, emitted tree
  `9ef2fddbdded131c5f70036e3c2e6a57c794c523`. Transform tests: 23 passed.

### FFNO calibration and holdout

- RED before the producer fix:
  `test_public_synthetic_ffno_gs1_ci_five_epoch_quality` failed after the full
  run because the bundle manifest lacked `checkpoint_selection`
  (`1 failed in 207.99s`).
- Frozen producer commit:
  `90f0825c186d2c40cc7ee192aac1df9b044b1475`.
- Two independent fresh-process fit runs used empty roots and identical locked
  CLI arguments:
  `/tmp/refactor-ffno-serving-run02-v274ufXz/.../ffno-gs1-ci-5ep` and
  `/tmp/refactor-ffno-calibration-run02-EzKWCSac/.../ffno-gs1-ci-5ep`.
  Their dataset manifests, train/test NPZ array SHA-256 maps, seed lineage,
  root-normalized resolved workflows, runtime fingerprints, and deterministic
  execution records were all exactly equal.
- Locked environment: NVIDIA GeForce RTX 3090, compute capability 8.6, driver
  580.173.02, CUDA 12.8, cuDNN 91002, Torch 2.9.1+cu128, Lightning 2.5.5,
  Python 3.11.13, NumPy 1.26.4, precision `32-true`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms enabled.
- Both fit runs produced exactly:
  amplitude SSIM `0.5976732191880556`, phase SSIM
  `0.878507971092135`, absolute amplitude MAE
  `0.22102645985677952`, wrapped phase MAE
  `0.23077204823493958`, final train loss `175.2398681640625`, and
  final validation loss `148.75523376464844`. The selected serving checkpoint
  was epoch 4/global step 1405 with score `148.75523376464844`; the bundled
  weights source was `checkpoint`.
- The frozen envelope is the documented fit-only formula:
  amplitude/phase SSIM minima minus `0.035`/`0.015`, error maxima plus
  maxima times `1.10` plus `1e-6`. Resulting ceilings/floors are recorded in
  `tests/fixtures/synthetic_ffno_gs1_ci_5ep_metrics.json`; its SHA-256 after
  fit and after holdout was
  `0af6f4ae7ff2615fd585b4c0fe5736a89a6a1fcc40c35e17f27852ed954fe521`.
- Raw metrics recomputed from each `reconstruction.npz` were exactly equal to
  the recorded metrics. All four fit/holdout/sealed images were byte-identical:
  SHA-256 `5cbdfd7fa6bdb9711e7defa7a8bce6a5f26e6cec767b2e468552c975382cb249`,
  1,685,403 bytes, 2100x1350. Manual review passed line morphology and rejected
  flatness, collapse, checkerboarding, mirroring/transposition, saturation,
  seams/holes, and crop errors.
- Untouched holdout Run 03 used
  `/tmp/refactor-ffno-calibration-run03-UXTlCwUl/.../ffno-gs1-ci-5ep` and passed
  all six frozen thresholds with the exact fit metrics and losses. The fixture
  hash remained unchanged.
- The committed fixture is `8476470333243a2eb4c544a54690f96215ac3d76`.
  Sealed Run 04 used
  `/tmp/refactor-ffno-calibration-run04-zgy9SnU1/.../ffno-gs1-ci-5ep`;
  the public selector passed in 229.08s, raw metrics matched, all thresholds
  passed, and the rendered artifact matched the fit and holdout byte for byte.
