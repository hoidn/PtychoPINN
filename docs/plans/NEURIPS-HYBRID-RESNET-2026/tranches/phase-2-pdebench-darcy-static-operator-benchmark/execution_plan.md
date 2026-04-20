# Phase 2 PDEBench Darcy Static Operator Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a paper-relevant PDEBench Darcy Flow beta `1.0` static operator benchmark comparing Hybrid ResNet against strong FNO and U-Net baselines.

**Architecture:** Extend the shared `scripts/studies/pdebench_image128/` suite, reusing the existing supervised SWE model-builder and metric components where sensible. Keep Darcy-specific data loading, splits, normalization, literature-context, and result reporting explicit so the benchmark cannot be confused with smoke, capped-pilot, or dynamic one-step evidence.

**Tech Stack:** Python 3.11 via PATH `python`, PyTorch, h5py, NumPy, optional `neuralop` for FNO, existing `ptycho_torch` Hybrid ResNet components, tmux plus `ptycho311` for long runs.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Tranche ID: `phase-2-pdebench-darcy-static-operator-benchmark`
- Status: implemented through readiness; full benchmark pending
- Date: 2026-04-20
- Scope owner: Roadmap Phase 2 PDEBench `128x128` image-suite execution
- Source design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Source roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- Suite plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- Current preflight: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_preflight.md`
- Experiment root: `/home/ollie/Documents/PtychoPINN/`
- Data root: `/home/ollie/Documents/pdebench-data/`
- Manuscript artifact root: `/home/ollie/Documents/neurips/` (future Phase 5 root; do not create or write there in this tranche)
- Implementation summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md`

## Compliance Matrix

- [ ] **Repo Guidance:** `CLAUDE.md` and `docs/index.md` must be read before edits; plans stay under `docs/plans/`; long runs use tmux; no worktrees.
- [ ] **Design Contract:** PDE competitiveness runs must use the supervised real-channel Hybrid ResNet adapter, not the CDI physics wrapper.
- [ ] **Design Contract:** Smoke and readiness runs are not benchmark-performance evidence and must be labeled accordingly.
- [ ] **Design Contract:** Meaningful benchmark rows train on the full available training split after validation/test holdout.
- [ ] **Training Recipe:** Use the grid-lines-derived recipe unless this plan records a justified, pre-run override: `fno_modes=12`, hidden width `32`, `fno_blocks=4`, MAE loss, Adam `lr=2e-4`, and `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-4, threshold=0.0)`.
- [ ] **Darcy Data Contract:** Treat Darcy as a static operator map `nu[i] -> tensor[i]`; do not synthesize a time axis or one-step pairs.
- [ ] **Baseline Contract:** A Darcy comparison is not a strong-baseline comparison unless it includes both FNO and a non-toy U-Net baseline, or records a blocker before interpreting Hybrid ResNet.
- [ ] **Artifact Policy:** Write run artifacts under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/`; write durable planning/summary docs under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`; do not write manuscript tables or prose.

## Literature Context and Calibration Targets

Primary sources checked on 2026-04-20:

- PDEBench repository: `https://github.com/pdebench/PDEBench`
- PDEBench supplementary PDF, Table 8: `https://proceedings.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Supplemental-Datasets_and_Benchmarks.pdf`
- FNO/JMLR neural-operator paper: `https://jmlr.org/papers/volume24/21-1524/21-1524.pdf`
- HAMLET OpenReview PDF: `https://openreview.net/pdf/2a77316b975457831c5d7c21021f5bb87c99eff1.pdf`
- PDEBench Darcy config reference: `https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/models/config/args/config_Darcy.yaml`

PDEBench official Darcy baseline context:

| Source | Protocol caveat | Model | Metric target for Darcy beta `1.0` |
| --- | --- | --- | --- |
| PDEBench supplement Table 8 | Published PDEBench baseline table; config path indicates `reduced_resolution: 2`, so values are calibration for the PDEBench baseline protocol, not an exact native-`128x128` local pass/fail threshold. | U-Net | RMSE `6.4e-3`, nRMSE `3.3e-2` |
| PDEBench supplement Table 8 | Same caveat. | FNO | RMSE `1.2e-2`, nRMSE `6.4e-2` |
| HAMLET Table 1 | Later literature context following PDEBench protocol; Appendix C says Darcy samples are x2 subsampled to `64x64` with `9000/1000` train/test. Use as aspirational SOTA context, not same-protocol local reproduction. | OFormer | nRMSE `2.05e-2` |
| HAMLET Table 1 | Same caveat. | HAMLET | nRMSE `1.40e-2` |
| FNO/JMLR Table 2 | Canonical Darcy neural-operator benchmark, but not the exact PDEBench beta-file protocol. | FNO | relative error about `1.08e-2` to `9.8e-3` across listed resolutions |

Interpretation for this tranche:

- U-Net is the stronger official PDEBench baseline than FNO on beta `1.0`; do not frame FNO as the only strong baseline.
- The current local `SmallUNet` in `scripts/studies/pdebench_swe/models.py` is a smoke-scale model. A quick local parameter-count probe on 2026-04-20 gave about `18,465` trainable parameters for `hidden_channels=16`, while PDEBench's U-Net implementation/logged Darcy run is around `7.76M` parameters and the current Hybrid ResNet base is about `7.76M`. This plan therefore requires an official-style or otherwise parameter-credible U-Net profile before any strong-baseline claim.
- Local native-`128x128` rows may not numerically match the literature because PDEBench/HAMLET commonly report x2 spatial subsampling and different train/test policy. Treat literature values as sanity bands and context:
  - If local U-Net nRMSE is much worse than `3.3e-2` after a full split run, debug U-Net capacity, loss, normalization, metric implementation, and split/protocol before interpreting Hybrid ResNet.
  - If local FNO nRMSE is much worse than `6.4e-2` after a full split run, debug FNO wiring, modes, normalization, metric implementation, and training budget before interpreting Hybrid ResNet.
  - If Hybrid ResNet lands near or below the U-Net official band, the result is likely paper-interesting. If it only beats a weak local U-Net, it is not.

## Data Contract

Current verified local Darcy file:

- Path: `/home/ollie/Documents/pdebench-data/darcy/2D_DarcyFlow_beta1.0_Train.hdf5`
- Size: `1,310,724,488` bytes
- HDF5 attr: `beta = 1.0`
- Input dataset: `nu`, shape `(10000, 128, 128)`, dtype `float32`, logical axis `NHW`
- Target dataset: `tensor`, shape `(10000, 1, 128, 128)`, dtype `float32`, logical axis `NCHW`
- Supervision units: `10000` static samples
- Local sample contract: `x = nu[i]` converted to `(1,128,128)` and `y = tensor[i]` kept as `(1,128,128)`

Required split policy:

- Use deterministic sample-level splits over shared sample indices.
- Default split: `8000` train, `1000` validation, `1000` test using seed `20260420`.
- The same index list selects both `nu` and `tensor`.
- No overlap across splits.
- Full benchmark rows must train on all `8000` training samples. Smaller training sets are readiness, debug, or pilot rows only.

Required normalization policy:

- Fit input statistics on train-split `nu` only.
- Fit target statistics on train-split `tensor` only.
- Normalize inputs with input stats and targets with target stats.
- Denormalize predictions with target stats before reporting metrics.
- Record both stats files and the sample counts used to compute them.

Required metric policy:

- Primary local metric: denormalized target-space nRMSE / relative L2 computed as `sqrt(sum((pred-target)^2) / sum(target^2))` over the evaluated test split.
- Secondary metrics: denormalized RMSE, per-channel RMSE/nRMSE, wall time, peak GPU memory when available.
- Store enough metadata to compare the local metric to PDEBench Table 8, but do not claim exact PDEBench reproduction unless the reduced-resolution/split/training protocol also matches.

## Architecture and Interfaces

Expected new or modified files:

- Modify or create: `scripts/studies/pdebench_image128/data.py`
  - Lazy HDF5 `DarcyStaticOperatorDataset`.
  - Channel-first conversion from `NHW` and `NCHW`.
  - No full-file materialization.
- Modify or create: `scripts/studies/pdebench_image128/splits.py`
  - Deterministic sample-level split builder and split-manifest writer.
- Modify or create: `scripts/studies/pdebench_image128/normalization.py`
  - Separate train-only input and target stats for static operator tasks.
- Modify or create: `scripts/studies/pdebench_image128/metrics.py`
  - Denormalized RMSE, nRMSE/relative L2, and per-channel payloads.
- Modify or create: `scripts/studies/pdebench_image128/models.py`
  - Shared model builder derived from `scripts/studies/pdebench_swe/models.py`.
  - Supervised real-channel Hybrid ResNet.
  - Neuralop FNO.
  - Official-style or strong U-Net profile; local `SmallUNet` may remain only as `unet_tiny_smoke`.
- Modify or create: `scripts/studies/pdebench_image128/run_config.py`
  - Darcy full-run budget validation, primary profile list, literature-context metadata.
- Modify or create: `scripts/studies/pdebench_image128/darcy.py`
  - Darcy task wiring if keeping task-specific orchestration out of generic modules is cleaner.
- Modify: `scripts/studies/run_pdebench_image128_suite.py`
  - Add `--task darcy` benchmark/readiness entry points without breaking existing preflight.
- Test: `tests/studies/test_pdebench_darcy_data.py`
- Test: `tests/studies/test_pdebench_darcy_metrics.py`
- Test: `tests/studies/test_pdebench_image128_models.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`
- Durable summary after implementation: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md`

Reuse expectations:

- Reuse `HybridResnetSweModel`, `PadCropWrapper`, `ModelBuildBlocker`, and `count_parameters` concepts, but move/copy them behind image-suite names so Darcy is not semantically tied to SWE.
- Reuse the existing `err_nrmse` formula where it matches the operator metric, but add tests that prove denormalization uses target statistics.
- Reuse the project training recipe. Do not copy PDEBench's `lr=1e-3`, `epochs=500`, or `reduced_resolution=2` as the local default unless a later plan intentionally adds an official-protocol reproduction row.

## Phases

### Phase A - Contract Lock and Literature Payload

- [ ] A1: Add a machine-readable `literature_context.json` writer or static payload for Darcy beta `1.0`.
  - Include source URLs, access date `2026-04-20`, metric names, values, and protocol caveats.
  - Expected values: U-Net RMSE `6.4e-3`, U-Net nRMSE `3.3e-2`, FNO RMSE `1.2e-2`, FNO nRMSE `6.4e-2`, OFormer nRMSE `2.05e-2`, HAMLET nRMSE `1.40e-2`.
- [ ] A2: Add a test that fails if Darcy benchmark summaries omit literature values or omit protocol caveats.
- [ ] A3: Add a test that fails if a summary labels `unet_tiny_smoke` as a strong baseline.
- [ ] A4: Run the new literature/reporting tests and verify they fail before implementation.
- [ ] A5: Implement the minimal literature payload/reporting support.
- [ ] A6: Rerun the tests and verify they pass.

### Phase B - Darcy Dataset and Split Tests

- [ ] B1: Write tests for `DarcyStaticOperatorDataset` against a synthetic HDF5 file with `nu` `(N,H,W)` and `tensor` `(N,1,H,W)`.
  - Assert `len(dataset) == N`.
  - Assert sample keys include `input`, `target`, and `sample_index`.
  - Assert output tensors are channel-first `float32` shapes `(1,H,W)`.
  - Assert no time-axis or pair expansion appears in the dataset.
- [ ] B2: Write a test for the staged real file metadata when `/home/ollie/Documents/pdebench-data/darcy/2D_DarcyFlow_beta1.0_Train.hdf5` exists.
  - Assert `nu` shape `(10000,128,128)`.
  - Assert `tensor` shape `(10000,1,128,128)`.
  - Assert attr `beta` is `1.0`.
- [ ] B3: Write deterministic split tests.
  - Assert default split sizes are `8000/1000/1000`.
  - Assert train/val/test sample index sets are disjoint.
  - Assert split manifests record seed, file path, file size, beta, and split counts.
- [ ] B4: Run the new dataset/split tests and verify they fail.
- [ ] B5: Implement the lazy dataset, split builder, and manifest writer.
- [ ] B6: Rerun the new dataset/split tests and verify they pass.

### Phase C - Normalization and Metrics

- [ ] C1: Write tests proving input and target normalization stats are separate for static operator tasks.
- [ ] C2: Write tests proving stats are fit from train indices only.
- [ ] C3: Write tests proving predictions are denormalized with target stats before RMSE and nRMSE.
- [ ] C4: Write tests for nRMSE/relative-L2 formula against a hand-computed tensor.
- [ ] C5: Run the normalization/metric tests and verify they fail.
- [ ] C6: Implement normalization and metric payloads.
- [ ] C7: Rerun the normalization/metric tests and verify they pass.

### Phase D - Strong Model Profiles

- [ ] D1: Move or mirror the generic SWE supervised model builder into `scripts/studies/pdebench_image128/models.py`.
- [ ] D2: Add tests that build `hybrid_resnet_base`, `fno_base`, `unet_strong`, and `unet_tiny_smoke` for `in_channels=1`, `out_channels=1`, `spatial_shape=(128,128)`.
- [ ] D3: Add tests that parameter counts are recorded for every profile.
- [ ] D4: Add a strong U-Net gate:
  - `unet_tiny_smoke` may be used only for readiness.
  - `unet_strong` must be official-style `UNet2d(init_features=32)` or an explicitly documented equivalent.
  - If `unet_strong` cannot be built, full strong-baseline comparison is blocked.
- [ ] D5: Add a FNO dependency gate:
  - `neuralop` missing may block FNO, but it must be reported as a blocker, not silently skipped.
- [ ] D6: Run the model tests and verify they fail.
- [ ] D7: Implement model profiles and gates.
- [ ] D8: Rerun the model tests and verify they pass.

### Phase E - Runner, Artifacts, and Readiness

- [ ] E1: Extend `scripts/studies/run_pdebench_image128_suite.py` with a Darcy mode that supports at least `inspect`, `readiness`, and `benchmark`.
- [ ] E2: Add a run-budget validator for Darcy with separate readiness and benchmark contracts.
  - Readiness may cap samples and epochs, and must emit `evidence_scope: smoke_feasibility_only`.
  - Benchmark must require full train split and the three primary profiles: `hybrid_resnet_base`, `fno_base`, `unet_strong`.
- [ ] E3: Add artifact writers:
  - `dataset_manifest.json`
  - `hdf5_metadata.json`
  - `split_manifest.json`
  - `normalization_stats_input.json`
  - `normalization_stats_target.json`
  - `model_profile_<profile>.json`
  - `metrics_<profile>.json`
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `literature_context.json`
  - `invocation.json`
  - `invocation.sh`
- [ ] E4: Add runner/reporting tests using a tiny synthetic HDF5 file and toy budgets.
- [ ] E5: Run the runner tests and verify they fail.
- [ ] E6: Implement the runner and artifact writers.
- [ ] E7: Rerun runner tests and verify they pass.
- [ ] E8: Run a Darcy readiness pass on tiny data or a tiny cap of the staged file. This verifies plumbing only and cannot be used for model performance.

### Phase F - Full Benchmark Execution Plan

- [ ] F1: Before launching long training, write a full-run budget JSON under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json`.
- [ ] F2: The benchmark budget must include:
  - `task_id: darcy`
  - train/val/test counts `8000/1000/1000`
  - primary profiles `hybrid_resnet_base`, `fno_base`, `unet_strong`
  - training seed `20260420`
  - loss decision and rationale
  - optimizer/scheduler recipe
  - batch size, epochs, precision, device, and num_workers
  - no capped training set
- [ ] F3: Loss decision gate:
  - Default to the project recipe MAE for same-protocol local comparison.
  - If switching to relative L2 or MSE for operator-learning alignment, record the override before training and apply it to all profiles.
- [ ] F4: Launch full runs in tmux with `ptycho311`, one profile at a time if needed for the RTX 3090 budget.
  - Priority order under tight runtime: `hybrid_resnet_base`, `unet_strong`, `fno_base`.
  - Rationale: U-Net is the stronger official PDEBench beta `1.0` local baseline; FNO remains the canonical operator baseline.
- [ ] F5: If any primary profile fails, record a blocker and do not claim a complete strong-baseline comparison.
- [ ] F6: If local U-Net or FNO is more than roughly `2x` worse than the PDEBench nRMSE calibration after a full run, debug protocol and implementation before interpreting Hybrid ResNet.

Example tmux launch pattern for the eventual long run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python scripts/studies/run_pdebench_image128_suite.py \
  --task darcy \
  --mode benchmark \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark \
  --run-budget .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json
```

### Phase G - Summary and Roadmap Update

- [ ] G1: Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md`.
- [ ] G2: The summary must separate:
  - readiness evidence
  - full benchmark evidence
  - capped/pilot/debug evidence
  - literature calibration targets
  - blockers and failed profiles
- [ ] G3: Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md` with the completed Darcy status and next selector recommendation.
- [ ] G4: Update `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` only if gates, order, or baseline definitions changed during execution.
- [ ] G5: Update `docs/index.md` and `docs/studies/index.md` so the Darcy plan and summary are discoverable.
- [ ] G6: Append a selector-facing update to `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`.

## Verification Commands

Initial plan/doc validation:

```bash
python - <<'PY'
from pathlib import Path
path = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md")
text = path.read_text(encoding="utf-8")
required = [
    "Darcy",
    "beta `1.0`",
    "nu[i] -> tensor[i]",
    "8000",
    "unet_strong",
    "6.4e-3",
    "3.3e-2",
    "1.2e-2",
    "6.4e-2",
    "1.40e-2",
    "smoke_feasibility_only",
]
missing = [term for term in required if term not in text]
if missing:
    raise SystemExit(f"Darcy plan missing required terms: {missing}")
print("Darcy execution plan contains required terms")
PY
```

After implementation adds tests:

```bash
python -m pytest --collect-only tests/studies/test_pdebench_darcy_data.py tests/studies/test_pdebench_darcy_metrics.py tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q
python -m pytest tests/studies/test_pdebench_image128_preflight.py tests/studies/test_pdebench_darcy_data.py tests/studies/test_pdebench_darcy_metrics.py tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q
```

Readiness only, not performance:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task darcy \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/readiness
```

Doc/state checks:

```bash
python -m json.tool state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json >/dev/null
git diff --check -- \
  docs/index.md \
  docs/studies/index.md \
  docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md
```

## Completion Criteria

- [ ] Darcy data, split, normalization, and metric contracts are implemented and tested.
- [ ] Hybrid ResNet, FNO, and `unet_strong` build and record parameter counts for `1x128x128 -> 1x128x128`.
- [ ] Readiness run proves load/train/eval/write plumbing and labels metrics as non-performance evidence.
- [ ] Full benchmark run trains on all `8000` train samples for each completed primary profile.
- [ ] Comparison summary includes local full-run metrics and literature calibration values with protocol caveats.
- [ ] No paper-facing artifact is written to `/home/ollie/Documents/neurips/` during this tranche.
