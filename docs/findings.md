# PtychoPINN Knowledge Base (NanoBragg Standard)

This ledger captures the most important lessons, conventions, and recurring issues discovered across the project. Before starting new work—or especially when debugging—consult this table to avoid repeating known mistakes.

| Finding ID | Date | Keywords | Synopsis | Evidence Pointer | Status |
| INFERENCE-OVERSAMPLE-001 | 2026-01-13 | dose-response, inference, oversampling, n_groups, stitching | The dose_response_study pipeline reused training `n_groups` (2000) for test data with only 128 raw images while oversampling was enabled. This triggered K-choose-C oversampling during inference and led to stitching shape errors and garbage reconstructions. Fix: cap inference groups to `len(test_data.diff3d)` and use the standard inference bundle path (`load_inference_bundle_with_backend` → `nbutils.reconstruct_image` → `tf_helper.reassemble_position`). | [Link](scripts/studies/dose_response_study.py) | Resolved |
| REASSEMBLY-BATCH-001 | 2025-11-13 | batched-reassembly, translation-layer, shape-mismatch, dense-view, graph-mode | Batched patch reassembly failed when processing large (5k+) dense datasets in TensorFlow graph mode (tf.while_loop). The issue was using `tf.shape(canvas)` (dynamic runtime values) instead of the static `padded_size` parameter for resize operations. TensorFlow's `shape_invariants` requirement in while_loop demands static shape specifications at graph compilation time. Fixed by using the static `padded_size` integer directly in `tf.image.resize_with_crop_or_pad` instead of extracting canvas dimensions dynamically. This ensures shape consistency between the canvas and batch_result tensors during while_loop compilation and execution. `ReassemblePatchesLayer` now auto-selects batched reassembly when `total_patches > 64` to keep GPU memory bounded. | [Link](ptycho/tf_helper.py:939-965, ptycho/custom_layers.py:149-164, tests/study/test_dose_overlap_comparison.py:760-817) | Resolved |
| XLA-VECTORIZE-001 | 2025-11-13 | xla, vectorization, batch-mismatch, mk_reassemble_position_real | The `mk_reassemble_position_real` function's vectorized path failed with "Dimensions must be equal, but are 5088 and 159" when processing dense datasets (>1000 patches) through `translate_xla`. Root cause: the `_vectorised()` branch processes all patches in a single call to `translate()`, which is decorated with `@complexify_function`. For complex tensors (gridsize=2 → 4 channels), this splits into real/imag parts and the XLA bilinear interpolation path receives mismatched batch dimensions during the internal chunking. Fixed by adding an explicit patch count limit (`max_vectorized_patches=1000`) alongside the existing memory cap, forcing large patch counts to use the `_streaming()` path with tf.while_loop chunking (chunk_size=1024). | [Link](ptycho/tf_helper.py:1183-1223) | Resolved |
| XLA-DYN-DOT-001 | 2025-11-13 | xla, einsum, dynamic-shape, projective-warp | TF XLA JIT failed during inference with DynamicPadder RET_CHECK on dot/einsum when applying homography to the sampling grid (dynamic H×W). Replaced `tf.einsum("bij,bhwj->bhwi", M, grid)` with explicit broadcasted multiplies/adds to compute `(sx, sy, w)`; removes dynamic dot and unblocks XLA. Also removed a `tf.print` in Translation layer that emitted a PrintV2 op. | ptycho/projective_warp_xla.py:90, ptycho/tf_helper.py:824 | Resolved |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CONVENTION-001 | 2025-10-16 | architecture, legacy, global-state | The codebase consists of a legacy grid-based system tied to `params.cfg` and a modern coordinate-based system; pick the right side before changing anything. | [Link](docs/DEVELOPER_GUIDE.md#1-the-core-concept-a-two-system-architecture) | Active |
| POLICY-001 | 2025-10-17 | policy, PyTorch, dependencies, mandatory | PyTorch (torch>=2.2) is now a mandatory dependency for PtychoPINN as of Phase F (INTEGRATE-PYTORCH-001). All code in `ptycho_torch/` and `tests/torch/` assumes PyTorch is installed. Torch-optional execution paths were removed; modules raise actionable RuntimeError if torch import fails. Tests in `tests/torch/` are automatically skipped in TensorFlow-only CI environments via directory-based pytest collection rules in `tests/conftest.py`, but will fail with actionable ImportError messages if PyTorch is missing in local development. Migration rationale and implementation evidence documented in governance decision and Phase F implementation logs. | [Link](plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md) | Active |
| ANTIPATTERN-001 | 2025-10-16 | imports, side-effects, debugging | Import-time side effects (e.g., loading data in module scope) caused hidden crashes; always push work into functions with explicit arguments. | [Link](docs/DEVELOPER_GUIDE.md#21-anti-pattern-side-effects-on-import) | Active |
| CONFIG-001 | 2025-10-16 | params.cfg, initialization-order | `update_legacy_dict(params.cfg, config)` must run before any legacy module executes; missing this broke gridsize sync and legacy interop. | [Link](docs/debugging/QUICK_REFERENCE_PARAMS.md#⚠️-the-golden-rule) | Active |
| CONFIG-002 | 2025-10-20 | execution-config, cli, params.cfg | PyTorch execution configuration (PyTorchExecutionConfig) controls runtime behavior only and MUST NOT populate params.cfg. Only canonical configs (TrainingConfig, InferenceConfig) bridge via CONFIG-001. CLI helpers auto-detect accelerator default='auto' and validate execution config fields via factory integration. Execution config applied at priority level 2 (between explicit overrides and dataclass defaults). | [Link](plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_redline.md) | Active |
| CONFIG-LOGGER-001 | 2025-10-24 | logger, execution-config, lightning, mlflow | PyTorch training uses CSVLogger by default (`PyTorchExecutionConfig.logger_backend='csv'`) to capture train/validation metrics from Lightning `self.log()` calls, replacing prior `logger=False` which discarded metrics. Allowed backends: `csv` (zero deps, CI-friendly), `tensorboard` (requires tensorboard from TF install), `mlflow` (requires mlflow package + server), `none` (disable). Legacy `--disable_mlflow` flag deprecated with DeprecationWarning mapping to `--logger none`. MLflow migration to Lightning MLFlowLogger tracked as Phase EB3.C4 backlog. Decision rationale and implementation evidence in governance decision approval record. | [Link](plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md) | Active |
| EXEC-ACCUM-001 | 2025-11-13 | pytorch, lightning, manual-optimization | Lightning manual optimization (PtychoPINN_Lightning sets `automatic_optimization=False`) is incompatible with `Trainer(accumulate_grad_batches>1)`. Passing `--torch-accumulate-grad-batches` > 1 causes `MisconfigurationException: Automatic gradient accumulation is not supported for manual optimization`. Treat the flag as unsupported for PyTorch backend until Lightning support lands: fail fast or clamp to 1 with a clear warning. | [Link](plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/red/blocked_20251113T184100Z_manual_optim_accum.md) | Active |
| DATA-SUP-001 | 2025-11-13 | supervised, labels, data-contract | Supervised PyTorch mode expects dataloader batches with `label_amp`/`label_phase`. Experimental fly001 NPZ datasets and the minimal CLI fixture omit those keys, leading to `KeyError: 'label_amp'` in `PtychoPINN_Lightning.training_step`. Either supply labeled synthetic NPZ (per DATA-001) or block supervised runs with an actionable error until labeled data is available. | [Link](plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/red/blocked_20251113T184300Z_supervised_data_contract.md) | Active |
| DEVICE-MISMATCH-001 | 2025-11-13 | pytorch, inference, accelerator | RESOLVED — Commit 85478a67 updates `scripts/inference/inference.py` and `ptycho_torch/inference.py` to call `model.to(device)`/`model.eval()` in both the CLI and helper, adds regression tests, and proves CUDA inference succeeds (`cli/pytorch_cli_smoke_training/inference_cuda.log`, `green/pytest_pytorch_inference_device.log`). | [Link](plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt) | Resolved |
| BUG-TF-001 | 2025-10-16 | shape-mismatch, gridsize, tests | Gridsize > 1 yields channel mismatches unless `params.cfg['gridsize']` is populated before `generate_grouped_data`; verify config vs params values. | [Link](docs/debugging/TROUBLESHOOTING.md#shape-mismatch-errors) | Active |
| DATA-001 | 2025-10-16 | data-contract, npz, io | All NPZ datasets must follow the canonical specification (keys, dtypes, normalization); deviations caused silent inference failures. | [Link](../specs/data_contracts.md) | Active |
| NORMALIZATION-001 | 2025-10-16 | scaling, physics, preprocessing | Three independent normalization systems (physics, statistical, display) must never be mixed; applying scaling in the wrong stage created double-scaling bugs. | [Link](docs/DEVELOPER_GUIDE.md#35-normalization-architecture-three-distinct-systems) | Active |
| STUDY-001 | 2025-10-16 | fly64, baseline, generalization | On fly64 experiments the baseline model outperformed PtychoPINN by ~6–10 dB, contradicting expectations and motivating architecture review. | [Link](docs/FLY64_GENERALIZATION_STUDY_ANALYSIS.md#key-findings) | Active |
| MODULE-SINGLETON-001 | 2026-01-07 | model, singleton, import-time, gridsize, shape-mismatch | `ptycho.model.autoencoder` is a module-level singleton created at import time using current `params.cfg['gridsize']`. Changing gridsize after import does NOT recreate the model—the architecture is frozen with the import-time gridsize. This causes shape mismatches when simulation uses gridsize=1 but training uses gridsize=2. **CONFIG-001 alone is insufficient**—you must use factory functions to create fresh models with the correct architecture. **FULLY RESOLVED (2026-01-07):** Phase B lazy loading via `__getattr__` (ptycho/model.py:867-890) prevents import-time model creation. Phase C removed all XLA workarounds (`USE_XLA_TRANSLATE=0`, `TF_XLA_FLAGS`, `run_functions_eagerly`) from scripts and tests. XLA re-enablement verified by spike test. For custom workflows, use `create_compiled_model()` (returns compiled model ready for training) or `create_model_with_gridsize()` (uncompiled). | [Link](ptycho/model.py:867-890, tests/test_model_factory.py, docs/debugging/TROUBLESHOOTING.md#model-architecture-mismatch-after-changing-gridsize) | Resolved |
| ACCEPTANCE-001 | 2025-11-11 | phase-d, geometry, spacing, dense-view | Dense fly64 overlap runs cannot meet the legacy 10 % minimum acceptance—bounding-box math caps the dense view at ≈0.96 % (42/5088) for the 38.4 px threshold—so `generate_overlap_views` must compute a geometry-aware acceptance floor (area ÷ packing discs) and log `geometry_acceptance_bound` + `effective_min_acceptance` in the metrics bundle before proceeding. | [Link](plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md) | Active |
| OVERSAMPLING-001 | 2025-10-16 | oversampling, gridsize, combinatorics | Oversampling only works when `gridsize > 1` and `K > C`; otherwise requested groups can never exceed raw images. | [Link](docs/debugging/TROUBLESHOOTING.md#oversampling-not-working) | Active |
| MIGRATION-001 | 2025-10-16 | params-removal, refactor, strategy | 66+ files still depend on `params.cfg`; migration plan is to eliminate new uses, document remaining ones, then remove the dependency. | [Link](docs/debugging/QUICK_REFERENCE_PARAMS.md#the-66-file-problem) | Active |
| PROCEDURE-001 | 2025-10-16 | review, defensive-coding | Flag hidden `params` reads and undocumented dependencies during code review; insist on explicit parameters or documented prerequisites. | [Link](docs/debugging/QUICK_REFERENCE_PARAMS.md#red-flags-in-code-review-🚩) | Active |
| FORMAT-001 | 2025-10-17 | data-contract, npz, legacy-format, transpose | Some NPZ datasets use legacy (H,W,N) diffraction array format instead of DATA-001 compliant (N,H,W); caused IndexError in PyTorch dataloader when nn_indices referenced global positions beyond first dimension. Auto-transpose heuristic added to both `_get_diffraction_stack()` and `npz_headers()` to detect and fix at runtime. | [Link](plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T230724Z/callchain/summary.md) | Active |
| TYPE-PATH-001 | 2025-11-06 | pytorch, path, type-safety, config | PyTorch workflows failed with AttributeError/TypeError when string paths from TrainingConfig were passed to functions expecting Path objects. Root cause: ptycho_torch/workflows/components.py:650,682 called functions with raw string paths (config.train_data_file, config.output_dir) instead of wrapping with Path(). Symptom: 'str' object has no attribute 'exists' and unsupported operand type(s) for /. Fix: Wrap path parameters with Path() at call sites before invoking downstream helpers. Prevention: Normalize TrainingConfig/InferenceConfig path fields via Path() in constructors or apply runtime coercion at module boundaries. | [Link](plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/summary.md) | Active |
| PHASEC-METADATA-001 | 2025-11-06 | phase-c, metadata, validator, orchestration | `run_phase_g_dense.py` still expects Phase C outputs in legacy `dose_*_{train,test}/fly64_<split>_simulated.npz` directories; the refreshed generator now writes `data/phase_c/dose_<dose>/{patched,patched_train,patched_test}.npz` with `_metadata`. The guard falsely blocks dense relaunches until it scans the new layout and ensures `_metadata` on the patched splits. | [Link](plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T081826Z/phase_c_metadata_guard_blocker/cli/blocker_phase_c_metadata.log) | Active |
| TEST-CLI-001 | 2025-11-10 | tdd, cli-validation, orchestration, regression-guard | Orchestrator-level CLI validation requires explicit test fixtures for RED/GREEN cycles and must enforce complete log bundles (phase banners + SUCCESS sentinel + dose/view-specific filenames). When adding validation to artifact verifiers, create both RED (missing/wrong patterns) and GREEN (complete artifacts) test cases with realistic log content including all required markers. **Filename patterns MUST match orchestrator output**: phase_e_baseline_gs1_dose{dose}.log, phase_e_{view}_gs2_dose{dose}.log, phase_f_{view}_train.log, etc. (not generic phase_e_baseline.log). Helper logs (aggregate_report_cli.log, metrics_digest_cli.log) and completion sentinels ("complete" marker) are also required. Test isolation tip: GREEN tests should validate only the specific check under test, not entire bundle completeness (use focused assertions on check['valid'] field rather than exit_code==0). | [Link](plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:437-596) | Active |
| PREVIEW-PHASE-001 | 2025-11-11 | highlights, preview, validator, phase-only | The dense Phase G preview artifact (`analysis/metrics_delta_highlights_preview.txt`) must contain **only** the four phase deltas (MS-SSIM/MAE vs Baseline/PtyChi) with explicit ± signs; any `amplitude` text or extra tokens indicates corruption and must fail validation. Existing verifier logic only checked for the presence of formatted numbers, so previews that regressed back to amplitude-inclusive lines could slip through. Harden `validate_metrics_delta_highlights` (and tests) to enforce phase-only content and surface actionable metadata in reports. | [Link](plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T005802Z/phase_g_dense_full_execution_real_run/plan/plan.md) | Active |
| METRICS-NAMING-001 | 2025-11-14 | phase-g, metrics, canonical-models | `report_phase_g_dense_metrics.py` only recognizes the canonical model IDs `PtychoPINN`, `Baseline`, and `PtyChi`. When `scripts.compare_models.py` writes friendly labels such as `"Pty-chi (pty-chi)"`, the reporter treats `PtyChi` as missing and aborts even if the data exists. Keep the canonical names (or add a mapping) before invoking the reporter, otherwise PREVIEW/verification evidence never materializes. | [Link](plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/aggregate_report_cli.log) | Active |
| BASELINE-OFFSET-001 | 2025-11-13 | baseline, offsets, numerical-stability, compare-models | Baseline model inference returned all-zero outputs on test split due to uncentered offsets causing numerical instability. Train/test splits had very different offset means (train≈185px, test≈273px, 87px delta), and the baseline model (trained with zero-mean offsets) failed on the large distribution shift. Fixed by centering `baseline_offsets` to zero-mean in `prepare_baseline_inference_data()` before inference: `centered_offsets = flattened_offsets_np - offset_mean`. After centering, test split baseline outputs changed from mean=0.000000 (all zeros) to mean=0.079082 (16086 nonzero pixels). Added `--baseline-debug-limit` and `--baseline-debug-dir` CLI flags for fast debugging with NPZ+JSON artifact dumps. | [Link](scripts/compare_models.py:247-258, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/blocker_resolved_offset_centering.md) | Resolved |
| BASELINE-CHUNKED-001 | 2025-11-13 | baseline, gpu-memory, chunked-inference, compare-models | Dense test split Baseline inference failed with TensorFlow `ResourceExhaustedError` when processing 5088 groups in a single batch, leaving blank Baseline metric rows and blocking Phase G pipeline. Train split succeeded with 78.7M nonzero pixels, but test split OOM'd before completing inference. Root cause: TensorFlow GPU memory exhaustion during large-batch inference. Fixed by implementing chunked Baseline inference with configurable `--baseline-chunk-size` (number of groups per chunk) and `--baseline-predict-batch-size` (batch size within each chunk, default 32). Chunks are processed sequentially with explicit `tf.keras.backend.clear_session()` between chunks to release GPU memory. Per-chunk DIAGNOSTIC logging added to track mean/max/nonzero stats. Automatic fallback: when `--baseline-chunk-size` is None, uses original single-shot path. Chunked path includes try/except for `ResourceExhaustedError` with actionable guidance to reduce chunk/batch sizes. | [Link](scripts/compare_models.py:1077-1140, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251116T010000Z_test_baseline_oom.md) | Active |
| BASELINE-CHUNKED-002 | 2025-11-16 | baseline, gpu-memory, data-loading, chunked-container | Chunked Baseline inference alone is insufficient: `scripts/compare_models.py` still instantiates a full `PtychoDataContainer` before entering the chunk loop, so the dense-test rerun continues to crash with `ResourceExhaustedError: failed to allocate memory [Op:Cast]` inside `ptycho/loader.py:141 → combine_complex()` (`analysis/dose_1000/dense/test/logs/logs/debug.log:299-360`). **RESOLVED 2025-11-16**: Refactored chunked Baseline mode to slice RawData per chunk (`slice_raw_data`, `dataclasses.replace(final_config, n_groups=n_chunk)`), create chunk-scoped `PtychoDataContainer`s, and use concatenated `pinn_offsets` for alignment so the full container never materializes on GPU. Implementation at scripts/compare_models.py:1152-1197 (Baseline chunking) and 1431-1442 (alignment with pinn_offsets). | [Link](scripts/compare_models.py:1152-1197,1431-1442, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251116T010000Z_test_baseline_oom.md) | Resolved |
| SINGLETON-SAVE-001 | 2026-01-06 | model_manager, save, singleton, train_pinn, untrained-model | `model_manager.save()` hardcodes saving `model.autoencoder` and `model.diffraction_to_obj` (module-level singletons). When `train_pinn.train()` was changed (commit d17a351f) to create a NEW model via `create_compiled_model()` and train it, the trained model was returned but NEVER assigned back to the singletons. Thus `model_manager.save()` saved the UNTRAINED singleton, and inference loaded garbage (saturated ~0.5 amplitude, flat phase). **Root cause identified via git bisect:** last good commit 7d558ead used singleton directly, first bad commit d17a351f used factory but didn't update singleton. **Fixed by updating module singletons after model creation in `train_pinn.train()`: `model.autoencoder = model_instance; model.diffraction_to_obj = diffraction_to_obj`.** | [Link](ptycho/train_pinn.py:86-90, ptycho/model_manager.py:463-464) | Resolved |
| BUG-TF-REASSEMBLE-001 | 2026-01-09 | tf_helper, mixed-types, tensor-shape, batched-reassembly | `_reassemble_position_batched()` fails with `ValueError: Can't convert Python sequence with mixed types to Tensor` at line 936 when constructing `tf.zeros((1, padded_size, padded_size, 1))`. Root cause: `padded_size` may be a numpy scalar (`numpy.int64`) rather than Python `int`, creating mixed types in the shape tuple. Affects large datasets (>64 patches) that trigger batched processing. Related to REASSEMBLY-BATCH-001. | [Link](docs/bugs/BUG-TF-REASSEMBLE-MIXED-TYPES-001.md) | Open |
| PARAMS-N-DIV4-001 | 2026-01-13 | params, padding, centermask, reassembly | `params.cfg['N']` must be divisible by 4; `CenterMaskLayer` uses `N//2` and `N//4` to build the normalization mask, and non-divisible values cause size drift (e.g., 138 → 137) that breaks reassembly. Enforced via `params.validate()` to fail fast. | [Link](ptycho/params.py:83-92, tests/study/test_dose_overlap_comparison.py:774-802) | Active |
| SIM-LINES-CONFIG-001 | 2026-01-20 | sim_lines_4x, CONFIG-001, NaN, training | The sim_lines_4x pipeline and any plan-local runners MUST call `update_legacy_dict(params.cfg, config)` before training or inference to prevent NaN failures. Without this sync, legacy modules read stale gridsize/intensity values causing training collapse. Fix applied in C4f (scripts/studies/sim_lines_4x/pipeline.py + DEBUG-SIM-LINES-DOSE-001 runner). | [Link](plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103144Z/final_summary.md) | Resolved |
| DEBUG-SIM-LINES-DOSE-001-NAN | 2026-01-20 | sim_lines_4x, CONFIG-001, NaN, training-crash | **NaN CRASH RESOLVED (partial fix).** Root Cause: CONFIG-001 violation (stale `params.cfg` values not synced before training/inference). Fix: C4f bridging — `update_legacy_dict(params.cfg, config)` calls added to `scripts/studies/sim_lines_4x/pipeline.py` and plan-local runner. Verification: All four scenarios now train without NaN. **CRITICAL: Initiative NOT complete** — reconstructions still have ~3-6x amplitude bias vs ground truth. "No NaN" ≠ "working reconstruction". Phase D in progress to achieve actual dose_experiments parity. | [Link](plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103144Z/final_summary.md) | Active |
| H-NEPOCHS-001 | 2026-01-20 | sim_lines_4x, training, epochs, amplitude | Extending gs2_ideal from 5 epochs to 60 (plan-local runner, EarlyStopping halted at epoch 30) produced <1% change in MAE/pearson_r and left the ~6.7× prediction↔truth gap intact, so insufficient training length is **not** the cause of the amplitude collapse. Future work must focus on normalization/loss wiring or architecture. | [Link](plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T140531Z/summary.md) | Active |

## DATA-002 - NPZ Metadata Pickle Requirement
**Category:** Data I/O  
**Impact:** Phase C→D→E pipeline execution  
**Location:** studies/fly64_dose_overlap/{overlap.py:388, training.py:409,416}  
**Date:** 2025-11-11

### Issue
Phase C datasets include a `_metadata` field (dtype=object) containing JSON provenance strings. NumPy 1.16+ defaults to `allow_pickle=False` for security, causing `ValueError: Object arrays cannot be loaded when allow_pickle=False` when consuming these NPZ files downstream.

### Solution
Add `allow_pickle=True` parameter to all `np.load()` calls that read Phase C outputs:
```python
with np.load(path, allow_pickle=True) as data:
    data_dict = {k: data[k] for k in data.keys()}
```

### Context
- Phase C generation (studies/fly64_dose_overlap/generation.py) saves metadata as object array
- Phase D (overlap.py) and Phase E (training.py) validation consume these NPZs  
- Security: object arrays are trusted since they're generated internally, not user-supplied

### Related
- DATA-001 (data contract compliance)
- Commit: 5cd130d3

## PINN-CHUNKED-001 - PINN Chunked Inference Architecture Limitation
**Category:** Architecture, GPU Memory, Refactor Required
**Impact:** compare_models.py large-dataset PINN inference
**Location:** scripts/compare_models.py:1026-1146, ptycho/loader.py (PtychoDataContainer)
**Date:** 2025-11-13 (Reported), 2026-01-07 (Resolved)

### Issue
Dense test-split PINN inference fails with OOM (`ResourceExhaustedError` in `ptycho/loader.py:141 → combine_complex()`) when processing 5216 already-grouped diffraction patterns. Attempted chunked PINN inference (similar to BASELINE-CHUNKED-001) hits architectural limitation: NPZ files contain pre-grouped data, and `create_ptycho_data_container()` eagerly converts ALL groups to GPU tensors before inference begins.

### Resolution (2026-01-07)
**FEAT-LAZY-LOADING-001 Phase B** implemented lazy tensor allocation in `PtychoDataContainer`:
1. Data stored as NumPy arrays internally (`_X_np`, `_Y_I_np`, etc.)
2. Lazy property accessors (`.X`, `.Y`, etc.) convert to tensors on first access with caching
3. `as_tf_dataset(batch_size)` method provides memory-efficient streaming for training
4. `load()` function updated to pass NumPy arrays (no eager tensorification)
5. Backward compatible: existing code accessing `.X`, `.Y` still works (triggers lazy conversion)

For large datasets, use `container.as_tf_dataset(batch_size)` instead of accessing `.X`, `.Y` directly.

### Evidence
- Fix implementation: `ptycho/loader.py:97-321` (PtychoDataContainer with lazy loading)
- Tests: `tests/test_lazy_loading.py::TestLazyLoading` (5 tests, all passing)
- G-scaled verification: `tests/test_lazy_loading.py::TestCompareModelsChunking::test_container_numpy_slicing_for_chunked_inference` — verifies `_X_np`/`_coords_nominal_np` slicing enables chunked inference without GPU allocation
- Intensity scale: `ptycho/train_pinn.py:165-237` (D4d–D4f enforce dataset_intensity_stats → `_X_np` → closed-form priority; tests: `tests/test_train_pinn.py::TestIntensityScale::{test_lazy_container_does_not_materialize,test_uses_dataset_stats}`)
- Artifacts: `plans/active/FEAT-LAZY-LOADING-001/reports/2026-01-07T220000Z/`, `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T200000Z/`, `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T004455Z/`

### Status
**Resolved** — Lazy tensor allocation implemented. Large datasets can now be processed without OOM at container construction time. G-scaled verification confirms chunked inference via NumPy slicing is supported. Dataset-derived intensity scale handling now prioritizes `dataset_intensity_stats` (attached by loader and manual constructors) before `_X_np`, and only falls back to the `(N/2)` closed form when both sources are missing or near-zero (D4d–D4f, 2026-01-21). Any caller that instantiates `PtychoDataContainer` manually must either route through `loader.load()` or provide `dataset_intensity_stats` so the preferred CPU path stays available.

## TF-NON-XLA-SHAPE-001 - Non-XLA Translation Batch Dimension Mismatch
**Category:** TensorFlow, Translation, Shape Handling
**Impact:** TensorFlow training/inference with gridsize > 1 when XLA disabled (`USE_XLA_TRANSLATE=0`)
**Location:** ptycho/tf_helper.py:702-839 (`translate_core`, `Translation.call`)
**Date:** 2025-11-14

### Issue
When XLA is disabled (`USE_XLA_TRANSLATE=0`) and gridsize > 1, TensorFlow training crashes during the first epoch with shape mismatch in `translate_core`:
```
Shapes of all inputs must match: values[0].shape = [4] != values[2].shape = [128]
 [[{{node functional_1/padded_objs_with_offsets_1/translation_36_1/stack_1}}]]
```
**Root cause:** The non-XLA `ImageProjectiveTransformV3` fast path builds transformation matrices using `tf.stack` on tensors (`ones`, `zeros`, `dx`, `dy`) that must all have the same batch dimension. When `gridsize > 1`, `_channel_to_flat` expands images from shape `(b, N, N, c)` to `(b*c, N, N, 1)`, but `translations` remains `(b, 2)`. The mismatch causes `tf.stack` to fail.

### Fix (2025-11-14)
**Phase 1 (Immediate):** Disabled the `ImageProjectiveTransformV3` fast path entirely when `use_xla=False`. The function now always uses the fallback `_translate_images_simple` implementation for non-XLA mode, which includes broadcast logic to tile translations to match the images batch dimension via `tf.repeat`.

**Location:** ptycho/tf_helper.py:735-753, 756-839
- Lines 742-744: Compute `images_batch` and `trans_batch` for guard check
- Lines 746-753: Skip fast path when `use_xla=False` (unconditional fallback)
- Lines 813-828: Broadcast translations using `tf.cond` and `tf.repeat` when batch dimensions mismatch

**Rationale:** Attempting to conditionally use the fast path in graph mode (via `tf.cond` or `tf.debugging.assert_equal`) introduced complexity:
- `tf.debugging.assert_equal` raises at graph execution time, bypassing `try/except`
- Nested `tf.cond` with the existing XLA path adds control flow complexity
- The performance benefit of `ImageProjectiveTransformV3` vs `_translate_images_simple` is minor for typical batch sizes
- Disabling the fast path entirely when `use_xla=False` is the simplest, most reliable solution

### Outcome
- **Training:** TensorFlow training with gridsize=2, batch_size=4, n_groups=32 completes 1 full epoch without crashing (verified 2025-11-14)
- **Inference blocker:** A separate issue in `_translate_images_simple` (reshape with 0 values → shape [4]) occurs during post-training eval/inference. This is tracked separately and does not affect the core training crash fix.

### Tests
Regression tests added in `tests/tf_helper/test_translation_shape_guard.py`:
- `test_non_xla_translation_guard`: Mismatched batch dimensions fall back to `_translate_images_simple`
- `test_non_xla_translation_matching_batch`: Matching batch dimensions use fast path when available
- `test_reassemble_patches_position_real_gridsize2`: Full reassembly integration with gridsize=2

All 3 tests pass (pytest run 2025-11-14, 5.13s).

### Evidence
- Original blocker: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/.../tf_baseline/phase_c1/red/blocked_20251114T074039Z_tf_non_xla_shape_error.md`
- Scaled training GREEN (1 epoch): `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/.../tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log` (lines showing "8/8 steps" completion)
- Regression tests: `tests/tf_helper/test_translation_shape_guard.py` (3/3 passed)

### Status
**Resolved** (training), **Inference blocker open** (reshape 0→4 error in `_translate_images_simple` during eval)

## TF-XLA-BATCH-BROADCAST-001 - XLA Translation Batch Dimension Mismatch
**Category:** TensorFlow, XLA, Translation, Shape Handling
**Impact:** TensorFlow training/inference with gridsize > 1 when XLA enabled (default)
**Location:** ptycho/projective_warp_xla.py:270-285 (`translate_xla`)
**Date:** 2026-01-06

### Issue
When XLA is enabled (default) and gridsize > 1, TensorFlow training crashes with shape mismatch:
```
Input to reshape is a tensor with 389376 values, but the requested shape has 24336
```
Error signature: 389376 = 64 (batch) × 78 × 78 vs 24336 = 4 (gridsize²) × 78 × 78

**Root cause:** `translate_xla` built homography matrices M using `B = tf.shape(translations)[0]`, but when gridsize > 1, images are flattened from (b, N, N, C) to (b*C, N, N, 1). The M matrix batch dimension (from translations) didn't match the images batch dimension, causing reshape failures in the homography application.

### Fix (2026-01-06)
XLA-compatible batch broadcast using modular indexing with `tf.gather`:
```python
indices = tf.range(images_batch) % trans_batch
translations = tf.gather(translations, indices)
```

**Important:** Initial approach using `tf.repeat`/`tf.cond` failed XLA compilation with "Repeat/Tile must be a compile-time constant" error. The modular indexing approach avoids this by using ops that XLA can trace without requiring compile-time constant arguments.

**Location:** ptycho/projective_warp_xla.py:270-285

### Tests
Regression tests in `tests/tf_helper/test_translation_shape_guard.py`:
- `test_translate_xla_gridsize_broadcast`: Verifies broadcast with `use_jit=False`
- `test_translate_xla_gridsize_broadcast_jit`: Verifies broadcast with `use_jit=True` (XLA)

All 8 tests pass (pytest run 2026-01-06, 27.75s).

### Evidence
- Implementation: `plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/pytest_all_tests.log`
- Tests: `tests/tf_helper/test_translation_shape_guard.py` (5/5 passed)
- Model factory regression: `tests/test_model_factory.py` (3/3 passed)

### Status
**Resolved** — XLA batch broadcast fix implemented and verified
