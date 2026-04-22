# PtychoPINN Knowledge Base (NanoBragg Standard)

This ledger captures the most important lessons, conventions, and recurring issues discovered across the project. Before starting new work—or especially when debugging—consult this table to avoid repeating known mistakes.

| Finding ID | Date | Keywords | Synopsis | Evidence Pointer | Status |
| TORCH-PADDED-SIZE-001 | 2026-02-05 | pytorch, padding, jitter, config | Torch `get_padded_size` ignores `max_position_jitter` and uses `bigN` only; the jitter buffer is not part of the Torch padded-size contract. | `ptycho_torch/helper.py:310-320, tests/torch/test_padded_size_no_jitter.py` | Resolved |
| TORCH-REASSEMBLY-NORM-001 | 2026-02-05 | pytorch, reassembly, centermask, normalization, parity | Torch `reassemble_patches_position_real` must mirror TF `mk_norm` behavior: normalize with centermask counts plus 0.001 epsilon and **do not** hard-mask outputs. Hard-masking diverges from TF and changes C=1 object_big reassembly results. | `ptycho_torch/helper.py:100-137, tests/torch/test_reassemble_patches_position_real_c1.py` | Resolved |
| GRIDLINES-OBJECT-BIG-001 | 2026-02-05 | grid-lines, pytorch, config, object_big, parity | The grid-lines Torch runner must set `object_big=False` to match TF grid_lines defaults (`configure_legacy_params`). Leaving the ModelConfig default `object_big=True` triggers object-big reassembly on C=1 data and regresses hybrid_resnet integration metrics. | `scripts/studies/grid_lines_torch_runner.py:278-292, ptycho/workflows/grid_lines_workflow.py:184-205` | Resolved |
| GRIDLINES-PROBE-BIG-001 | 2026-02-05 | grid-lines, pytorch, config, probe_big, parity | The grid-lines Torch runner must set `probe_big=False` to match TF params.cfg defaults (`probe.big=False`). Propagating the PyTorch `probe_big=True` default flips the decoder path and regresses hybrid_resnet integration metrics. | `scripts/studies/grid_lines_torch_runner.py:278-293, ptycho/params.py:69-70` | Resolved |
| GRIDLINES-PROBE-PIPELINE-001 | 2026-03-31 | grid-lines, probe, preprocessing, metadata, dataset-contract | Probe preprocessing should be modeled as a normalized ordered pipeline (for example `smooth:0.5|pad:128|interp:256`) rather than as an ever-growing `probe_scale_mode` enum list. Different normalized pipelines define different dataset contracts and require fresh baselines/accepted-state lineages. Legacy `probe_scale_mode` values should normalize into the same canonical provenance record. | `ptycho/workflows/grid_lines_workflow.py, tests/test_grid_lines_workflow.py, docs/studies/lines_256_dataset.md` | Active |
| PDEBENCH-CNS-UPSAMPLER-001 | 2026-04-21 | pdebench, cns, hybrid_resnet, upsampler, pixelshuffle, bilinear, checkerboard | On the post-skip-add capped same-shell PDEBench `2d_cfd_cns` upsampler compare, `pixelshuffle` beat both transpose and bilinear on aggregate denormalized error, while bilinear showed the worst high-frequency penalty and stripe-like directional artifacts. Promote `pixelshuffle` into `hybrid_resnet_cns`; keep transpose and bilinear manual-only. | `docs/plans/2026-04-21-hybrid-upsampler-artifact-study-results.md, .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep-20260421T221400Z/comparison_summary.json, .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep-20260421T221400Z/gallery_sample0_full_compare.png` | Active |
| PDEBENCH-CNS-BOTTLENECK-001 | 2026-04-21 | pdebench, cns, hybrid_resnet, spectral_resnet, ffno, bottleneck, skip-add | On the first capped same-shell PDEBench `2d_cfd_cns` compare, the shared-spectral bottleneck beat both the canonical local `hybrid_resnet_cns` bottleneck and the new FFNO-close bottleneck on aggregate denormalized error. The FFNO-close row did not justify promotion into default CNS bundles from this run. | `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md, .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep/comparison_summary.json` | Active |
| PDEBENCH-CNS-SPECTRAL-40EP-001 | 2026-04-21 | pdebench, cns, spectral_resnet, fno, hybrid_resnet, unet, 40-epoch, capped-compare | On the capped 40-epoch PDEBench `2d_cfd_cns` follow-up, `spectral_resnet_bottleneck_base` beat the earlier 40-epoch `hybrid_resnet_base`, `fno_base`, and `unet_strong` rows on the main reported eval metrics (`err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low/mid/high`). Treat this as decision-support evidence, not benchmark-complete CNS ranking. | `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md, .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep/comparison_summary.json, .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep/gallery_sample0_spectral_vs_fno_unet_40ep.png` | Active |
| PDEBENCH-CNS-SPECTRAL-SHARE-001 | 2026-04-21 | pdebench, cns, spectral_resnet, weight-sharing, bottleneck, skip-add | On the capped same-shell PDEBench `2d_cfd_cns` shared-vs-non-shared spectral compare, disabling spectral weight sharing improved every tracked eval metric (`err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low/mid/high`) over the shared row, but both rows still showed visible pressure striping. Keep the non-shared row manual-only and treat it as decision-support evidence, not a default-profile promotion. | `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_weight_sharing_summary.md, .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/comparison_summary.json, .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/gallery_sample0_error.png` | Active |
| PDEBENCH-CNS-GNOT-001 | 2026-04-22 | pdebench, cns, gnot, spectral_resnet, equal-footing, dgl, environment | Official GNOT can run locally on the PDEBench `2d_cfd_cns` contract only after moving to a CUDA-compatible `torch 2.4.1 + dgl 2.4.0` environment (`ptycho311_2`). On the first capped equal-footing `512/64/64`, `8`-window, `10`-epoch compare, `gnot_cns_base` trailed `spectral_resnet_bottleneck_base` badly on aggregate denormalized error (`relative_l2 0.24565` vs `0.08598`). Treat that result as a fairness-probe outcome only; subsequent GNOT reruns should use the paper-style defaults (`hidden=128`, `relative_l2`, `AdamW`, `OneCycleLR`, `lr=1e-3`). | `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md, .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z/comparison_summary.json, .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot_source.json` | Active |
| TF-REPEATED-MODEL-OOM-001 | 2026-04-13 | tensorflow, gpu-memory, keras, repeated-training, model-singleton, probe-mischaracterization | Repeated TensorFlow model training runs in one Python process can OOM/fragment the GPU allocator even on normal Conv2D activation-gradient tensors. In the probe-mischaracterization full run, `[16,64,64,64]` came from `conv_grad_input_ops.cc` during `model.fit` with `gridsize=1` and input shape `(16,64,64,1)`, so it is not a data/coordinate broadcast error. For repeated condition studies, clear Keras and delete `ptycho.model.autoencoder`, `ptycho.model.diffraction_to_obj`, and `ptycho.model.autoencoder_no_nll` module references before rebuilding models; drop unneeded retained models such as smoke `baseline_model`; use per-condition subprocess isolation if cleanup is insufficient. | `artifacts/revision_studies/probe_mischaracterization/logs/full_20260412T232814Z.log, scripts/studies/probe_mischaracterization_stress_test.py:816-833, ptycho/train_pinn.py:85-90, .artifacts/revision_studies/probe_mischaracterization/implementation_review.md` | Active |
| LINES256-PROPOSAL-RECOVERY-001 | 2026-04-01 | lines_256, controller, proposal, resume, workflow | The v2 `lines_256` controller must treat proposal generation as a transaction. Setting `proposal_running` before durable candidate metadata exists and then hard-requiring `candidate_metadata.json` on resume makes provider-capacity or interruption failures unrecoverable. Fix by writing controller-owned `proposal_attempt.json` / `proposal_result.json`, making missing proposal metadata retryable, and moving smoke execution back under deterministic controller ownership. | `scripts/studies/lines_256_session_controller.py, tests/studies/test_lines_256_session_controller.py, state/lines_256_arch_improvement_v2/sessions/20260331T015545Z/iterations/035/*` | Active |
| LINES256-SESSION-BRANCH-001 | 2026-04-06 | lines_256, controller, git, session-branch, detached-head, provenance | The v2 `lines_256` controller should keep each dedicated run checkout on a named session branch such as `lines256/session/<session_id>`. Branch names are operational handles for human inspection and resume; exact commit SHAs remain the authoritative scientific provenance for accepted state and source candidates. Detached checkouts are recoverable, but they should not be the intended steady state. | `scripts/studies/lines_256_session_controller.py, tests/studies/test_lines_256_session_controller.py, docs/studies/lines_256_controller_loop.md` | Active |
| LINES256-IMPORT-PROVENANCE-001 | 2026-04-01 | lines_256, controller, pythonpath, import-provenance, source-candidates, resume | Controller-owned smoke/scored child runs must treat the session repo root as the authoritative import root. Preserving ambient `PYTHONPATH` can make a run-checkout source candidate score code from a different checkout, producing false ties or misleading results without crashing. Keep PATH `python` per `PYTHON-ENV-001`, but overwrite `PYTHONPATH` with the session repo root and record runtime provenance in invocation artifacts. | `scripts/studies/lines_256_session_controller.py, scripts/studies/invocation_logging.py, scripts/studies/grid_lines_torch_runner.py, scripts/studies/run_lines_256_arch_experiment.py, tests/studies/test_lines_256_session_controller.py, tests/test_grid_lines_invocation_logging.py, tests/torch/test_grid_lines_torch_runner.py` | Active |
| LINES256-CONTROLLER-PROVENANCE-001 | 2026-04-01 | lines_256, controller, runtime-provenance, compatibility, recovery, invalid_execution | Source-execution integrity for `lines_256` must be proven by controller-owned launch/import-probe artifacts, not by child `invocation.json` schema alone. Missing child runtime-provenance fields in an older run checkout are a recoverable compatibility problem, not scientific evidence and not a reason to auto-complete the session. | `scripts/studies/lines_256_session_controller.py, tests/studies/test_lines_256_session_controller.py, state/lines_256_arch_improvement_v2/sessions/20260331T015545Z/iterations/036/*` | Active |
| LINES256-CTRL-PATH-001 | 2026-03-30 | lines_256, controller, interpreter, subprocess, path, tensorflow | The v2 `lines_256` controller must stabilize the scored-command subprocess PATH so plain `python ...` resolves to the same runtime as the controller itself. `bash -lc` plus ambient shell startup can drift to a different interpreter, causing false `ModuleNotFoundError: tensorflow` crashes even when smoke runs succeed. Fix the controller boundary, not the persisted command strings. | `scripts/studies/lines_256_session_controller.py, tests/studies/test_lines_256_session_controller.py, state/lines_256_arch_improvement_v2/sessions/20260330T001026Z/iterations/046/*.log` | Active |
| PROBE-MASK-DEFAULT-001 | 2026-02-20 | pytorch, probe-mask, hybrid_resnet, regression, integration-test | Hybrid-resnet metric regression on `fno-stable` was introduced when Torch probe masking changed from effectively off-by-default to on-by-default (`db1e43f9`). Hard-vs-soft edge is secondary; disabling default masking restores the integration gate. | `db1e43f9 vs 8dac52fc, ptycho_torch/config_params.py, ptycho_torch/model.py, tests/torch/test_grid_lines_hybrid_resnet_integration.py` | Active |
| REASSEMBLY-M-CONTRACT-001 | 2026-03-04 | pytorch, reassembly, external_raw_npz, M, crop-border, shift-sum | Position reassembly must have a single trim owner: forward full patches to `tf_helper.reassemble_position` and derive `M_effective` from `M_requested` + `position_crop_border`. Runner-side pre-trim plus tf_helper trim caused broadcast-shape failures. | `scripts/studies/grid_lines_torch_runner.py, tests/torch/test_grid_lines_torch_runner.py, docs/workflows/pytorch.md, docs/debugging/TROUBLESHOOTING.md` | Resolved |
| REASSEMBLY-BATCH-001 | 2025-11-13 | batched-reassembly, translation-layer, shape-mismatch, dense-view, graph-mode | Batched patch reassembly failed when processing large (5k+) dense datasets in TensorFlow graph mode (tf.while_loop). The issue was using `tf.shape(canvas)` (dynamic runtime values) instead of the static `padded_size` parameter for resize operations. TensorFlow's `shape_invariants` requirement in while_loop demands static shape specifications at graph compilation time. Fixed by using the static `padded_size` integer directly in `tf.image.resize_with_crop_or_pad` instead of extracting canvas dimensions dynamically. This ensures shape consistency between the canvas and batch_result tensors during while_loop compilation and execution. `ReassemblePatchesLayer` now auto-selects batched reassembly when `total_patches > 64` to keep GPU memory bounded. | `ptycho/tf_helper.py:939-965, ptycho/custom_layers.py:149-164, tests/study/test_dose_overlap_comparison.py:760-817` | Resolved |
| XLA-VECTORIZE-001 | 2025-11-13 | xla, vectorization, batch-mismatch, mk_reassemble_position_real | The `mk_reassemble_position_real` function's vectorized path failed with "Dimensions must be equal, but are 5088 and 159" when processing dense datasets (>1000 patches) through `translate_xla`. Root cause: the `_vectorised()` branch processes all patches in a single call to `translate()`, which is decorated with `@complexify_function`. For complex tensors (gridsize=2 → 4 channels), this splits into real/imag parts and the XLA bilinear interpolation path receives mismatched batch dimensions during the internal chunking. Fixed by adding an explicit patch count limit (`max_vectorized_patches=1000`) alongside the existing memory cap, forcing large patch counts to use the `_streaming()` path with tf.while_loop chunking (chunk_size=1024). | `ptycho/tf_helper.py:1183-1223` | Resolved |
| XLA-DYN-DOT-001 | 2025-11-13 | xla, einsum, dynamic-shape, projective-warp | TF XLA JIT failed during inference with DynamicPadder RET_CHECK on dot/einsum when applying homography to the sampling grid (dynamic H×W). Replaced `tf.einsum("bij,bhwj->bhwi", M, grid)` with explicit broadcasted multiplies/adds to compute `(sx, sy, w)`; removes dynamic dot and unblocks XLA. Also removed a `tf.print` in Translation layer that emitted a PrintV2 op. | ptycho/projective_warp_xla.py:90, ptycho/tf_helper.py:824 | Resolved |
| PTYCHOVIT-FRAME-001 | 2026-02-10 | ptychovit, interop, coords, object-shape, frame | PtychoViT interop quality regressed when para `object` used clipped `YY_ground_truth` while scan positions came from full-frame global offsets, and when absolute top-left coords were passed without converting to the centered frame expected by upstream loader (`data.py` adds object origin internally). This caused severe patch-footprint mismatch/OOB sampling. Contract fix: prefer `YY_full` for `object` when available and normalize positions into centered frame before writing HDF5. | `ptycho/interop/ptychovit/convert.py:52-169, docs/workflows/ptychovit.md` | Resolved |
| PTYCHOVIT-ASSEMBLY-001 | 2026-02-10 | ptychovit, bridge, reconstruction, stitching, parity | Bridge inference currently reconstructs with scan-wise patch mean instead of position-aware stitching. Upstream PtychoViT reconstruction/visualization uses scan-position placement (`place_patches_fourier_shift`) and occupancy normalization. Mean aggregation violates this assumption and can yield flat low-information reconstructions even when normalization/probe contracts are valid. | `scripts/studies/ptychovit_bridge_entrypoint.py:254-269, /home/ollie/Documents/ptycho-vit/training.py:165-223` | Active |
| PROBE-MULTIMODE-001 | 2026-03-04 | nersc, probe, multimode, incoherent-aggregate, compatibility | NERSC paired-HDF5 conversion now defaults to `probe_mode_policy=incoherent_aggregate` (instead of mode-0 truncation) when collapsing multimode probes for external-raw NPZ. Compatibility fallback `first_mode` remains available and prep manifests record `probe_mode_policy`, `probe_source_shape`, and `probe_mode_power_weights` for auditability. | `scripts/studies/nersc_pair_adapter.py, scripts/studies/prepare_nersc_hybrid_dataset.py, scripts/studies/nersc_orchestration.py, docs/workflows/ptychovit.md` | Active |
| CONVENTION-001 | 2025-10-16 | architecture, legacy, global-state | The codebase consists of a legacy grid-based system tied to `params.cfg` and a modern coordinate-based system; pick the right side before changing anything. | [Link](DEVELOPER_GUIDE.md#1-the-core-concept-a-two-system-architecture) | Active |
| POLICY-001 | 2025-10-17 | policy, PyTorch, dependencies, mandatory | PyTorch (torch>=2.2) is now a mandatory dependency for PtychoPINN as of Phase F (INTEGRATE-PYTORCH-001). All code in `ptycho_torch/` and `tests/torch/` assumes PyTorch is installed. Torch-optional execution paths were removed; modules raise actionable RuntimeError if torch import fails. Tests in `tests/torch/` are automatically skipped in TensorFlow-only CI environments via directory-based pytest collection rules in `tests/conftest.py`, but will fail with actionable ImportError messages if PyTorch is missing in local development. Migration rationale and implementation evidence documented in governance decision and Phase F implementation logs. | [Link](../docs/plans/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md) | Active |
| ANTIPATTERN-001 | 2025-10-16 | imports, side-effects, debugging | Import-time side effects (e.g., loading data in module scope) caused hidden crashes; always push work into functions with explicit arguments. | [Link](DEVELOPER_GUIDE.md#21-anti-pattern-side-effects-on-import) | Active |
| CONFIG-001 | 2025-10-16 | params.cfg, initialization-order | `update_legacy_dict(params.cfg, config)` must run before any legacy module executes; missing this broke gridsize sync and legacy interop. | [Link](debugging/QUICK_REFERENCE_PARAMS.md#⚠️-the-golden-rule) | Active |
| CONFIG-002 | 2025-10-20 | execution-config, cli, params.cfg | PyTorch execution configuration (PyTorchExecutionConfig) controls runtime behavior only and MUST NOT populate params.cfg. Only canonical configs (TrainingConfig, InferenceConfig) bridge via CONFIG-001. CLI helpers auto-detect accelerator default='auto' and validate execution config fields via factory integration. Execution config applied at priority level 2 (between explicit overrides and dataclass defaults). | [Link](../docs/plans/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_redline.md) | Active |
| CONFIG-LOGGER-001 | 2025-10-24 | logger, execution-config, lightning, mlflow | PyTorch training uses CSVLogger by default (`PyTorchExecutionConfig.logger_backend='csv'`) to capture train/validation metrics from Lightning `self.log()` calls, replacing prior `logger=False` which discarded metrics. Allowed backends: `csv` (zero deps, CI-friendly), `tensorboard` (requires tensorboard from TF install), `mlflow` (requires mlflow package + server), `none` (disable). Legacy `--disable_mlflow` flag deprecated with DeprecationWarning mapping to `--logger none`. MLflow migration to Lightning MLFlowLogger tracked as Phase EB3.C4 backlog. Decision rationale and implementation evidence in governance decision approval record. | [Link](../docs/plans/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md) | Active |
| EXEC-ACCUM-001 | 2025-11-13 | pytorch, lightning, manual-optimization | Lightning manual optimization (PtychoPINN_Lightning sets `automatic_optimization=False`) is incompatible with `Trainer(accumulate_grad_batches>1)`. Passing `--torch-accumulate-grad-batches` > 1 causes `MisconfigurationException: Automatic gradient accumulation is not supported for manual optimization`. Treat the flag as unsupported for PyTorch backend until Lightning support lands: fail fast or clamp to 1 with a clear warning. | `../docs/plans/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/red/blocked_20251113T184100Z_manual_optim_accum.md` | Active |
| DATA-SUP-001 | 2025-11-13 | supervised, labels, data-contract | Supervised PyTorch mode expects dataloader batches with `label_amp`/`label_phase`. Experimental fly001 NPZ datasets and the minimal CLI fixture omit those keys, leading to `KeyError: 'label_amp'` in `PtychoPINN_Lightning.training_step`. Either supply labeled synthetic NPZ (per DATA-001) or block supervised runs with an actionable error until labeled data is available. | `../docs/plans/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/red/blocked_20251113T184300Z_supervised_data_contract.md` | Active |
| DEVICE-MISMATCH-001 | 2025-11-13 | pytorch, inference, accelerator | RESOLVED — Commit 85478a67 updates `scripts/inference/inference.py` and `ptycho_torch/inference.py` to call `model.to(device)`/`model.eval()` in both the CLI and helper, adds regression tests, and proves CUDA inference succeeds (`cli/pytorch_cli_smoke_training/inference_cuda.log`, `green/pytest_pytorch_inference_device.log`). | `../docs/plans/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt` | Resolved |
| DEVICE-HANDOFF-001 | 2026-02-16 | pytorch, lightning, inference, accelerator, performance | In train→inference pipelines, never infer runtime device from the model’s current parameter device alone. After `Trainer.fit(...)`, explicitly resolve the intended inference accelerator and call `model.to(device)` before any forward loop. Missing this handoff can silently push inference to CPU and create major runtime regressions. | `scripts/studies/grid_lines_torch_runner.py, ptycho_torch/workflows/components.py:1115-1167` | Active |
| BUG-TF-001 | 2025-10-16 | shape-mismatch, gridsize, tests | Gridsize > 1 yields channel mismatches unless `params.cfg['gridsize']` is populated before `generate_grouped_data`; verify config vs params values. | [Link](debugging/TROUBLESHOOTING.md#shape-mismatch-errors) | Active |
| DATA-001 | 2025-10-16 | data-contract, npz, io | All NPZ datasets must follow the canonical specification (keys, dtypes, normalization); deviations caused silent inference failures. | [Link](../specs/data_contracts.md) | Active |
| NORMALIZATION-001 | 2025-10-16 | scaling, physics, preprocessing | Three independent normalization systems (physics, statistical, display) must never be mixed; applying scaling in the wrong stage created double-scaling bugs. | [Link](DEVELOPER_GUIDE.md#35-normalization-architecture-three-distinct-systems) | Active |
| STUDY-001 | 2025-10-16 | fly64, baseline, generalization | On fly64 experiments the baseline model outperformed PtychoPINN by ~6–10 dB, contradicting expectations and motivating architecture review. | [Link](FLY64_GENERALIZATION_STUDY_ANALYSIS.md#key-findings) | Active |
| MODULE-SINGLETON-001 | 2026-01-07 | model, singleton, import-time, gridsize, shape-mismatch | `ptycho.model.autoencoder` is a module-level singleton created at import time using current `params.cfg['gridsize']`. Changing gridsize after import does NOT recreate the model—the architecture is frozen with the import-time gridsize. This causes shape mismatches when simulation uses gridsize=1 but training uses gridsize=2. **CONFIG-001 alone is insufficient**—you must use factory functions to create fresh models with the correct architecture. **FULLY RESOLVED (2026-01-07):** Phase B lazy loading via `__getattr__` (ptycho/model.py:867-890) prevents import-time model creation. Phase C removed all XLA workarounds (`USE_XLA_TRANSLATE=0`, `TF_XLA_FLAGS`, `run_functions_eagerly`) from scripts and tests. XLA re-enablement verified by spike test. For custom workflows, use `create_compiled_model()` (returns compiled model ready for training) or `create_model_with_gridsize()` (uncompiled). | `ptycho/model.py:867-890, tests/test_model_factory.py, docs/debugging/TROUBLESHOOTING.md#model-architecture-mismatch-after-changing-gridsize` | Resolved |
| ACCEPTANCE-001 | 2025-11-11 | phase-d, geometry, spacing, dense-view | Dense fly64 overlap runs cannot meet the legacy 10 % minimum acceptance—bounding-box math caps the dense view at ≈0.96 % (42/5088) for the 38.4 px threshold—so `generate_overlap_views` must compute a geometry-aware acceptance floor (area ÷ packing discs) and log `geometry_acceptance_bound` + `effective_min_acceptance` in the metrics bundle before proceeding. | [Link](../docs/plans/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/summary.md) | Active |
| OVERSAMPLING-001 | 2025-10-16 | oversampling, gridsize, combinatorics | Oversampling only works when `gridsize > 1` and `K > C`; otherwise requested groups can never exceed raw images. | [Link](debugging/TROUBLESHOOTING.md#oversampling-not-working) | Active |
| MIGRATION-001 | 2025-10-16 | params-removal, refactor, strategy | 66+ files still depend on `params.cfg`; migration plan is to eliminate new uses, document remaining ones, then remove the dependency. | [Link](debugging/QUICK_REFERENCE_PARAMS.md#the-66-file-problem) | Active |
| PROCEDURE-001 | 2025-10-16 | review, defensive-coding | Flag hidden `params` reads and undocumented dependencies during code review; insist on explicit parameters or documented prerequisites. | [Link](debugging/QUICK_REFERENCE_PARAMS.md#red-flags-in-code-review-🚩) | Active |
| FORMAT-001 | 2025-10-17 | data-contract, npz, legacy-format, transpose | Some NPZ datasets use legacy (H,W,N) diffraction array format instead of DATA-001 compliant (N,H,W); caused IndexError in PyTorch dataloader when nn_indices referenced global positions beyond first dimension. Auto-transpose heuristic added to both `_get_diffraction_stack()` and `npz_headers()` to detect and fix at runtime. | [Link](../docs/plans/INTEGRATE-PYTORCH-001/reports/2025-10-17T230724Z/callchain/summary.md) | Active |
| TYPE-PATH-001 | 2025-11-06 | pytorch, path, type-safety, config | PyTorch workflows failed with AttributeError/TypeError when string paths from TrainingConfig were passed to functions expecting Path objects. Root cause: ptycho_torch/workflows/components.py:650,682 called functions with raw string paths (config.train_data_file, config.output_dir) instead of wrapping with Path(). Symptom: 'str' object has no attribute 'exists' and unsupported operand type(s) for /. Fix: Wrap path parameters with Path() at call sites before invoking downstream helpers. Prevention: Normalize TrainingConfig/InferenceConfig path fields via Path() in constructors or apply runtime coercion at module boundaries. | [Link](../docs/plans/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T150500Z/phase_e_training_bundle_real_runs_exec/summary.md) | Active |
| PHASEC-METADATA-001 | 2025-11-06 | phase-c, metadata, validator, orchestration | `run_phase_g_dense.py` still expects Phase C outputs in legacy `dose_*_{train,test}/fly64_<split>_simulated.npz` directories; the refreshed generator now writes `data/phase_c/dose_<dose>/{patched,patched_train,patched_test}.npz` with `_metadata`. The guard falsely blocks dense relaunches until it scans the new layout and ensures `_metadata` on the patched splits. | [Link](../docs/plans/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T081826Z/phase_c_metadata_guard_blocker/cli/blocker_phase_c_metadata.log) | Active |
| TEST-CLI-001 | 2025-11-10 | tdd, cli-validation, orchestration, regression-guard | Orchestrator-level CLI validation requires explicit test fixtures for RED/GREEN cycles and must enforce complete log bundles (phase banners + SUCCESS sentinel + dose/view-specific filenames). When adding validation to artifact verifiers, create both RED (missing/wrong patterns) and GREEN (complete artifacts) test cases with realistic log content including all required markers. **Filename patterns MUST match orchestrator output**: phase_e_baseline_gs1_dose{dose}.log, phase_e_{view}_gs2_dose{dose}.log, phase_f_{view}_train.log, etc. (not generic phase_e_baseline.log). Helper logs (aggregate_report_cli.log, metrics_digest_cli.log) and completion sentinels ("complete" marker) are also required. Test isolation tip: GREEN tests should validate only the specific check under test, not entire bundle completeness (use focused assertions on check['valid'] field rather than exit_code==0). | [Link](../docs/plans/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/verify_dense_pipeline_artifacts.py:437-596) | Active |
| PREVIEW-PHASE-001 | 2025-11-11 | highlights, preview, validator, phase-only | The dense Phase G preview artifact (`analysis/metrics_delta_highlights_preview.txt`) must contain **only** the four phase deltas (MS-SSIM/MAE vs Baseline/PtyChi) with explicit ± signs; any `amplitude` text or extra tokens indicates corruption and must fail validation. Existing verifier logic only checked for the presence of formatted numbers, so previews that regressed back to amplitude-inclusive lines could slip through. Harden `validate_metrics_delta_highlights` (and tests) to enforce phase-only content and surface actionable metadata in reports. | [Link](../docs/plans/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T005802Z/phase_g_dense_full_execution_real_run/plan/plan.md) | Active |
| METRICS-NAMING-001 | 2025-11-14 | phase-g, metrics, canonical-models | `report_phase_g_dense_metrics.py` only recognizes the canonical model IDs `PtychoPINN`, `Baseline`, and `PtyChi`. When `scripts.compare_models.py` writes friendly labels such as `"Pty-chi (pty-chi)"`, the reporter treats `PtyChi` as missing and aborts even if the data exists. Keep the canonical names (or add a mapping) before invoking the reporter, otherwise PREVIEW/verification evidence never materializes. | `../docs/plans/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/cli/aggregate_report_cli.log` | Active |
| BASELINE-OFFSET-001 | 2025-11-13 | baseline, offsets, numerical-stability, compare-models | Baseline model inference returned all-zero outputs on test split due to uncentered offsets causing numerical instability. Train/test splits had very different offset means (train≈185px, test≈273px, 87px delta), and the baseline model (trained with zero-mean offsets) failed on the large distribution shift. Fixed by centering `baseline_offsets` to zero-mean in `prepare_baseline_inference_data()` before inference: `centered_offsets = flattened_offsets_np - offset_mean`. After centering, test split baseline outputs changed from mean=0.000000 (all zeros) to mean=0.079082 (16086 nonzero pixels). Added `--baseline-debug-limit` and `--baseline-debug-dir` CLI flags for fast debugging with NPZ+JSON artifact dumps. | `scripts/compare_models.py:247-258, docs/plans/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/blocker_resolved_offset_centering.md` | Resolved |
| BASELINE-CHUNKED-001 | 2025-11-13 | baseline, gpu-memory, chunked-inference, compare-models | Dense test split Baseline inference failed with TensorFlow `ResourceExhaustedError` when processing 5088 groups in a single batch, leaving blank Baseline metric rows and blocking Phase G pipeline. Train split succeeded with 78.7M nonzero pixels, but test split OOM'd before completing inference. Root cause: TensorFlow GPU memory exhaustion during large-batch inference. Fixed by implementing chunked Baseline inference with configurable `--baseline-chunk-size` (number of groups per chunk) and `--baseline-predict-batch-size` (batch size within each chunk, default 32). Chunks are processed sequentially with explicit `tf.keras.backend.clear_session()` between chunks to release GPU memory. Per-chunk DIAGNOSTIC logging added to track mean/max/nonzero stats. Automatic fallback: when `--baseline-chunk-size` is None, uses original single-shot path. Chunked path includes try/except for `ResourceExhaustedError` with actionable guidance to reduce chunk/batch sizes. | `scripts/compare_models.py:1077-1140, docs/plans/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251116T010000Z_test_baseline_oom.md` | Active |
| BASELINE-CHUNKED-002 | 2025-11-16 | baseline, gpu-memory, data-loading, chunked-container | Chunked Baseline inference alone is insufficient: `scripts/compare_models.py` still instantiates a full `PtychoDataContainer` before entering the chunk loop, so the dense-test rerun continues to crash with `ResourceExhaustedError: failed to allocate memory [Op:Cast]` inside `ptycho/loader.py:141 → combine_complex()` (`analysis/dose_1000/dense/test/logs/logs/debug.log:299-360`). **RESOLVED 2025-11-16**: Refactored chunked Baseline mode to slice RawData per chunk (`slice_raw_data`, `dataclasses.replace(final_config, n_groups=n_chunk)`), create chunk-scoped `PtychoDataContainer`s, and use concatenated `pinn_offsets` for alignment so the full container never materializes on GPU. Implementation at scripts/compare_models.py:1152-1197 (Baseline chunking) and 1431-1442 (alignment with pinn_offsets). | `scripts/compare_models.py:1152-1197,1431-1442, docs/plans/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251116T010000Z_test_baseline_oom.md` | Resolved |
| SINGLETON-SAVE-001 | 2026-01-06 | model_manager, save, singleton, train_pinn, untrained-model | `model_manager.save()` hardcodes saving `model.autoencoder` and `model.diffraction_to_obj` (module-level singletons). When `train_pinn.train()` was changed (commit d17a351f) to create a NEW model via `create_compiled_model()` and train it, the trained model was returned but NEVER assigned back to the singletons. Thus `model_manager.save()` saved the UNTRAINED singleton, and inference loaded garbage (saturated ~0.5 amplitude, flat phase). **Root cause identified via git bisect:** last good commit 7d558ead used singleton directly, first bad commit d17a351f used factory but didn't update singleton. **Fixed by updating module singletons after model creation in `train_pinn.train()`: `model.autoencoder = model_instance; model.diffraction_to_obj = diffraction_to_obj`.** | `ptycho/train_pinn.py:86-90, ptycho/model_manager.py:463-464` | Resolved |
| STITCH-GRIDSIZE-001 | 2026-01-26 | stitching, gridsize, guard, data-preprocessing | `data_preprocessing.stitch_data()` has an incorrect ValueError guard that rejects gridsize=1: `if gridsize == 1: raise ValueError("stitch_data requires gridsize >= 2")`. This guard was added in commit aa80f15b (July 2025) but the stitching math works correctly for gridsize=1—it produces a 1x1 grid. The guard prevents valid gridsize=1 workflows that were working before this commit. **Workaround:** `scripts/studies/grid_resolution_study.py` implements `stitch_predictions()` that bypasses this guard. **Fix required:** Remove the ValueError guard from `ptycho/data_preprocessing.py:152-156`. | `ptycho/data_preprocessing.py:152-156, scripts/studies/grid_resolution_study.py:stitch_predictions` | Active |
| DATALOADER-EXPID-001 | 2026-01-27 | pytorch, dataloader, experiment_id, tensor-shape | Lightning dataloader `PtychoLightningDataset` returned `experiment_id` as `torch.zeros(1, dtype=torch.long)` (shape `(1,)`), which after DataLoader collation became `(batch_size, 1)`. The model's `compute_loss` expected `experiment_id` with shape `(batch_size,)` for indexing into alpha/beta parameters. **Fixed:** Changed to `torch.tensor(0, dtype=torch.long)` (scalar), which collates to `(batch_size,)`. | `ptycho_torch/workflows/components.py:456` | Resolved |
| DATALOADER-SCALE-001 | 2026-01-27 | pytorch, dataloader, scaling, broadcasting | Lightning dataloader returned scaling constants (`rms_scaling_constant`, `physics_scaling_constant`) as scalars after squeezing, which collated to shape `(batch_size,)`. The model's `scale()` function multiplies `x * scale_factor` where `x` has shape `(batch, C, H, W)`, requiring `scale_factor` to be `(batch, 1, 1, 1)` for broadcasting. **Fixed:** Scaling constants now have shape `(1, 1, 1)` which collates to `(batch_size, 1, 1, 1)`. | `ptycho_torch/workflows/components.py:437-449` | Resolved |
| LIGHTNING-STRATEGY-001 | 2026-01-27 | pytorch, lightning, strategy, execution-config | `_train_with_lightning` checked `pt_training_config.strategy` (default `'ddp'`) to determine whether to unpack dataloaders, but should use `execution_config.strategy` (runtime setting). When `execution_config.strategy='auto'` but `pt_training_config.strategy='ddp'`, the dataloader unpacking was skipped incorrectly, causing `UnboundLocalError: train_loader`. **Fixed:** Now uses `execution_config.strategy` for the runtime check. | `ptycho_torch/workflows/components.py:808-813` | Resolved |
| FORWARD-SIG-001 | 2026-01-26 | pytorch, fno, hybrid, forward-signature, inference | FNO and Hybrid architectures require **single-input forward signature** `model(X)` where X is diffraction patterns only. Unlike CNN architectures which may accept `model(X, coords)` for position encoding, FNO/Hybrid models learn spatial relationships implicitly through spectral convolutions and do NOT accept coordinate inputs. Passing coords to FNO/Hybrid causes shape errors or silent misuse. **Contract:** `run_torch_inference()` in `grid_lines_torch_runner.py` checks `cfg.architecture in ('fno', 'hybrid')` and passes X only. | `scripts/studies/grid_lines_torch_runner.py:246-255, tests/torch/test_grid_lines_torch_runner.py:TestForwardSignatureEnforcement` | Active |
| OUTPUT-COMPLEX-001 | 2026-01-26 | pytorch, fno, hybrid, output-contract, complex | FNO/Hybrid models output predictions in **real/imag format** with shape `(..., 2)` where last dimension contains `[real, imag]`. The `to_complex_patches()` helper in `grid_lines_torch_runner.py` converts this to complex64 for downstream physics consistency checks. Runner returns `predictions_complex` key when conversion is applied. **Contract:** If `predictions.shape[-1] == 2`, apply `to_complex_patches()` before metrics computation. | `scripts/studies/grid_lines_torch_runner.py:32-49, tests/torch/test_grid_lines_torch_runner.py:TestOutputContractConversion` | Active |
| STABLE-GAMMA-001 | 2026-01-29 | pytorch, stable_hybrid, initialization, instance-norm | Zero-gamma InstanceNorm in `StablePtychoBlock` keeps residual branches near-identity for tens of epochs: Stage A runs showed all `norm.weight` tensors stayed |0.09|, amplitude std collapsed to ~3e-8, and reconstructions became constant despite healthy gradients elsewhere. **Resolved by LayerScale** (Phase 5): replacing zero-gamma with InstanceNorm(weight=1) + LayerScale(init=1e-3) freed norm weights (mean~0.82-1.0). See STABLE-LS-001 for the follow-on failure mode. | [Link](../docs/plans/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/stage_a_summary.md) | Resolved |
| STABLE-LS-001 | 2026-01-29 | pytorch, stable_hybrid, layerscale, training-collapse | LayerScale (init=1e-3) in `StablePtychoBlock` solved zero-gamma stagnation (STABLE-GAMMA-001) — norm weights train to healthy values (mean~0.82-1.0). However, the stable arm still collapses to constant amplitude (std=0, SSIM=0.277). **Phase 7 LR sweep (3 arms) confirmed collapse is LR-independent.** **Phase 8 optimizer sweep confirmed collapse is optimizer-independent:** SGD (mom=0.9, LR=3e-4, WarmupCosine) and AdamW (wd=0.01, LR=3e-4, WarmupCosine) both produce identical metrics to Adam — best_val=0.0237, final_val=0.198, amp_ssim=0.277. This eliminates all training-dynamics hypotheses (LR, clipping, optimizer). **Conclusion: collapse is architectural.** The Norm-Last + LayerScale topology in `StablePtychoBlock` is fundamentally incompatible with this physics task. The control arm (`hybrid` PtychoBlock with PreNorm topology) trains stably. Next: crash hunt (Phase 9) to test stochastic stability of the control arm at depth, or topology revert. | [Link](../docs/plans/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T050000Z/stage_a_optimizer_summary.md) | Active |
| FNO-DEPTH-001 | 2026-01-28 | pytorch, fno, hybrid, scalability, memory, channel-doubling | `HybridUNOGenerator` doubles hidden channels at each encoder level (`ch *= 2`), causing exponential parameter growth: `fno_blocks=4` → 17M params (0.07 GB), `fno_blocks=8` → 4.4B params (17.7 GB). At 8 blocks, bottleneck reaches 4096 channels and spectral conv weights (`channels^2 × modes^2`) dominate. Model cannot load on RTX 3090 (24 GB). Deep stress tests require either capping channel growth (e.g., max 512), using constant-width blocks, or reducing `hidden_channels`. `fno_blocks=6` (bottleneck=1024, ~280M params) is a feasible intermediate. | [Link](../docs/plans/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/stage_b_summary.md) | Active |
| FNO-DEPTH-002 | 2026-01-29 | pytorch, fno, hybrid, channel-cap, depth-scaling, stability | `max_hidden_channels=512` cap stabilizes `fno_blocks=6` (276M params, 1.1 GB model, ~10.4 GB VRAM on RTX 3090). 50 epochs with no NaNs, grad_norm median=8.56, p99=22.85. `fno_blocks=8` with same cap still OOMs (18 GiB activation memory at model-to-device). Practical depth limit on 24 GB GPU: 6 blocks with cap=512, hidden_channels=64. 8 blocks needs gradient checkpointing or larger GPU. | [Link](../docs/plans/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T210000Z/stage_b_summary.md) | Active |
| STABLE-CRASH-DEPTH-001 | 2026-01-29 | pytorch, hybrid, stochastic-crash, depth-sweep, seed-sensitivity | **Crash Hunt (Phase 9):** Control hybrid (norm clip 1.0, max_hidden_channels=512) depth sweep across seeds 20260128/29/30. **DEPTH_CRASH=4** (unexpected — planning assumption was depth 6). P_crash by depth: 4=33% (1/3 crashed), 6=0% (all stable), 8=100% (all OOM). Crashed seed (20260129 @ depth 4): amp_ssim=0.277, final_val=0.181 (constant amplitude collapse). Depth 6 (276M params) was paradoxically more stable — all 3 seeds converged (amp_ssim 0.78–0.80). The channel cap at depth 6 constrains parameter space in a potentially stabilizing way, whereas depth 4 (17M params, no cap effect) leaves dynamics unregularized. Alternatively, 3 seeds may be insufficient to observe depth-6 failures. Depth 8 remains blocked by CUDA OOM (18 GiB allocation). | [Link](../docs/plans/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/crash_hunt_summary.md) | Active |

## PROBE-MASK-DEFAULT-001 - Torch Probe Mask Default Caused Hybrid Metric Regression
**Category:** PyTorch, Config Default, Integration Regression
**Impact:** `tests/torch/test_grid_lines_hybrid_resnet_integration.py::test_grid_lines_hybrid_resnet_metrics`
**Location:** `ptycho_torch/config_params.py`, `ptycho_torch/model.py`, `ptycho_torch/helper.py`, `ptycho_torch/config_factory.py`
**Date:** 2026-02-20

### Issue
`fno-stable` started failing the hybrid-resnet integration metric gate after commit `db1e43f9`.

### Root Cause
The regression was introduced by changing Torch probe-mask semantics from effectively **off by default** to **on by default**:
- Newest passing commit: `8dac52fc`
- Oldest failing commit: `db1e43f9` (`feat(torch): default soft probe mask semantics across workflows`)

In `8dac52fc`, default behavior is no mask (`probe_mask=None` -> ones mask in forward path).  
In `db1e43f9`, defaults switched to `probe_mask=True` and mask resolution is always applied unless explicitly disabled.

### Validation
- Hard-edge default alone (`probe_mask_sigma=0.0`) did **not** restore pass (`mae_amp=0.12561041`, gate limit `0.0996316`).
- Disabling default masking (`probe_mask=False`) restored the integration pass (`mae_amp=0.07790681`).

### Conclusion
Primary regression driver is **default mask enablement**, not Gaussian edge smoothing.

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
- Artifacts: `docs/plans/FEAT-LAZY-LOADING-001/reports/2026-01-07T220000Z/`, `docs/plans/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T200000Z/`

### Status
**Resolved** — Lazy tensor allocation implemented. Large datasets can now be processed without OOM at container construction time. G-scaled verification confirms chunked inference via NumPy slicing is supported.

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
- Original blocker: `docs/plans/FIX-PYTORCH-FORWARD-PARITY-001/reports/.../tf_baseline/phase_c1/red/blocked_20251114T074039Z_tf_non_xla_shape_error.md`
- Scaled training GREEN (1 epoch): `docs/plans/FIX-PYTORCH-FORWARD-PARITY-001/reports/.../tf_baseline/phase_c1_scaled/cli/train_tf_phase_c1_scaled.log` (lines showing "8/8 steps" completion)
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
- Implementation: `docs/plans/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/pytest_all_tests.log`
- Tests: `tests/tf_helper/test_translation_shape_guard.py` (5/5 passed)
- Model factory regression: `tests/test_model_factory.py` (3/3 passed)

### Status
**Resolved** — XLA batch broadcast fix implemented and verified

## REPORTING-ARTIFACT-BOUNDARY-001 - Optional Reporting Artifacts Must Not Decide Scored Experiment Failure
**Category:** Experiment Orchestration, Artifact Contracts, Robustness
**Impact:** Long-running study loops can misclassify successful experiments as crashes and waste debug budget
**Date:** 2026-03-29

### Rule
For scored experiment workflows, only core execution evidence should determine
`KEEP`/`DISCARD`/`TIMEOUT`/`CRASH`:
- launcher exit status
- primary metrics
- any required comparability contract such as randomness metadata

Optional reporting artifacts such as comparison galleries, probe-inclusive
rerenders, or convenience visual exports may add warnings, but they must not by
themselves change a successful scored run into `CRASH`.

### Why
Post-processing and publication helpers are much more brittle than the actual
training/inference path. If they are treated as fatal, the workflow can:
- stop autonomous search even though the experiment finished
- trigger pointless crash-debug attempts against candidate code
- hide the real experiment outcome by failing after metrics were already
  produced

### Recommended Pattern
- Persist launcher result before later harvest or publication logic.
- Classify from core artifacts first.
- Record optional publication status separately, for example
  `published`, `fallback_plain_compare`, or `missing_nonfatal`.
- Surface optional-artifact problems as warnings in run/assessment artifacts.

### Status
**Recommended practice** — adopted for the `lines_256` controller path

## LINES256-CTRL-PATH-001 - Scored Controller Runs Must Stabilize PATH `python`
**Category:** Controller Runtime, Subprocess Boundary, Interpreter Drift
**Impact:** `lines_256` v2 scored candidates can false-crash before model execution
**Location:** `scripts/studies/lines_256_session_controller.py`
**Date:** 2026-03-30

### Issue
Iteration `046` of the `set_phi=True` `lines_256` v2 session crashed on
`ModuleNotFoundError: No module named 'tensorflow'` even though the matching
smoke run for the same candidate succeeded.

### Root Cause
The controller launched scored commands with `bash -lc`, which re-entered login
shell startup and let ambient PATH state choose a different `python` than the
controller runtime. The proposal artifacts were still correct to use plain
`python ...`, but the controller boundary was not holding that command to the
controller's own runtime.

### Required Rule
- Keep persisted commands and docs as plain `python ...` per `PYTHON-ENV-001`.
- Before scored controller launches, prepend the controller runtime's bin dir to
  `PATH`.
- Prefer `bash -c` over `bash -lc` for internal scored-command execution so
  login-shell startup cannot drift the interpreter.
- Persist a short crash excerpt in controller assessment artifacts so debug
  sees the real root-cause line.

### Evidence
- `state/lines_256_arch_improvement_v2/sessions/20260330T001026Z/iterations/046/20260330T191252Z_33407af0ab81.log`
- `outputs/lines_256_arch_improvement_v2/sessions/20260330T001026Z/candidates/20260330T191252Z_21bb106786e1/driver_stderr.log`
- `state/lines_256_arch_improvement_v2/sessions/20260330T001026Z/iterations/046/20260330T191252Z_adaff131ade6__source_smoke.log`

## LINES256-IMPORT-PROVENANCE-001 - Child Study Imports Must Resolve from the Session Repo Root
**Category:** Controller Runtime, Import Provenance, Cross-Checkout Isolation
**Impact:** `lines_256` v2 source candidates can be scored against the wrong source tree without crashing
**Location:** `scripts/studies/lines_256_session_controller.py`, `scripts/studies/invocation_logging.py`
**Date:** 2026-04-01

### Issue
Iteration `035` of session `20260331T015545Z` proposed a real source candidate
(`0fdceed...`) that replaced scalar `gated_add` skip weights with conditioned
per-channel gates. The scored run tied the champion exactly and produced the
old scalar-gate checkpoint state, even though the candidate commit existed and
smoke had succeeded.

### Root Cause
The controller's child-run environment preserved ambient `PYTHONPATH`. When the
session was resumed from the main repo against a dedicated run checkout, child
study commands imported `ptycho_torch` from the main repo instead of the
session checkout. Git validation alone was therefore insufficient: `HEAD`
matched the candidate commit in the run checkout, but the executed Python
modules came from a different checkout.

### Required Rule
- Keep PATH `python` and persisted command strings plain `python ...` per
  `PYTHON-ENV-001`.
- For controller-owned smoke, scored, and deterministic debug child runs, set
  `PYTHONPATH` explicitly to the session repo root instead of inheriting
  ambient launcher state.
- Record runtime provenance in invocation artifacts so `python_executable`,
  `cwd`, `PYTHONPATH`, and resolved `ptycho_torch.__file__` are inspectable
  when debugging.
- If a `source` candidate's runtime provenance does not resolve to the session
  repo root, classify the scored result as `INVALID_EXECUTION` rather than a
  scientific `DISCARD`.
- Exact source-candidate metric ties are a red flag worth warning on, but they
  are not sufficient proof by themselves; deterministic provenance checks are
  the authoritative gate.

### Evidence
- `state/lines_256_arch_improvement_v2/sessions/20260331T015545Z/iterations/035/proposal_result.json`
- `outputs/lines_256_arch_improvement_v2/sessions/20260331T015545Z/candidates/20260401T162432Z_0fdceed5370e/checkpoints/last.ckpt`
- `scripts/studies/lines_256_session_controller.py`
- `scripts/studies/invocation_logging.py`

## LINES256-CONTROLLER-PROVENANCE-001 - Controller-Owned Provenance Must Be Authoritative and Recoverable
**Category:** Controller Runtime, Compatibility Drift, Infra Recovery
**Impact:** `lines_256` v2 sessions can stop early on stale run checkouts even when the scored command itself succeeds
**Location:** `scripts/studies/lines_256_session_controller.py`
**Date:** 2026-04-01

### Issue
Iteration `036` of session `20260331T015545Z` scored a real source candidate
successfully, but the controller classified it as `INVALID_EXECUTION` solely
because the dedicated run checkout still emitted older `invocation.json`
artifacts without `extra.runtime_provenance`. The controller then treated that
infra-invalid result as terminal and marked the session `completed`.

### Root Cause
The controller had already moved integrity ownership in the right direction,
but it still depended on child-script runtime-provenance schema as the decisive
artifact. That made the controller brittle against run-checkout drift:
- new controller expectations
- old wrapper/runner artifact schema
- successful scored command
- terminal session stop anyway

### Required Rule
- Controller-owned launch metadata and import probes are the authoritative
  source-execution evidence for `source` candidates.
- Wrapper/runner runtime provenance is useful corroboration when present, but
  absence of those fields is only a compatibility warning.
- If controller-owned provenance is missing or incomplete, classify the result
  as recoverable `INVALID_EXECUTION` and yield for remediation instead of
  auto-completing the session.
- Only affirmative repo-root mismatches from controller-owned provenance should
  be treated as hard invalid execution.

### Evidence
- `state/lines_256_arch_improvement_v2/sessions/20260331T015545Z/iterations/036/candidate_assessment.json`
- `state/lines_256_arch_improvement_v2/sessions/20260331T015545Z/iterations/036/candidate_run_result.json`
- `outputs/lines_256_arch_improvement_v2/sessions/20260331T015545Z/candidates/20260401T183743Z_89f86c4a78f8/invocation.json`
- `outputs/lines_256_arch_improvement_v2/sessions/20260331T015545Z/candidates/20260401T183743Z_89f86c4a78f8/runs/pinn_hybrid_resnet/invocation.json`

## LINES256-PROPOSAL-CANONICALIZATION-001 - Provider Git SHAs Are Advisory, Not Authoritative
**Category:** Proposal Handoff, Source Provenance, Resume Robustness
**Impact:** `lines_256` v2 proposal steps can stop before scoring when the provider writes a wrong or hallucinated `candidate_commit`
**Location:** `scripts/studies/lines_256_session_controller.py`
**Date:** 2026-04-03

### Issue
Iteration `052` of session `20260331T015545Z` prepared a real source candidate
and passed smoke, but the proposal agent wrote a bogus full SHA into
`candidate_metadata.json`. The controller then trusted that metadata field
during proposal validation and stopped the session before any scored run
started.

### Root Cause
The provider was still treated as the authority for final git provenance even
though commit resolution is deterministic controller state, not prompt-owned
judgment. That made the proposal handoff brittle:
- the agent created the right source change
- repo `HEAD` was on a valid candidate commit
- provider metadata contained the wrong SHA
- controller trusted the string instead of the repo state and aborted

### Required Rule
- Treat provider `candidate_commit` as optional/advisory only for `source`
  proposals.
- Resolve the authoritative source candidate commit from repo `HEAD` in the
  controller and persist it in `proposal_resolution.json`.
- Validate/scored downstream source candidates against the controller-resolved
  commit, not the raw provider-authored SHA.
- If the controller cannot prove a valid source candidate state, write a
  retryable `proposal_result.json`, reset to `proposal_pending`, and allow
  resume to continue instead of leaving stale `proposal_running`.

### Evidence
- `state/lines_256_arch_improvement_v2/sessions/20260331T015545Z/iterations/052/candidate_metadata.json`
- `state/lines_256_arch_improvement_v2/sessions/20260331T015545Z/iterations/052/proposal_agent.log`
- `scripts/studies/lines_256_session_controller.py`

## LINES256-SESSION-BRANCH-001 - Session Checkouts Should Stay on Named Branches While SHAs Remain Authoritative
**Category:** Controller Runtime, Git Ergonomics, Resume Semantics
**Impact:** detached `HEAD` makes live `lines_256` run checkouts harder to inspect and resume even though branch names are not the real provenance contract
**Location:** `scripts/studies/lines_256_session_controller.py`, `docs/studies/lines_256_controller_loop.md`
**Date:** 2026-04-06

### Issue
The v2 controller already uses dedicated run checkouts, which is correct for
source-candidate rollback and cleanup. But leaving those checkouts detached by
default makes the operational state harder for humans to inspect and makes
resume semantics more confusing than necessary.

### Root Cause
The controller historically coupled source-candidate validation to exact commit
identity and let that bleed into the runtime checkout model. Commit SHAs are
the right scientific provenance surface, but detached `HEAD` is not required to
preserve that property.

### Required Rule
- Keep one named session branch per run checkout, for example
  `lines256/session/<session_id>`.
- Treat that branch name as an operator-facing runtime handle only.
- Keep `accepted_ref` and resolved source candidate SHAs as the authoritative
  provenance for scoring, ledgers, and resume validation.
- Detached checkouts are recoverable, but they should not be the intended
  steady-state runtime mode.

### Evidence
- `scripts/studies/lines_256_session_controller.py`
- `tests/studies/test_lines_256_session_controller.py`
- `docs/studies/lines_256_controller_loop.md`
