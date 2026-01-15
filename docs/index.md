# PtychoPINN Documentation Hub

This index provides a comprehensive overview of all available documentation with detailed descriptions to help you quickly find relevant information.

## Critical Gotchas ⚠️

These are the most common pitfalls that cause subtle, hard-to-debug failures. **Read these first when debugging.**

| Gotcha | Symptom | Fix | Reference |
|--------|---------|-----|-----------|
| **MODULE-SINGLETON-001** | Model expects shape (B,N,N,1) but data has (B,N,N,4) after changing gridsize | Use `create_model_with_gridsize()` factory instead of `model.autoencoder` singleton | [Troubleshooting](debugging/TROUBLESHOOTING.md#model-architecture-mismatch-after-changing-gridsize) |
| **CONFIG-001** | Shape mismatch - gridsize not synced | Call `update_legacy_dict(params.cfg, config)` before data loading | [Quick Reference](debugging/QUICK_REFERENCE_PARAMS.md) |
| **CONFIG-001 Exception** | CONFIG-001 doesn't fix model architecture | Module-level singletons are created at import time; use factory functions | [Quick Reference](debugging/QUICK_REFERENCE_PARAMS.md#️-critical-exception-module-level-singletons-module-singleton-001) |
| **ANTIPATTERN-001** | Hidden crashes from import-time side effects | Push work into functions with explicit arguments | [Developer Guide](DEVELOPER_GUIDE.md#21-anti-pattern-side-effects-on-import) |

---

## Quick Start

### [README](../README.md) - Project Overview
**Description:** High-level project introduction with installation instructions and basic usage examples.  
**Keywords:** installation, overview, setup, quickstart  
**Use this when:** First time setting up PtychoPINN or need basic installation instructions.

### [Quick Reference: Parameters](debugging/QUICK_REFERENCE_PARAMS.md) CRITICAL
**Description:** Essential cheatsheet for params.cfg initialization - covers the critical `update_legacy_dict()` call required before data operations and debugging shape mismatch errors.  
**Keywords:** params.cfg, initialization, gridsize, shape-mismatch, troubleshooting  
**Use this when:** Getting shape mismatch errors, debugging configuration issues, or need to understand params.cfg initialization pattern.

### [Workflow Guide](WORKFLOW_GUIDE.md)
**Description:** Comprehensive guide covering complete PtychoPINN workflows from training through evaluation and model comparison, including common patterns and troubleshooting.  
**Keywords:** workflow, training, evaluation, model-comparison, troubleshooting  
**Use this when:** Starting a new project, planning experiments, or need to understand the complete train→evaluate→compare workflow.

### [Model Comparison Guide](MODEL_COMPARISON_GUIDE.md)
**Description:** How to run 2-way/3-way comparisons, pick entry points (direct compare vs wrappers vs studies), reuse existing runs, plug in PtyChi/Tike reconstructions, apply subsampling consistently, and handle fixed-canvas alignment.  
**Keywords:** comparison, studies, ptychi, subsampling, artifacts, alignment  
**Use this when:** You need to compare models, reuse existing outputs, or decide which comparison script to run.

### [PyTorch Workflow Guide](workflows/pytorch.md)
**Description:** End-to-end instructions for configuring, training, and running inference with the PyTorch implementation, highlighting differences from the TensorFlow pipelines and reusing the shared data contracts.  
**Keywords:** pytorch, lightning, mlflow, configuration, training  
**Use this when:** Working on the `ptycho_torch/` stack or porting TensorFlow workflows to PyTorch.

### [Knowledge Base Ledger](findings.md)
**Description:** Centralized record of critical discoveries, conventions, and recurring bugs, maintained as the agent's long-term memory.  
**Keywords:** knowledge-base, lessons-learned, debugging, conventions  
**Use this when:** Investigating an issue, planning a change, or verifying whether a problem has prior art.

### [Compare Models Spec](../specs/compare_models_spec.md)
**Description:** Interface/behavior contract for `compare_models.py` (CLI + future API), including inputs, outputs, sampling, registration, stitching, metrics, and recon NPZ expectations.  
**Keywords:** spec, comparison, interface, contract  
**Use this when:** You need authoritative details on `compare_models.py` behavior or are refactoring the compare pipeline.

## Project Management

### [PROJECT_STATUS](PROJECT_STATUS.md)
**Description:** Current development status tracker and active initiatives overview.  
**Keywords:** project-status, tracking, initiatives  
**Use this when:** Need to see current project status and active development work.

### [CLAUDE Instructions](../CLAUDE.md)
**Description:** AI agent guidance defining critical data management rules, configuration requirements, and development conventions.  
**Keywords:** ai-agent, conventions, guidelines, critical-rules  
**Use this when:** Working as an AI agent or need to understand project conventions and critical requirements.

### [Initiative Workflow Guide](INITIATIVE_WORKFLOW_GUIDE.md)
**Description:** Comprehensive guide for AI-assisted development workflow using structured planning documents and phase-based execution with templates and best practices.  
**Keywords:** ai-assisted-development, phase-based-workflow, planning, project-management, test-driven  
**Use this when:** Starting any new development initiative or structuring complex work into manageable phases.

### [Implementation Plan Template](../plans/templates/implementation_plan.md)
**Description:** Repo-specific phased plan template with initiative header, exit criteria, per‑phase checklists, Do Now block, artifacts index, and plan‑update protocol reminder.  
**Keywords:** template, plan, phased, checklist, exit-criteria  
**Use this when:** Creating or reshaping an initiative plan to match project conventions.

### [Agent Git Setup (Runbook)](../prompts/git_setup_agent.md)
**Description:** Step-by-step, idempotent setup for Git in supervisor/loop/orchestrator environments. Covers global config defaults, submodule hygiene for .claude/claude, .gitignore hardening for logs/outputs/data, safe pull wrappers, and recovery playbooks.  
**Keywords:** git, automation, supervisor, loop, submodules, rebase, setup  
**Use this when:** Bringing the orchestration stack to a new repo or stabilizing pull/rebase behavior.

### [Git Hygiene Guidelines](../prompts/git_hygiene.md)
**Description:** Reusable practices to keep automation-friendly repositories clean and conflict-free: submodule policy (ignore=dirty for tooling), ignore lists, safe pull patterns, CI guidance, and verification checklist.  
**Keywords:** git-hygiene, submodules, ignore, CI, rerere, autosquash  
**Use this when:** Maintaining multiple repos that run the same supervisor/loop and you want consistent, low-friction Git behavior.

## Architecture & Development

### Core Documentation

#### [Developer Guide](DEVELOPER_GUIDE.md) ⚠️ ESSENTIAL
**Description:** Comprehensive architectural guide covering the "two-system" architecture (legacy grid-based vs. modern coordinate-based), critical anti-patterns, data pipeline contracts, TDD methodology, configuration management best practices, and **inference pipeline patterns (§12)** with canonical load→inference→stitch code examples. Includes PYTHON-ENV-001 (Interpreter & Subprocess Policy) as the single source of truth for interpreter selection.
**Keywords:** architecture, TDD, anti-patterns, configuration, data-pipeline, inference, stitching
**Use this when:** Starting any development work, debugging shape mismatches, understanding the codebase architecture, implementing new features using TDD methodology, or writing inference workflows.

#### [Architecture Overview](architecture.md)
**Description:** Shared core architecture (data contracts, grouping, configuration, orchestration concepts) across both backends. Includes Scripts Overview and stable-modules policy, plus config lifecycle snippet (`update_legacy_dict(params.cfg, config)`). Backend-specific sequences and diagrams are in the TF/PyTorch pages below.  
**Keywords:** components, data-flow, scripts, shared-architecture, stable-modules, configuration  
**Use this when:** Getting oriented with shared architecture and how scripts map to orchestrators.

#### [Architecture — TensorFlow](architecture_tf.md)
**Description:** TensorFlow-specific architecture: component diagram, training and inference sequences, component reference, stable modules, and function/container mapping (TF ↔ PyTorch).  
**Keywords:** tensorflow, components, training, inference, containers, mapping  
**Use this when:** Implementing or debugging the TensorFlow backend and its workflows.

#### [Architecture — PyTorch](architecture_torch.md)
**Description:** PyTorch-specific architecture: component diagram, Lightning-based training and inference sequences, component reference, and function/container mapping (PyTorch ↔ TF).
**Keywords:** pytorch, lightning, components, training, inference, containers, mapping
**Use this when:** Implementing or debugging the PyTorch backend and its workflows.

#### [Architecture — Inference Pipeline](architecture_inference.md) NEW
**Description:** Comprehensive inference pipeline architecture: component diagrams, data flow (load→inference→stitch), tensor format system (Grid/Channel/Flat), backend dispatch pattern, coordinate conventions, and performance characteristics.
**Keywords:** inference, stitching, reassembly, data-flow, tensor-formats, coordinates
**Use this when:** Understanding or implementing the load→inference→stitch workflow, debugging shape mismatches during inference, or optimizing reconstruction performance.

#### [Testing Guide](TESTING_GUIDE.md)
**Description:** Comprehensive testing strategy covering unit tests, integration tests, TDD methodology, regression testing practices, and specific guidance for testing CLI parameters and backward compatibility.  
**Keywords:** testing, TDD, integration, regression, CLI-testing  
**Use this when:** Writing new tests, running the test suite, implementing TDD cycles, or ensuring backward compatibility.

#### <doc-ref type="test-index">docs/development/TEST_SUITE_INDEX.md</doc-ref>
**Description:** Machine-generated catalog of every `tests/` module with purpose statements, key test names, and direct execution commands.  
**Keywords:** test-coverage, discovery, regression, navigation  
**Use this when:** Locating the right test to extend, auditing coverage during code reviews, or coordinating TDD cycles across the suite.

#### [Troubleshooting Guide](debugging/TROUBLESHOOTING.md)
**Description:** Practical debugging guide for common issues including shape mismatch errors, configuration precedence problems, oversampling setup, and quick debugging commands with solutions.  
**Keywords:** debugging, shape-mismatch, configuration, oversampling, quick-fixes  
**Use this when:** Encountering shape mismatch errors (especially gridsize-related), debugging configuration issues, or need quick diagnostic commands.

#### [Debugging Methodology](debugging/debugging.md)
**Description:** Standard four-step process (verify contracts → sync configuration → isolate component → capture failing test) required for all new investigations.  
**Keywords:** methodology, workflow, testing, triage  
**Use this when:** Beginning any new bug hunt or postmortem to ensure consistent, auditable steps.

#### [Undocumented Conventions](debugging/undocumented_conventions.md)
**Description:** Living list of subtle behaviors (e.g., two-system assumptions, legacy sync order) that commonly cause regressions when overlooked.  
**Keywords:** conventions, gotchas, params.cfg, legacy-system  
**Use this when:** Reviewing legacy code, onboarding new teammates, or double-checking implicit assumptions.

### Configuration & Data

#### [Configuration Guide](CONFIGURATION.md)
**Description:** Canonical reference for the modern dataclass-based configuration system with comprehensive parameter documentation for ModelConfig, TrainingConfig, and InferenceConfig classes.  
**Keywords:** configuration, parameters, dataclass, YAML, command-line  
**Use this when:** Setting up training or inference runs, understanding parameter precedence, or creating reproducible experiment configurations.

#### [Data Contracts](../specs/data_contracts.md) CRITICAL
**Description:** Official format specifications for NPZ datasets including required keys, data types, shapes, and normalization requirements.
**Keywords:** NPZ-format, data-contracts, normalization, diffraction, amplitude
**Use this when:** Creating or validating datasets, troubleshooting data format errors, or understanding amplitude vs intensity requirements.

#### [Data Management Guide](DATA_MANAGEMENT_GUIDE.md)
**Description:** Best practices for managing NPZ and HDF5 data files, git hygiene rules, and Ptychodus product export workflow with metadata parameters and raw data inclusion options.
**Keywords:** data-management, git-hygiene, file-types, NPZ, HDF5, ptychodus-export
**Use this when:** Need to export reconstructions to Ptychodus format, understanding data file organization, or ensuring data files are not committed to git.

#### [Data Normalization Guide](DATA_NORMALIZATION_GUIDE.md)
**Description:** Explains the three distinct types of normalization (physics, statistical, display) and their proper application throughout the data pipeline to avoid common scaling errors.
**Keywords:** normalization, intensity_scale, physics, statistical, pipeline
**Use this when:** Debugging normalization issues, implementing new data loading features, or resolving scaling-related bugs.

#### [Data Generation Guide](DATA_GENERATION_GUIDE.md) CRITICAL
**Description:** Comprehensive guide to the two data generation pipelines: grid-based (`mk_simdata`) for notebook-compatible workflows and nongrid (`generate_simulated_data`) for production scripts. Covers parameter mappings, entry points, container construction, and **alternative data creation flows (§4)** for programmatic data generation without NPZ files.
**Keywords:** simulation, synthetic-data, grid, nongrid, mk_simdata, generate_simulated_data, params.cfg, no-npz, programmatic
**Use this when:** Implementing dose studies, generating synthetic datasets, choosing between grid and nongrid simulation, creating data programmatically without NPZ files, or debugging data generation issues.

#### [GridSize & n_groups Guide](GRIDSIZE_N_GROUPS_GUIDE.md) CRITICAL
**Description:** Explains the unified n_groups parameter behavior across different gridsize values, eliminating confusion between individual images vs groups.  
**Keywords:** n_groups, gridsize, sampling, groups, patterns  
**Use this when:** Understanding group formation with different gridsize values or troubleshooting unexpected dataset sizes.

### Specialized Topics

#### [Sampling User Guide](SAMPLING_USER_GUIDE.md)
**Description:** Comprehensive guide to flexible sampling control using n_subsample and n_groups parameters for memory management and training efficiency.  
**Keywords:** sampling, n_subsample, n_groups, memory, reproducibility  
**Use this when:** Controlling memory usage during training or implementing diverse sampling strategies.

#### [Pty-Chi Migration Guide](PTYCHI_MIGRATION_GUIDE.md)
**Description:** Complete migration guide for replacing Tike with pty-chi in three-way comparisons.
**Keywords:** pty-chi, tike-replacement, performance-optimization, reconstruction  
**Use this when:** Want to speed up iterative ptychographic reconstructions or migrate from Tike to pty-chi.

#### [Memory Optimization Guide](memory.md)
**Description:** Technical context document for Phase 5 of the Independent Sampling Control initiative detailing current implementation state and documentation gaps.  
**Keywords:** sampling-control, phase5-context, memory-management, parameter-interpretation  
**Use this when:** Working on Phase 5 of the sampling control initiative or need context on current implementation state.

## Workflows & Scripts

### Core Workflows

#### [Training](../scripts/training/README.md)
**Description:** Comprehensive guide for training PtychoPINN models with NPZ data, including configuration, sampling modes (random/sequential), logging, and next steps for evaluation.  
**Keywords:** training, configuration, sampling, logging, model-artifacts  
**Use this when:** Training a PtychoPINN model from scratch with proper configuration and understanding all training options.

#### [Inference](../scripts/inference/README.md)
**Description:** Guide for running inference with trained PtychoPINN models on test data, generating reconstruction visualizations and probe analysis.  
**Keywords:** inference, reconstruction, visualization, model-loading, testing  
**Use this when:** Have a trained model and want to generate reconstructions from new test data.

#### Evaluation (via Studies) — [Studies Guide](../scripts/studies/README.md)
**Description:** Use the Studies tooling to run evaluations and aggregate metrics across runs; for single reconstructions, see Inference.  
**Keywords:** evaluation, studies, metrics, analysis, aggregation  
**Use this when:** Running evaluations across datasets or comparing models using existing study tools.

#### [Simulation](../scripts/simulation/README.md)
**Description:** Two-stage modular simulation architecture for generating ptychographic datasets: Stage 1 creates object/probe inputs, Stage 2 simulates diffraction patterns.  
**Keywords:** simulation, diffraction, modular, synthetic-data, workflow  
**Use this when:** Need to generate synthetic ptychographic datasets from custom objects or existing reconstructions.

### Advanced Workflows

#### [Model Comparison](../scripts/studies/README.md)
**Description:** Systematic generalization study framework for training models across dataset sizes, supporting synthetic/experimental data with statistical robustness.  
**Keywords:** generalization, multi-trial, statistical-analysis, comparison  
**Use this when:** Conducting rigorous performance studies across training set sizes with statistical robustness.

#### [Reconstruction with Pty-Chi](../scripts/reconstruction/README.md)
**Description:** Traditional iterative reconstruction methods (Tike, Pty-chi) for baseline comparison against neural network approaches with configurable algorithms.  
**Keywords:** iterative-reconstruction, tike, pty-chi, baseline-comparison  
**Use this when:** Need traditional iterative reconstructions as baselines for comparison with PtychoPINN.

#### [Data Preprocessing Tools](../scripts/tools/README.md)
**Description:** Essential data preprocessing pipeline tools for format conversion, dataset preparation, splitting, and visualization with recovery patterns.  
**Keywords:** preprocessing, format-conversion, dataset-preparation, visualization  
**Use this when:** Have raw experimental data that needs format standardization or want to prepare datasets for training.

### Command Reference

#### [Commands Reference](COMMANDS_REFERENCE.md)
**Description:** Quick reference guide for essential PtychoPINN workflows including data preparation golden paths, training, inference, evaluation, and troubleshooting commands.  
**Keywords:** quick-reference, command-line, workflows, best-practices  
**Use this when:** Need rapid lookup of common command patterns and want to follow established golden paths.

## Studies & Analysis

### Study Guides

#### [Generalization Study Guide](studies/GENERALIZATION_STUDY_GUIDE.md)
**Description:** Complete guide for running multi-model generalization studies comparing PtychoPINN and baseline performance across different training set sizes.  
**Keywords:** generalization-study, model-comparison, training-sizes, automated-workflow  
**Use this when:** Systematically comparing model performance across different amounts of training data.

#### [Studies Guide](../scripts/studies/README.md)
**Description:** Study scripts and workflows for generalization experiments, comparisons, and metrics aggregation.  
**Keywords:** studies, generalization, comparisons, workflows  
**Use this when:** Running study workflows or comparing multiple models.

#### [SIM-LINES-4X Runner](../scripts/studies/sim_lines_4x/README.md)
**Description:** Four-scenario nongrid "lines" simulation with TensorFlow training and reconstruction (gs1/gs2 x idealized/custom probes), plus a gs2 ideal probe-scale sweep helper.  
**Keywords:** lines, nongrid, simulation, reconstruction, gs1, gs2, probe-scale  
**Use this when:** Generating the SIM-LINES-4X datasets, reconstructions, or a gs2 ideal probe-scale sweep.

## Datasets & Experiments

#### [FLY64 Dataset Guide](FLY64_DATASET_GUIDE.md)
**Description:** Comprehensive guide for working with the FLY64 experimental ptychography dataset, including preprocessing requirements and specialized subset datasets.  
**Keywords:** fly64, experimental-data, preprocessing, dataset-variants  
**Use this when:** Working with real experimental ptychography data (fly64) or need to understand dataset preprocessing requirements.

#### [FLY64 Generalization Analysis](FLY64_GENERALIZATION_STUDY_ANALYSIS.md)
**Description:** Detailed analysis of a complete generalization study on fly64 dataset revealing unexpected results where baseline models significantly outperform PtychoPINN.  
**Keywords:** fly64-analysis, unexpected-results, baseline-superiority, methodology-validation  
**Use this when:** Need to understand why baseline models might outperform PINNs on experimental data.

#### [Tike Reassembly Artifact Fix](TIKE_REASSEMBLY_ARTIFACT_FIX.md)
**Description:** Implementation guide for fixing visual reassembly artifacts in Tike reconstructions processed through PtychoPINN's comparison pipeline.  
**Keywords:** tike-integration, artifact-fixing, boundary-handling, coordinate-systems  
**Use this when:** Encounter visual grid artifacts in Tike reconstruction comparisons.

## Core Module Documentation

### Neural Network & Physics (`ptycho/`)

#### `ptycho/model.py` - Neural Network Architecture ⚠️ SINGLETON WARNING
**Description:** U-Net-based physics-informed neural network combining deep learning with differentiable ptychographic forward modeling. Features custom Keras layers for physics constraints.
**Key Dependencies:** Global `params.cfg` state at import time, `tf_helper` for tensor operations
**Critical For:** Training workflows, inference, physics-informed reconstruction
**⚠️ CRITICAL:** `model.autoencoder` and `model.diffraction_to_obj` are **module-level singletons** created at import time. They capture `params.cfg['gridsize']` when imported, NOT when used. If you change gridsize after importing this module, use `create_model_with_gridsize(gridsize, N)` to get correctly-sized models. See [MODULE-SINGLETON-001](findings.md) and [Troubleshooting](debugging/TROUBLESHOOTING.md#model-architecture-mismatch-after-changing-gridsize).

#### `ptycho/diffsim.py` - Physics Simulation Layer
**Description:** Core forward physics implementation simulating the complete ptychographic measurement process: object illumination → coherent diffraction → Poisson photon noise.  
**Key Dependencies:** `params.cfg` for physics parameters (nphotons, N), `tf_helper` for differentiable operations  
**Critical For:** Training data generation, physics loss constraints, synthetic dataset creation

#### `ptycho/cache.py` - RawData Cache Helpers
**Description:** Hosts `memoize_raw_data` and the hashing utilities that back the RawData caching workflow so nongrid simulations can reuse expensive `RawData.generate_grouped_data()` payloads without living in `scripts/`. Aligned with the shared **Data Pipeline** rules in [docs/architecture.md §3](architecture.md#3-a-deep-dive-the-data-loading--preprocessing-pipeline).
**Key Dependencies:** `ptycho.raw_data.RawData` for grouped data construction and `.artifacts/synthetic_helpers/cache` as the default cache root.  
**Critical For:** Synthetic helper scripts that enable quick iteration on nongrid simulations and any future workflows that need a stable, side-effect-free cache primitive. Legacy imports continue to work through the shim `scripts/simulation/cache_utils.py`, which now re-exports the core helper.

### Specifications

#### [Inference Pipeline Specification](specs/spec-inference-pipeline.md) NEW
**Description:** Normative IDL-style contracts for the inference pipeline: function signatures, preconditions/postconditions, tensor shape invariants, data container contracts (RawData, GroupedDataDict, PtychoDataContainer), model loading, inference, and stitching APIs with error taxonomy.
**Keywords:** spec, inference, contracts, IDL, shapes, invariants, API
**Use this when:** Need authoritative API contracts for inference functions, debugging shape mismatches, implementing new inference workflows, or understanding data container requirements.

#### [PtychoPINN Spec — Index](specs/spec-ptychopinn.md)
**Description:** Index of normative spec shards for the TensorFlow‑based physics‑informed ptychography pipeline (core physics, runtime, workflow, interfaces, conformance, tracing).
**Keywords:** spec, ptychography, TensorFlow, physics‑informed, index
**Use this when:** You need the top‑level map of all PtychoPINN specifications.

#### [PtychoPINN Core Physics & Data Contracts](specs/spec-ptycho-core.md)
**Description:** Normative definition of the forward model (object·probe→FFT→|F|²/N²→sqrt), Poisson observation, intensity scaling symmetry, coordinates/patch extraction, probe/masking/smoothing, valid inputs, losses, and outputs.  
**Keywords:** physics, FFT, Poisson, scaling, offsets, probe, contracts  
**Use this when:** Implementing or auditing the physical/mathematical operations and strict data shapes.

#### [PtychoPINN Runtime & Execution](specs/spec-ptycho-runtime.md)
**Description:** TensorFlow runtime guardrails: dtype/device policy, XLA translation/compile modes, vectorization/streaming, graph hygiene, environment flags, and error conditions.  
**Keywords:** runtime, TensorFlow, XLA, vectorization, determinism  
**Use this when:** Tuning performance, enabling XLA, or validating execution safety constraints.

#### [PtychoPINN Workflow](specs/spec-ptycho-workflow.md)
**Description:** End‑to‑end pipeline: NPZ ingest → grouping → normalization → model → loss/optimization → inference → stitching → evaluation; staging knobs and guards.  
**Keywords:** workflow, grouping, normalization, training, inference, stitching, evaluation  
**Use this when:** Building or verifying the full training/inference pipeline.

#### [PtychoPINN Interfaces](specs/spec-ptycho-interfaces.md)
**Description:** Public API surface and data/file interfaces (RawData, loader, models, training/eval), model I/O contracts, precedence rules for params/env, and error conditions.  
**Keywords:** API, data‑contracts, shapes, params, precedence  
**Use this when:** Integrating modules, writing loaders, or consuming model interfaces.

#### [PtychoPINN Conformance Tests](specs/spec-ptycho-conformance.md)
**Description:** Acceptance tests PTY‑AT‑XXX: forward amplitude equivalence, Poisson semantics, grouping/coords shapes, translation round‑trip, intensity scaling symmetry, positive intensity for NLL, loader contracts, inference determinism, stitch border math.  
**Keywords:** conformance, acceptance‑tests, parity, validation  
**Use this when:** Certifying a build or diagnosing regressions against the spec.

#### [PtychoPINN Tracing & Debug](specs/spec-ptycho-tracing.md)
**Description:** Tracing obligations for physics/intermediate tensors, coordinate/translation traces, scaling invariants, and first‑divergence workflow with artifact guidance.  
**Keywords:** tracing, debug, parity, diagnostics  
**Use this when:** Investigating numerical/physics divergences or instrumentation gaps.

#### [Ptychodus Integration API Spec](../specs/ptychodus_api_spec.md)
**Description:** Normative API contract for integrating PtychoPINN with Ptychodus, covering configuration bridging, backend selection, data ingestion, lifecycle, and persistence expectations.  
**Keywords:** spec, API, integration, config-bridge, backend  
**Use this when:** Implementing or validating a backend used by Ptychodus, or wiring configs through `update_legacy_dict()` per the contract.

#### [Ptychodus Data Contracts](../specs/data_contracts.md)
**Description:** Normative HDF5 product format (metadata, positions, probe, object, loss history) used by Ptychodus product readers/writers; includes shapes, dtypes, and units.  
**Keywords:** data-contracts, HDF5, product, metadata, probe, object  
**Use this when:** Writing/reading product files or converting datasets to the Ptychodus product format.

#### [Config Bridge (TensorFlow ↔ PyTorch)](specs/spec-ptycho-config-bridge.md)
**Description:** Normative mapping between TensorFlow dataclass configs and PyTorch config singletons, including field transformations (grid_size→gridsize, epochs→nepochs, mode→model_type), defaults/overrides, update_legacy_dict flow, and validation rules.  
**Keywords:** config, bridge, translation, params.cfg, dataclasses, pytorch  
**Use this when:** Translating configuration between backends, ensuring CONFIG‑001 compliance, or verifying field mappings.

#### [Overlap Metrics Spec](specs/overlap_metrics.md)
**Description:** Overlap-driven sampling and reporting for Phase D. Defines three 2D disc-overlap metrics (group-based, image-based, and group↔group COM-based), explicit controls via `s_img` and `n_groups`, and removes spacing/packing acceptance gates.  
**Keywords:** overlap, metrics, s_img, n_groups, probe-diameter, gridsize  
**Use this when:** Implementing or validating Phase D overlap behavior and reporting measured overlaps instead of geometry gating.

### Configuration & Workflows

#### `ptycho/config/` - Configuration System
**Description:** Modern dataclass-based configuration with one-way translation to legacy `params.cfg`. Provides type-safe configuration management.  
**Key Dependencies:** `KEY_MAPPINGS` for legacy compatibility, YAML loading utilities  
**Critical For:** All workflows requiring parameter management, CLI argument parsing

#### `ptycho/workflows/` - High-Level Workflow Functions
**Description:** Orchestration layer bridging CLI scripts and core modules. Chains together complete pipelines including data loading, configuration management, and training.  
**Key Dependencies:** Core modules integration, configuration system, data pipeline  
**Critical For:** End-to-end workflows, `run_cdi_example()`, training orchestration

### Data Pipeline

#### `ptycho/loader.py` - Data Loading Utilities
**Description:** NumPy-to-TensorFlow conversion layer transforming grouped data into GPU-ready tensors. Handles dtype conversion and tensor reshaping for multi-channel architectures.  
**Key Dependencies:** `raw_data.py` for grouped data, TensorFlow tensor operations  
**Critical For:** Model training, inference data preparation, tensor pipeline

#### `ptycho/raw_data.py` - Raw Data Handling
**Description:** First-stage data pipeline transforming NPZ files into structured containers with efficient coordinate grouping. Implements "sample-then-group" strategy for 10-100x performance improvement.  
**Key Dependencies:** `params.cfg` for grouping parameters, scikit-learn for nearest neighbors  
**Critical For:** Data preprocessing, coordinate grouping, NPZ file ingestion

#### `ptycho/tf_helper.py` - TensorFlow Utilities
**Description:** Essential tensor transformation operations including three-format conversion system (Grid/Channel/Flat), patch extraction/reassembly, and batched operations.  
**Key Dependencies:** Global `params.cfg` for tensor dimensions (N, gridsize, offset)  
**Critical For:** Data format conversion, patch operations, model training/inference

### Evaluation & Baselines

#### `ptycho/evaluation.py` - Evaluation Metrics
**Description:** Central quality assessment orchestrating multiple metrics (SSIM, MS-SSIM, FRC, MAE). Handles complex data preprocessing, phase alignment, and standardized interfaces.  
**Key Dependencies:** `ptycho.FRC`, `ptycho.image` for registration, scikit-image metrics  
**Critical For:** Training validation, model comparison, research analysis

#### `ptycho/baselines.py` - Baseline Models
**Description:** Supervised learning baseline using dual-output U-Net architecture for pure data-driven reconstruction without physics constraints.  
**Key Dependencies:** Standard TensorFlow/Keras layers, `tf_helper` for compatibility  
**Critical For:** Model comparison studies, benchmarking, performance evaluation

#### `ptycho/misc.py` - Utility Functions
**Description:** Caching decorators (`@memoize_disk_and_memory`) for expensive computations, output path generation, and image processing helpers.  
**Key Dependencies:** `params.cfg` for configuration state, filesystem operations  
**Critical For:** Performance optimization, caching, utility operations across modules

## Finding Information

### By Task
- **Starting a new feature**: [Developer Guide](DEVELOPER_GUIDE.md) → [Initiative Workflow](INITIATIVE_WORKFLOW_GUIDE.md)
- **Running experiments**: [Workflow Guide](WORKFLOW_GUIDE.md) → [Commands Reference](COMMANDS_REFERENCE.md)
- **Debugging issues**: [Troubleshooting](debugging/TROUBLESHOOTING.md) → [Quick Reference Params](debugging/QUICK_REFERENCE_PARAMS.md)
- **Understanding data**: [Data Contracts](../specs/data_contracts.md) → [Data Normalization](DATA_NORMALIZATION_GUIDE.md)
- **Fixing shape mismatches**: [Quick Reference Params](debugging/QUICK_REFERENCE_PARAMS.md) → [Troubleshooting](debugging/TROUBLESHOOTING.md)
- **Training models**: [Training README](../scripts/training/README.md) → [Configuration Guide](CONFIGURATION.md)
- **Running inference**: [Developer Guide §12](DEVELOPER_GUIDE.md#12-inference-pipeline-patterns) → [Inference Spec](specs/spec-inference-pipeline.md) → [Inference Architecture](architecture_inference.md)
- **Creating data without NPZ**: [Data Generation Guide §4](DATA_GENERATION_GUIDE.md#4-alternative-data-creation-flows-no-npz-required)
- **Evaluating models**: [Evaluation README](../scripts/evaluation/README.md) → [Model Comparison](../scripts/studies/README.md)

### By User Type
- **New Users**: [README](../README.md) → [Workflow Guide](WORKFLOW_GUIDE.md) → [Training](../scripts/training/README.md)
- **Developers**: [Developer Guide](DEVELOPER_GUIDE.md) → [Testing Guide](TESTING_GUIDE.md) → [Architecture](architecture.md)
- **Researchers**: [Generalization Studies](studies/GENERALIZATION_STUDY_GUIDE.md) → [Model Comparison](../scripts/studies/README.md)
- **AI Agents**: [CLAUDE.md](../CLAUDE.md) → [Initiative Workflow](INITIATIVE_WORKFLOW_GUIDE.md) → [Developer Guide](DEVELOPER_GUIDE.md)

## Documentation Standards

When adding new documentation:
1. Update this index with detailed descriptions (1-2 lines minimum)
2. Include keywords/tags for searchability
3. Add "Use this when..." guidance
4. Use the `<doc-ref>` XML tagging system for cross-references
5. Ensure bidirectional linking
6. Add to [PROJECT_STATUS.md](PROJECT_STATUS.md) if it's an initiative document

## 🔗 External Resources

- **Paper**: [Nature Scientific Reports Publication](https://www.nature.com/articles/s41598-023-48351-7)
- **GitHub Issues**: Report bugs and request features
- **License**: [LICENSE](../LICENSE)

---

*Last updated: September 2025*
*added detailed descriptions for improved navigation*
