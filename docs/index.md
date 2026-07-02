# PtychoPINN Documentation Hub

This index provides a comprehensive overview of all available documentation with detailed descriptions to help you quickly find relevant information.

## Quick Start

### [README](../README.md) - Project Overview
**Description:** High-level project introduction with installation instructions and basic usage examples.  
**Keywords:** installation, overview, setup, quickstart  
**Use this when:** First time setting up PtychoPINN or need basic installation instructions.

### [Quick Reference: Parameters](QUICK_REFERENCE_PARAMS.md) CRITICAL
**Description:** Essential cheatsheet for params.cfg initialization - covers the critical `update_legacy_dict()` call required before data operations and debugging shape mismatch errors.  
**Keywords:** params.cfg, initialization, gridsize, shape-mismatch, troubleshooting  
**Use this when:** Getting shape mismatch errors, debugging configuration issues, or need to understand params.cfg initialization pattern.

## Project Management

### [CLAUDE Instructions](../CLAUDE.md)
**Description:** AI agent guidance defining critical data management rules, configuration requirements, and development conventions.  
**Keywords:** ai-agent, conventions, guidelines, critical-rules  
**Use this when:** Working as an AI agent or need to understand project conventions and critical requirements.

### [Initiative Workflow Guide](INITIATIVE_WORKFLOW_GUIDE.md)
**Description:** Comprehensive guide for AI-assisted development workflow using structured planning documents and phase-based execution with templates and best practices.  
**Keywords:** ai-assisted-development, phase-based-workflow, planning, project-management, test-driven  
**Use this when:** Starting any new development initiative or structuring complex work into manageable phases.

## Architecture & Development

### Core Documentation

#### [Developer Guide](DEVELOPER_GUIDE.md) ⚠️ ESSENTIAL
**Description:** Comprehensive architectural guide covering the "two-system" architecture (legacy grid-based vs. modern coordinate-based), critical anti-patterns, data pipeline contracts, TDD methodology, and configuration management best practices.  
**Keywords:** architecture, TDD, anti-patterns, configuration, data-pipeline  
**Use this when:** Starting any development work, debugging shape mismatches, understanding the codebase architecture, or implementing new features using TDD methodology.

#### [Architecture Overview](architecture.md)
**Description:** High-level component diagram and data flow visualization of the `ptycho/` core library, showing module relationships and typical workflow sequences.  
**Keywords:** components, data-flow, modules, workflow, visualization  
**Use this when:** Getting oriented with the codebase structure, understanding how modules interact, or planning new component integration.

#### [Testing Guide](TESTING_GUIDE.md)
**Description:** Comprehensive testing strategy covering unit tests, integration tests, TDD methodology, regression testing practices, and specific guidance for testing CLI parameters and backward compatibility.  
**Keywords:** testing, TDD, integration, regression, CLI-testing  
**Use this when:** Writing new tests, running the test suite, implementing TDD cycles, or ensuring backward compatibility.

#### [Troubleshooting Guide](TROUBLESHOOTING.md)
**Description:** Practical debugging guide for common issues including shape mismatch errors, configuration precedence problems, oversampling setup, and quick debugging commands with solutions.  
**Keywords:** debugging, shape-mismatch, configuration, oversampling, quick-fixes  
**Use this when:** Encountering shape mismatch errors (especially gridsize-related), debugging configuration issues, or need quick diagnostic commands.

### [CI Test Gate](ci.md)
**Description:** Authoritative CI policy for main: PR-only merges gated by the required `pytest-cpu` check, exclusion-baseline ratchet policy, and the red-PR playbook.
**Keywords:** ci, testing-gate, github-actions, ruleset, ratchet
**Use this when:** Your PR is red, you need to run the CI suite locally, or you are editing the exclusion baseline.

### Configuration & Data

#### [Configuration Guide](CONFIGURATION.md)
**Description:** Canonical reference for the modern dataclass-based configuration system with comprehensive parameter documentation for ModelConfig, TrainingConfig, and InferenceConfig classes.  
**Keywords:** configuration, parameters, dataclass, YAML, command-line  
**Use this when:** Setting up training or inference runs, understanding parameter precedence, or creating reproducible experiment configurations.

#### [Data Contracts](data_contracts.md) CRITICAL
**Description:** Official format specifications for NPZ datasets including required keys, data types, shapes, and normalization requirements.  
**Keywords:** NPZ-format, data-contracts, normalization, diffraction, amplitude  
**Use this when:** Creating or validating datasets, troubleshooting data format errors, or understanding amplitude vs intensity requirements.

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

#### `ptycho/model.py` - Neural Network Architecture
**Description:** U-Net-based physics-informed neural network combining deep learning with differentiable ptychographic forward modeling. Features custom Keras layers for physics constraints.  
**Key Dependencies:** Global `params.cfg` state at import time, `tf_helper` for tensor operations  
**Critical For:** Training workflows, inference, physics-informed reconstruction

#### `ptycho/diffsim.py` - Physics Simulation Layer
**Description:** Core forward physics implementation simulating the complete ptychographic measurement process: object illumination → coherent diffraction → Poisson photon noise.  
**Key Dependencies:** `params.cfg` for physics parameters (nphotons, N), `tf_helper` for differentiable operations  
**Critical For:** Training data generation, physics loss constraints, synthetic dataset creation

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
- **Debugging issues**: [Troubleshooting](TROUBLESHOOTING.md) → [Quick Reference Params](QUICK_REFERENCE_PARAMS.md)
- **Understanding data**: [Data Contracts](data_contracts.md) → [Data Normalization](DATA_NORMALIZATION_GUIDE.md)
- **Fixing shape mismatches**: [Quick Reference Params](QUICK_REFERENCE_PARAMS.md) → [Troubleshooting](TROUBLESHOOTING.md)
- **Training models**: [Training README](../scripts/training/README.md) → [Configuration Guide](CONFIGURATION.md)
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
