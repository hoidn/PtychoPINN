# PtychoPINN Documentation Hub

Welcome to the PtychoPINN documentation. This index provides a comprehensive overview of all available documentation, organized by category and use case.

## 🚀 Quick Start

- **[README](../README.md)** - Project overview and basic usage
- **[Installation Guide](../README.md#installation)** - Setting up PtychoPINN
- **[Quick Reference: Parameters](QUICK_REFERENCE_PARAMS.md)** - Essential params.cfg initialization guide
- **[Workflow Guide](WORKFLOW_GUIDE.md)** - Complete workflow walkthrough

## 📋 Project Management

- **[PROJECT_STATUS](PROJECT_STATUS.md)** - Current development status and active initiatives
- **[CLAUDE Instructions](../CLAUDE.md)** - AI agent guidance and conventions
- **[Initiative Workflow Guide](INITIATIVE_WORKFLOW_GUIDE.md)** - How to plan and execute development initiatives

## 🏗️ Architecture & Development

### Core Documentation
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Comprehensive guide for contributors
- **[Architecture Overview](architecture.md)** - System design and components
- **[Testing Guide](TESTING_GUIDE.md)** - Testing strategy and practices
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions

### Configuration & Data
- **[Configuration Guide](CONFIGURATION.md)** - Complete configuration system documentation
- **[Data Contracts](data_contracts.md)** - Data format specifications and requirements
- **[Data Normalization Guide](DATA_NORMALIZATION_GUIDE.md)** - Understanding normalization conventions
- **[GridSize & n_groups Guide](GRIDSIZE_N_GROUPS_GUIDE.md)** - Critical parameter interactions

### Specialized Topics
- **[Sampling User Guide](SAMPLING_USER_GUIDE.md)** - Independent sampling control
- **[Pty-Chi Migration Guide](PTYCHI_MIGRATION_GUIDE.md)** - Migrating from Tike to pty-chi
- **[Memory Optimization Guide](memory.md)** - Memory management and optimization

## 🔧 Workflows & Scripts

### Core Workflows
- **[Training](../scripts/training/README.md)** - Training models with ptycho_train
- **[Inference](../scripts/inference/README.md)** - Running inference on trained models
- **[Evaluation](../scripts/evaluation/README.md)** - Single model evaluation with metrics
- **[Simulation](../scripts/simulation/README.md)** - Simulating ptychographic data

### Advanced Workflows
- **[Model Comparison](../scripts/studies/README.md)** - Comparing PtychoPINN vs baseline models
- **[Generalization Studies](studies/GENERALIZATION_STUDY_GUIDE.md)** - Running comprehensive studies
- **[Reconstruction with Pty-Chi](../scripts/reconstruction/README.md)** - Traditional reconstruction methods
- **[Data Preprocessing Tools](../scripts/tools/README.md)** - Dataset preparation utilities

### Command Reference
- **[Commands Reference](COMMANDS_REFERENCE.md)** - All available CLI commands
- **[Custom Claude Commands](./../.claude/commands/)** - AI-powered development commands

## 📊 Studies & Analysis

### Study Guides
- **[Studies Overview](studies/CLAUDE.md)** - Study workflows and organization
- **[Generalization Study Guide](studies/GENERALIZATION_STUDY_GUIDE.md)** - Running large-scale studies
- **[Quick Reference for Studies](../scripts/studies/QUICK_REFERENCE.md)** - Study script reference

### Phase Checklists
- **[Phase 1 Checklist](studies/phase_1_checklist.md)** - Initial study phase
- **[Phase 2 Checklist](studies/phase_2_checklist.md)** - Implementation phase
- **[Phase 3 Checklist](studies/phase_3_checklist.md)** - Validation phase

## 🔬 Datasets & Experiments

- **[FLY64 Dataset Guide](FLY64_DATASET_GUIDE.md)** - Experimental dataset documentation
- **[FLY64 Generalization Analysis](FLY64_GENERALIZATION_STUDY_ANALYSIS.md)** - Study results analysis
- **[Tike Reassembly Artifact Fix](TIKE_REASSEMBLY_ARTIFACT_FIX.md)** - Known issues and fixes

## 🛠️ Development Initiatives

### Active Initiatives
- **[Initiatives Overview](initiatives/CLAUDE.md)** - Current development initiatives
- **[Smart Subsampling](initiatives/smart-subsampling/implementation.md)** - Advanced sampling implementation
- **[Independent Sampling Control](initiatives/independent_sampling_control/phase6_plan.md)** - Sampling control system

### Migration & Legacy
- **[Migration: Coordinate Grouping](migration/coordinate_grouping.md)** - Legacy system migration

## 📚 Module Documentation

### Core Library (`ptycho/`)
- **[Core Library Guide](../ptycho/CLAUDE.md)** - Internal library development guide
- Key modules:
  - `ptycho/model.py` - Neural network architecture
  - `ptycho/diffsim.py` - Physics simulation layer
  - `ptycho/config/` - Configuration system
  - `ptycho/workflows/` - High-level workflow functions
  - `ptycho/loader.py` - Data loading utilities

### Script Categories
- **Training Scripts** (`scripts/training/`)
- **Inference Scripts** (`scripts/inference/`)
- **Evaluation Scripts** (`scripts/evaluation/`)
- **Simulation Scripts** (`scripts/simulation/`)
- **Study Scripts** (`scripts/studies/`)
- **Tool Scripts** (`scripts/tools/`)
- **Reconstruction Scripts** (`scripts/reconstruction/`)

## 🔍 Finding Information

### By Task
- **Starting a new feature**: Read [Developer Guide](DEVELOPER_GUIDE.md) and [Initiative Workflow](INITIATIVE_WORKFLOW_GUIDE.md)
- **Running experiments**: See [Workflow Guide](WORKFLOW_GUIDE.md) and [Commands Reference](COMMANDS_REFERENCE.md)
- **Debugging issues**: Check [Troubleshooting](TROUBLESHOOTING.md) and [Quick Reference Params](QUICK_REFERENCE_PARAMS.md)
- **Understanding data**: Review [Data Contracts](data_contracts.md) and [Data Normalization](DATA_NORMALIZATION_GUIDE.md)

### By User Type
- **New Users**: Start with [README](../README.md) → [Workflow Guide](WORKFLOW_GUIDE.md)
- **Developers**: [Developer Guide](DEVELOPER_GUIDE.md) → [Testing Guide](TESTING_GUIDE.md) → [Architecture](architecture.md)
- **Researchers**: [Generalization Studies](studies/GENERALIZATION_STUDY_GUIDE.md) → [Model Comparison](../scripts/studies/README.md)
- **AI Agents**: [CLAUDE.md](../CLAUDE.md) → [Initiative Workflow](INITIATIVE_WORKFLOW_GUIDE.md)

## 📝 Documentation Standards

When adding new documentation:
1. Update this index with appropriate links
2. Use the `<doc-ref>` XML tagging system for cross-references
3. Follow the categorization structure above
4. Ensure bidirectional linking (new doc links here, this links to new doc)
5. Add to [PROJECT_STATUS.md](PROJECT_STATUS.md) if it's an initiative document

## 🔗 External Resources

- **Paper**: [Nature Scientific Reports Publication](https://www.nature.com/articles/s41598-023-48351-7)
- **GitHub Issues**: Report bugs and request features
- **License**: [LICENSE](../LICENSE)

---

*Last updated: September 2025*
*Documentation version: 1.0.0*