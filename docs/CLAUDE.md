# Documentation Agent Guide

## Quick Context
- **Purpose**: Central hub for all project documentation
- **Organization**: Guides (architectural) vs. workflows (procedural) vs. contracts (data specs)
- **Navigation**: Follow document type to find relevant information

## Document Categories

### Architectural Guides
| Document | Purpose | When to Use |
|----------|---------|-------------|
| `DEVELOPER_GUIDE.md` | Core architecture, anti-patterns, lessons learned | Understanding project design, debugging complex issues |
| `CONFIGURATION_GUIDE.md` | Parameter management, YAML configs | Setting up experiments, troubleshooting configs |
| `MODEL_COMPARISON_GUIDE.md` | Evaluation workflows, metrics | Running model comparisons, interpreting results |

### Workflow Guides  
| Document | Purpose | When to Use |
|----------|---------|-------------|
| `TOOL_SELECTION_GUIDE.md` | Decision matrix for tools | Choosing right workflow, avoiding common mistakes |
| `FLY64_DATASET_GUIDE.md` | Experimental dataset usage | Working with fly64 data |
| `INITIATIVE_WORKFLOW_GUIDE.md` | Planning and execution system | Managing development initiatives |
| `PROJECT_ORGANIZATION_GUIDE.md` | File organization conventions | Structuring project files |

### Data Contracts
| Document | Purpose | When to Use |
|----------|---------|-------------|
| `data_contracts.md` | NPZ file format specifications | Creating/modifying datasets, debugging data errors |

### Project Tracking
| Document | Purpose | When to Use |
|----------|---------|-------------|
| `PROJECT_STATUS.md` | Current initiative status | **ALWAYS** read before starting new tasks |

## Navigation Rules

### For Architecture Questions
**Start with**: `DEVELOPER_GUIDE.md`  
**Covers**: Design principles, anti-patterns, critical lessons  
**Follow up**: Specific guides based on area

### For Configuration Help
**Start with**: `CONFIGURATION_GUIDE.md`  
**Covers**: YAML configs, parameter validation, examples  
**Troubleshooting**: Modern vs legacy config systems

### For Model Evaluation
**Start with**: `MODEL_COMPARISON_GUIDE.md`  
**Covers**: Training workflows, metrics, registration, debug visualization  
**Advanced**: SSIM/MS-SSIM, FRC analysis

### For Tool Selection
**Start with**: `TOOL_SELECTION_GUIDE.md`  
**Covers**: Decision matrix, common mistakes, workflow patterns  
**Integration**: Links to specific tool workflows

### For Data Format Issues
**Start with**: `data_contracts.md`  
**Critical**: Most errors stem from incorrect data formats  
**Authority**: Single source of truth for NPZ specifications

### For Current Project Status
**Start with**: `PROJECT_STATUS.md`  
**Required**: Read before any new task  
**Updates**: Active initiatives, completed work

## Quick Decision Matrix

| I need to... | Read this first | Follow up with |
|---------------|----------------|----------------|
| Understand project architecture | `DEVELOPER_GUIDE.md` | Specific module docs |
| Set up an experiment | `CONFIGURATION_GUIDE.md` | Script-specific guides |
| Compare model performance | `MODEL_COMPARISON_GUIDE.md` | `scripts/studies/CLAUDE.md` |
| Choose the right tool | `TOOL_SELECTION_GUIDE.md` | Tool-specific docs |
| Fix data format errors | `data_contracts.md` | `scripts/tools/CLAUDE.md` |
| Start a new task | `PROJECT_STATUS.md` | Initiative-specific docs |

## Subdirectory Organization

### `docs/refactor/`
- **Purpose**: Completed major refactoring initiatives
- **Content**: Registration system, evaluation enhancements
- **Status**: ✅ Complete (archived for reference)
- **Access**: <doc-ref type="workflow-guide">docs/refactor/CLAUDE.md</doc-ref>

### `docs/studies/` 
- **Purpose**: Research study planning and analysis
- **Content**: Generalization studies, multirun statistics
- **Status**: Mix of completed and active
- **Access**: <doc-ref type="workflow-guide">docs/studies/CLAUDE.md</doc-ref>

### `docs/initiatives/`
- **Purpose**: Active development initiatives
- **Content**: Current planning and execution documents
- **Status**: Check `PROJECT_STATUS.md` for current state
- **Access**: <doc-ref type="workflow-guide">docs/initiatives/CLAUDE.md</doc-ref>

### `docs/sampling/`
- **Purpose**: Sampling study initiative
- **Content**: Spatially-biased randomized sampling analysis
- **Status**: Check `PROJECT_STATUS.md` for current phase
- **Access**: <doc-ref type="workflow-guide">docs/sampling/CLAUDE.md</doc-ref>

## Cross-References

- **Active initiatives**: <doc-ref type="status">PROJECT_STATUS.md</doc-ref>
- **Initiative planning**: <doc-ref type="workflow-guide">docs/initiatives/CLAUDE.md</doc-ref>
- **Study planning**: <doc-ref type="workflow-guide">docs/studies/CLAUDE.md</doc-ref>
- **Completed refactors**: <doc-ref type="workflow-guide">docs/refactor/CLAUDE.md</doc-ref>
- **Sampling studies**: <doc-ref type="workflow-guide">docs/sampling/CLAUDE.md</doc-ref>