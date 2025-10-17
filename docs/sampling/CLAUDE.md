# Sampling Study Agent Guide

## Quick Context
- **Purpose**: Spatially-biased randomized sampling study planning
- **Status**: Check `PROJECT_STATUS.md` for current phase
- **Goal**: Enable generalization studies on random samples from specific spatial regions
- **Focus**: Understanding how spatial sampling affects model performance

## Key Deliverables

### Tools
- **`shuffle_dataset_tool.py`**: Randomizing dataset order
- **Purpose**: Remove spatial bias from training data selection
- **Integration**: Part of preprocessing pipeline

### Documentation Updates
- **Updated `scripts/tools/README.md`**: Document shuffle tool usage
- **Integration guidance**: How to incorporate into existing workflows

### Study Execution
- **Complete generalization study**: Apply to fly64 dataset regions
- **Spatial analysis**: Compare spatially-biased vs. random sampling
- **Statistical validation**: Multi-trial analysis of sampling effects

## Document Structure

### Plan Document
- **File**: `plan_sampling_study.md`
- **Content**: Research objectives, spatial bias hypothesis
- **Purpose**: Understand why spatial sampling matters

### Implementation Document
- **File**: `implementation_sampling_study.md`  
- **Content**: Technical implementation of shuffle tool
- **Purpose**: Development roadmap and technical specifications

### Phase Checklists
- **Files**: `phase_*_checklist.md`
- **Content**: Step-by-step execution tasks
- **Purpose**: Track progress through implementation phases

### Final Validation
- **File**: `final_phase_validation_checklist.md`
- **Content**: Complete testing and validation procedures
- **Purpose**: Ensure study deliverables meet objectives

## Research Context

### Spatial Bias Problem
- **Issue**: Sequential scan positions create spatial correlation
- **Impact**: Training on early scans may miss features in later regions
- **Solution**: Randomize scan order to eliminate spatial bias

### Generalization Study Integration
- **Workflow**: Use shuffled datasets in generalization studies
- **Analysis**: Compare performance with/without spatial randomization
- **Metrics**: Standard evaluation metrics across spatial sampling strategies

## Study Methodology

### Baseline Comparison
1. **Standard sampling**: Use original scan order
2. **Random sampling**: Use shuffle_dataset_tool.py output
3. **Statistical comparison**: Multi-trial analysis of both approaches

### Spatial Region Analysis
1. **Regional subsets**: Extract specific spatial regions from fly64
2. **Cross-regional testing**: Train on one region, test on another
3. **Generalization assessment**: Quantify spatial transfer performance

## Tool Usage

### shuffle_dataset_tool.py
```bash
# Basic usage
python scripts/tools/shuffle_dataset_tool.py input.npz output.npz

# With specific random seed for reproducibility
python scripts/tools/shuffle_dataset_tool.py input.npz output.npz --seed 42
```

### Integration with Studies
```bash
# 1. Shuffle dataset
python scripts/tools/shuffle_dataset_tool.py fly64_data.npz fly64_shuffled.npz

# 2. Run generalization study on shuffled data
./scripts/studies/run_complete_generalization_study.sh \
    --dataset fly64_shuffled.npz \
    --output-dir spatial_bias_study

# 3. Compare with non-shuffled baseline
./scripts/studies/run_complete_generalization_study.sh \
    --dataset fly64_data.npz \
    --output-dir spatial_bias_baseline
```

## Current Status

### Check Progress
- **Primary source**: `docs/PROJECT_STATUS.md`
- **Current phase**: Look for sampling study status
- **Completion**: Check phase checklist progress

### Implementation Status
- **Tool development**: Check if shuffle_dataset_tool.py exists
- **Documentation**: Verify scripts/tools/README.md updates
- **Testing**: Review validation checklist completion

## Cross-References

- **Current status**: <doc-ref type="status">docs/PROJECT_STATUS.md</doc-ref>
- **Tool usage**: <doc-ref type="workflow-guide">scripts/tools/CLAUDE.md</doc-ref>
- **Study workflows**: <doc-ref type="workflow-guide">scripts/studies/CLAUDE.md</doc-ref>
- **Research context**: <doc-ref type="workflow-guide">docs/studies/CLAUDE.md</doc-ref>