# Research Studies Agent Guide

## Quick Context
- **Purpose**: Planning documents for research studies and statistical analysis
- **Organization**: By study type (generalization, multirun statistics)
- **Status**: Mix of completed and active studies
- **Focus**: Publication-quality research and statistical validation

## Study Categories

### Model Generalization Studies
- **Purpose**: Performance vs. training set size analysis
- **Method**: Multiple training sizes with statistical trials
- **Output**: Scaling laws, training efficiency curves
- **Status**: ✅ Complete

### Statistical Multirun Analysis  
- **Purpose**: Multiple trials with uncertainty quantification
- **Method**: Repeated experiments with statistical aggregation
- **Output**: Mean ± IQR reporting, confidence intervals
- **Status**: ✅ Complete

### Active Studies
- **Check**: See `PROJECT_STATUS.md` for current research work
- **Access**: Active study documents in `docs/initiatives/` or `docs/sampling/`

## Document Structure  

### Plan Documents (plan_*.md)
- **Purpose**: Research objectives and experimental design
- **Content**: Hypotheses, methodology, success criteria
- **Example**: `plan_model_generalization.md`

### Implementation Documents (implementation_*.md)
- **Purpose**: Technical implementation details
- **Content**: Workflow automation, data collection, analysis scripts
- **Example**: `implementation_model_generalization.md`

### Phase Checklists (phase_*_checklist.md)
- **Purpose**: Execution tracking and validation
- **Content**: Step-by-step tasks, verification procedures
- **Progress**: Track completion status

### Study Guides (*.md)
- **Purpose**: User-facing workflow documentation
- **Content**: How to run studies, interpret results
- **Example**: `GENERALIZATION_STUDY_GUIDE.md`

## Completed Study Results

### Model Generalization Study
- **Deliverables**: 
  - `scripts/studies/run_complete_generalization_study.sh`
  - `scripts/studies/aggregate_and_plot_results.py`
  - Statistical analysis with multiple trials
- **Impact**: Established training size vs. performance relationships
- **Usage**: Foundation for determining optimal training set sizes

### Multirun Statistical Framework
- **Deliverables**: 
  - Multi-trial support in study scripts
  - Mean/IQR reporting instead of median-based
  - Uncertainty quantification
- **Impact**: Robust statistical validation of model comparisons
- **Usage**: Standard approach for all comparative studies

## Study Execution Patterns

### Research Study Lifecycle
1. **Hypothesis Formation** → Research plan document
2. **Experimental Design** → Implementation specification  
3. **Workflow Automation** → Script development
4. **Data Collection** → Multi-trial execution
5. **Statistical Analysis** → Aggregation and plotting
6. **Publication Preparation** → Result interpretation

### Statistical Best Practices
- **Multiple trials**: 3-5 trials minimum for robustness
- **Mean ± IQR**: More robust than median for small samples
- **Uncertainty quantification**: Always report confidence intervals
- **Cross-validation**: Multiple datasets when available

## Active Research Areas

### Current Focus
- Check `docs/PROJECT_STATUS.md` for active research initiatives
- Look for studies in `docs/initiatives/` or `docs/sampling/`

### Future Research Directions
- Spatial sampling bias analysis
- Domain adaptation studies  
- Noise robustness evaluation
- Scaling to larger datasets

## Integration with Workflows

### Study Execution
- **Scripts**: Use `scripts/studies/` for standardized workflows
- **Data**: Follow data contracts for consistent input/output
- **Analysis**: Use `aggregate_and_plot_results.py` for visualization

### Publication Pipeline
- **Raw data**: Stored in study output directories
- **Processed results**: CSV files with statistical summaries
- **Visualizations**: Publication-ready plots and figures
- **Documentation**: Complete methodology in plan documents

## Cross-References

- **Active studies**: <doc-ref type="status">docs/PROJECT_STATUS.md</doc-ref>
- **Study workflows**: <doc-ref type="workflow-guide">scripts/studies/CLAUDE.md</doc-ref>
- **Study execution**: <doc-ref type="workflow-guide">scripts/studies/README.md</doc-ref>
- **Quick reference**: <doc-ref type="workflow-guide">scripts/studies/QUICK_REFERENCE.md</doc-ref>
- **Sampling studies**: <doc-ref type="workflow-guide">docs/sampling/CLAUDE.md</doc-ref>