# Refactor Initiative Agent Guide

## Quick Context  
- **Purpose**: Planning documents for completed major refactoring initiatives
- **Status**: All initiatives here are ✅ Complete (archived for reference)
- **Organization**: plan_*.md (R&D plans), implementation_*.md (technical plans), phase_*_checklist.md (execution tracking)
- **Use Case**: Historical reference, pattern learning for future initiatives

## Initiative Categories

### Image Registration System
- **Status**: ✅ Complete
- **Goal**: Automatic alignment for fair model comparison
- **Key Deliverables**:
  - `ptycho/image/registration.py` with sub-pixel phase correlation
  - Integration into `scripts/compare_models.py` with `--skip-registration` flag
  - Unified NPZ file format for reconstruction data

### Evaluation Enhancements  
- **Status**: ✅ Complete
- **Goal**: Advanced perceptual metrics and improved phase alignment
- **Key Deliverables**:
  - SSIM/MS-SSIM metrics implementation
  - Enhanced phase preprocessing (plane fitting vs mean subtraction)
  - Debug visualization capabilities with `--save-debug-images` flag
  - Enhanced FRC with configurable smoothing

## Document Structure

### Plan Documents (plan_*.md)
- **Purpose**: High-level objectives and experimental design
- **Content**: Problem statement, success criteria, research approach
- **Audience**: R&D planning, understanding motivations

### Implementation Documents (implementation_*.md)  
- **Purpose**: Technical specifications and detailed steps
- **Content**: Phase breakdown, technical architecture, file modifications
- **Audience**: Developers, technical implementation

### Phase Checklists (phase_*_checklist.md)
- **Purpose**: Step-by-step execution tracking
- **Content**: Specific tasks, file paths, verification steps
- **Audience**: Implementation execution, progress tracking

### Validation Reports
- **Purpose**: Final verification and testing results
- **Content**: Success criteria verification, performance analysis
- **Audience**: Quality assurance, completion validation

## Completed Initiative Details

### Registration System Files
- **Context**: `docs/refactor/context_priming_registration.md`
- **Plan**: `docs/refactor/plan_registration.md`  
- **Impact**: Enabled fair model comparison by correcting translational misalignments

### Evaluation Enhancement Files
- **Plan**: `docs/refactor/eval_enhancements/plan_eval_enhancements.md`
- **Implementation**: `docs/refactor/eval_enhancements/implementation_eval_enhancements.md`
- **Impact**: Added perceptual metrics (SSIM/MS-SSIM) beyond traditional MAE/MSE/PSNR

## Learning Patterns

### Initiative Lifecycle Pattern
1. **Context/Problem Identification** → Context document
2. **R&D Planning** → plan_*.md document  
3. **Technical Planning** → implementation_*.md document
4. **Phase-by-Phase Execution** → phase_*_checklist.md files
5. **Validation & Completion** → validation reports
6. **Archive** → Move to completed status

### Technical Implementation Pattern
- **Modular approach**: New functionality in separate modules
- **Integration points**: Clear interfaces with existing code
- **Backward compatibility**: Flags to enable/disable new features
- **Validation**: Test cases and verification procedures

### Documentation Pattern
- **Problem context**: Why was this needed?
- **Design decisions**: What approach was chosen and why?
- **Implementation details**: How was it built?
- **Usage examples**: How to use the new functionality?
- **Validation results**: How do we know it works?

## Reference Value

### For Future Initiatives
- **Planning templates**: Use existing plan structures as templates
- **Implementation patterns**: Learn from successful technical approaches  
- **Phase decomposition**: Examples of breaking complex work into manageable phases
- **Validation strategies**: Proven approaches for verifying success

### For Understanding Current System
- **Feature origins**: Why do certain features exist?
- **Design rationale**: Historical context for architectural decisions
- **Evolution tracking**: How did the system reach its current state?

## Cross-References

- **Current initiatives**: <doc-ref type="status">docs/PROJECT_STATUS.md</doc-ref>
- **Active initiatives**: <doc-ref type="workflow-guide">docs/initiatives/CLAUDE.md</doc-ref>
- **Registration usage**: <doc-ref type="guide">docs/MODEL_COMPARISON_GUIDE.md</doc-ref>
- **Evaluation metrics**: <doc-ref type="workflow-guide">scripts/studies/CLAUDE.md</doc-ref>