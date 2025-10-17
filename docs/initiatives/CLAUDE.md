# Active Initiatives Agent Guide

## Quick Context
- **Purpose**: Planning documents for current/active development initiatives
- **Check Status**: Always consult `PROJECT_STATUS.md` for current initiative status
- **Organization**: By initiative name (plan_*.md, implementation_*.md, phase_*_checklist.md)
- **Dynamic**: Content changes based on active work

## Current Initiatives

**⚠️ Always check `docs/PROJECT_STATUS.md` for up-to-date initiative list and status**

As of last update:
- **No active initiative** - Project awaiting new R&D plan
- Check `PROJECT_STATUS.md` for any newly started initiatives

## Document Structure

### Plan Documents (plan_*.md)
- **Purpose**: R&D objectives and experimental design
- **Content**: Problem statement, success criteria, research approach
- **Lifecycle**: Created at initiative start, guides all subsequent work

### Implementation Documents (implementation_*.md)  
- **Purpose**: Technical specifications and implementation steps
- **Content**: Phase breakdown, technical architecture, file modifications
- **Lifecycle**: Created after R&D planning, before execution

### Phase Checklists (phase_*_checklist.md)
- **Purpose**: Execution tracking and validation
- **Content**: Specific tasks, file paths, verification steps
- **Lifecycle**: Generated per phase, updated during execution

## Workflow Integration

### For Initiative Execution
1. **Read plan document** → Understand objectives and success criteria
2. **Review implementation document** → Understand technical approach
3. **Work through phase checklists** → Execute step-by-step
4. **Verify success tests** → Ensure phase completion
5. **Update progress tracking** → Mark completed tasks

### For Technical Details
- **Architecture decisions**: Reference implementation documents
- **Task breakdown**: Use phase checklists for specific actions
- **Success criteria**: Verify against plan document objectives

### For Context/Objectives
- **Problem understanding**: Start with plan documents
- **Design rationale**: Implementation documents explain approach
- **Progress tracking**: Phase checklists show current status

## Initiative Management

### Starting New Initiatives
1. **Create plan document** → Define R&D objectives
2. **Create implementation document** → Break down technical work
3. **Generate phase checklists** → Create execution roadmap
4. **Update PROJECT_STATUS.md** → Mark as active initiative

### During Initiative Execution
1. **Work through checklists** → Complete tasks in order
2. **Update progress** → Mark completed tasks
3. **Verify success tests** → Ensure quality gates
4. **Document decisions** → Add notes to checklists

### Completing Initiatives
1. **Final validation** → Complete final phase checklist
2. **Archive documents** → Move to appropriate archive location
3. **Update PROJECT_STATUS.md** → Mark as completed
4. **Document outcomes** → Record final results

## Historical Context

### Previous Active Initiatives
- **MS-SSIM Correction**: Completed metric validation initiative
- **Image Registration**: Moved to completed (docs/refactor/)
- **Evaluation Enhancements**: Moved to completed (docs/refactor/)

### Initiative Patterns
- **Focused scope**: 1-2 weeks typical duration
- **Phase-based execution**: 2-4 phases plus final validation
- **Quality gates**: Success tests at each phase boundary
- **Documentation-driven**: Plan → Implementation → Execution

## Cross-References

- **Initiative status**: <doc-ref type="status">docs/PROJECT_STATUS.md</doc-ref>
- **Completed initiatives**: <doc-ref type="workflow-guide">docs/refactor/CLAUDE.md</doc-ref>
- **Research studies**: <doc-ref type="workflow-guide">docs/studies/CLAUDE.md</doc-ref>
- **Sampling studies**: <doc-ref type="workflow-guide">docs/sampling/CLAUDE.md</doc-ref>
- **Initiative workflow**: <doc-ref type="workflow-guide">docs/INITIATIVE_WORKFLOW_GUIDE.md</doc-ref>