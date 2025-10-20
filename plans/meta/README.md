# Meta-Level Process Documents

**Purpose:** This directory contains process improvement plans, workflow optimizations, and meta-templates for development methodology - not implementation plans for specific features.

---

## What Goes Here vs plans/active/

### ✅ Plans/Meta (This Directory)
**Process improvement and methodology:**
- Process optimization proposals
- CLAUDE.md update proposals
- Development workflow improvements
- Meta-templates for initiative planning
- Quality gate definitions
- Retrospective analyses
- Agent workflow refinements

**Examples:**
- How we should plan initiatives (templates)
- How we should test (test_strategy_template.md)
- How we should analyze constraints (constraint_analysis_template.md)
- Improvements to supervisor/engineer workflow
- Documentation structure improvements

### ✅ Plans/Active
**Implementation plans for specific features/initiatives:**
- INTEGRATE-PYTORCH-001 (feature implementation)
- TEST-SUITE-TRIAGE (test organization)
- VECTOR-TRICUBIC (performance optimization)
- Specific bugs, features, refactorings

**Examples:**
- Building the PyTorch backend
- Fixing a specific bug
- Adding a new feature
- Optimizing a module

---

## Current Meta-Level Documents

### Process Optimizations
**File:** `PROCESS_OPTIMIZATIONS.md`
**Purpose:** Comprehensive analysis of agentic development efficiency based on 15 iterations
**Key Content:**
- 4 critical optimizations (OPT-1 through OPT-4)
- ROI analysis and implementation roadmap
- Expected 2.6x speedup with proposed changes
- Risk mitigation and success metrics

**Status:** PROPOSED - pending review and approval

### CLAUDE.md Updates
**File:** `CLAUDE_MD_UPDATES.md`
**Purpose:** Specific directive additions to prevent test infrastructure issues
**Key Content:**
- 3 new critical directives to add to CLAUDE.md
- Updated test harness compatibility section
- Rollout strategy and success metrics
- Exact text ready to copy-paste

**Status:** PROPOSED - ready for implementation

### Templates Directory
**Location:** `templates/`
**Purpose:** Reusable templates for initiative planning and quality gates

**Available Templates:**
1. `test_strategy_template.md` - Test infrastructure design
2. `constraint_analysis_template.md` - Phase 0 environment analysis

**Usage:** Copy template to `plans/active/<initiative>/` when starting new work

---

## How to Use This Directory

### For Process Improvements
1. Document analysis in this directory (like PROCESS_OPTIMIZATIONS.md)
2. Create specific proposals (like CLAUDE_MD_UPDATES.md)
3. Get review/approval from maintainers
4. Implement changes to CLAUDE.md, prompts/, docs/
5. Update templates as needed
6. Archive successful improvements for reference

### For New Initiative Templates
1. Start with templates in `templates/`
2. Copy to `plans/active/<initiative>/`
3. Fill in specific details
4. Reference back to template if issues arise

### For Retrospectives
1. Create `YYYY-MM_retrospective.md` in this directory
2. Analyze what worked/didn't work
3. Propose improvements as new documents
4. Update templates based on learnings

---

## Meta vs Implementation Decision Tree

**Ask yourself:** Is this about...

**HOW we develop?** → `plans/meta/`
- Process improvements
- Workflow changes
- Template updates
- Quality gate definitions
- Agent coordination improvements

**WHAT we're building?** → `plans/active/`
- Feature implementations
- Bug fixes
- Refactorings
- Specific optimizations
- Integration work

**Still unclear?** Ask:
- "Will other initiatives use this template?" → meta/
- "Is this specific to one feature?" → active/
- "Does this change how we work in general?" → meta/
- "Is this changing agent prompts or CLAUDE.md?" → meta/

---

## Directory Structure

```
plans/
├── meta/                          # THIS DIRECTORY - Process & methodology
│   ├── README.md                  # This file
│   ├── PROCESS_OPTIMIZATIONS.md   # Efficiency analysis & proposals
│   ├── CLAUDE_MD_UPDATES.md       # Directive updates
│   └── templates/                 # Meta-templates for initiatives
│       ├── test_strategy_template.md
│       └── constraint_analysis_template.md
│
├── active/                        # Feature implementation plans
│   ├── INTEGRATE-PYTORCH-001/     # Specific feature
│   ├── TEST-SUITE-TRIAGE/         # Specific improvement
│   └── ...
│
└── [other planning docs]          # Legacy or one-off plans
```

---

## Maintenance

### When to Add Documents
- After completing major initiatives (retrospective)
- When proposing workflow improvements
- When creating new reusable templates
- After identifying recurring patterns

### When to Archive
- Process improvements implemented and validated
- Templates superseded by better versions
- Proposals rejected or no longer relevant

### Review Cadence
- **Monthly:** Review proposals, approve for implementation
- **Quarterly:** Retrospective on process improvements
- **Annually:** Major template and workflow review

---

## References

**Analysis Sources:**
- `logs/feature-torchapi/galph-summaries/META_ANALYSIS.md` - Full iteration analysis
- `logs/feature-torchapi/galph-summaries/GAPS_AND_ISSUES.md` - Issue catalog
- `logs/feature-torchapi/galph-summaries/DECISION_LOG.md` - Decision evolution

**Related Documentation:**
- `docs/DEVELOPER_GUIDE.md` - Current development workflow
- `docs/INITIATIVE_WORKFLOW_GUIDE.md` - Current initiative structure
- `prompts/supervisor.md` - Supervisor agent workflow
- `CLAUDE.md` - Core project directives

---

**Last Updated:** 2025-10-16
**Maintainer:** Project leads / supervisor agent
**Status:** Active - contains proposed improvements pending approval
