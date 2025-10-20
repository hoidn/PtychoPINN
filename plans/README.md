# Plans Directory Organization

This directory contains both active implementation plans and meta-level process documentation.

---

## Directory Structure

```
plans/
├── meta/                           # Process improvements & meta-templates
│   ├── README.md                   # Guide to meta-level documents
│   ├── PROCESS_OPTIMIZATIONS.md    # Efficiency analysis & proposals
│   ├── CLAUDE_MD_UPDATES.md        # Directive update proposals
│   └── templates/                  # Reusable initiative templates
│       ├── test_strategy_template.md
│       └── constraint_analysis_template.md
│
├── active/                         # Active feature/bug implementation plans
│   ├── INTEGRATE-PYTORCH-000/      # Planning refresh initiative
│   ├── INTEGRATE-PYTORCH-001/      # PyTorch backend integration
│   ├── TEST-SUITE-TRIAGE/          # Test organization
│   └── [other active initiatives]
│
└── [legacy plans]                  # One-off or archived plans
    ├── pytorch_integration_test_plan.md
    ├── ptychodus_pytorch_integration_plan.md
    └── vectorization.md
```

---

## Quick Reference

### I want to...

**Start a new feature/bug fix:**
→ Go to `active/` and create `<INITIATIVE-NAME>/implementation.md`
→ Use templates from `meta/templates/` for test strategy and constraints

**Improve the development process:**
→ Go to `meta/` and create a proposal document
→ Reference existing analyses in `logs/feature-torchapi/galph-summaries/`

**Find a template for planning:**
→ Go to `meta/templates/`
→ Copy template to `active/<initiative>/` and fill in

**Review process improvement proposals:**
→ Go to `meta/` and check `PROCESS_OPTIMIZATIONS.md` or `CLAUDE_MD_UPDATES.md`

---

## Key Distinctions

### Meta (Process & Methodology)
**Location:** `plans/meta/`

**Purpose:** How we plan, develop, and test in general
- Process improvements
- Workflow optimizations
- Quality gate definitions
- Reusable templates
- Agent coordination improvements
- Retrospectives and lessons learned

**Examples:**
- "We should design test infrastructure before implementation"
- "Add constraint analysis to Phase 0 of all initiatives"
- "Template for test strategy documents"

### Active (Feature Implementation)
**Location:** `plans/active/`

**Purpose:** What we're building right now
- Feature implementations
- Bug fixes
- Refactorings
- Specific optimizations
- Active development work

**Examples:**
- "Build PyTorch backend integration"
- "Fix config bridge field translation"
- "Optimize tricubic interpolation"

### Legacy (Historical/Archived)
**Location:** `plans/` (root)

**Purpose:** Completed or superseded plans
- One-off plans that don't fit elsewhere
- Historical planning documents
- Superseded by newer initiatives
- Reference material

---

## Workflow

### Starting New Work

1. **Check if process improvement or feature:**
   - Process improvement? → `meta/`
   - Feature/bug? → `active/`

2. **For new features:**
   ```bash
   # Create initiative directory
   mkdir -p plans/active/<INITIATIVE-NAME>

   # Copy templates
   cp plans/meta/templates/test_strategy_template.md \
      plans/active/<INITIATIVE-NAME>/test_strategy.md
   cp plans/meta/templates/constraint_analysis_template.md \
      plans/active/<INITIATIVE-NAME>/constraint_analysis.md

   # Create implementation plan
   # (use examples in active/ as reference)
   ```

3. **For process improvements:**
   ```bash
   # Create analysis document
   vim plans/meta/<PROPOSAL_NAME>.md

   # Reference meta-analysis if available
   # logs/feature-torchapi/galph-summaries/META_ANALYSIS.md
   ```

### Completing Work

**For active initiatives:**
- Move `active/<INITIATIVE>/` to `completed/<INITIATIVE>/` (or archive)
- Update `docs/fix_plan.md` with final status
- Extract lessons learned to `meta/` if applicable

**For meta improvements:**
- Implement changes (update CLAUDE.md, prompts/, docs/)
- Archive proposal in `meta/archive/` or delete if implemented
- Update templates based on learnings

---

## See Also

- `meta/README.md` - Detailed guide to process documents
- `docs/INITIATIVE_WORKFLOW_GUIDE.md` - How to run initiatives
- `docs/DEVELOPER_GUIDE.md` - Development best practices
- `CLAUDE.md` - Core project directives

---

**Last Updated:** 2025-10-16
**Maintainer:** Project leads
