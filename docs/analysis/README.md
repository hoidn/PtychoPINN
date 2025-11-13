# Iteration Analysis Reports

This directory contains comprehensive audits of engineering progress in the PtychoPINN repository.

## Available Reports

### Iteration Audit 280-320
**File**: `iteration_audit_280-320.md`  
**Date**: 2025-11-12  
**Scope**: 40 iterations (Nov 11-12, 2025)  
**Method**: Three-pass scoring (Summary + Code Diff + Semantic)

**Key Findings**:
- Prompt interventions increased productivity by **168-205%** (mean: 18.0 → 54.8)
- Three major interventions: Dwell persistence (iter 290), Three-tier gates (iter 293), Artifact policy (iter 306)
- Most effective mechanism: **Dwell persistence** forcing execution after 3 planning loops
- Top iterations score 70-79, characterized by: multiple key files, test coverage, semantic score 85+

**Contents**:
1. Complete iteration timeline with scores (280-320)
2. ASCII score trend visualization with intervention markers
3. Three scoring pass methodologies (Summary, Code, Semantic)
4. Detailed prompt change analysis with git diffs
5. Prompt rule → Code implementation mapping table
6. Pre/post statistics with effect sizes
7. Key patterns and insights (5 major patterns identified)
8. Limitations and confounds discussion
9. Three actionable recommendations
10. Appendix with high-score and low-score sample evidence

## Methodology

### Three-Pass Scoring System

**Pass A: Summary-Based (0-100)**
- Reads Ralph summaries from `logs/feature-torchapi-newprompt/ralph-summaries/`
- Heuristic scoring based on keywords (tests passing, implementation, blocked, etc.)
- Weight: 30% of aggregate score

**Pass B: Code Diff Heuristic (0-100)**
- Objective metrics from `git diff --name-status --stat`
- Scores based on volume of changes in key paths (dbex/, scripts/, tests/)
- Weight: 30% of aggregate score

**Pass C: Deep Semantic (0-100)**
- Reads actual diffs with context (`git diff -U3`)
- Pattern matching for functions, classes, tests, core logic
- Weight: 40% of aggregate score (emphasizes actual code changes)

**Aggregate Score**
- Weighted average: 0.3×A + 0.3×B + 0.4×C
- Range: 0-100
- Rubric:
  - 90-100: Core correctness aligned to spec with validating tests
  - 70-89: Significant functional progress or high-leverage diagnostics
  - 50-69: Useful hardening/refactors, test alignment
  - 30-49: Limited movement, metadata/harness tweaks
  - 0-29: No visible product movement

## Usage

To reproduce the analysis:

```bash
# Run the analysis script
python3 /tmp/analyze_iterations.py

# Generate prompt analysis
python3 /tmp/prompt_analysis.py

# Create prompt-to-code mapping
python3 /tmp/prompt_code_map.py
```

Scripts are available in the report or can be extracted from commit history.

## Related Documentation

- `/docs/INITIATIVE_WORKFLOW_GUIDE.md` - Initiative planning and artifact storage
- `/docs/fix_plan.md` - Master task ledger
- `/prompts/supervisor.md` - Galph (supervisor) prompt
- `/prompts/main.md` - Ralph (engineer) prompt
- `/CLAUDE.md` - Repository constitution for AI agents

## Future Work

- **Longitudinal analysis**: Track iterations 1-280 to identify earlier patterns
- **Confound analysis**: Control for task complexity by categorizing initiatives
- **A/B testing**: Experimental design for prompt changes with control groups
- **Predictive modeling**: Use early-iteration features to predict final scores
- **Cross-repo validation**: Apply methodology to other projects with similar two-agent loops

---

**Maintained by**: Iteration Analysis Auditor  
**Last Updated**: 2025-11-12
