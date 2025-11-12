# Iteration Analysis Audit — Index and Guide

**Audit Completion Date:** 2025-11-11
**Auditor:** iteration-analysis-auditor agent (read-only, Environment Freeze respected)
**Repository:** PtychoPINN (feature/torchapi-newprompt branch)
**Analysis Scope:** Iterations 262–293 (30 iterations, ~3 hours of agent work)

---

## Quick Start

**For quick overview:** Read `iteration_audit_executive_summary.md` (2-page summary, 5-minute read)

**For detailed analysis:** Read `iteration_analysis_audit_full.md` (comprehensive report, 8 sections)

**For raw data:** Import `iteration_scores_262-293_extended.csv` (30 rows × 9 columns, suitable for plotting/analysis)

---

## Document Hierarchy

```
docs/
├── ITERATION_AUDIT_INDEX.md                    ← YOU ARE HERE (index + guide)
├── iteration_audit_executive_summary.md        ← Start here (executive summary)
├── iteration_analysis_audit_full.md            ← Full audit (all 8 sections)
├── iteration_scores_262-293_extended.csv       ← Score data (CSV format)
└── iteration_scores_262-291.csv                ← Original partial scores (archived)
```

---

## Section Map (Full Audit Report)

Navigate directly to specific sections in `iteration_analysis_audit_full.md`:

| Section | Title | Key Content |
|---------|-------|-------------|
| 1 | Per-Iteration Scores and Timeline | Extended score table (30 rows), ASCII plot, scoring methodology |
| 2 | Deep Semantic Analysis | Intent/effect/impact analysis for i=285, i=287, i=289, i=292 with `path:line` anchors |
| 3 | Diff-Only Pass | Objective code/test changes extracted via `git diff --numstat` |
| 4 | Prompt Changes and Attribution | Rule → Code map (PYTHON-ENV-001, DATA-001, TEST-CLI-001, etc.) |
| 5 | Pre/Post Statistical Check | Descriptive stats, limitations, small-n caveats |
| 6 | Key Trends and Inflection Points | 5 trends (planning saturation, blockers, breakthroughs), 4 inflection points |
| 7 | Next Steps | Automation proposals (score_iterations.py, check_dwell_violations.py, prompt_rule_index) |
| 8 | Conclusions | Strengths, weaknesses, recommendations (6 prioritized) |

---

## Key Findings (At a Glance)

### Quantitative Summary
- **30 iterations analyzed** (i=262–293)
- **26 planning-only loops** (87%) with no production code
- **4 implementation loops** (13%) with code changes
- **Mean aggregate score:** 24.5 / 100 (moderate-low)
- **Highest score:** 44 (i=289: pytest GREEN + pipeline launch)

### Critical Blockers
1. **Planning saturation** (i=262–284): 23 consecutive planning loops violated dwell enforcement rules
2. **Workspace mismatch** (i=277–281): 5 iterations consumed by PtychoPINN vs PtychoPINN2 confusion
3. **Git rebase thrashing** (i=282–288): 200+ consecutive rebase failures due to unstaged changes

### High-Impact Commits
- **5cd130d3** (i=287): NPZ allow_pickle fix → `studies/fly64_dose_overlap/overlap.py:232`
- **811f4264** (i=292): PYTHON-ENV-001 policy → `docs/DEVELOPER_GUIDE.md:127` + 8 files
- **c8eb30bb** (i=292): Dose filtering → `studies/fly64_dose_overlap/generation.py:42`
- **854b0014** (i=289): Phase G pytest GREEN + orchestration launch

---

## How to Use This Audit

### For Project Managers
**Goal:** Understand overall progress and blockers

1. Read **executive summary** → Section "Key Findings"
2. Review **blocker analysis table** → Identify process bottlenecks
3. Check **recommendations** → Prioritize fixes (dwell enforcement, workspace guards, autostash)

### For Prompt Engineers
**Goal:** Assess prompt rule effectiveness and adoption

1. Read **Section 4** (Prompt Changes and Attribution) → See which rules landed in code
2. Review **Section 6** (Trends) → Understand delayed adoption patterns
3. Check **Section 5** (Statistical Check) → Note limitations for future pre/post comparisons

### For Agent Developers
**Goal:** Improve agent autonomy and reduce planning saturation

1. Study **Section 2** (Deep Semantic Analysis) → Learn from successful implementation loops (i=285, i=289, i=292)
2. Review **Section 6 Trend 1** → Understand planning saturation failure mode
3. Implement **Section 7 recommendations** → Automate scoring, dwell enforcement, rule traceability

### For Researchers
**Goal:** Extract quantitative data for analysis

1. Import **CSV file** → `docs/iteration_scores_262-293_extended.csv`
2. Columns: `iter, summary_score, diff_score, semantic_score, aggregate_score, rationale, key_changes, commit_hash, timestamp`
3. Suitable for: Time series plots, correlation analysis, blocker clustering, score distribution histograms

---

## Scoring Rubric (Quick Reference)

**0–29 (Low):** Planning-only, git sync, minimal artifacts
- Example: i=263 (score=12) — test artifacts only, no implementation

**30–49 (Moderate):** Execution prep, partial implementation, hardening
- Example: i=285 (score=34) — Phase C initiated, 4.7GB dataset generated

**50–69 (Good):** Significant functional progress, refactors with tests
- No examples in analyzed window (i=262–293)

**70–89 (Very Good):** High-leverage progress, multiple acceptance criteria met
- Example: i=289 (score=44, nearest) — pytest GREEN + full pipeline launch

**90–100 (Excellent):** Core correctness improvements with validating tests
- No examples in analyzed window (ceiling: i=289 at 44)

---

## Prompt Rule → Code Index (Quick Lookup)

For complete attribution map with rationales, see **Section 4** of full audit.

| Rule ID | Prompt Source | Code Implementation | Iteration | Status |
|---------|---------------|---------------------|-----------|--------|
| PYTHON-ENV-001 | `prompts/main.md:37` (interpreter policy) | `docs/DEVELOPER_GUIDE.md:127` | i=292 | ✅ Landed |
| DATA-001 | `docs/findings.md` (allow_pickle) | `overlap.py:232`, `training.py:409` | i=287 | ✅ Landed |
| TEST-CLI-001 | `prompts/main.md:100` (pytest before CLI) | `green/pytest_*.log` | i=285+ | ✅ Applied |
| TYPE-PATH-001 | `docs/findings.md` (hub-relative paths) | `run_phase_g_dense.py:234` | i=270 | ✅ Landed |
| Dwell enforcement | `prompts/supervisor.md:59-66,262+` (three-tier 2/4/6) | `galph_memory.md` + `ralph_last_commit` | i=290+ | ✅ Updated |
| Implementation floor | `prompts/main.md:32` (code task req) | — | i=290+ | ⚠️ Violated i=262-284 |

**Legend:**
- ✅ **Landed:** Code change committed and merged
- ✅ **Applied:** Process rule followed in agent behavior
- ⚠️ **Delayed:** Rule adopted after 200+ iterations post-prompt revision
- ⚠️ **Violated:** Rule not enforced during analyzed window

---

## Limitations and Caveats

### Statistical Limitations
- **Small n:** Only 30 iterations analyzed; insufficient for robust statistical inference
- **No pre-group data:** Cannot compute pre/post prompt revision comparison (need iterations 1–44)
- **Autocorrelation:** All iterations clustered on single initiative (STUDY-SYNTH-FLY64)
- **Confounds:** Blockers (workspace, git) distort score signal

### Methodological Constraints
- **Environment Freeze respected:** No packages installed/upgraded; missing tool blockers recorded but not resolved
- **Read-only analysis:** No code modifications; audit cannot fix detected issues
- **Scoring subjectivity:** Aggregate score is auditor's best judgment (not algorithmic)
- **Coverage:** Only iterations 262–293 examined; earlier/later work not assessed

### Causal Inference
- **No controlled experiment:** Cannot attribute score changes to prompt revisions (correlation ≠ causation)
- **Learning curve effects:** Analyzed window post-dates prompt revision by 200+ iterations
- **External interventions:** Commit 317c360a "unblock execution" may confound dwell enforcement attribution

---

## Next Actions (For Repository Maintainers)

### Immediate (High Priority)
1. ✅ **Review audit findings** — Read executive summary with project team
2. 🔧 **Implement workspace guards** — Add `assert_workspace()` to all CLI entry points (prevent PtychoPINN2 confusion)
3. 🔧 **Enable autostash by default** — Update `scripts/orchestration/git_bus.py` to always stash before pull/rebase

### Short-term (1-2 weeks)
4. 🤖 **Automate iteration scoring** — Build `scripts/tools/score_iterations.py` per Section 7 recommendations
5. 📊 **Create dwell dashboard** — Implement `scripts/tools/check_dwell_violations.py` for compliance monitoring
6. 📚 **Build prompt rule index** — Generate `docs/prompt_rule_index.md` with bidirectional Rule ↔ Code map

### Long-term (1-3 months)
7. 📈 **Collect pre-revision baseline** — Analyze iterations 1–44 for pre/post comparison (need n≥30 per group)
8. 🎯 **Real-time score dashboard** — Web UI or CLI tool for live iteration tracking
9. 🧪 **A/B test prompt variants** — Controlled experiment with randomized prompt assignment

---

## Contact and Feedback

**Questions about the audit methodology?**
- See **Section 1** (Scoring Methodology) in full audit report
- Review **Appendix A** (Methodology Details) for data sources, tools, rubric

**Found an error or inconsistency?**
- Check commit hashes in CSV against `git log` output
- Verify `path:line` anchors in attribution map (Section 4)
- Report discrepancies with iteration number + expected vs actual values

**Want to extend the analysis?**
- Use CSV file as starting point for custom analysis
- Import into Jupyter, R, or spreadsheet tool
- Contact repository maintainers for access to pre-i=262 data

---

**End of Index**

For immediate action, start with: `docs/iteration_audit_executive_summary.md`
