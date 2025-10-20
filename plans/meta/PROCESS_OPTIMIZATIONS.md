# Agentic Development Process Optimizations
**Status:** PROPOSED
**Based On:** Meta-analysis of galph iterations 1-15 (feature/torchapi branch)
**Expected Impact:** 2.6x speedup, improved quality, reduced rework
**Priority:** HIGH (implements learnings from 6.5 hours of development)

---

## Executive Summary

Analysis of 15 supervisor iterations revealed that **50% of development time was spent on rework** due to:
1. Test infrastructure designed during implementation (not upfront)
2. CI constraints discovered after code written
3. Test execution status ambiguity (SKIPPED vs PASSED)
4. Late feedback cycles masking bugs

This document proposes **4 critical optimizations** that would have saved 8-10 iterations (~60% improvement).

---

## Optimization Roadmap

### Tier 1: Must-Fix (ROI > 3x)
- **OPT-1:** Test Infrastructure Design Phase (saves 4-5 iterations)
- **OPT-2:** Test Execution Quality Gate (saves 2-3 iterations)

### Tier 2: High-Value (ROI > 2x)
- **OPT-3:** Upfront Constraint Analysis (saves 2-3 iterations)
- **OPT-4:** Evidence-Planning Fusion (saves 1-2 iterations)

**Total savings:** 9-10 iterations out of 15 (60% efficiency gain)

---

## OPT-1: Test Infrastructure Design Phase

### Problem
**Observed in:** Iterations 9, 12, 13, 15
**Cost:** 4-5 iterations of rework

Test framework incompatibilities discovered after implementation:
- unittest.TestCase + pytest.parametrize mixing (Iter 13)
- PyTorch dependency blocking CI (Iter 11-14)
- No torch-optional design upfront
- 38 parity tests blocked on framework refactor

### Root Cause
Test infrastructure treated as implementation detail, not architectural decision.

### Solution
**Add mandatory Test Infrastructure Design review before Phase B (implementation).**

**Location:** `plans/active/<initiative>/test_strategy.md`

**Template:** See `plans/templates/test_strategy_template.md`

**Required Content:**
1. Framework Selection & Compatibility
2. CI/CD Constraint Analysis
3. Test Tier Definitions (unit/integration/smoke)
4. Execution Proof Requirements
5. Mock/Stub Strategy

### Implementation

#### 1. Add to CLAUDE.md
```xml
<directive level="critical" purpose="Design test infrastructure before implementation">
  Before Phase B (implementation) of any initiative involving tests, you **MUST**
  complete a test infrastructure design review documented in
  `plans/active/<initiative>/test_strategy.md`.

  Required coverage:
  - Framework selection and compatibility (pytest vs unittest, parametrization)
  - CI/CD constraints and optional dependency handling
  - Test tier definitions (unit/integration/smoke)
  - Execution proof requirements (PASSED vs SKIPPED criteria)
  - Mock/stub strategy for unavailable dependencies

  Reference this design in Phase B planning and validate before writing first test.
</directive>
```

#### 2. Create Template
Place template at: `plans/templates/test_strategy_template.md`

#### 3. Update Workflow
- Supervisor must check for test_strategy.md before approving Phase B start
- Ralph must reference test_strategy.md when writing tests
- Quality gate: No tests written without approved test strategy

### Success Metrics
- ✅ Test framework issues discovered in Phase A (planning)
- ✅ CI constraints known before writing tests
- ✅ Zero test refactoring iterations
- ✅ All tests RUN (not SKIP) from first execution

---

## OPT-2: Test Execution Quality Gate

### Problem
**Observed in:** Iterations 10-11
**Cost:** 2-3 iterations of false confidence

Tests showing SKIPPED status treated as success:
- Iter 10: Implementation marked "complete" ✅
- Iter 11: Actually broken (TypeError, bugs masked) ❌
- Gap: Tests never actually ran

### Root Cause
No distinction between:
- Tests PASSED (validated)
- Tests SKIPPED (not validated)
- Tests not run (unknown)

### Solution
**Enforce execution proof requirement for task completion.**

### Implementation

#### 1. Update Supervisor Prompt
Add to `prompts/supervisor.md`:

```markdown
## Task Completion Quality Gate

A task is NOT complete if:
- ❌ Tests show SKIPPED status without justification
- ❌ No pytest execution log in artifacts
- ❌ Test output not proven in reports
- ❌ Tests not actually executed (framework import-only check)

Required evidence from ralph:
- ✅ Pytest output showing PASSED (not SKIPPED)
- ✅ Test execution log at `plans/active/<initiative>/reports/<timestamp>/pytest.log`
- ✅ Explicit justification for any SKIPPED tests in summary.md
- ✅ Proof tests actually ran their assertions (not just imports)

Acceptable SKIP justifications:
- Hardware unavailable (GPU tests on CPU-only CI)
- External service unavailable (integration tests)
- Long-running tests (performance benchmarks)

UNACCEPTABLE:
- Missing optional dependency (use torch-optional pattern)
- Framework incompatibility (design error, not acceptable)
```

#### 2. Update input.md Template
Add required artifacts section:

```markdown
## Artifacts Required

Mapped tests: pytest <path>::<test> -vv
└─> MUST show PASSED or justified XFAIL (not SKIPPED)

Reports directory: plans/active/<initiative>/reports/<timestamp>/
├─ pytest.log         # REQUIRED: Full pytest output
├─ summary.md         # REQUIRED: Test execution summary
└─ [additional artifacts]

Quality Gate:
- [ ] Tests executed (not skipped)
- [ ] pytest.log shows PASSED status
- [ ] Any SKIPs explicitly justified
```

#### 3. Supervisor Validation Step
Before marking task complete, galph must:
1. Read pytest.log from artifacts
2. Verify PASSED status (not SKIPPED)
3. Count and justify any SKIPs
4. Confirm assertions actually ran

### Success Metrics
- ✅ Zero false "complete" iterations
- ✅ Bugs discovered in same iteration as implementation
- ✅ All tests proven to execute
- ✅ Skip justifications documented

---

## OPT-3: Upfront Constraint Analysis

### Problem
**Observed in:** Iterations 11, 13, 14
**Cost:** 3-4 iterations implementing workarounds

CI constraints discovered during implementation:
- No PyTorch in CI environment
- GPU unavailable
- Framework dependencies assumed available
- Workarounds designed reactively (torch-optional harness)

### Root Cause
Environment constraints analyzed during implementation, not planning.

### Solution
**Add Phase 0: Constraint Analysis to initiative template.**

**Location:** Section in `plans/active/<initiative>/implementation.md`
**Template:** See `plans/templates/constraint_analysis_template.md`

### Implementation

#### 1. Update Initiative Template
Add Phase 0 to standard structure:

```markdown
# <INITIATIVE-NAME> Implementation Plan

## Phase 0: Constraint Analysis ⏳ [Status]
**Exit Criteria:** All constraints documented, mitigations planned

### Environment Constraints
- [ ] Python version requirements
- [ ] Framework availability (torch/tf/jax in dev vs CI)
- [ ] Hardware requirements (CPU/GPU/TPU)
- [ ] CI environment matches dev? Document gaps
- [ ] External dependencies and API access

### Integration Constraints
- [ ] Upstream/downstream API compatibility
- [ ] Data format requirements (ref specs/)
- [ ] Performance requirements (latency/throughput)
- [ ] Persistence/serialization constraints

### Testing Constraints
- [ ] CI test execution capability (full/limited/offline)
- [ ] Mock/stub requirements
- [ ] Test data availability and size
- [ ] Long-running test handling

**Deliverable:** constraints.md or inline documentation above
**Artifacts:** `plans/active/<initiative>/reports/<timestamp>/constraint_analysis.md`

## Phase A: [Next phase...]
```

#### 2. Supervisor Checklist
Before moving to Phase A, galph must verify:
- Constraint analysis complete
- Gaps documented
- Mitigations planned
- Test strategy references constraints

#### 3. Ralph Execution
When directed to Phase 0:
- Document current environment (python --version, pip freeze)
- Test CI capabilities (what runs, what skips)
- Identify mismatches with dev environment
- Propose mitigations (mocks, optional imports, skip rules)

### Success Metrics
- ✅ CI constraints known in Phase 0
- ✅ Torch-optional design from start (not retrofit)
- ✅ Zero "discovered we can't run this" iterations
- ✅ Mitigations designed proactively

---

## OPT-4: Evidence-Planning Fusion

### Problem
**Observed in:** Iterations 4, 8, 14
**Cost:** 3 iterations of pure evidence gathering (non-implementation)

Separate iterations for:
- Evidence collection (run analysis scripts)
- Planning (synthesize evidence into plan)
- Implementation (execute plan)

Creates context-switching overhead and delays implementation.

### Root Cause
Rigid separation of evidence vs planning action types.

### Solution
**Fuse evidence collection into planning iterations where feasible.**

### Implementation

#### 1. Update Supervisor Guidelines
Add to `prompts/supervisor.md`:

```markdown
## Evidence-Planning Fusion

PREFER: Combine evidence gathering with planning iteration
- Run analysis scripts during planning
- Include evidence artifacts in planning deliverables
- Single iteration produces: evidence + plan + directive

DEFER to separate iteration when:
- Analysis expected to take >30 minutes
- Complex debugging required (multiple hypotheses)
- Evidence gathering is blocking (need data before any planning)

EXAMPLE - Good Fusion:
Action: Planning
Deliverables:
- Run config field mapping script
- Analyze output (12 missing fields, 6 mismatches)
- Create MVP scope based on evidence
- Write plan referencing analysis
Artifacts: field_mapping.csv + mvp_plan.md

EXAMPLE - Separate Needed:
Action: Evidence Collection (Debug)
Deliverables:
- Reproduce bug with minimal test case
- Test 3 hypotheses systematically
- Identify root cause
- Defer planning to next iteration (requires stakeholder input)
```

#### 2. Practical Application
When supervisor planning:
1. Identify evidence needs (e.g., field mapping, config analysis)
2. If < 30 min: Include script execution in planning iteration
3. If > 30 min: Separate evidence iteration
4. If debugging: Always separate

### Success Metrics
- ✅ Average 1 iteration saved per initiative
- ✅ Faster implementation start
- ⚠️ Monitor: Evidence quality must not degrade

### Caveat
**Evidence-first approach prevented bad decisions** - don't eliminate systematic analysis, just reduce context switching where safe.

---

## Implementation Roadmap

### Week 1: Critical Infrastructure (OPT-1, OPT-2)
**Goal:** Prevent test framework issues in future initiatives

- [ ] Create `plans/templates/test_strategy_template.md`
- [ ] Add test infrastructure directive to `CLAUDE.md`
- [ ] Update `prompts/supervisor.md` with quality gate
- [ ] Document in `docs/DEVELOPER_GUIDE.md`

**Deliverables:**
- Test strategy template
- Updated CLAUDE.md (new directive)
- Updated supervisor prompt
- Developer guide section

**Validation:**
- Next initiative with tests must have test_strategy.md in Phase 0
- No test written without framework compatibility check
- All tests must show PASSED (not SKIPPED) for completion

### Week 2: Constraint Analysis (OPT-3)
**Goal:** Front-load environment constraint discovery

- [ ] Create `plans/templates/constraint_analysis_template.md`
- [ ] Update initiative template with Phase 0
- [ ] Add constraint checklist to supervisor workflow
- [ ] Document CI environment baseline

**Deliverables:**
- Constraint analysis template
- Updated initiative template
- CI environment documentation

**Validation:**
- Next initiative starts with Phase 0 constraint analysis
- CI gaps documented before Phase A
- Mitigations planned proactively

### Week 3: Process Refinement (OPT-4)
**Goal:** Reduce evidence-planning overhead

- [ ] Update supervisor guidelines for fusion
- [ ] Define <30 min threshold criteria
- [ ] Add examples to supervisor prompt
- [ ] Monitor evidence quality metrics

**Deliverables:**
- Updated supervisor guidelines
- Fusion criteria documented
- Quality monitoring plan

**Validation:**
- Evidence iterations reduced 30-50%
- Evidence quality maintained
- Implementation velocity increased

---

## Metrics & Success Criteria

### Lead Indicators (Week 1-3)
- Templates created and in use
- CLAUDE.md updated with directives
- First initiative uses new templates

### Lag Indicators (Month 1-2)
- **Efficiency:** Iterations to MVP reduced 40-60%
- **Quality:** Zero test refactoring iterations
- **Confidence:** Zero false "complete" iterations
- **Velocity:** Time to MVP reduced 50%+

### Baseline (feature/torchapi branch)
```
Iterations to MVP: 15
Time to MVP: 6.5 hours
Rework iterations: 5 (33%)
Overhead iterations: 3 (20%)
Efficiency: 47%
```

### Target (with optimizations)
```
Iterations to MVP: 6-8
Time to MVP: 2.5-3.5 hours
Rework iterations: 1 (12%)
Overhead iterations: 1 (12%)
Efficiency: 75%
```

---

## Risk Mitigation

### Risk: Upfront Design Overhead
**Concern:** Phase 0 adds planning time
**Mitigation:**
- Template-driven (30 min max)
- Saves 8-10 iterations (net positive)
- Can be parallelized with other work

### Risk: Process Rigidity
**Concern:** Too many required steps
**Mitigation:**
- Phase 0 required only for initiatives with tests
- Templates are guides, not checklists
- Supervisor has discretion for simple work

### Risk: False Confidence in New Process
**Concern:** New templates become box-checking
**Mitigation:**
- Supervisor validates, not just checks existence
- Quality gates enforce actual execution proof
- Monthly retrospectives to assess effectiveness

---

## Retrospective Plan

### After First Initiative Using New Process
**Questions:**
1. Did test_strategy.md prevent framework issues?
2. Was constraint analysis accurate?
3. Were quality gates effective?
4. Time saved vs overhead added?

**Adjustments:**
- Refine templates based on usage
- Update quality gates if gaps found
- Document lessons learned

### Quarterly Review
**Metrics:**
- Average iterations to MVP
- Percentage of rework iterations
- Test framework issues (should be zero)
- Developer satisfaction survey

---

## References

**Analysis Source:**
- `logs/feature-torchapi/galph-summaries/META_ANALYSIS.md`
- `logs/feature-torchapi/galph-summaries/GAPS_AND_ISSUES.md`
- `logs/feature-torchapi/galph-summaries/DECISION_LOG.md`

**Related Documents:**
- `docs/DEVELOPER_GUIDE.md` - For process guidelines
- `docs/INITIATIVE_WORKFLOW_GUIDE.md` - For initiative structure
- `prompts/supervisor.md` - For supervisor workflow
- `CLAUDE.md` - For critical directives

**Templates Created:**
- `plans/templates/test_strategy_template.md` (OPT-1)
- `plans/templates/constraint_analysis_template.md` (OPT-3)

---

**Document Status:** PROPOSED
**Next Action:** Review and approve, then implement Week 1 deliverables
**Owner:** Project maintainer / supervisor agent
**Last Updated:** 2025-10-16
