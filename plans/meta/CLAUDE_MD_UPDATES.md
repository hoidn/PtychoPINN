# Proposed CLAUDE.md Updates
**Purpose:** Add critical directives to prevent test infrastructure issues
**Based On:** Analysis of 15 galph iterations showing 50% rework rate
**Impact:** Estimated 2.6x speedup, reduced rework from 50% to 15%

---

## What to Add

These directives should be added to `/home/ollie/Documents/PtychoPINN/CLAUDE.md` Section 2 (Core Project Directives).

---

## Directive 1: Test Infrastructure Design (OPT-1)

**Add after existing directives, before Section 3:**

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

  Use the template at `plans/templates/test_strategy_template.md`.

  Reference this design in Phase B planning and validate before writing first test.
</directive>
```

**Rationale:**
- Prevents unittest + pytest.parametrize mixing (4-5 iterations saved)
- Addresses CI constraints upfront (PyTorch unavailability)
- Defines execution proof requirements
- Blocks: Iterations 9, 12, 13, 15 rework

---

## Directive 2: Constraint Analysis (OPT-3)

**Add after test infrastructure directive:**

```xml
<directive level="critical" purpose="Analyze constraints before planning">
  Every initiative **MUST** begin with Phase 0: Constraint Analysis documenting
  environment, integration, and testing constraints in
  `plans/active/<initiative>/implementation.md` or separate
  `plans/active/<initiative>/constraint_analysis.md`.

  Required analysis:
  - Environment constraints (Python version, framework availability, hardware)
  - CI/dev environment gaps (PyTorch in dev but not CI, GPU availability)
  - Integration constraints (upstream/downstream APIs, data formats)
  - Testing constraints (CI capabilities, data size limits, mock requirements)

  Use the template at `plans/templates/constraint_analysis_template.md`.

  No implementation may begin until constraints are documented and mitigations planned.
</directive>
```

**Rationale:**
- Discovers CI limitations in Phase 0 (not during implementation)
- Enables torch-optional design from start (not retrofit)
- Plans mitigations proactively
- Blocks: Iterations 11, 13, 14 reactive workarounds

---

## Directive 3: Test Execution Proof (OPT-2)

**Update existing test-related directive or add new:**

```xml
<directive level="critical" purpose="Require test execution proof">
  A task involving tests is **NOT** complete unless tests show PASSED status
  (not SKIPPED) with execution proof in artifacts.

  Required evidence:
  - pytest execution log at `plans/active/<initiative>/reports/<timestamp>/pytest.log`
  - Test summary in `summary.md` with pass/fail/skip counts
  - Explicit justification for any SKIPPED tests (hardware, long-running, etc.)
  - Proof tests executed assertions (not just imports)

  Acceptable SKIP reasons:
  - Hardware unavailable (GPU tests on CPU-only CI) - mark @pytest.mark.gpu
  - External service unavailable (integration tests) - document in summary
  - Long-running tests (>5 min benchmarks) - mark @pytest.mark.slow

  UNACCEPTABLE:
  - Missing optional dependency (use torch-optional pattern instead)
  - Framework incompatibility (design error, must fix)
  - "Tests don't work in CI" (fix CI or test design, not acceptable)
</directive>
```

**Rationale:**
- Prevents false "complete" status from skipped tests
- Discovered bugs in same iteration (not next iteration)
- Eliminates false confidence
- Blocks: Iteration 10-11 gap (implementation "done" but actually broken)

---

## Updated Section 4.3: Test Harness Compatibility

**Replace existing Section 4.3 with enhanced version:**

```markdown
### 4.3. Test Harness Compatibility

**Framework Standardization:**
Write all new tests using **pure pytest** style. Do NOT mix `unittest.TestCase`
with pytest features (parametrization, fixtures).

**Pattern - Correct:**
```python
# tests/torch/test_config_bridge.py
import pytest

@pytest.fixture
def sample_config():
    return ModelConfig(...)

@pytest.mark.parametrize("field,expected", [
    ("N", 128),
    ("gridsize", 2),
])
def test_field_translation(sample_config, field, expected):
    result = translate_field(sample_config, field)
    assert result == expected
```

**Pattern - INCORRECT (causes TypeError):**
```python
import unittest
import pytest

class TestConfigBridge(unittest.TestCase):  # ❌ Don't mix!
    @pytest.mark.parametrize("field", [...])  # ❌ Incompatible!
    def test_field(self, field):
        ...
```

**Torch-Optional Pattern:**
For tests that depend on PyTorch (unavailable in CI), use the torch-optional
harness pattern documented in `tests/conftest.py`:

```python
# tests/conftest.py
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def pytest_collection_modifyitems(config, items):
    # Auto-skip tests requiring torch when unavailable
    skip_torch = pytest.mark.skip(reason="PyTorch not available")
    for item in items:
        if "torch" in item.keywords and not TORCH_AVAILABLE:
            item.add_marker(skip_torch)
```

**When modernizing existing tests:** Move entirely to pytest in a single loop
(don't leave mixed state). Refactoring test framework is acceptable technical debt
to address proactively.
```

**Rationale:**
- Documents the specific pattern that caused issues
- Provides correct and incorrect examples
- References the torch-optional pattern
- Makes explicit the "no mixing" rule

---

## Summary of Changes

### New Directives (3 total)
1. **Test Infrastructure Design** - Mandatory test_strategy.md before Phase B
2. **Constraint Analysis** - Mandatory Phase 0 with environment/CI analysis
3. **Test Execution Proof** - PASSED status required, not SKIPPED

### Updated Sections (1 total)
4. **Test Harness Compatibility** - Enhanced with patterns and torch-optional

### Templates Created (2 total)
- `plans/templates/test_strategy_template.md`
- `plans/templates/constraint_analysis_template.md`

---

## Implementation Checklist

**Week 1: Critical Directives**
- [ ] Add Directive 1 (Test Infrastructure Design) to CLAUDE.md Section 2
- [ ] Add Directive 2 (Constraint Analysis) to CLAUDE.md Section 2
- [ ] Add Directive 3 (Test Execution Proof) to CLAUDE.md Section 2
- [ ] Update Section 4.3 with enhanced test patterns
- [ ] Validate templates exist: test_strategy_template.md, constraint_analysis_template.md

**Week 1: Validation**
- [ ] Next initiative must create test_strategy.md (enforce in supervisor review)
- [ ] Next initiative must complete Phase 0 constraint analysis
- [ ] Test completion requires pytest.log artifact showing PASSED
- [ ] Document in DEVELOPER_GUIDE.md

**Week 2: Refinement**
- [ ] Update prompts/supervisor.md with quality gate details
- [ ] Add examples to TESTING_GUIDE.md
- [ ] Create pre-commit hook for smoke tests

---

## Expected Impact

### Before (Measured from feature/torchapi)
```
Iterations to MVP: 15
Time to MVP: 6.5 hours
Rework iterations: 5 (33%)
Efficiency: 47%
```

### After (Projected with new directives)
```
Iterations to MVP: 6-8
Time to MVP: 2.5-3.5 hours
Rework iterations: 1 (12%)
Efficiency: 75%+
```

### Prevented Issues
- ✅ unittest/pytest mixing (saves 4-5 iterations)
- ✅ Test skipping false confidence (saves 2-3 iterations)
- ✅ CI constraint surprises (saves 2-3 iterations)
- ✅ Late feedback cycles (saves 1-2 iterations)

**Total:** 9-13 iterations saved per initiative (60% improvement)

---

## Rollout Strategy

### Phase 1: Documentation (Week 1)
1. Add directives to CLAUDE.md
2. Create templates
3. Update DEVELOPER_GUIDE.md

### Phase 2: Enforcement (Week 2)
1. Supervisor must check for test_strategy.md before Phase B
2. Supervisor must verify Phase 0 constraint analysis
3. Supervisor must validate pytest.log shows PASSED (not SKIPPED)

### Phase 3: Validation (Week 3-4)
1. Monitor next initiative using new process
2. Measure: iterations to MVP, rework rate
3. Collect feedback from agents and developers
4. Refine templates based on usage

### Phase 4: Retrospective (Month 1)
1. Compare metrics: before vs after
2. Identify gaps in templates
3. Update documentation based on learnings
4. Celebrate wins, iterate on process

---

## Risk Mitigation

**Risk:** Upfront design overhead slows initial progress
**Mitigation:**
- Templates keep Phase 0 under 30 minutes
- Saves 8-10 iterations downstream (net positive)
- Can parallelize constraint analysis with other planning

**Risk:** Process becomes bureaucratic checklist
**Mitigation:**
- Supervisor validates quality, not just existence
- Templates are guides, not rigid requirements
- Focus on preventing known failure modes (not theoretical)

**Risk:** Developers circumvent process
**Mitigation:**
- Document WHY (show the 15-iteration analysis)
- Make templates easy to use
- Enforce in supervisor quality gates
- Collect feedback and iterate

---

## Success Metrics

**Lead Indicators (Week 1-2):**
- [ ] All 3 directives added to CLAUDE.md
- [ ] Templates created and accessible
- [ ] Next initiative uses templates

**Lag Indicators (Month 1-2):**
- [ ] Iterations to MVP reduced 40-60%
- [ ] Zero test framework refactoring iterations
- [ ] Zero "tests were skipped, actually broken" iterations
- [ ] Positive feedback from agents/developers

**Validation:**
Run similar meta-analysis after next major initiative:
- Compare iteration count (target: 6-8 vs previous 15)
- Compare rework rate (target: <15% vs previous 33%)
- Compare efficiency (target: 75% vs previous 47%)

---

## References

**Analysis Documents:**
- `logs/feature-torchapi/galph-summaries/META_ANALYSIS.md` - Full analysis
- `logs/feature-torchapi/galph-summaries/GAPS_AND_ISSUES.md` - Issues catalog
- `plans/PROCESS_OPTIMIZATIONS.md` - Detailed optimization proposals

**Templates:**
- `plans/templates/test_strategy_template.md` - Test infrastructure design
- `plans/templates/constraint_analysis_template.md` - Environment constraints

**Related Docs:**
- `docs/DEVELOPER_GUIDE.md` - Development workflow
- `docs/TESTING_GUIDE.md` - Testing methodology
- `prompts/supervisor.md` - Supervisor workflow

---

**Status:** PROPOSED
**Next Action:** Review and approve, then update CLAUDE.md
**Owner:** Project maintainer
**Expected Completion:** Week 1 (critical directives)
