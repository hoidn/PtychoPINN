# Phase 0: Constraint Analysis
**Initiative:** <INITIATIVE-NAME>
**Created:** <YYYY-MM-DD>
**Status:** <IN PROGRESS | COMPLETE>
**Analyst:** <Name/Agent>

---

## Purpose

Document all environmental, integration, and testing constraints BEFORE implementation begins. This ensures surprises are minimized and mitigations are designed proactively, not reactively.

**Why This Matters:**
- Prevents "discovered we can't do this" mid-implementation
- Enables torch-optional / GPU-optional design from start
- Identifies CI/dev environment mismatches early
- Plans mitigations before coding, not after

---

## 1. Environment Constraints

### 1.1 Development Environment

**Python Environment:**
```bash
# Captured on: <DATE>
Python version: <version>
pip --version: <version>
Virtual env: <path or N/A>
```

**Installed Frameworks:**
```bash
# Run: pip list | grep -E '(torch|tensorflow|jax|numpy)'
torch==<version or NOT INSTALLED>
tensorflow==<version or NOT INSTALLED>
numpy==<version>
[other key packages]
```

**Hardware:**
- CPU: <model, cores>
- GPU: <model, CUDA version or NOT AVAILABLE>
- Memory: <RAM>
- Storage: <available space>

**Operating System:**
- OS: <Linux / macOS / Windows>
- Version: <specific version>
- Architecture: <x86_64 / arm64>

### 1.2 CI Environment

**CI Platform:** <GitHub Actions / GitLab CI / Jenkins / etc.>

**Python Environment:**
```bash
# Captured from CI logs or .github/workflows/*.yml
Python version: <version>
Runner OS: <typically ubuntu-latest>
```

**Installed Frameworks:**
```bash
# What's available in CI by default?
torch: ❌ NOT INSTALLED (key gap!)
tensorflow: ✅ <version>
numpy: ✅ <version>
[other packages]
```

**Hardware:**
- CPU: <CI runner type>
- GPU: ❌ NOT AVAILABLE (key constraint!)
- Memory: <typically 7GB on GitHub Actions>
- Ephemeral: ✅ (clean env each run)

**Operating System:**
- OS: <typically Ubuntu LTS>
- Version: <20.04, 22.04, etc.>

### 1.3 Environment Gap Analysis

| Resource | Dev | CI | Mismatch? | Impact | Mitigation |
|----------|-----|----|-----------| -------|------------|
| PyTorch | ✅ v2.0 | ❌ None | **YES** | Tests can't run | Torch-optional harness |
| GPU | ✅ CUDA 12 | ❌ CPU-only | **YES** | GPU tests skip | @pytest.mark.gpu, CI skip acceptable |
| TensorFlow | ✅ v2.15 | ✅ v2.15 | NO | N/A | N/A |
| Test Data | ✅ 10GB | ⚠️ 100MB limit | **YES** | Large data unavailable | Small fixtures, download on-demand |
| Python | 3.10 | 3.10 | NO | N/A | N/A |

**Critical Gaps:**
1. **PyTorch unavailable in CI** → Requires torch-optional test design
2. **GPU unavailable in CI** → GPU tests must be @pytest.mark.skipif
3. **Test data size limits** → Cannot use full datasets in CI

**Mitigations Planned:**
1. Implement torch-optional harness (see test_strategy.md)
2. Design tests to run with stubs when torch absent
3. Create small test fixtures (<100MB) for CI
4. Document GPU tests as local-validation only

---

## 2. Integration Constraints

### 2.1 Upstream Dependencies

**Systems We Depend On:**

| System | Type | Version | Stability | API Contract |
|--------|------|---------|-----------|--------------|
| Ptychodus | External API | v1.0 | Stable | specs/ptychodus_api_spec.md |
| TensorFlow | Framework | 2.15+ | Stable | Public API |
| nanoBragg | Optional | dev | Unstable | No formal spec |

**Constraint Details:**

**Ptychodus API:**
- **Contract:** `specs/ptychodus_api_spec.md` §5.1-5.3
- **Compatibility:** Must accept TensorFlow dataclasses (ModelConfig, TrainingConfig)
- **Breaking Changes:** None expected in v1.x
- **Validation:** Integration tests required

**TensorFlow:**
- **Version Range:** 2.12-2.15 (pin to 2.15 for stability)
- **Breaking Changes:** None expected in 2.x
- **GPU Support:** Optional (CPU fallback acceptable)

**nanoBragg:**
- **Status:** Development, unstable API
- **Constraint:** Cannot assume specific version
- **Mitigation:** Isolate nanoBragg dependencies, version pin when stable

### 2.2 Downstream Consumers

**Systems That Depend On Us:**

| System | Type | Constraint | Impact |
|--------|------|-----------|--------|
| Ptychodus | External | Expects config_bridge module | Breaking change requires coordination |
| Training Scripts | Internal | Relies on params.cfg format | Must maintain backward compatibility |
| Inference Pipeline | Internal | Needs checkpoint loading | Persistence format must be stable |

**Backward Compatibility:**
- **params.cfg:** MUST maintain legacy format (66+ files depend on it)
- **Checkpoint format:** MUST remain TensorFlow-compatible (migration plan needed for PyTorch)
- **Data contracts:** MUST follow specs/data_contracts.md (diffraction=amplitude, not intensity)

### 2.3 Data Format Constraints

**Input Data:**
```python
# specs/data_contracts.md requirements
.npz format with:
  - 'diffraction': amplitude (sqrt of intensity), NOT intensity ⚠️
  - 'objectGuess': complex64, MUST be larger than probeGuess
  - 'probeGuess': complex64
  - 'Y': complex64 patches (historical bug if float64!)
```

**Output Data:**
```python
# Model checkpoints
TensorFlow: .h5 or SavedModel format
PyTorch: .pt or .pth (planned)
```

**Compatibility Requirement:**
- PyTorch backend must produce results within tolerance of TensorFlow
- Data format must be identical (no breaking changes)

### 2.4 API Compatibility Matrix

| Our Module | Ptychodus API | TensorFlow API | Legacy params.cfg |
|------------|---------------|----------------|-------------------|
| config_bridge | ✅ Produces ModelConfig | ✅ Consumes dataclasses | ✅ Via update_legacy_dict |
| ptycho_torch.model | ⏳ Planned integration | ❌ Not used | ✅ Reads params.cfg |
| Data loaders | ✅ Compatible | ✅ Compatible | N/A |

**API Contract Risks:**
- Config bridge changes could break Ptychodus (coordinate changes)
- PyTorch model must maintain parity with TensorFlow (critical!)

---

## 3. Testing Constraints

### 3.1 CI Test Execution Capability

**What Can Run in CI:**
- ✅ Unit tests (CPU-only, no heavy frameworks)
- ✅ Integration tests (if dependencies available)
- ⚠️ Smoke tests (limited by data size)
- ❌ GPU tests (hardware unavailable)
- ❌ Long-running tests (>10 min timeout)
- ❌ Tests requiring PyTorch (unless mocked)

**CI Limits:**
```
Max execution time: 6 hours (per workflow)
Max per-job time: 6 hours
Recommended: <10 minutes for fast feedback
Memory: 7GB available
Disk: 14GB available
Network: Outbound allowed, rate limits apply
```

**Implications:**
- Fast unit tests (<5 sec) run on every push
- Integration tests (<30 sec) run on every push
- GPU tests run locally only (document requirement)
- Long benchmarks run on-demand (manual trigger)

### 3.2 Mock/Stub Requirements

**What Must Be Mocked:**

| Dependency | Reason | Strategy |
|-----------|--------|----------|
| PyTorch | Unavailable in CI | Torch-optional harness, stubs |
| GPU | Unavailable in CI | @pytest.mark.skipif, CPU fallback |
| Large datasets | Size limits | Small fixtures, download on-demand |
| External APIs | Network unreliable | requests-mock, responses library |

**What Should NOT Be Mocked:**
- ❌ Core config bridge logic (test the real thing!)
- ❌ Type conversions (need real behavior validation)
- ❌ Validation logic (mocking defeats purpose)

### 3.3 Test Data Availability

**Small Fixtures (include in repo):**
```
tests/data/
├── minimal_config.json      # <1KB, minimal valid config
├── sample_diffraction.npz   # 100KB, single pattern
└── tiny_model.h5            # 500KB, minimal weights
```

**Large Datasets (download on-demand):**
```
External storage (S3, GCS, etc.):
├── full_training_data.npz   # 10GB, production dataset
├── benchmark_data.npz       # 5GB, performance testing
└── validation_suite.tar.gz  # 2GB, comprehensive tests
```

**Download Strategy:**
```python
# pytest-datafiles or custom fixture
@pytest.fixture(scope="session")
def large_dataset(tmp_path_factory):
    if os.getenv("CI"):
        pytest.skip("Large dataset not available in CI")
    # Download if not present
    return download_or_use_cached(url, tmp_path_factory)
```

### 3.4 Performance Requirements

**Test Execution Time Budgets:**

| Test Type | Target | Max Acceptable | Action if Exceeded |
|-----------|--------|----------------|-------------------|
| Unit | <5 sec total | 10 sec | Optimize or split |
| Integration | <30 sec total | 60 sec | Mark @pytest.mark.slow |
| Smoke | <10 sec | 15 sec | Must stay fast |
| E2E | <5 min | 10 min | Run on-demand only |

**CI Feedback Loop:**
- Target: <2 minutes total (including setup)
- Acceptable: <5 minutes
- Unacceptable: >10 minutes (developer friction)

### 3.5 Test Coverage Requirements

**Minimum Coverage:**
- New code: 80% line coverage
- Critical paths (config bridge): 90%+ branch coverage
- MVP scope: 100% (all 9 fields tested)

**Exclusions:**
- Type stubs (TYPE_CHECKING blocks)
- Debug code (__debug__ blocks)
- Deprecated code (marked for removal)

**Enforcement:**
```bash
# CI must pass:
pytest --cov=ptycho_torch --cov-fail-under=80
```

---

## 4. Performance Constraints

### 4.1 Latency Requirements

| Operation | Target | Max Acceptable | Notes |
|-----------|--------|----------------|-------|
| Config translation | <1ms | 10ms | Per config object |
| params.cfg update | <5ms | 50ms | Includes KEY_MAPPINGS |
| Test execution | <5sec | 10sec | Full unit suite |
| Model init | <1sec | 5sec | With params.cfg |

**Measurement:**
```python
import time
start = time.perf_counter()
result = config_bridge.to_model_config(...)
elapsed = time.perf_counter() - start
assert elapsed < 0.01, f"Too slow: {elapsed:.3f}s"
```

### 4.2 Resource Constraints

**Memory:**
- Max per test: 1GB (avoid memory leaks)
- Full suite: <2GB total
- CI limit: 7GB available

**Disk:**
- Test artifacts: <100MB per run
- CI limit: 14GB available
- Cleanup: Automated in fixtures

**Network:**
- Minimize external calls (use mocks)
- Rate limits: GitHub Actions has limits
- Fallback: Offline mode for CI

---

## 5. Security & Privacy Constraints

### 5.1 Sensitive Data

**What's Sensitive:**
- [ ] API keys (if external services used)
- [ ] User data (if applicable)
- [ ] Proprietary algorithms (if applicable)
- [ ] N/A (config bridge has no sensitive data)

**Handling:**
- API keys: Environment variables, never committed
- Test data: Synthetic only, no real user data
- Secrets: Use GitHub Secrets, not hardcoded

### 5.2 Dependency Security

**Scan Dependencies:**
```bash
# Run periodically
pip-audit
safety check
```

**Pin Versions:**
```
# requirements.txt
torch>=2.0,<3.0  # Allow minor updates
tensorflow>=2.15,<2.16  # Pin to specific minor
```

**Supply Chain:**
- Use verified packages (PyPI, conda-forge)
- Check for known vulnerabilities
- Review transitive dependencies

---

## 6. Constraint Summary & Mitigations

### 6.1 Critical Constraints (Must Address)

| # | Constraint | Impact | Mitigation | Status |
|---|-----------|--------|------------|--------|
| 1 | PyTorch unavailable in CI | Tests skip, false confidence | Torch-optional harness | Planned |
| 2 | GPU unavailable in CI | GPU tests can't run | Skip in CI, validate locally | Acceptable |
| 3 | Test data size limits | Can't use full datasets | Small fixtures (<100MB) | Planned |
| 4 | params.cfg backward compat | 66+ files depend on it | Maintain legacy format | Required |

### 6.2 Important Constraints (Should Address)

| # | Constraint | Impact | Mitigation | Status |
|---|-----------|--------|------------|--------|
| 5 | TensorFlow version range | Compatibility issues | Pin to 2.15, test 2.12-2.15 | Planned |
| 6 | Test execution time | Developer friction | Optimize, use @pytest.mark.slow | Ongoing |
| 7 | Coverage requirements | Quality assurance | Target 80%, critical paths 90% | Planned |

### 6.3 Nice-to-Have (Can Defer)

| # | Constraint | Impact | Mitigation | Status |
|---|-----------|--------|------------|--------|
| 8 | nanoBragg API stability | Future integration | Version pin when stable | Deferred |
| 9 | Performance benchmarks | Optimization baseline | Add in Phase C/D | Deferred |

---

## 7. Constraint-Driven Design Decisions

### 7.1 Architecture Implications

**Decision:** Use torch-optional harness
**Driven By:** PyTorch unavailable in CI (Constraint #1)
**Implication:** All torch imports must be optional, tests must work without it

**Decision:** Small test fixtures only
**Driven By:** CI data size limits (Constraint #3)
**Implication:** Synthetic data generation, not real production data

**Decision:** Maintain params.cfg format
**Driven By:** Backward compatibility (Constraint #4)
**Implication:** Cannot simplify legacy dictionary, must keep KEY_MAPPINGS

### 7.2 Test Strategy Implications

**Decision:** Skip GPU tests in CI
**Driven By:** GPU unavailable in CI (Constraint #2)
**Implication:** GPU validation is local-only, document in test_strategy.md

**Decision:** Fast unit tests only in CI
**Driven By:** CI time limits, developer feedback loop
**Implication:** Long-running tests marked @pytest.mark.slow, run on-demand

---

## 8. Validation Checklist

**Before proceeding to Phase A:**

- [ ] Environment gap analysis complete
- [ ] All critical constraints (1-4) have mitigations planned
- [ ] Test strategy references constraints (see test_strategy.md)
- [ ] CI capabilities documented and validated
- [ ] Backward compatibility requirements understood
- [ ] Data format constraints documented
- [ ] Performance requirements defined
- [ ] No showstopper constraints discovered

**Showstopper Check:**
- [ ] Can we implement the MVP with these constraints? **YES / NO**
- If NO: Document blocker and escalate

---

## 9. Next Steps

**Immediate Actions:**
1. Create test_strategy.md referencing these constraints
2. Set up torch-optional harness pattern
3. Create small test fixtures (<100MB)
4. Validate CI can run basic tests

**Before Phase A Completion:**
1. Prove tests run in CI (not just skip)
2. Validate pytest.log artifact captures execution
3. Confirm backward compatibility with params.cfg

**Before Phase B Implementation:**
1. All constraints documented
2. Mitigations validated with proof-of-concept
3. Test infrastructure designed (test_strategy.md approved)

---

## 10. References

**Project Specs:**
- `specs/data_contracts.md` - Data format requirements
- `specs/ptychodus_api_spec.md` - API compatibility

**Related Documents:**
- `plans/active/<initiative>/test_strategy.md` - References constraints
- `plans/active/<initiative>/implementation.md` - Phase definitions
- `docs/DEVELOPER_GUIDE.md` - Development workflow

**CI Configuration:**
- `.github/workflows/test.yml` - CI pipeline definition
- `.pre-commit-config.yaml` - Pre-commit hooks

---

**Status:** <IN PROGRESS | COMPLETE>
**Completed By:** <Name/Agent>
**Date:** <YYYY-MM-DD>
**Approved By:** <Supervisor/Lead>
**Next Phase:** Phase A (can proceed when checklist complete)
