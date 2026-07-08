# Phase -1: Cross-Cutting Concerns Analysis
**Project:** _____________
**Created:** YYYY-MM-DD
**Status:** [ ] IN PROGRESS [ ] COMPLETE
**Analyst:** _____________

---

## Purpose

This analysis identifies constraints, requirements, and design decisions that affect multiple phases of implementation BEFORE detailed planning begins. It prevents expensive rework from discovering infrastructure limitations during implementation.

**Why Phase -1 (Before Planning):**
- Infrastructure decisions made during implementation cost 2-5x more to fix
- Environment constraints discovered late cause expensive retrofits
- Tool incompatibilities found during testing require complete rewrites
- Validation strategy defined late leads to false confidence and rework

**Time Investment:** 30-60 minutes
**Time Saved:** 2-5x reduction in rework iterations

---

## 1. Validation Strategy
*"How will we prove this works?"*

### Testing Approach
**What testing levels will we use?**
- [ ] Unit tests
- [ ] Integration tests
- [ ] End-to-end tests
- [ ] Manual testing
- [ ] Other: ___________

**What framework/tools?**
- Testing framework: ___________
- Assertion library: ___________
- Mocking/stubbing: ___________
- Coverage tool: ___________

**Are these compatible with each other?**
- [ ] ✅ Compatibility verified (sample test written and run)
- [ ] ⚠️ Potential issues: ___________
- [ ] ❌ Incompatible: ___________

### Validation Evidence Required
**What artifacts prove completion?**
- [ ] Test execution logs
- [ ] Coverage reports
- [ ] Performance metrics
- [ ] Screenshots/recordings
- [ ] Deployment verification
- [ ] Other: ___________

**Format and location:**
```
Required artifacts:
- ___________: [format], [location]
- ___________: [format], [location]
```

### Skip Criteria
**What's acceptable to skip?**
- ✅ Acceptable: ___________
- ✅ Acceptable: ___________

**What's NOT acceptable to skip?**
- ❌ Unacceptable: ___________
- ❌ Unacceptable: ___________

### Completion Proof
**How will we distinguish "works on my machine" from actually validated?**
- Proof required: ___________
- False positive prevention: ___________

**Sample validation:**
- [ ] Created sample test with chosen framework
- [ ] Verified it actually runs (not just imports)
- [ ] Confirmed evidence artifacts are produced
- [ ] No warnings or compatibility issues

---

## 2. Environment Constraints
*"Where will this run and what's available there?"*

### Environment Inventory

**Development Environment:**
```
OS: ___________
Language/Runtime: ___________
Key Dependencies: ___________
Hardware: ___________
Network Access: ___________
```

**Production/CI/Deployment Environment:**
```
OS: ___________
Language/Runtime: ___________
Key Dependencies: ___________
Hardware: ___________
Network Access: ___________
```

### Gap Analysis

| Resource | Dev | Prod/CI | Gap? | Impact | Mitigation |
|----------|-----|---------|------|--------|------------|
| OS/Platform | _____ | _____ | Y/N | _____ | _____ |
| Language/Runtime | _____ | _____ | Y/N | _____ | _____ |
| Dependency 1 | _____ | _____ | Y/N | _____ | _____ |
| Dependency 2 | _____ | _____ | Y/N | _____ | _____ |
| Hardware (CPU/GPU/Memory) | _____ | _____ | Y/N | _____ | _____ |
| Network/Services | _____ | _____ | Y/N | _____ | _____ |

### Critical Gaps

**List any showstopper gaps:**
1. ___________
   - Impact: ___________
   - Mitigation: ___________

2. ___________
   - Impact: ___________
   - Mitigation: ___________

**Validation:**
- [ ] No showstopper gaps, or all have mitigations
- [ ] Mitigations proven with sample (actually tested)

---

## 3. Proof of Completion
*"What evidence proves this task is actually done?"*

### Completion Checklist

**For this work to be "done", we must have:**
- [ ] ___________
- [ ] ___________
- [ ] ___________
- [ ] ___________

### Artifact Requirements

**Required artifacts (must be produced):**
```
Artifact 1: [name]
  - Format: [e.g., .log, .json, .md, .png]
  - Location: [path]
  - Content: [what it must contain]
  - Validation: [how to verify it's valid]

Artifact 2: [name]
  - Format: ___________
  - Location: ___________
  - Content: ___________
  - Validation: ___________
```

### False Positive Prevention

**How will we prevent claiming "done" when it's not really validated?**
- Check 1: ___________
- Check 2: ___________
- Check 3: ___________

**Verification method:**
- [ ] Automated (script/tool checks artifacts)
- [ ] Manual (reviewer validates)
- [ ] Hybrid: ___________

---

## 4. Tool Compatibility
*"What tools/versions/patterns are we mixing?"*

### Tool Matrix

| Tool/Library | Version | Purpose | Source |
|-------------|---------|---------|--------|
| ___________ | _____ | _____ | [Official/Community/Internal] |
| ___________ | _____ | _____ | [Official/Community/Internal] |
| ___________ | _____ | _____ | [Official/Community/Internal] |
| ___________ | _____ | _____ | [Official/Community/Internal] |

### Compatibility Verification

**Are these compatible with each other?**

| Tool A | Version | Tool B | Version | Compatible? | Verified How? | Notes |
|--------|---------|--------|---------|-------------|---------------|-------|
| _____ | _____ | _____ | _____ | ✅/⚠️/❌ | _____ | _____ |
| _____ | _____ | _____ | _____ | ✅/⚠️/❌ | _____ | _____ |

### Pattern Compatibility

**Are we mixing different paradigms or patterns?**
- Pattern 1: ___________
- Pattern 2: ___________
- Compatible? ✅ / ⚠️ / ❌
- Verified how? ___________

**Examples to check:**
- Old + new patterns (e.g., class-based + functional)
- Different testing frameworks (e.g., unittest + pytest)
- Async + sync code
- Different state management approaches

### Deprecation Check

**Are any tools/versions deprecated or being sunset?**
- [ ] All tools/versions checked against roadmaps
- [ ] No deprecated versions in use
- [ ] Upgrade path identified if deprecated

**Deprecation warnings:**
1. ___________
   - Sunset date: ___________
   - Migration plan: ___________

### Sample Integration

- [ ] Created minimal integration of all tools together
- [ ] Verified no warnings or errors
- [ ] Confirmed expected behavior

---

## 5. Data Constraints
*"What data requirements and limitations exist?"*

### Size Constraints

| Environment | Memory Limit | Disk Limit | Network Limit |
|-------------|--------------|------------|---------------|
| Development | _____ | _____ | _____ |
| CI/Testing | _____ | _____ | _____ |
| Production | _____ | _____ | _____ |

**Data size requirements:**
- Smallest test case: ___________
- Typical workload: ___________
- Maximum workload: ___________

**Fit analysis:**
- [ ] ✅ All workloads fit in all environments
- [ ] ⚠️ Need reduced dataset for: ___________
- [ ] ❌ Showstopper: ___________

### Data Availability

**What data do we need?**
- Production data? [ ] Yes [ ] No - If yes, how to access? ___________
- Synthetic data? [ ] Yes [ ] No - If yes, how to generate? ___________
- Test fixtures? [ ] Yes [ ] No - If yes, where stored? ___________

**Data access constraints:**
- Privacy requirements: ___________
- Security requirements: ___________
- Licensing requirements: ___________

### Format Requirements

**Data format specifications:**
- Input format: ___________
- Output format: ___________
- Schema/contract: ___________
- Validation: ___________

**Compatibility:**
- [ ] Format supported in all environments
- [ ] Conversion tools available if needed
- [ ] Backward compatibility maintained

### Privacy & Security

**Sensitive data handling:**
- [ ] No sensitive data required
- [ ] Anonymization required: ___________
- [ ] Encryption required: ___________
- [ ] Access controls required: ___________

**Compliance:**
- [ ] GDPR/privacy regulations reviewed
- [ ] Data retention policy defined
- [ ] Audit logging required: ___________

---

## 6. Integration Constraints
*"What external dependencies and APIs exist?"*

### Upstream Dependencies

**Systems/APIs we depend on:**

| System | Type | Version | API Contract | Stability | Contact |
|--------|------|---------|--------------|-----------|---------|
| _____ | [Internal/External/Library] | _____ | _____ | [Stable/Beta/Unstable] | _____ |
| _____ | [Internal/External/Library] | _____ | _____ | [Stable/Beta/Unstable] | _____ |

**Critical contracts:**
1. System: ___________
   - Contract: ___________
   - Validation: ___________
   - Breaking change risk: [ ] Low [ ] Medium [ ] High

### Downstream Consumers

**Systems/users that depend on us:**

| Consumer | Dependency | Contract | Can Break? | Migration Plan |
|----------|-----------|----------|------------|----------------|
| _____ | _____ | _____ | Y/N | _____ |
| _____ | _____ | _____ | Y/N | _____ |

**Backward compatibility:**
- [ ] ✅ Can make breaking changes (no downstream consumers)
- [ ] ⚠️ Must maintain compatibility: ___________
- [ ] ❌ Compatibility required, migration needed: ___________

### Version Compatibility

**Version constraints:**
- Minimum version: ___________
- Maximum version: ___________
- Tested versions: ___________

**Compatibility matrix:**
| Our Version | Dependency Version | Compatible? | Notes |
|-------------|-------------------|-------------|-------|
| _____ | _____ | ✅/⚠️/❌ | _____ |

### API Rate Limits

**External service limits:**
- Service 1: _____ requests per _____
- Service 2: _____ requests per _____

**Mitigations:**
- Caching: ___________
- Rate limiting: ___________
- Fallback: ___________

---

## 7. Exit Criteria & Validation

### Completion Checklist

**All sections must be complete:**
- [ ] 1. Validation Strategy - Testing approach defined, framework chosen, sample test works
- [ ] 2. Environment Constraints - Dev vs prod gaps identified, mitigations planned
- [ ] 3. Proof of Completion - Artifact requirements defined, false positive prevention planned
- [ ] 4. Tool Compatibility - Tools verified compatible, sample integration successful
- [ ] 5. Data Constraints - Size/format/access requirements documented, compliant
- [ ] 6. Integration Constraints - API contracts documented, compatibility verified

### Showstopper Check

**Critical question: Can we implement with these constraints?**
- [ ] ✅ YES - All constraints manageable, mitigations planned
- [ ] ⚠️ MAYBE - High-risk areas need stakeholder decision: ___________
- [ ] ❌ NO - Showstopper identified: ___________

**If NO or MAYBE:**
- Blocker: ___________
- Stakeholders to involve: ___________
- Decision needed by: ___________

### Sample Validations Performed

**Proof that analysis is accurate:**
- [ ] Created sample with chosen testing framework
- [ ] Ran sample in target environment (CI/prod-like)
- [ ] Verified tool compatibility with sample integration
- [ ] Tested data format with sample input
- [ ] Called sample API/dependency to verify contract

**Issues found during validation:**
1. ___________
   - Resolution: ___________
2. ___________
   - Resolution: ___________

---

## 8. Sign-Off

**Analyst:** ___________
**Date Completed:** ___________
**Time Spent:** _____ minutes

**Reviewer:** ___________
**Date Approved:** ___________

**Approval:**
- [ ] All 6 sections complete
- [ ] No unresolved showstoppers
- [ ] Sample validations successful
- [ ] Ready to proceed to Phase 0 (Planning)

**Next Steps:**
1. Proceed to Phase 0 (Requirements/Planning)
2. Reference this analysis when making design decisions
3. Update if new constraints discovered
4. Review at project retrospective

---

## 9. Notes & Discoveries

**Key insights from this analysis:**
- ___________
- ___________

**Risks to monitor:**
- ___________
- ___________

**Questions for stakeholders:**
- ___________
- ___________

---

**Template Version:** 1.0
**Applicability:** Universal (any project, any domain)
**Customization Required:** None - fill in your answers
**Time to Complete:** 30-60 minutes
**Expected ROI:** 2-5x reduction in rework
