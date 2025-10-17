# PyTorch Workflow Assets Inventory

**Generated:** 2025-10-17  
**Analyst:** Claude Code (Haiku 4.5)  
**Project:** PtychoPINN2 PyTorch Backend  
**Goal:** Assess readiness for Ptychodus integration  

---

## Documents in This Analysis

### 1. **PYTORCH_INVENTORY_SUMMARY.txt** (218 lines, 8.2 KB)
**Executive summary for quick reference**

Start here if you have 5-10 minutes. Contains:
- Key findings (reusability score: 65/100)
- 3 critical integration blockers
- Immediate wins (quick fixes)
- Critical path to integration
- 4-5 day effort estimate

### 2. **PYTORCH_WORKFLOW_INVENTORY.md** (660 lines, 27 KB)
**Comprehensive module-by-module analysis**

Read this for technical details. Contains 11 sections:
1. Training surface (`train.py`) - entry points, gaps
2. Inference surface (`inference.py`) - architecture, gaps
3. API layer (`ptycho_torch/api/`) - 6 high-level classes
4. Lightning integration - model, data module, utilities
5. Dataloader (`dataloader.py`) - memory mapping, schema
6. Config system (`config_params.py`) - 5 dataclasses
7. Cross-system gap analysis - integration blockers matrix
8. Reusability assessment - by component, by pattern
9. Phase B recommendations - immediate actions
10. Implementation priority matrix - impact vs effort
11. File reference guide - all 7 core files

---

## Key Findings at a Glance

### Reusability Score: **65/100**

| Component | Status | Reusability |
|-----------|--------|-------------|
| Training loop (Lightning) | 95% | HIGH |
| Data loading (memory-mapped) | 85% | PARTIAL |
| Config translation (bridge) | 100% | HIGH |
| Inference stitching | 60% | PARTIAL |
| Model persistence | 20% | NONE |

### 3 Critical Integration Blockers

1. **params.cfg not populated** (config_bridge.py exists but never called)
   - Fix: Add 5 lines of code
   - Time: < 1 day

2. **Data format incompatible** (PyTorch TensorDict vs TF NPZ)
   - Fix: Add NPZ export method
   - Time: 1-2 days

3. **Model persistence missing** (MLflow-only, no cross-system format)
   - Fix: Implement PyTorch .pth + metadata
   - Time: 1-2 days

### Critical Path: 4-5 Days to Full Integration

1. Bridge connection (< 1 day)
2. Data export (1-2 days)
3. Model persistence (1-2 days)
4. Integration test (1 day)

---

## Quick Navigation

### I want to...

**Understand the overall architecture**
→ Read: PYTORCH_INVENTORY_SUMMARY.txt (Section: "KEY FINDINGS")

**See what's reusable**
→ Read: PYTORCH_WORKFLOW_INVENTORY.md (Section 8: "REUSABILITY ASSESSMENT MATRIX")

**Identify integration gaps**
→ Read: PYTORCH_WORKFLOW_INVENTORY.md (Section 7: "CROSS-SYSTEM GAP ANALYSIS")

**Find quick wins**
→ Read: PYTORCH_INVENTORY_SUMMARY.txt (Section: "IMMEDIATE WINS")

**Understand config system**
→ Read: PYTORCH_WORKFLOW_INVENTORY.md (Section 6: "CONFIG SYSTEM")

**See all the details on a specific file**
→ Read: PYTORCH_WORKFLOW_INVENTORY.md (search filename)

**Get actionable next steps**
→ Read: PYTORCH_INVENTORY_SUMMARY.txt (Section: "NEXT STEPS")

---

## Core Files Analyzed

Prioritized by integration importance:

### Tier 1: Critical for Ptychodus Integration
- `ptycho_torch/config_bridge.py` (377 lines) - config translation
- `ptycho_torch/train.py` (255 lines) - training entry point
- `ptycho_torch/config_params.py` (160 lines) - config definitions
- `ptycho_torch/dataloader.py` (783 lines) - data loading

### Tier 2: Important for Full Integration
- `ptycho_torch/model.py` (1268 lines) - Lightning model
- `ptycho_torch/inference.py` (212 lines) - inference pipeline
- `ptycho_torch/train_utils.py` (441 lines) - utilities

### Tier 3: Secondary Infrastructure
- `ptycho_torch/api/base_api.py` (995 lines) - high-level API
- `ptycho_torch/api/trainer_api.py` (50 lines) - trainer factory

---

## What's 100% Ready vs What Needs Work

### Production-Ready (Use As-Is)
- **config_bridge.py** - Full translation logic, just needs invocation
- **model.py (Lightning)** - Mature, well-tested training loop
- **train_utils.py** - Utilities broadly applicable
- **trainer_api.py** - Simple, clean factory function

### 80%+ Ready (Small Fixes Needed)
- **train.py** - Add config bridge invocation
- **dataloader.py** - Add NPZ export method
- **config_params.py** - Add missing field defaults

### 50-80% Ready (Needs Integration Work)
- **inference.py** - Decouple reassembly, add config bridge
- **api/base_api.py** - Complete save/load implementations

### < 50% Ready (Skip or Rewrite)
- **api/base_api.py** (model persistence) - Has stub implementations
- Current model persistence strategy - MLflow-only, not cross-system

---

## Implementation Roadmap

### Phase 1: Config Bridge Connection (< 1 day) ← START HERE
```
1. Add update_legacy_dict() call to train.py::main()
2. Add update_legacy_dict() call to inference.py::load_and_predict()
3. Add n_groups and test_data_file to config_params.py defaults
4. Test that params.cfg is populated correctly
```

### Phase 2: Data Export (1-2 days)
```
5. Implement NPZ export from dataloader.py memory-mapped data
6. Validate against specs/data_contracts.md schema
7. Test with downstream RawData.from_file()
```

### Phase 3: Model Persistence (1-2 days)
```
8. Implement PyTorch native save format (pth + manifest.json)
9. Implement PyTorch native load format
10. Verify with Ptychodus model lifecycle
```

### Phase 4: Integration Test (1 day)
```
11. Create test reconstructor using PyTorch backend
12. Verify params.cfg flows to all legacy consumers
13. Test full training + inference round-trip
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Total PyTorch codebase analyzed | ~4,800 lines |
| Modules with high reusability | 4 / 9 |
| Config fields supported | 9 / 23 (MVP scope) |
| Critical blockers | 3 |
| Quick-fix tasks | 3 |
| Estimated integration effort | 4-5 days |
| Components at 90%+ readiness | 3 |

---

## Document Versions

- **PYTORCH_WORKFLOW_INVENTORY.md** v1.0 (660 lines)
  - Comprehensive module analysis
  - 11-section deep dive
  - Production-ready

- **PYTORCH_INVENTORY_SUMMARY.txt** v1.0 (218 lines)
  - Executive summary
  - Quick reference
  - Action-oriented

- **PYTORCH_INVENTORY_README.md** v1.0 (this document)
  - Navigation guide
  - Quick reference
  - Meta-overview

---

## How to Use This Inventory

### For Managers / Decision Makers
1. Read PYTORCH_INVENTORY_SUMMARY.txt completely (5 minutes)
2. Review the "CRITICAL PATH TO INTEGRATION" section
3. Use the "Reusability Score" and "REUSABILITY RECOMMENDATIONS" for planning

### For Engineers Starting Implementation
1. Read PYTORCH_INVENTORY_SUMMARY.txt (5 minutes)
2. Read PYTORCH_WORKFLOW_INVENTORY.md Section 7 (Gap Analysis) (10 minutes)
3. Pick a task from Section 9 (Recommendations) (30 minutes)
4. Reference specific files as needed during coding

### For Architecture Review
1. Read PYTORCH_WORKFLOW_INVENTORY.md Section 8 (Reusability Matrix)
2. Review Section 3 (API Layer)
3. Review Section 7 (Cross-System Gaps)

---

## Contact & Questions

**Generated by:** Claude Code (Haiku 4.5) on 2025-10-17  
**Analysis scope:** ptycho_torch/ directory  
**Integration context:** Ptychodus Phase B, specs/ptychodus_api_spec.md §4  

For questions about findings, refer to specific sections in the comprehensive inventory document.

