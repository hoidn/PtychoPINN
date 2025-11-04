# PtychoPINN Project Status

*Last Updated: 2025-08-29*

## 📍 Current Active Initiative

**Name:** Independent Sampling and Grouping Control
**Path:** `docs/initiatives/independent_sampling_control/`
**Branch:** `feature/fix-sampling`
**Started:** 2025-08-28
**Current Phase:** Phase 5 of 6 Complete (Documentation & Examples)
**Progress:** ████████████████████████░░░░░░░░ 83% 
**Status:** 🚧 IN PROGRESS - Phase 6 Planning Complete
**Implementation Plan:** `docs/initiatives/independent_sampling_control/implementation.md`
**Test Tracking:** `docs/initiatives/independent_sampling_control/test_tracking.md`
**Phase 6 Plan:** `docs/initiatives/independent_sampling_control/phase6_plan.md`

### Initiative Goal
Enable independent control of data subsampling and neighbor grouping in PtychoPINN by adding a `--n-subsample` parameter to CLI scripts. This allows users to separately control how many images are selected from the dataset vs. how many groups are created for training/inference.

### Completed Phases
- ✅ **Phase 1: Core Infrastructure** - Added subsampling to `components.py`, updated configs, created tests (11/11 passing)
- ✅ **Phase 2: Training Integration** - Updated training script with `interpret_sampling_parameters()`, fixed argument parsing
- ✅ **Phase 3: Inference Integration** - Added similar functionality to inference.py, made InferenceConfig consistent
- ✅ **Phase 4: Comparison Script Updates** - Updated compare_models.py with n_subsample support
- ✅ **Phase 5: Documentation & Examples** - Fixed help text, created examples, wrote [SAMPLING_USER_GUIDE.md](docs/SAMPLING_USER_GUIDE.md)

### Next Phase
- ⏳ **Phase 6: Oversampling Hardening & Docs** – The TF data loader already implements K‑choose‑C oversampling (auto when `n_groups > n_subsample` and `gridsize>1`). Phase 6 will focus on documentation alignment, examples, performance characterization, and guardrails (K≥C validation, memory guidance).

### Key Benefits
- **Flexibility**: Control data density vs computational cost independently
- **Memory Management**: Subsample large datasets before grouping
- **Minimal Changes**: ~60 lines of code implemented, no changes to core `raw_data.py`
- **Backward Compatible**: All existing workflows continue unchanged, all tests passing

## 📊 Development Activity

### Current Focus
Adding independent control for subsampling and grouping operations that are currently coupled through the single `n_images` parameter. The solution leverages the existing architecture's separation between data loading and grouping.

### Recent Completed Initiatives
- **Efficient Coordinate Grouping** (2025-08-15): 200-1000x performance improvement, 300 lines removed
- **Selective Registration for Model Comparison** (2025-08-28): Fair comparison with `--register-ptychi-only` flag
- **GridSize Compatibility Analysis** (2025-08-27): Discovered baseline model architecture limitations
- Codebase Housekeeping (Phase 3 paused at 50%)
- Infrastructure and workflow improvements
- Documentation standardization
- Testing framework establishment

## 🎯 Upcoming Initiatives

- Complete Independent Sampling and Grouping Control implementation
- Complete remaining codebase housekeeping tasks
- *Additional initiatives to be defined*

---

*For detailed project documentation, see `CLAUDE.md` and `docs/DEVELOPER_GUIDE.md`*
