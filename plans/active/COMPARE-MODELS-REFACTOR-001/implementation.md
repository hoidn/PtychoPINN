## Initiative: Compare Models Refactor (COMPARE-MODELS-REFACTOR-001)

### Goal
Refactor `scripts/compare_models.py` into a thin CLI over a modular Python API, conforming to `specs/compare_models_spec.md`. Preserve CLI behavior while improving composability, alignment consistency, and testability.

### Do Now
- Design the API surface (function signature, return dataclass) matching current CLI capabilities (2-way/3-way, sampling/seed, registration toggle, stitching, chunking, logging).
- Map CLI args to API parameters; ensure backward-compatible defaults.
- Plan fixed-canvas alignment and seed-sharing semantics for external recon.

### Phases
1) **API Design & Skeleton**
   - Define `compare_models.compare(...)` (inputs: pinn_dir, baseline_dir, test_npz, recon_npz, sampling, registration, stitching, chunking, logging/env toggles).
   - Define results dataclass (metrics, plot path, npz paths, offsets).
2) **CLI Thin Wrapper**
   - Rework `scripts/compare_models.py` to parse args, set env/logging, call API.
   - Keep existing flags/defaults; route new flags as needed.
3) **Alignment & Subsampling Consistency**
   - Implement fixed-canvas option; expose seed pass-through so recon subsets can be matched.
   - Ensure gridsize enforcement, stitching M, and registration workflow remain intact.
4) **IO & Data Contracts**
   - Centralize NPZ schema validation (test NPZ, recon NPZ) per `specs/data_contracts.md`.
   - Normalize output artifact naming (metrics CSV, plot, raw/aligned NPZs, FRC curves).
5) **Error Handling & Logging**
   - Standardize errors for missing files/keys, shape/gridsize mismatches, invalid stitch size.
   - Maintain logging toggles; document CPU/XLA hints.
6) **Tests & Examples**
   - Add unit/integration coverage for 2-way/3-way, sampling/seed, fixed-canvas on/off, registration on/off, chunked/unchunked paths.
   - Provide a minimal API usage example in docs.
7) **Docs & Spec**
   - Update `specs/compare_models_spec.md` if needed; cross-link in docs.
   - Ensure `docs/MODEL_COMPARISON_GUIDE.md` and `docs/COMMANDS_REFERENCE.md` point to the spec/API usage.
8) **Migration & Cleanup**
   - Deprecate embedded utilities as needed; keep CLI compatibility.
   - Note impacts for study scripts/wrappers; optionally add guidance for API adoption.

### Artifacts
- `plans/active/COMPARE-MODELS-REFACTOR-001/summary.md` (this file).
- Updated spec/docs (see phases 7).
- Tests demonstrating API behavior.
