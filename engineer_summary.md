# Engineer Summary — Phase 5: LayerScale Stable Hybrid

**Focus:** FNO-STABILITY-OVERHAUL-001, Phase 5 (LayerScale residual unlock)
**Branch:** fno2
**Date:** 2026-01-29

---

## What I Did

### Task 1: LayerScale-Enhanced StablePtychoBlock
Replaced zero-gamma InstanceNorm initialization with a LayerScale-gated residual:
- InstanceNorm affine weights now default to 1 (not zero)
- Added `self.layerscale = nn.Parameter(torch.full((channels,), 1e-3))` as a per-channel gate
- Forward: `update = norm(act(spectral(x) + local(x))); return x + layerscale * update`
- Updated tests: relaxed `test_identity_init` tolerance to `atol=1e-2`, added `test_layerscale_grad_flow`
- All 4 `TestStablePtychoBlock` tests pass, plus all mapped regression tests (7/7)

### Task 2: Stage A Rerun
Reused control arm datasets. Ran stable_hybrid with LayerScale, 50 epochs, MAE loss, no clipping.

**Results:**
| Metric | LayerScale | Control (ref) | Zero-Gamma (prev) |
|--------|-----------|---------------|-------------------|
| Best val_loss | **0.024** | 0.014 | 0.178 |
| Final val_loss | 0.179 | — | 0.178 |
| Amp SSIM | 0.277 | 0.925 | 0.277 |
| Norm weights near zero | **No** (mean~0.9) | N/A | Yes (|0.09|) |

**Key finding:** LayerScale solved the norm-weight stagnation (STABLE-GAMMA-001 resolved). The model now converges early (val_loss=0.024 at epoch ~4, near control) but then collapses by epoch 50. Failure mode shifted from "can't learn" to "learns then collapses" — a training dynamics issue, not architecture.

### Task 3: Docs & Findings Sync
Updated all required docs with Phase 5 results and metrics.

---

## Files Changed
- `ptycho_torch/generators/fno.py` — LayerScale in StablePtychoBlock
- `tests/torch/test_fno_generators.py` — Updated/added tests
- `docs/strategy/mainstrategy.md` — LayerScale arm metrics table
- `docs/fix_plan.md` — Phase 5 execution log
- `docs/findings.md` — STABLE-GAMMA-001 resolved, STABLE-LS-001 added
- `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` — Phase 5 status
- `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T230000Z/` — Artifacts (log, stats, metrics, model, history)

## Tests Run
- `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock -v` — 4/4 PASSED
- `pytest tests/torch/test_fno_generators.py::TestStableHybridUNOGenerator::test_stable_hybrid_generator_output_shape -v` — PASSED
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v` — PASSED
- `pytest tests/torch/test_fno_generators.py::TestGeneratorRegistry::test_resolve_stable_hybrid_generator -v` — PASSED

## Verification Turn (2026-01-29, iteration 2)
All plan tasks were already completed by the prior session (commit `1be27b5f`). This turn re-ran all 6 mapped regression tests — all passed. No code changes needed.

## Blockers / Open Questions
- **STABLE-LS-001:** LayerScale stable_hybrid learns early then collapses. Next step: investigate LR warmup/cosine schedule or reduced learning rate. The architecture is no longer the bottleneck — training dynamics need tuning.
