# FNO-STABILITY-OVERHAUL-001 — Phase 5: LayerScale Stable Hybrid

**Summary:** Implement the LayerScale-gated StablePtychoBlock and rerun the Stage A stable arm with shared datasets to see if the architecture finally beats control.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 5 (LayerScale residual unlock)

**Branch:** fno2

**Mapped tests:**
- `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock -v`
- `pytest tests/torch/test_fno_generators.py::TestStableHybridUNOGenerator::test_stable_hybrid_generator_output_shape -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T230000Z/`

**Next Up (optional):** If LayerScale succeeds, prep gradient-checkpointing spike for 8-block depth.

---

## Do Now — Execute plan_layerscale.md Tasks

1. **Follow the LayerScale plan:** Read `plans/active/FNO-STABILITY-OVERHAUL-001/plan_layerscale.md` (same as `docs/plans/2026-01-29-layerscale-stable-hybrid.md`) and work through Tasks 1–3 in order.
2. **Task 1 (LayerScale block + tests):**
   - Update `StablePtychoBlock` per plan (layerscale parameter, InstanceNorm defaults, docstring) and refresh the associated tests in `tests/torch/test_fno_generators.py`.
   - Run `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock -v` and archive the log to the artifacts hub.
3. **Task 2 (Stage A rerun with LayerScale):**
   - Reuse the Stage A control datasets (rsync) and run the stable arm command from the plan, teeing stdout/stderr to `stage_a_arm_stable_layerscale.log` in the artifacts hub.
   - Copy `history.json`, `metrics.json`, `model.pt`, and run `scripts/internal/stage_a_dump_stats.py` into the same hub.
4. **Task 3 (Docs + metrics sync):**
   - Update `stage_a_metrics.json` and `stage_a_summary.md` with the new arm’s numbers, then reflect the results in `docs/strategy/mainstrategy.md`, `docs/fix_plan.md`, `docs/findings.md`, and `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md`.
   - Run the mapped tests list to prove no regressions and archive each log under the artifacts path.
5. **Knowledge base + hygiene:** If LayerScale yields a durable lesson, add/update a Finding (e.g., extend `STABLE-GAMMA-001`). Keep artifacts limited to README/logs/metrics per CLAUDE.md; do not commit datasets.
