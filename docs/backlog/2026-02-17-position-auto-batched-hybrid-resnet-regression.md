# Backlog: `position_reassembly_backend=batched` Corrupts External N=128 Reconstructions

**Created:** 2026-02-17  
**Status:** Open (partially mitigated)  
**Priority:** High  
**Related:** `scripts/studies/grid_lines_torch_runner.py`, `tests/torch/test_grid_lines_torch_runner.py`, `docs/plans/2026-02-17-hybrid-resnet-trash-recon-debug.md`  
**Impacts:** External grid-lines Torch studies using `reassembly_mode=position` with `N=128` (notably `pinn_hybrid_resnet`)

## Summary

A deterministic reconstruction regression exists in the `batched` position-reassembly path for external `N=128` data. `batched` produces trash-looking outputs (large quadrant artifact + striped/checker phase), while `shift_sum` produces plausible reconstructions from the same checkpoint and predictions.

The immediate routing issue (`auto` selecting `batched`) has been mitigated by making `auto` prefer `shift_sum`, but the underlying `batched` correctness bug remains.

## Evidence

1. Known-good visual reference:
   - `outputs/grid_lines_external_fly001_n128_top_train_full_test_e20_seed3_cnn_hybrid_resnet/visuals/amp_phase_pinn_hybrid_resnet.png`
2. Regressed visual reference:
   - `outputs/grid_lines_external_fly001_n128_top_train_full_test_e40_seed3_hybrid_resnet/visuals/amp_phase_pinn_hybrid_resnet.png`
3. Controlled backend ablation on the same bad checkpoint:
   - `tmp/debug/hybrid_resnet_trash_recon/position_backend_ablation/auto/visuals/amp_phase_pinn_hybrid_resnet.png`
   - `tmp/debug/hybrid_resnet_trash_recon/position_backend_ablation/batched/visuals/amp_phase_pinn_hybrid_resnet.png`
   - `tmp/debug/hybrid_resnet_trash_recon/position_backend_ablation/shift_sum/visuals/amp_phase_pinn_hybrid_resnet.png`
4. Numeric summary from ablation:
   - `tmp/debug/hybrid_resnet_trash_recon/position_backend_ablation/summary.json`
   - `auto` and `batched` amplitudes exploded (`amp_max ~= 413881.78`), while `shift_sum` stayed in plausible range (`amp_max ~= 6.57`).

## Root Cause

There are two distinct issues:

1. **Routing regression (mitigated):** `_choose_position_backend()` previously selected `batched` for large jobs (`batch >= 1024` or `patch_n >= 128`), sending most external `N=128` runs through the broken path.
2. **Batched-path correctness bug (still open):** `reassemble_whole_object`/`_reassemble_position_batched` does not match `shift_sum` behavior for external/global-frame `coords_offsets`.
   - `shift_sum` explicitly recenters offsets by subtracting center-of-mass before translation.
   - `batched` translates with raw offsets and diverges under this data contract.

This mismatch is what creates the corner/stripe failure pattern when `batched` is used.

The heuristic was introduced in commit:

- `230de56958cc86a8aa08912284ab75ebf03c8862`
- `feat(studies): add strategy-based position reassembly for external torch datasets`

## Applied Mitigation

Changed `auto` behavior to prefer `shift_sum` for parity/correctness, while retaining the existing OOM fallback path:

- `auto` now returns `shift_sum` unconditionally.
- If `shift_sum` raises `tf.errors.ResourceExhaustedError`, code already falls back to `batched`.
- Explicit `--torch-position-reassembly-backend batched` remains available.

## Verification

Targeted runner tests covering backend selection and fallback pass:

- `python -m pytest -q tests/torch/test_grid_lines_torch_runner.py -k "auto_backend_prefers_shift_sum_for_large_position_jobs or explicit_batched_backend_overrides_auto_preference or shift_sum_oom_falls_back_to_batched"`
- `python -m pytest -q tests/torch/test_grid_lines_torch_runner.py -k "position_reassembly or position_backend or auto_backend"`

Regenerated visuals/recons for affected outputs after mitigation:

- `outputs/grid_lines_external_fly001_n128_top_train_full_test_e40_seed3_hybrid_resnet`
- `outputs/grid_lines_external_fly001_n128_top_train_full_test_e40_seed4_cnn_hybrid_resnet`
- `outputs/grid_lines_external_fly001_n128_top_train_full_test_e20_seed3_cnn_hybrid_resnet_rerun_20260216_165013_pty`
- Report: `tmp/debug/hybrid_resnet_trash_recon/regeneration_report.json`

## Required Fix Work (Open)

1. Fix `batched` offset handling to match `shift_sum` semantics for external offsets (center/reframe offsets consistently before translation).
2. Add parity regression test(s) that fail when `batched` diverges from `shift_sum` on fixed external-style offsets.
3. Validate both quality and scale parity (`amp_mean`, `amp_q99`, `amp_max`) on a checkpoint replay fixture.
4. Only after parity passes, reconsider recommending explicit `batched` in docs.
