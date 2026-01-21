# User Input — 2026-01-20

## Summary
- Root-level orchestration config is still missing even though docs/fix_plan.md treats it as present, so reviewers cannot read `router.review_every_n` / `logs_dir` from the documented source of truth.
- `scripts/orchestration/supervisor.py` exposes a `--no-git` flag but never checks it, so “no git” runs still pull/push, auto-commit, and will fail in environments without git access.
- Prompt templates still reference non-existent documentation roots (`docs/architecture/data-pipeline.md`, `docs/spec-shards/*.md`), undermining the doc-hygiene fixes that were declared complete.

## Evidence
- docs/fix_plan.md:520-535 claims the repository now ships `orchestration.yaml`, but no such file exists anywhere under the repo root (see `rg --files -g 'orchestration.yaml'` returning nothing).
- scripts/orchestration/supervisor.py:34-140 adds `--no-git`, yet the file never reads `args.no_git`, so git pulls/pushes/auto-commits always run.
- prompts/arch_writer.md:38-55 and prompts/spec_reviewer.md:20-34 still reference `docs/architecture/data-pipeline.md` and `docs/spec-shards/*.md`, neither of which exists (specs live under `specs/` and architecture docs are flat files such as `docs/architecture.md`).

## Requested Next Steps
1. Add the promised root-level `orchestration.yaml` (router.review_every_n/state_file/logs_dir) and commit it so reviewer tooling has an authoritative config source.
2. Wire the supervisor’s `--no-git` flag through the pull/push/auto-commit code paths (and guard against git usage when it is set) so local-only/spec bootstrap runs stop failing.
3. Update `prompts/arch_writer.md` and `prompts/spec_reviewer.md` to reference valid docs (`docs/architecture.md`, `../specs/*.md`), keeping docs/index.md as the canonical map.

## 2026-01-21 Reviewer Addendum

### Summary
1. The new dataset-derived intensity scale in `train_pinn.calculate_intensity_scale()` now forces `PtychoDataContainer.X` to materialize on the GPU and casts it to float64, which defeats the lazy-loading design and will OOM large Phase G datasets before training even begins.

### Evidence
- `ptycho/train_pinn.py:165-192` calls `ptycho_data_container.X`, which caches a full tf.Tensor copy and then casts it to float64 to compute the scale.
- `ptycho/loader.py:117-134` explicitly warns that accessing `.X` eagerly loads the full tensor into GPU memory and instructs callers to avoid it when handling large grouped datasets.

### Plan Update Needed
- Extend DEBUG-SIM-LINES-DOSE-001 Phase D with a D4d checklist item to recompute the dataset-derived scale from the container’s NumPy arrays (or a streaming reducer) so lazy loading remains effective. Add a regression test that ensures `calculate_intensity_scale()` leaves `_tensor_cache` empty and does not allocate GPU tensors when invoked on large containers.

### Next Steps
1. Supervisor assigns D4d to the DEBUG-SIM-LINES team with explicit guardrails on memory use (NumPy reduction or CPU tensor path only).
2. Implement the fix and capture evidence (memory telemetry + pytest) under a fresh Phase D4d hub before rerunning the gs2_ideal scenario.
