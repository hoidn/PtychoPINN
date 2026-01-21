# Reviewer Result — 2026-01-21T003626Z

## Issues Identified
1. `ptycho/train_pinn.py:165-192` now calls `ptycho_data_container.X`, which the container docs explicitly warn will eagerly allocate the entire dataset on the GPU (`ptycho/loader.py:117-134`). This defeats the lazy-loading/streaming architecture and risks immediate OOM for Phase G–sized runs; the dataset-derived scale should be computed from the stored NumPy arrays (or a CPU reducer) so `_tensor_cache` remains empty.
2. `prompts/arch_writer.md:260-261` changed the spec links to `specs/...`, but because the file lives under `prompts/` those relative paths now point to `prompts/specs/...` (missing). They need to include `../` again so the template links back to the repo-root `specs/` directory.
3. `prompts/arch_reviewer.md` still references non-existent files: line 25 instructs readers to use `docs/architecture/` (there is no such directory) and the JSON example at line 221 points to `docs/architecture/data-pipeline.md`. Both need to be updated to the actual flat docs (`docs/architecture.md`, etc.) per the doc-hygiene plan.
4. `prompts/main.md:104` continues to cite `docs/architecture/pytorch_design.md`, another missing file. The acceptance instructions should reference the real PyTorch architecture doc (`docs/architecture_torch.md`) to keep the golden-path pointers valid.
5. `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:360-378` still lists D4c as unchecked and even details an outdated implementation snippet (`np.mean(X.numpy()**2)`), despite the dataset-derived fix already landing with tests. The plan/summary need to be updated so future loops don’t reimplement the same change or miss the regression evidence.

## Integration Test
- **Command:** `RUN_TS=$(date -u +%Y-%m-%dT%H%M%SZ) RUN_LONG_INTEGRATION=1 INTEGRATION_OUTPUT_DIR=.artifacts/integration_manual_1000_512/${RUN_TS}/output pytest tests/test_integration_manual_1000_512.py -v`
- **Outcome:** PASS on first attempt; logs + metrics under `.artifacts/integration_manual_1000_512/2026-01-21T002845Z/output/`
- **Key excerpt:** `tests/test_integration_manual_1000_512.py::test_train_infer_cycle_1000_train_512_test PASSED`

## Review Window and Context
- `review_every_n`: 3 (from `orchestration.yaml`), router disabled; inspected commits from 2026-01-20T23:39Z review tag through HEAD (d5671c85 → 45412d4e)
- `state_file`: `sync/state.json`
- `logs_dir`: `logs/`
- Logs consulted: none beyond the passing integration output because the long test succeeded
