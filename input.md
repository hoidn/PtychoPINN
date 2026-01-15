# PARALLEL-API-INFERENCE — Task 2: align TF demo path with `_run_tf_inference_and_reconstruct`

**Summary:** Update the programmatic demo so its TensorFlow backend uses the new inference helper, keeping TF/PyTorch parity per the API spec and capturing quick regression evidence.

**Focus:** PARALLEL-API-INFERENCE — Programmatic TF/PyTorch API parity (Task 2 demo parity + smoke test)

**Branch:** paper

**Mapped tests:**
- `pytest tests/scripts/test_api_demo.py::TestPyTorchApiDemo::test_demo_module_importable -v`
- `pytest tests/scripts/test_api_demo.py::TestPyTorchApiDemo::test_run_backend_function_signature -v`
- `pytest tests/scripts/test_tf_inference_helper.py::TestTFInferenceHelper::test_helper_signature_matches_spec -v`

**Artifacts:** `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-15T233622Z/`

---

## Do Now

- PARALLEL-API-INFERENCE.Task2
  - Implement: `scripts/pytorch_api_demo.py::run_backend` — import `_run_tf_inference_and_reconstruct` + `extract_ground_truth` from `scripts.inference.inference`, reuse the already-loaded `RawData`, and replace the TensorFlow branch so it calls the helper with the bundle’s restored config (`params_dict` from `load_inference_bundle_with_backend`), passes `nsamples=infer_cfg.n_groups` (or `None` for “all”), and pipes the resulting amplitude/phase through `tf_components.save_outputs`. Leave the PyTorch branch untouched aside from deleting now-unused TensorFlow-specific helpers.
  - Validate: run the fast pytest selectors listed above and archive logs with `tee` under the artifacts hub (collect-only is covered by the TF helper suite); slow backend executions stay deselected for now.

## How-To Map
1. In `scripts/pytorch_api_demo.py`, add:
   ```python
   from scripts.inference.inference import (
       _run_tf_inference_and_reconstruct,
       extract_ground_truth,
   )
   ```
2. Within `run_backend`, keep the shared `train_data`/`test_data` setup. For the TF branch:
   - Call `_run_tf_inference_and_reconstruct(model, test_data, params_dict, K=4, nsamples=infer_cfg.n_groups, quiet=True)` and capture `(amp, phase)`.
   - Optionally fetch `gt = extract_ground_truth(test_data)` for future reporting; the current flow can still pass `{}` as the `results` dict when invoking `tf_components.save_outputs`.
   - Remove the direct `tf_components.create_ptycho_data_container()`/`perform_inference()` usage so the demo exercises the helper specified in `specs/ptychodus_api_spec.md §4.8`.
3. Keep the PyTorch execution path unchanged (still calls `_run_inference_and_reconstruct` + `save_individual_reconstructions`).
4. Commands (repo root, PYTHON-ENV-001 compliant):
   ```bash
   pytest tests/scripts/test_api_demo.py::TestPyTorchApiDemo::test_demo_module_importable -v \
     | tee plans/active/PARALLEL-API-INFERENCE/reports/2026-01-15T233622Z/pytest_api_demo_import.log

   pytest tests/scripts/test_api_demo.py::TestPyTorchApiDemo::test_run_backend_function_signature -v \
     | tee plans/active/PARALLEL-API-INFERENCE/reports/2026-01-15T233622Z/pytest_api_demo_signature.log

   pytest tests/scripts/test_tf_inference_helper.py::TestTFInferenceHelper::test_helper_signature_matches_spec -v \
     | tee plans/active/PARALLEL-API-INFERENCE/reports/2026-01-15T233622Z/pytest_tf_helper_signature.log
   ```

## Pitfalls To Avoid
1. Do not reintroduce `tf_components.perform_inference`; `_run_tf_inference_and_reconstruct` is the spec’d surface (see `scripts/inference/inference.py:353`).
2. Keep `RawData` loading deterministic — no extra sampling or CLI flags inside the demo (ANTIPATTERN-001).
3. Treat `params_dict` as read-only — CONFIG-001 compliance relies on the bundle restoring `params.cfg` before inference.
4. Avoid importing heavy TensorFlow modules at top level beyond the helper import.
5. Preserve the existing `WORKDIR` layout (`tmp/api_demo/<backend>/...`).
6. Leave the PyTorch branch untouched; we’re only modernizing the TF path this loop.
7. Run pytest from the repo root and capture full stdout/stderr via `tee`.
8. If a selector unexpectedly runs the `@pytest.mark.slow` tests, abort and document rather than letting them hang.

## If Blocked
- If `_run_tf_inference_and_reconstruct` raises due to missing config keys, capture the traceback plus the offending config dict in `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-15T233622Z/blocker.md`, add the reference to `docs/fix_plan.md` Attempts History, and stop.
- If any mapped pytest selector fails, keep the log in the artifacts hub, summarize the failure (command + exit code) in the same blocker file, and halt so we can triage before continuing.

## Findings Applied
| Finding ID | Adherence |
|------------|-----------|
| CONFIG-001 | Use the bundle-restored config (`params_dict`) when calling `_run_tf_inference_and_reconstruct` so legacy modules stay synchronized. |
| ANTIPATTERN-001 | Keep all heavy processing inside functions; no new import-time data loading or implicit globals. |

## Pointers
- `scripts/pytorch_api_demo.py:1` — demo script that still calls the deprecated TF path.
- `scripts/inference/inference.py:323-520` — canonical `_run_tf_inference_and_reconstruct` + `extract_ground_truth` implementation.
- `tests/scripts/test_api_demo.py:1` — smoke tests you’ll re-run.
- `tests/scripts/test_tf_inference_helper.py:1` — helper regression selector.
- `specs/ptychodus_api_spec.md:200-360` — normative inference API & backend dispatch contract for programmatic callers.

## Next Up
- After the TF branch uses the helper and quick tests pass, schedule a loop to run the slow backend smoke tests and document the example flow in `docs/workflows/pytorch.md`.
