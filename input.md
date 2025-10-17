Summary: Implement torch-optional data container so Phase C.C2 parity tests go green.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 Phase C.C2 (PtychoDataContainerTorch)
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_data_pipeline.py -k "data_container" -vv; pytest tests/torch/test_data_pipeline.py -k "Y dtype" -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T075914Z/{data_container_requirements.md,trace_baseline.txt}
Do Now:
1. Phase C.C2 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md — implement torch-optional `PtychoDataContainerTorch` bridging grouped data to tensors; tests: none.
2. Phase C.C2 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md — run `pytest tests/torch/test_data_pipeline.py -k "data_container" -vv` and `pytest tests/torch/test_data_pipeline.py -k "Y dtype" -vv`; stash logs as `pytest_data_container_green.log` and `pytest_y_dtype_green.log`; tests: targeted selectors.
3. Phase C.C2 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md — update docs/fix_plan.md Attempts History with Attempt #35 referencing new artifacts and summarize implementation notes; tests: none.
If Blocked: Capture failing selector output (`pytest tests/torch/test_data_pipeline.py -k "data_container" -vv`) and save as `pytest_data_container_blocked.log` under the artifact directory; note blocker in docs/fix_plan.md Attempt log.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:164-188 requires backend containers to expose TensorFlow-compatible tensors; PyTorch adapter must match shapes/dtypes.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T075914Z/data_container_requirements.md documents required attribute table and dtype guardrails.
- tests/torch/test_data_pipeline.py:222-280 encodes the acceptance criteria; making it pass unblocks Phase C progression and TEST-PYTORCH-001 dependency.
- RawData delegation already green (Attempt #33); container work is the remaining gating task before memmap integration.
How-To Map:
- Module skeleton: create `ptycho_torch/data_container_bridge.py` (torch-optional guard) exporting `PtychoDataContainerTorch`; re-export in `ptycho_torch/__init__.py`.
- Construction path: accept grouped dict from `RawDataTorch.generate_grouped_data`; branch on `TORCH_AVAILABLE` to return torch tensors (`torch.from_numpy(...).to(torch.float32)` / `.to(torch.complex64)`) or NumPy arrays (`np.asarray(..., dtype=...)`).
- Rebuild complex tensors: `Y = combine_complex(Y_I, Y_phi)` analogue using torch or numpy; for torch use `torch.polar(Y_I, Y_phi)` if available or `Y_I * torch.exp(1j * Y_phi)`; amplitude/phase computations rely on consistent dtype conversions.
- Preserve `nn_indices`, `global_offsets`, `local_offsets` without casting unless explicitly required; keep float64 to match baseline.
- Log storage: append targeted selectors’ verbose output to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T075914Z/pytest_data_container_green.log` and `.../pytest_y_dtype_green.log`.
- After implementation rerun `pytest tests/torch/test_data_pipeline.py -vv` if time permits to ensure entire module stays green.
Pitfalls To Avoid:
- Do not import torch at module top without guards (`TORCH_AVAILABLE` pattern from raw_data_bridge).
- Avoid mutating `params.cfg` inside container; rely on RawDataTorch configuration.
- Ensure `Y` remains complex64; never drop to float32/float64 real-only arrays.
- Keep tests torch-optional: do not require torch-specific APIs when TORCH_AVAILABLE is False.
- Preserve grouped dict keys; don’t rename `coords_offsets` / `coords_relative`.
- Don’t downcast offsets to float32 unless parity assertions updated; follow baseline dtype.
- Remember to update docs/fix_plan.md attempts and reference artifact filenames explicitly.
- Avoid mixing pytest parametrization with unittest constructs (tests already pytest-style).
- Capture deterministic seeds (42) to keep parity comparisons stable.
- No ad-hoc scripts outside repo; reuse existing modules.
Pointers:
- ptycho/loader.py:93 — TensorFlow `PtychoDataContainer` constructor and tensor composition.
- tests/torch/test_data_pipeline.py:220 — Acceptance tests for container parity and dtype validation.
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md:48 — Phase C.C2 checklist and artifact references.
- ptycho_torch/raw_data_bridge.py:1 — Torch-optional pattern to mirror in container bridge.
Next Up: Phase C.C3 (PtychoDataContainerTorch ↔ memory map integration) if time remains.
