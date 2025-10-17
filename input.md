Summary: Bridge PyTorch memory-mapped dataset to RawDataTorch/PtychoDataContainerTorch using the config bridge and capture parity evidence.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 Phase C.C3 (memmap bridge)
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_data_pipeline.py -k "memmap" -vv; pytest tests/torch/test_data_pipeline.py -k "cache_reuse" -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T082035Z/{memmap_bridge_analysis.md,pytest_memmap_red.log,pytest_memmap_green.log,cache_validation.md}
Do Now:
1. Phase C.C3 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md — extend `tests/torch/test_data_pipeline.py` with torch-optional memmap parity + cache reuse cases; run `pytest tests/torch/test_data_pipeline.py -k "memmap or cache_reuse" -vv` to capture the red state and store output as `pytest_memmap_red.log`; tests: targeted selector (expected fail before implementation).
2. Phase C.C3 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md — refactor `ptycho_torch/dset_loader_pt_mmap.py` (and related dataset entry points) to delegate grouping to RawDataTorch via config_bridge, emit grouped-data dicts for `PtychoDataContainerTorch`, and keep torch-optional import guards; tests: none.
3. Phase C.C3 + C.D2 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md — rerun `pytest tests/torch/test_data_pipeline.py -k "memmap or cache_reuse" -vv`, store passing logs as `pytest_memmap_green.log`, and document cache reuse evidence in `cache_validation.md` (hash/timestamp of `.groups_cache.npz`); tests: targeted selector.
4. Phase C.C3 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md — update docs/fix_plan.md Attempts History (Attempt #36→#37) and mark checklist rows (C.C3 guidance consumed) in `phase_c_data_pipeline.md` + `implementation.md`; tests: none.
If Blocked: Capture failing selector output (`pytest tests/torch/test_data_pipeline.py -k "memmap or cache_reuse" -vv`) as `pytest_memmap_blocked.log`, log the blocker and partial findings in docs/fix_plan.md Attempts History, and leave implementation changes staged but uncommitted.
Priorities & Rationale:
- specs/data_contracts.md:38-123 — grouped-data keys/dtypes must stay identical when produced via memmap bridge.
- specs/ptychodus_api_spec.md:164-215 — reconstructor contract requires params.cfg to be initialized via dataclass bridge before data pipeline execution.
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md:44-47 — C.C2 complete, C.C3 now depends on memmap delegation and cache reuse evidence.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T082035Z/memmap_bridge_analysis.md — documents duplicate grouping risks and recommended RawDataTorch delegation path.
- docs/findings.md:CONFIG-001 & DATA-001 — enforce `update_legacy_dict` ordering and complex64 Y patches for parity.
How-To Map:
- Before editing, review `ptycho_torch/dset_loader_pt_mmap.py` + `ptycho_torch/dataloader.py` to decide whether to wrap existing TensorDict outputs or introduce a new adapter module; reuse RawDataTorch.generate_grouped_data().
- Use `ptycho_torch/config_bridge.to_training_config()` to convert singleton settings to dataclasses, then call `update_legacy_dict` once per dataset init per CONFIG-001.
- For tests, parameterize minimal fixture (dataset `datasets/fly/fly001_transposed.npz`) and assert equality between memmap bridge outputs and baseline RawDataTorch (`np.allclose` for diffraction/coords, exact match for indices); include cache timestamp/hash assertions in `cache_validation.md`.
- Store red log after Step 1 as `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T082035Z/pytest_memmap_red.log`; overwrite with `_green.log` in Step 3 while keeping both for traceability.
- When verifying cache reuse, capture `ls -l data/memmap`, `stat` output, or hashing snippet and embed results in `cache_validation.md` alongside narrative; ensure `.groups_cache.npz` path noted.
- After completion, refresh `phase_c_data_pipeline.md` row C.C3 (state) and update implementation plan row C5 if cache strategy finalized; log Attempt #37 with artifact links.
Pitfalls To Avoid:
- Do not bypass RawDataTorch or reimplement grouping; reuse existing adapter to keep parity.
- Avoid hard torch imports; guard with `TORCH_AVAILABLE` and keep NumPy fallback functional.
- Do not mutate `params.cfg` multiple times; initialize via config bridge once at dataset creation.
- Keep dtype fidelity: diffraction float32, coords float32, offsets float64, Y complex64; document any casting.
- Ensure tests remain torch-optional: skip gracefully when torch unavailable.
- Preserve existing cache directories; avoid deleting user data under `data/memmap/`.
- Maintain deterministic seeds for parity comparisons (use `seed=42`).
- Do not modify stable core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Keep artifacts under the specified timestamped directory; clean temporary scratch files.
Pointers:
- ptycho_torch/dset_loader_pt_mmap.py:1-220 — current memmap implementation to refactor.
- ptycho_torch/raw_data_bridge.py:1-170 — RawDataTorch adapter and config bridge usage pattern.
- tests/torch/test_data_pipeline.py:120-210, 320-420 — parity tests framework to extend for memmap + cache reuse.
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md:42-48 — C.C3 requirements and evidence references.
- docs/findings.md:9-11 — CONFIG-001 and DATA-001 guardrails relevant to this work.
Next Up: Phase C.C4 (config touchpoint audit for dataset + container) or Phase C.D1 (targeted pytest selectors) once memmap bridge passes parity tests.
