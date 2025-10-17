Summary: Capture TensorFlow vs PyTorch data pipeline contract and gap analysis for Phase C kickoff.
Mode: Docs
Focus: INTEGRATE-PYTORCH-001 — Phase C.A data pipeline baseline
Branch: main
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/{data_contract.md,torch_gap_matrix.md,test_blueprint.md}
Do Now:
1. INTEGRATE-PYTORCH-001 (Phase C.A1+C.A2 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md) — produce `data_contract.md` + `torch_gap_matrix.md` in the artifact directory; tests: none.
2. INTEGRATE-PYTORCH-001 (Phase C.B1 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md) — draft `test_blueprint.md` describing torch-optional pytest structure; tests: none.
If Blocked: Capture findings and open questions in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/blocked.md`, then log Attempt in docs/fix_plan.md with blockers noted.
Priorities & Rationale:
- specs/data_contracts.md — authoritative NPZ key/dtype contract that Phase C must honor.
- specs/ptychodus_api_spec.md:§4 — defines RawData/PtychoDataContainer behaviour consumed by Ptychodus.
- docs/architecture.md:§3 — visual map of RawData → loader pipeline informing parity requirements.
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md — checklist IDs C.A1-C.A3 and C.B1 to complete this loop.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/parity_map.md — Gap #2 context for data pipeline shortfalls.
How-To Map:
- Read specs/data_contracts.md and specs/ptychodus_api_spec.md to extract required keys, shapes, and cache expectations; summarize in `data_contract.md`.
- Inspect `ptycho/raw_data.py` and `ptycho/loader.py` to document TensorFlow runtime behaviours (grouping, caching, dtype conversion) in the same file.
- Review `ptycho_torch/dset_loader_pt_mmap.py`, `ptycho_torch/patch_generator.py`, and config singletons to populate `torch_gap_matrix.md` with actual tensor names/shapes and highlight deltas vs TensorFlow contract.
- In `test_blueprint.md`, outline module/fixture structure for upcoming pytest module per Phase C.B guidance (torch-optional guards, ROI, expected selectors).
- Keep notes concise; prefer tables mirroring plan IDs (C.A1-C.A3, C.B1). No tests to run in this loop.
Pitfalls To Avoid:
- Do not run or modify PyTorch training scripts yet; evidence-only loop.
- Avoid adding torch imports at module top level in future test designs.
- Respect canonical NPZ normalization rules (amplitude data only) when proposing fixtures.
- Do not invent new cache formats—document reuse of `.groups_cache.npz` expectations.
- Ensure all artifacts live under the specified timestamped directory.
- Keep docs/fix_plan.md untouched until artifacts are ready to log next attempt.
- Maintain one focus per loop; defer implementation details to future phases.
- When quoting spec requirements, cite exact key names to prevent drift.
- Flag any uncertainties about dataset availability instead of guessing values.
- Do not mix unittest-style patterns into upcoming pytest blueprint.
Pointers:
- specs/data_contracts.md
- specs/ptychodus_api_spec.md
- docs/architecture.md
- ptycho/raw_data.py
- ptycho/loader.py
- ptycho_torch/dset_loader_pt_mmap.py
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md
Next Up:
- Phase C.B2-C.B3 — author failing torch-optional tests once the blueprint is approved.
