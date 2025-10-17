Summary: Capture Phase C.D3 documentation updates for memmap bridge parity.
Mode: Docs
Focus: INTEGRATE-PYTORCH-001 / Phase C.D3 — Update parity ledger & docs
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T084246Z/{parity_map_update.md,implementation_notes.md}
Do Now:
- INTEGRATE-PYTORCH-001 Phase C.D3 @ plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md (row C.D3) + implementation.md:C5 — refresh parity_map.md + plan status; tests: none
If Blocked: Capture a short note in plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T084246Z/blockers.md describing what context is missing and ping supervisor next loop.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md — checklist now shows C.D1/C.D2 done; close C.D3 to unblock Phase C exit.
- plans/active/INTEGRATE-PYTORCH-001/implementation.md — C5 marked [P]; needs final doc refresh before Phase D work.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T084500Z/cache_semantics.md — authoritative source proving cache-free semantics; cross-link in parity docs.
How-To Map:
- Read cache findings (`reports/2025-10-17T084500Z/cache_semantics.md`) and deterministic test log (`reports/2025-10-17T084500Z/pytest_memmap_green_final.log`).
- Update `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/parity_map.md` with a new bullet covering MemmapDatasetBridge status, deterministic generation evidence, and lack of `.groups_cache` files; cite spec sections where relevant.
- Refresh `plans/active/INTEGRATE-PYTORCH-001/implementation.md` to set C5 → [x] once docs are updated, referencing the new report directory.
- Mark C.D3 row as [x] in `phase_c_data_pipeline.md` with links to the new artifacts; keep wording aligned with deterministic-generation terminology (no cache reuse).
- Append Attempt #39 in `docs/fix_plan.md` summarizing the documentation updates and artifact paths; one bullet is enough (docs-only loop).
- Save concise summary in `reports/2025-10-17T084246Z/parity_map_update.md` noting edits + any open follow-ups.
Pitfalls To Avoid:
- Do not reintroduce cache language implying `.groups_cache.npz` exists.
- Keep tests untouched; this loop is documentation only.
- Preserve torch-optional narrative in parity docs (no hard torch import assumptions).
- Maintain ISO timestamps in artifact names.
- Reference existing artifacts rather than duplicating logs.
- Update only the relevant sections of parity_map.md; avoid broad rewrites.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md
- plans/active/INTEGRATE-PYTORCH-001/implementation.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T084500Z/cache_semantics.md
- docs/fix_plan.md
Next Up: consider Phase C.E documentation hand-off once C.D3 closes.
