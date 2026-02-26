# Backlog: PtychoModel Facade

**Created:** 2026-02-06
**Status:** Open
**Priority:** Medium
**Related:** `ptycho_torch/api/model.py`, `ptycho_torch/api/data_resolver.py`, `ptycho_torch/legacy_bridge.py`
**Impacts:** `tests/torch/test_ptycho_model_facade.py`, `docs/workflows/pytorch.md`, `docs/architecture_torch.md`

## Summary
Introduce a minimal OO façade (`PtychoModel`) that exposes `train`/`infer`/`save`/`load`, accepts either dataset objects or paths, and preserves the memmap data flow. The façade must remain policy‑safe: call the legacy bridge before legacy modules, keep probe as data with optional `probe_path`, and keep dataset physics scale separate from model‑internal scaling.

## Impact
- **Usability:** Cleaner API surface without changing existing workflows.
- **Safety:** Centralizes legacy mutation and makes probe/scaling precedence explicit.
- **Performance:** Keeps memory‑mapped dataset path intact for large runs.

## Evidence
- Plan: `docs/plans/2026-02-06-ptycho-model-facade-implementation-plan.md`
- Design: `docs/plans/2026-02-06-ptycho-model-facade-design.md`

## Outstanding Issues
1. Add `PtychoModel` façade and data resolver modules.
2. Enforce legacy bridge usage for path‑based inputs.
3. Support memmap datasets and container inputs.
4. Add façade tests and update PyTorch workflow docs.

## Suggested Direction
Execute the plan in `docs/plans/2026-02-06-ptycho-model-facade-implementation-plan.md`.

## Related Artifacts
- Plan: `docs/plans/2026-02-06-ptycho-model-facade-implementation-plan.md`
- Design: `docs/plans/2026-02-06-ptycho-model-facade-design.md`
