# Spec Redline Prep — ADR-003 Phase E.A2

## Focus
- Initiative: ADR-003-BACKEND-API
- Task: Phase E.A2 — Redline `specs/ptychodus_api_spec.md` §§4.7–4.9 to document PyTorch execution contract.
- Source plan row: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md#L15` (E.A2).

## Findings Snapshot
- Relevant findings: POLICY-001 (mandatory torch dependency), CONFIG-001 (params bridge ordering).
- Latest artifacts referenced: `phase_e_governance_adr_addendum/adr_addendum.md` §5 (execution knobs backlog), `phase_b_factories/override_matrix.md` §5 (PyTorchExecutionConfig field catalogue).
- Specs baseline: `specs/ptychodus_api_spec.md` §§4.7–4.8 & §7 (CLI tables).

## Observed Gaps
1. **Section 4.7 (TensorFlow-only):** currently enumerates TensorFlow lambda-layer requirements but omits any PyTorch execution guarantees (Lightning trainer, config factories, checkpoint format, failure modes). Need a parallel subsection specifying PyTorch requirements (Lightning availability, execution config validation, checkpoint structure `wts.h5.zip`, CLI helper usage).
2. **Missing PyTorch Execution Contract:** spec never names `PyTorchExecutionConfig`. Governance expects normative statements covering:
   - Field categories (trainer, dataloader, optimization, checkpoint/logging, inference).
   - Default values + validation rules (accelerator whitelist, non-negative workers, positive LR, etc.).
   - Override precedence: execution config sits between explicit overrides and CLI defaults.
   - Relationship to CONFIG-001 (should not populate `params.cfg`).
3. **Section 4.8 (Backend Selection):** already mandates CONFIG-001 order + runtime routing, but does not mention execution config merge or CLI helper delegation. Should add clauses:
   - Dispatcher MUST build execution config via `build_execution_config_from_args()` (CLI) or accept `PyTorchExecutionConfig` objects (programmatic).
   - Factories MUST log applied overrides (per `create_training_payload` contract) and raise `ValueError`/`FileNotFoundError` on validation failures (Phase C2 evidence).
4. **CLI defaults mismatch:** Section 7 tables list `--accelerator` default `'cpu'`, but CLI now defaults to `'auto'` in both train/inference wrappers (`ptycho_torch/train.py:379`, `ptycho_torch/inference.py:471`). Need to update defaults + descriptions to include `'auto'` semantics. Also table omits `--quiet` (maps to `enable_progress_bar=False`), `--no-deterministic`, and runtime knobs yet to be exposed (scheduler, checkpoint options). For this redline we will document current surfaced flags and note the backlog for Phase E.B when exposing additional ones.
5. **Batch size naming:** spec references `InferenceConfig` but should clarify inference batch override lives in execution config (`inference_batch_size`). Update normative text accordingly.

## PyTorchExecutionConfig Field Inventory (from `ptycho/config/config.py:178-258`)
| Field | Default | Category | Validation (per __post_init__) | Notes |
| --- | --- | --- | --- | --- |
| accelerator | 'cpu' → (CLI overrides default to 'auto') | Trainer | Must be in {auto,cpu,gpu,cuda,tpu,mps} | CLI helper resolves `--device` legacy flag into accelerator.
| strategy | 'auto' | Trainer | n/a (validated downstream) | Future work: expose via CLI (Phase E.B2 backlog).
| deterministic | True | Trainer | n/a | `--deterministic` / `--no-deterministic` toggles.
| gradient_clip_val | None | Trainer | n/a | Execution backlog.
| accum_steps | 1 | Trainer | Must be > 0 | Planned CLI knob (Phase E.B2).
| num_workers | 0 | DataLoader | Must be ≥ 0 | CLI flag `--num-workers`.
| pin_memory | False | DataLoader | n/a | GPU-specific; default safe for CPU.
| persistent_workers | False | DataLoader | n/a | Only valid when num_workers>0.
| prefetch_factor | None | DataLoader | n/a | Not yet exposed.
| learning_rate | 1e-3 | Optimizer | Must be > 0 | CLI flag `--learning-rate`.
| scheduler | 'Default' | Optimizer | n/a | CLI backlog (Phase E.B2).
| enable_progress_bar | False | UX | n/a | Derived from `--quiet` toggle.
| enable_checkpointing | True | Checkpoint | n/a | CLI backlog (Phase E.B1) to allow disabling.
| checkpoint_save_top_k | 1 | Checkpoint | Must be ≥ 0 | CLI backlog (Phase E.B1).
| checkpoint_monitor_metric | 'val_loss' | Checkpoint | n/a | CLI backlog (Phase E.B1).
| early_stop_patience | 100 | Checkpoint | Must be > 0 | CLI backlog (Phase E.B1).
| logger_backend | None | Logging | n/a | Pending governance decision (Phase E.B3).
| inference_batch_size | None | Inference | Must be > 0 if set | CLI flag `--inference-batch-size`.
| middle_trim | 0 | Inference | n/a | Not implemented yet (document as TODO).
| pad_eval | False | Inference | n/a | Not implemented yet.

*(Note: dataclass default for `accelerator` is `'cpu'`, but CLI builders replace with `'auto'` unless explicit override. Spec should clarify canonical default vs CLI default to avoid confusion.)*

## Proposed Redline Structure
1. Rename §4.7 to **“Backend-Specific Runtime Requirements”** with two sub-bullets:
   - **TensorFlow Path:** retain current content.
   - **PyTorch Path:** new bullet list covering Lightning availability, execution config validation, checkpoint format (`wts.h5.zip`), factory usage, torch dependency (POLICY-001).
2. Add new §4.9 **“PyTorch Execution Configuration Contract”** summarizing:
   - Dataclass definition + reference (`ptycho/config/config.py`), default values table (above), validation rules, override precedence narrative, prohibition on writing to `params.cfg`.
   - Integration with CLI helpers + factories (citations: `ptycho_torch/cli/shared.py`, `ptycho_torch/config_factory.py`).
3. Update §4.8 bullets to reference §4.9 for execution config merge order and re-state failure semantics (`ValueError` on invalid backend, `RuntimeError` on torch missing, `ValueError/FileNotFoundError` for path validation).
4. Refresh §7 tables:
   - Training: update `--accelerator` default to `'auto'`, explicitly mention `--no-deterministic`, note that `--quiet` toggles progress bar, keep backlog knobs in note referencing Phase E.B.
   - Inference: same default correction, add description tying `--inference-batch-size` to execution config, mention `--quiet` effect.
   - Add footnote referencing §4.9 for full list of execution-only parameters (programmatic override).
5. Cross-reference new spec text from governance addendum + workflow guide (§12) to keep documentation aligned.

## Next Steps for Implementation Loop
- Draft redline in `specs/ptychodus_api_spec.md` following structure above.
- Update `plan.md` row E.A2 to `[x]` once spec merged; include this notes file path in How/Why column.
- Log Attempt in `docs/fix_plan.md` referencing spec updates + artifact path `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/`.
- Ensure Do Now instructs engineer to regenerate CLI tables from authoritative sources (train.py / inference.py) and re-run docs lint if applicable.

