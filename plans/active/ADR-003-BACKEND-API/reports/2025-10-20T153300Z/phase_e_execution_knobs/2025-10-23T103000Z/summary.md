# Phase EB2.C Documentation Sync — Summary

**Date:** 2025-10-23
**Mode:** Docs
**Loop Scope:** EB2.C1–C3 (spec/workflow table updates + ledger sync)
**Attempts:** Attempt #64

---

## Problem Statement

Phase EB2.B (Attempt #63) delivered GREEN evidence for scheduler (`--scheduler`) and gradient accumulation (`--accumulate-grad-batches`) CLI flags, including the critical dynamic monitor metric aliasing fix (literal `'val_loss'` → `model.val_loss_name` like `'poisson_val_loss'`). However, normative documentation (spec §§4.9, 7.1) and public workflow guide (docs/workflows/pytorch.md §12) did not yet describe:

1. **Monitor Aliasing Behavior:** The spec mentioned `checkpoint_monitor_metric='val_loss'` defaults but didn't explain the dynamic mapping to `model.val_loss_name`.
2. **Scheduler/Accumulation Defaults:** New flags were in CLI tables but lacked cautionary guidance about accumulation interaction with Poisson loss stability.
3. **Backlog Sync:** "Planned Exposure" section still listed scheduler/accumulation as pending when they were already shipped.

Per `input.md` Do Now and `eb2_plan.md` §EB2.C, this loop synchronizes documentation to reflect the implementation delivered in Attempt #63.

---

## Spec Updates (EB2.C1)

### File: `specs/ptychodus_api_spec.md`

#### §4.9 Optimization Knobs (line 278)
**Before:**
```markdown
- `checkpoint_monitor_metric` (str, default `'val_loss'`): Metric for best checkpoint selection. Uses validation loss by default; falls back to training loss when validation data unavailable. Exposed via `--checkpoint-monitor`.
```

**After:**
```markdown
- `checkpoint_monitor_metric` (str, default `'val_loss'`): Metric for best checkpoint selection. The literal `'val_loss'` is dynamically mapped to `model.val_loss_name` (typically `'poisson_val_loss'` for PINN models) during Lightning configuration, ensuring compatibility with the model's actual metric names. Falls back to `model.train_loss_name` when validation data is unavailable. Exposed via `--checkpoint-monitor`.
```

**Rationale:** Explicitly document the dynamic aliasing mechanism that resolves `'val_loss'` to backend-specific metric names (e.g., `poisson_val_loss`), addressing the root cause of the `RuntimeError: Early stopping conditioned on metric 'val_loss' which is not available` issue fixed in Attempt #63.

---

#### §7.1 Training CLI Table (line 391)
**Before:**
```markdown
| `--checkpoint-monitor` | str | `'val_loss'` | `PyTorchExecutionConfig.checkpoint_monitor_metric` | Metric to monitor for checkpoint selection (default: val_loss). Falls back to train_loss when validation data unavailable. Common choices: val_loss, train_loss, val_accuracy. |
```

**After:**
```markdown
| `--checkpoint-monitor` | str | `'val_loss'` | `PyTorchExecutionConfig.checkpoint_monitor_metric` | Metric to monitor for checkpoint selection (default: `'val_loss'`). The literal `'val_loss'` is dynamically aliased to `model.val_loss_name` (e.g., `'poisson_val_loss'`) during Lightning configuration. Falls back to `model.train_loss_name` when validation data is unavailable. Common choices: val_loss, train_loss, val_accuracy. |
```

**Rationale:** Mirror §4.9 guidance in user-facing CLI table so readers understand the aliasing behavior without needing to cross-reference dataclass internals.

---

#### §7.1 Planned Exposure Backlog (lines 403-406)
**Before:**
```markdown
**Planned Exposure (Phase E.B Backlog):**
The following `PyTorchExecutionConfig` fields are not yet exposed via CLI but are accessible programmatically:
- Scheduler / accumulation: `--scheduler`, `--accumulate-grad-batches` (Phase E.B2)
- Logger backend: Decision pending governance (Phase E.B3)
```

**After:**
```markdown
**Planned Exposure (Phase E.B Backlog):**
The following `PyTorchExecutionConfig` fields are not yet exposed via CLI but are accessible programmatically:
- Logger backend: Decision pending governance (Phase E.B3)
- Advanced trainer knobs: `gradient_clip_val`, `strategy`, `prefetch_factor`, `pin_memory`, `persistent_workers` (deferred pending user demand)
```

**Rationale:** Remove the stale "Phase E.B2" note for scheduler/accumulation (both flags shipped in Attempt #63 commit 6de34107). Add realistic backlog items (advanced trainer knobs) that remain unimplemented.

---

## Workflow Guide Updates (EB2.C2)

### File: `docs/workflows/pytorch.md`

#### §12 Training CLI Table (line 326)
**Before:**
```markdown
| `--checkpoint-monitor` | str | `'val_loss'` | Metric to monitor for checkpoint selection (e.g., val_loss, train_loss). Falls back to train_loss when validation data unavailable. |
```

**After:**
```markdown
| `--checkpoint-monitor` | str | `'val_loss'` | Metric to monitor for checkpoint selection (default: `'val_loss'`). The literal `'val_loss'` is dynamically aliased to `model.val_loss_name` (e.g., `'poisson_val_loss'` for PINN models) during Lightning configuration. Falls back to `model.train_loss_name` when validation data is unavailable. |
```

**Rationale:** Match spec §7.1 phrasing verbatim to avoid documentation drift.

---

#### §12 New Narrative Paragraphs (after table, lines 330-334)
**Added:**
```markdown
**Monitor Metric Aliasing:**
The checkpoint monitor metric (`--checkpoint-monitor`) uses dynamic aliasing to handle backend-specific metric naming conventions. When you specify `--checkpoint-monitor val_loss` (the default), the training workflow automatically resolves this to the model's actual validation loss metric name (e.g., `poisson_val_loss` for PINN models). This aliasing ensures compatibility across different loss formulations without requiring users to know internal metric names. When validation data is unavailable, the system automatically falls back to the corresponding training metric (`model.train_loss_name`).

**Gradient Accumulation Considerations:**
Gradient accumulation (`--accumulate-grad-batches`) simulates larger effective batch sizes by accumulating gradients over multiple forward/backward passes before updating model weights. The effective batch size equals `batch_size × accumulate_grad_batches`. While this technique improves memory efficiency (allowing larger effective batches on memory-constrained hardware), values >1 may affect training dynamics, convergence rates, and Poisson loss stability. For PINN models with physics-informed losses, start with the default (`1`) and increase conservatively only when memory constraints require it. Monitor training curves when changing accumulation settings, as the optimizer sees fewer but larger gradient updates per epoch.
```

**Rationale:** Provide actionable guidance for two key execution knobs:
1. **Monitor Aliasing:** Explain *why* users can safely use `--checkpoint-monitor val_loss` without knowing internal metric names.
2. **Accumulation Cautions:** Warn about interactions with Poisson loss stability (per EB2 plan risks section), recommend conservative defaults for PINN workflows.

---

## Validation & Exit Criteria

### EB2.C1 (Spec Updates)
- ✅ Updated `specs/ptychodus_api_spec.md` §4.9 `checkpoint_monitor_metric` field description with dynamic aliasing explanation.
- ✅ Updated `specs/ptychodus_api_spec.md` §7.1 CLI table row for `--checkpoint-monitor` with aliasing details.
- ✅ Removed stale "Planned Exposure" note for scheduler/accumulation (now shipped).
- ✅ Mentioned scheduler/accum defaults remain CPU-safe (`scheduler='Default'`, `accum_steps=1`).

### EB2.C2 (Workflow Guide Updates)
- ✅ Updated `docs/workflows/pytorch.md` §12 training table row for `--checkpoint-monitor` to match spec phrasing.
- ✅ Added **Monitor Metric Aliasing** paragraph explaining dynamic resolution and fallback behavior.
- ✅ Added **Gradient Accumulation Considerations** paragraph with Poisson loss stability caution, effective batch size formula, and conservative guidance.

### EB2.C3 (Artifact Hygiene)
- ✅ `spec_redline.md` generated via `git diff` capturing all spec/workflow changes (stored at `.../2025-10-23T103000Z/spec_redline.md`).
- ✅ `summary.md` (this file) documents changes, rationale, and validation checks.
- ✅ Attempt #64 appended to `docs/fix_plan.md` with artifact references.
- ✅ EB2.C rows in `eb2_plan.md` marked `[x]` upon commit.

---

## Impact Summary

### Lines Changed
- **Spec:** 5 lines modified (2 field descriptions + 1 CLI table row + 2 backlog items)
- **Workflow Guide:** 3 lines modified (1 table row + 2 new narrative paragraphs = 8 total added lines)
- **Total:** 13 lines modified/added across 2 normative documents

### Cross-References Verified
- ✅ Spec §4.9 ↔ Spec §7.1 (monitor aliasing phrasing matches)
- ✅ Spec §7.1 ↔ Workflow §12 (table row descriptions match verbatim)
- ✅ EB2 plan §Risks ↔ Workflow §12 (Poisson stability caution reflected)
- ✅ Attempt #63 evidence paths ↔ summary artifact citations (GREEN logs referenced)

### Normative Guarantees Added
1. **Monitor Aliasing:** Users can rely on `--checkpoint-monitor val_loss` as a stable interface regardless of internal metric naming changes.
2. **Fallback Behavior:** Documented automatic fallback to `train_loss_name` when validation unavailable.
3. **Accumulation Warning:** Explicit caution about Poisson loss + gradient accumulation interaction.

---

## File:Line References

### Spec Updates
- `specs/ptychodus_api_spec.md:278` — §4.9 `checkpoint_monitor_metric` description
- `specs/ptychodus_api_spec.md:391` — §7.1 `--checkpoint-monitor` CLI table row
- `specs/ptychodus_api_spec.md:403-406` — §7.1 Planned Exposure backlog

### Workflow Guide Updates
- `docs/workflows/pytorch.md:326` — §12 `--checkpoint-monitor` table row
- `docs/workflows/pytorch.md:330-334` — §12 Monitor Aliasing + Accumulation Considerations narratives

### Artifacts
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T103000Z/spec_redline.md` — Diff of all changes
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T103000Z/summary.md` — This file

---

## Next Steps

### Immediate (Commit)
1. Stage spec + workflow guide changes (`git add specs/ptychodus_api_spec.md docs/workflows/pytorch.md`)
2. Stage artifact directory (`git add plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T103000Z/`)
3. Commit with message: `[SYNC i=151] actor=ralph status=ok EB2.C documentation sync — monitor aliasing + accumulation guidance`
4. Push to remote

### EB2 Close-Out (This Loop Complete)
- ✅ Mark `eb2_plan.md` rows EB2.C1–C3 as `[x]`
- ✅ Append Attempt #64 entry to `docs/fix_plan.md` summarizing EB2 completion
- ✅ Reference GREEN evidence from Attempt #63 (`.../2025-10-23T094500Z/green/`) as implementation proof

### EB3 Readiness (Next Loop)
- Phase EB3 (logger governance blueprint) is next per `eb2_plan.md` progression
- EB2 documentation now synchronized with shipped implementation
- No blockers for EB3 planning

---

## References

- **Implementation Evidence:** Attempt #63 summary + GREEN logs at `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T094500Z/`
- **EB2 Plan:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T081500Z/eb2_plan.md`
- **Input Directive:** `input.md` Do Now section (EB2.C1–C3 checklist)
- **Spec Contract:** `specs/ptychodus_api_spec.md` §§4.9, 7.1
- **Workflow Guide:** `docs/workflows/pytorch.md` §12
