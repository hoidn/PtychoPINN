# Specification Documentation Updates — Phase EB1.F

**Loop:** Attempt #60, Mode: Docs, Initiative: ADR-003-BACKEND-API
**Date:** 2025-10-23
**Purpose:** Document checkpoint/early-stop CLI knobs added in commit 496a8ce3

## Files Modified

### 1. `specs/ptychodus_api_spec.md` §4.9 — PyTorchExecutionConfig Field Documentation

**Location:** Lines 273-280

**Changes:**
- Added `checkpoint_mode` field documentation (previously missing from spec)
- Removed "CLI backlog" wording from all checkpoint fields (now exposed via CLI)
- Updated field descriptions to reflect actual CLI exposure and behavior

**Specific Field Updates:**

| Field | Old Description | New Description |
|-------|----------------|-----------------|
| `enable_checkpointing` | "Lightning automatic checkpointing. CLI exposure planned (Phase E.B1)." | "Enable Lightning automatic checkpointing during training. Exposed via `--enable-checkpointing` / `--disable-checkpointing`." |
| `checkpoint_save_top_k` | "Number of best checkpoints to retain. MUST be ≥ 0. CLI backlog (Phase E.B1)." | "Number of best checkpoints to retain. MUST be ≥ 0. Set to -1 to save all checkpoints, 0 to disable saving. Exposed via `--checkpoint-save-top-k`." |
| `checkpoint_monitor_metric` | "Metric for best checkpoint selection. CLI backlog (Phase E.B1)." | "Metric for best checkpoint selection. Uses validation loss by default; falls back to training loss when validation data unavailable. Exposed via `--checkpoint-monitor`." |
| `checkpoint_mode` | *(MISSING)* | "Mode for checkpoint metric optimization. MUST be `'min'` (lower metric is better) or `'max'` (higher metric is better). Exposed via `--checkpoint-mode`." **(NEW)** |
| `early_stop_patience` | "Early stopping patience epochs. MUST be > 0. CLI backlog (Phase E.B1)." | "Early stopping patience epochs. MUST be > 0. Training stops if monitored metric doesn't improve for this many epochs. Exposed via `--early-stop-patience`." |

### 2. `specs/ptychodus_api_spec.md` §7.1 — Training CLI Execution Flags Table

**Location:** Lines 379-390

**Changes:**
- Added 5 new rows to CLI flags table documenting checkpoint knobs
- Updated "Planned Exposure" note (line 400-403) to remove checkpoint controls from backlog list

**New CLI Flags Documented:**

| CLI Flag | Type | Default | Config Field | Description |
|----------|------|---------|--------------|-------------|
| `--enable-checkpointing` / `--disable-checkpointing` | bool | `True` | `PyTorchExecutionConfig.enable_checkpointing` | Enable automatic model checkpointing during training (default: enabled). Checkpoints are saved based on monitored metric performance. Use `--disable-checkpointing` to turn off. |
| `--checkpoint-save-top-k` | int | `1` | `PyTorchExecutionConfig.checkpoint_save_top_k` | Number of best checkpoints to keep (default: 1). Set to -1 to save all checkpoints, 0 to disable saving. Best checkpoints are determined by `--checkpoint-monitor` metric. |
| `--checkpoint-monitor` | str | `'val_loss'` | `PyTorchExecutionConfig.checkpoint_monitor_metric` | Metric to monitor for checkpoint selection (default: val_loss). Falls back to train_loss when validation data unavailable. Common choices: val_loss, train_loss, val_accuracy. |
| `--checkpoint-mode` | str | `'min'` | `PyTorchExecutionConfig.checkpoint_mode` | Mode for checkpoint metric optimization (default: min). Use 'min' when lower metric values are better (e.g., loss), 'max' when higher values are better (e.g., accuracy). |
| `--early-stop-patience` | int | `100` | `PyTorchExecutionConfig.early_stop_patience` | Early stopping patience in epochs (default: 100). Training stops if monitored metric doesn't improve for this many consecutive epochs. Set to large value (e.g., 1000) to effectively disable early stopping. |

### 3. `docs/workflows/pytorch.md` §12 — Training Execution Flags Table

**Location:** Lines 315-326

**Changes:**
- Added 5 new rows to training flags table (identical content to spec §7.1)
- Updated note at line 357 to remove "checkpoint controls" from "programmatic-only parameters" list

**Before:**
```
**PyTorch Execution Configuration:** For the complete catalog of execution configuration fields (17 total, including programmatic-only parameters like checkpoint controls, scheduler, and logger backend), ...
```

**After:**
```
**PyTorch Execution Configuration:** For the complete catalog of execution configuration fields (17 total, including programmatic-only parameters like scheduler and logger backend), ...
```

## Validation

- All defaults match `ptycho/config/config.py:235-239` dataclass definition
- CLI flag names match argparse definitions in `ptycho_torch/train.py:478-538`
- Validation rules (`checkpoint_mode` must be 'min'/'max') documented per `__post_init__` line 264
- Fallback behavior (monitor→train_loss when val unavailable) captured from commit 496a8ce3 implementation

## Cross-References

- Implementation: `ptycho_torch/train.py:478-538` (argparse), `ptycho_torch/workflows/components.py` (callback wiring)
- Test evidence: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-20T160900Z/green/`
- Commit: 496a8ce3 ("ADR-003 EB1: Implement checkpoint/early-stop controls for PyTorch Lightning")

## Notes

- No production code changes in this loop (docs-only per input.md Mode: Docs)
- Tests not run (as expected for documentation-only update)
- Spec and workflow docs now synchronized with implemented CLI flags
- All "CLI backlog" references for checkpoint controls removed (Phase EB1 complete)
