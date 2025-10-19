# Phase D1b Checkpoint Inspection — 2025-10-19T123000Z

## Context
- Focus: INTEGRATE-PYTORCH-001-STUBS Phase D1b checkpoint diagnostics
- Objective: Inspect Lightning `last.ckpt` `hyper_parameters` payload to confirm config serialization and unblock remediation design.

## Completed Tasks
- [x] Ran training command to reproduce checkpoint (2 epochs, cpu, 64 groups) — checkpoint created at `checkpoint_run/checkpoints/last.ckpt`
- [x] Extracted checkpoint metadata dump to `checkpoint_dump.txt` (4 lines showing missing `hyper_parameters` key)
- [x] Documented findings + remediation hypotheses in `checkpoint_inspection.md`
- [x] Updated docs/fix_plan.md Attempts history with artifact pointers and next steps

## Key Findings

**ROOT CAUSE CONFIRMED:** The Lightning checkpoint **does not contain a `hyper_parameters` key**.

- Checkpoint keys present: `['callbacks', 'epoch', 'global_step', 'loops', 'lr_schedulers', 'optimizer_states', 'pytorch-lightning_version', 'state_dict']`
- `checkpoint.get('hyper_parameters')` returns `None`
- This explains the TypeError during inference load path: Lightning cannot reconstruct `PtychoPINN_Lightning` without the four config objects

**Implication:** The `self.save_hyperparameters()` call is either:
1. Missing from `PtychoPINN_Lightning.__init__()`
2. Present but failing silently due to non-serializable config attributes
3. Called in incorrect context (e.g., after state mutation)

## Next Steps (Phase D2 Remediation)

1. Inspect `ptycho_torch/model.py` for `save_hyperparameters()` presence/correctness
2. Add missing call or fix serialization issues (e.g., convert Path objects to str, handle enum fields)
3. Re-run training with fixed implementation to verify `hyper_parameters` key appears in checkpoint
4. Update load path to reconstruct configs from checkpoint and pass to Lightning module constructor
5. Verify integration test passes with state-free checkpoint reload

