### Turn Summary
Fixed batched reassembly shape mismatch in `ptycho/tf_helper.py:_reassemble_position_batched` that caused InvalidArgumentError when processing large (5k+) dense datasets.
Root cause: Translation layer output shrinks by 1px (147 vs 148) due to interpolation/rounding, causing broadcastable shapes error during canvas + batch_result addition.
Resolved by using `tf.image.resize_with_crop_or_pad` to align batch_summed to canvas dimensions after reduce_sum; both regression tests now PASS (2/2).
Artifacts: commit 087a9238, ptycho/tf_helper.py:941-957, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/green/pytest_compare_models_translation_fix.log, docs/findings.md (REASSEMBLY-BATCH-001)

Checklist:
- Files touched: ptycho/tf_helper.py, docs/findings.md
- Tests run: pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv (2 passed)
- Artifacts updated: green/pytest_compare_models_translation_fix.log, docs/findings.md, commit 087a9238

---

### Turn Summary (2025-11-14T020000Z - Galph handoff)
Tier 2 dwell triggered because no new Ralph commit landed while the Phase G rerun stayed ready_for_implementation, so I blocked STUDY-SYNTH-FLY64-DOSE-OVERLAP-001, captured the Translation failure evidence, and spun out FIX-COMPARE-MODELS-TRANSLATION-001.
Documented the new plan/Do Now (reproduce the train/test compare_models commands, batch ReassemblePatchesLayer/tf_helper, add a >=5k patch regression test, rerun pytest + both CLI commands) and updated docs/fix_plan.md, the initiative plans, input.md, and galph_memory so Ralph can jump straight into the fix.
Next: run the specified pytest selector plus the two `scripts/compare_models.py` commands, keep logs under `$HUB/cli/phase_g_dense_translation_fix_{train,test}.log`, and only hand back to STUDY-SYNTH once both exit 0 without Translation errors.
Artifacts: plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/implementation.md, plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/summary.md, input.md
