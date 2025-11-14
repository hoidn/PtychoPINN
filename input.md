Brief:
Refactor `scripts/compare_models.py` chunked Baseline mode so it slices `test_data_raw` per chunk (`slice_raw_data`) and creates chunk-scoped `PtychoDataContainer`s via `dataclasses.replace(final_config, n_groups=n_chunk)` instead of instantiating the full test container, then feed the concatenated `pinn_offsets` to the alignment path when chunking.
Record per-chunk DIAGNOSTIC stats, concatenate the chunk outputs/offsets, honor `--baseline-debug-limit`, and keep the single-shot code path untouched to avoid regressions.
After landing the change, rerun the translation guard, then execute the dense-test compare_models debug command (`--baseline-debug-limit 320 --baseline-chunk-size 160 --baseline-predict-batch-size 16`) followed by the full dense-test command (`--baseline-chunk-size 256 --baseline-predict-batch-size 16`), teeing each log under `$HUB/cli/`.
Stop and file `$HUB/red/blocked_<timestamp>.md` if any Baseline DIAGNOSTIC stats stay zero or if `analysis/dose_1000/dense/test/comparison_metrics.csv` / `analysis/metrics_summary.json` still lack Baseline rows; otherwise update `$HUB/summary/summary.md` so we can resume the Phase D selectors + counted Phase G rerun in the next loop.

Summary: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md
Plan: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md
Selector: pytest tests/study/test_dose_overlap_comparison.py::{test_pinn_reconstruction_reassembles_batched_predictions,test_pinn_reconstruction_reassembles_full_train_split} -vv
