# Ralph Input — DEBUG-SIM-LINES-DOSE-001 Phase D6 (Investigation)

**Summary:** Capture training label statistics (Y_amp/Y_I) to compare against inference ground truth and identify the amplitude gap source.

**Focus:** DEBUG-SIM-LINES-DOSE-001 — D6: Training target formulation analysis (investigation-only)

**Branch:** paper

**Mapped tests:** `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

**Artifacts:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T230000Z/`

---

## IMPORTANT CONSTRAINT

**Do NOT change or experiment with loss weights (CLAUDE.md directive).** The `realspace_weight=0` finding is documented but any loss-weight modification is OUT OF SCOPE. This loop is **investigation-only** — capture telemetry to understand the label vs ground-truth discrepancy.

---

## Do Now

### Context
D5b telemetry confirmed:
- IntensityScaler scales match to 7 significant figures — NOT the source
- Model amplifies inputs by ~7.45× (`0.085 → 0.634`)
- Truth requires ~31.8× amplification (`truth_mean=2.708`)
- `output_vs_truth_ratio=0.234` — predictions are ~4.3× smaller than truth

The amplitude gap is NOT in scaling layers. D6 hypothesis: the labels (Y_amp/Y_I) fed during training may be at a different scale than the ground truth used for comparison.

### Implement: `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::record_training_label_stats`

Extend the Phase C2 runner to capture **training label statistics** for analysis:

1. **Add function `record_training_label_stats(container)` after container construction:**
   ```python
   def record_training_label_stats(container) -> Dict[str, Any]:
       """Capture training label statistics for amplitude gap analysis.

       Per specs/spec-ptycho-core.md §Normalization Invariants:
       - Labels: Y_amp_scaled = s · X (amplitude), Y_int = (s · X)^2 (intensity)
       - Compare these against ground truth comparison targets.
       """
       stats = {}
       # Check available attributes on container
       for attr in ['Y_I', 'Y_phi', 'Y', 'Y_amp']:
           if hasattr(container, attr):
               val = getattr(container, attr)
               if val is not None:
                   arr = _ensure_numpy(val)
                   stats[attr] = format_array_stats(arr)
       return stats
   ```

2. **Call the function in `main()` right after container construction:**
   ```python
   # After: container = ptycho_data.create_container(raw_data, ...)
   training_labels = record_training_label_stats(container)
   ```

3. **Add to `intensity_stats` block in `run_metadata.json`:**
   ```json
   "training_labels": {
       "Y_I": { "min": ..., "max": ..., "mean": ..., "std": ..., "shape": [...], "dtype": "..." },
       "Y_phi": { ... },
       "Y": { ... }
   },
   "label_vs_truth_analysis": {
       "Y_I_mean": <float>,
       "ground_truth_amp_mean": <float>,  // from comparison metrics
       "ratio_truth_to_Y_I_sqrt_mean": <float>,  // truth_amp / sqrt(Y_I.mean) for intensity comparison
       "note": "Y_I is intensity (amp^2); compare sqrt(Y_I) to amplitude ground truth"
   }
   ```

4. **Run gs1_ideal scenario (it trains successfully):**
   ```bash
   mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T230000Z/logs
   python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py \
     --scenario gs1_ideal \
     --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T230000Z/gs1_ideal \
     --group-limit 64 --nepochs 5 --prediction-scale-source least_squares \
     2>&1 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T230000Z/logs/gs1_ideal_runner.log
   ```

5. **Archive and verify:**
   - `gs1_ideal/run_metadata.json` with new `training_labels` block
   - `pytest_cli_smoke.log`

### How-To Map

```bash
# Env setup
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md

# Create artifacts directory
mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T230000Z/logs

# Run gs1_ideal with extended telemetry
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py \
  --scenario gs1_ideal \
  --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T230000Z/gs1_ideal \
  --group-limit 64 --nepochs 5 --prediction-scale-source least_squares \
  2>&1 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T230000Z/logs/gs1_ideal_runner.log

# Pytest guard
pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
  2>&1 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T230000Z/logs/pytest_cli_smoke.log
```

### Pitfalls To Avoid

1. **Do NOT change loss weights** — CLAUDE.md directive explicitly prohibits this
2. **Do NOT modify core modules** (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`)
3. **Use `_ensure_numpy()` for TensorFlow tensors** — avoid `.numpy()` directly
4. **Record BEFORE training** — labels should be captured right after container construction
5. **Check attribute existence** — `PtychoDataContainer` may not expose all label types

### If Blocked

If container doesn't expose Y_I/Y_phi/Y:
1. Inspect `PtychoDataContainer` in `ptycho/loader.py` to find available attributes
2. Record whatever IS available
3. Document which attributes are missing for future investigation

### Findings Applied

- **CONFIG-001:** Legacy params.cfg bridging already wired in runner
- **NORMALIZATION-001:** Per spec, Y_amp_scaled = s · X and Y_int = (s · X)²
- **D5b (2026-01-21T210000Z):** Confirmed scaling layers match; gap is in learned weights

### Pointers

- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:408-413` — D6 hypothesis and scope
- `plans/active/DEBUG-SIM-LINES-DOSE-001/plan/parity_logging_spec.md` — Parity logging schema v1.0
- `ptycho/loader.py` — `PtychoDataContainer` class definition
- `specs/spec-ptycho-core.md:87-93` — Label formulas (Y_amp_scaled, Y_int)
- `docs/fix_plan.md:480-483` — D6 retraction note (no loss-weight changes)

### Expected Outcome

The telemetry will reveal:
- What scale Y_I/Y_phi labels are at during training
- Whether there's a mismatch between training labels and inference ground truth comparison
- This informs whether the gap is a label formulation issue or something else (architecture, loss function behavior)

### Next Up (optional)

After capturing label stats:
- Compare Y_I mean against ground_truth amplitude² to check intensity scaling
- If labels and truth are at same scale: the gap must be in model architecture or loss function behavior
- Document findings in implementation.md D6 entry without proposing loss-weight changes
