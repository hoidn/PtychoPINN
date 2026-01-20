# Implementation Plan: Fix Translation Layer Batch Dimension Mismatch for gridsize>1

## Initiative
- **ID:** FIX-GRIDSIZE-TRANSLATE-BATCH-001
- **Title:** Fix Translation Layer Batch Dimension Mismatch for gridsize>1
- **Owner:** Ralph
- **Spec Owner:** `specs/spec-ptycho-core.md` (Forward Model, Reassembly)
- **Status:** pending
- **Unblocks:** STUDY-SYNTH-DOSE-COMPARISON-001
- **Priority:** Critical
- **Working Plan:** this file
- **Reports Hub:** `plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/`

## Problem Statement

When `gridsize > 1`, the Translation layer receives mismatched batch dimensions:

1. **Images**: Flattened from `(b, N, N, C)` to `(b*C, N, N, 1)` via `_channel_to_flat()`
2. **Translations/Offsets**: Remain at `(b, 2)` or flattened to `(b*C, 2)` via `flatten_offsets()`

The error occurs in two code paths:

### XLA Path (`translate_xla` in `projective_warp_xla.py`)
- **Location:** `projective_warp_xla.py:271`
- **Root Cause:** `B = tf.shape(translations)[0]` uses translations batch, but homography matrices M are applied to images with different batch dimension
- **Error Signature:** `Input to reshape is a tensor with 389376 values, but the requested shape has 24336`
  - 389376 = 64 (batch) × 78 × 78 (mask_batch from M)
  - 24336 = 4 (gridsize²) × 78 × 78 (expected from images)

### Non-XLA Path (`_translate_images_simple` in `tf_helper.py`)
- **Location:** `tf_helper.py:198-199`
- **Root Cause:** Empty translations tensor causes reshape failure
- **Error Signature:** `Input to reshape is a tensor with 0 values, but the requested shape has 4`

## Root Cause Analysis

The issue traces to how `_reassemble_patches_position_real` and `_reassemble_position_batched` prepare inputs for the Translation layer:

```python
# From tf_helper.py:876-879 and 917-930
offsets_flat = flatten_offsets(offsets_xy)   # Should be (b*C, 2)
imgs_flat = _channel_to_flat(imgs)            # Is (b*C, N, N, 1)
Translation(...)([imgs_flat_padded, -offsets_flat])
```

**Hypothesis**: The `flatten_offsets` function may not be correctly flattening offsets from channel format `(B, 1, 2, C)` to `(B*C, 2)`. Or the offsets are arriving in wrong format.

**Call Chain to Verify:**
1. `train_cdi_model` → model training
2. During forward pass: `model.call()` → reassemble patches
3. `_reassemble_patches_position_real` or `_reassemble_position_batched` invoked
4. `flatten_offsets(offsets_xy)` → should produce `(b*C, 2)`
5. `Translation.call()` → `translate_core()` → `translate_xla()` or `_translate_images_simple()`

## Exit Criteria

1. `dose_response_study.py` with `gridsize=2` runs without Translation errors
2. `tests/test_model_factory.py::test_multi_n_model_creation` continues to pass
3. New regression test: `tests/tf_helper/test_translation_shape_guard.py::test_gridsize_2_translation` passes
4. Both XLA and non-XLA translation paths work with gridsize>1

## Compliance Matrix

- [ ] **Finding ID:** TF-NON-XLA-SHAPE-001 — Non-XLA batch broadcast already added at `tf_helper.py:779-794`
- [ ] **Finding ID:** CONFIG-001 — `update_legacy_dict` before legacy module usage
- [ ] **Finding ID:** BUG-TF-001 — `params.cfg['gridsize']` must be set before `generate_grouped_data`

## Spec Alignment

- **Normative Spec:** `specs/spec-ptycho-core.md`
- **Key Clauses:**
  - "Forward semantics: When gridsize > 1, forward-path reassembly semantics MUST match..."
  - Channel flattening/unflattening must preserve batch × channel correspondence

---

## Phase A — Diagnosis and Evidence Collection

### Checklist
- [ ] A1: **Shape Tracing:** Add debug logging to `_reassemble_patches_position_real` and `_reassemble_position_batched`:
  ```python
  tf.print("imgs shape:", tf.shape(imgs))
  tf.print("offsets_xy shape:", tf.shape(offsets_xy))
  tf.print("imgs_flat shape:", tf.shape(imgs_flat))
  tf.print("offsets_flat shape:", tf.shape(offsets_flat))
  ```
  - **Expected:** `imgs_flat` = `(b*C, N, N, 1)`, `offsets_flat` = `(b*C, 2)`
  - **Artifact:** Console log showing actual shapes

- [ ] A2: **Verify flatten_offsets:** Check `flatten_offsets` implementation at `tf_helper.py:853-854`:
  ```python
  def flatten_offsets(channels: tf.Tensor) -> tf.Tensor:
      return _channel_to_flat(channels)[:, 0, :, 0]
  ```
  - Input: `(B, 1, 2, C)` → `_channel_to_flat` → `(B*C, 1, 2, 1)` → `[:, 0, :, 0]` → `(B*C, 2)`
  - **Hypothesis:** This should work IF `offsets_xy` arrives in correct format `(B, 1, 2, C)`

- [ ] A3: **Check Input Format:** Trace what format `offsets_xy` arrives in at the reassembly functions
  - If `offsets_xy` is `(B, 2)` instead of `(B, 1, 2, C)`, `flatten_offsets` will fail silently

### Notes & Risks
- **Risk:** Debug logging may be stripped by XLA JIT or cause graph compilation issues
- **Mitigation:** Use eager mode for debugging (`tf.config.run_functions_eagerly(True)`)

---

## Phase B — Fix XLA Path (`translate_xla`)

### Checklist
- [ ] B1: **Add Batch Broadcast to translate_xla:** The fix should broadcast translations to match images batch:
  ```python
  # In translate_xla (projective_warp_xla.py:267-289)
  # After ensuring translations shape
  images_batch = tf.shape(images)[0]
  trans_batch = tf.shape(translations)[0]

  # Broadcast translations if needed
  def broadcast_trans():
      repeat_factor = images_batch // trans_batch
      return tf.repeat(translations, repeat_factor, axis=0)

  def keep_trans():
      return translations

  translations = tf.cond(
      tf.not_equal(images_batch, trans_batch),
      broadcast_trans,
      keep_trans
  )
  ```

- [ ] B2: **Verify Homography Batch:** Ensure M matrices have batch dimension matching images, not translations:
  - Current: `B = tf.shape(translations)[0]` (wrong)
  - Fixed: Should use images batch OR broadcast M to images batch

- [ ] B3: **Test XLA Path:** Run with `USE_XLA_TRANSLATE=1` (default) and verify gridsize=2 works

### Notes & Risks
- **Risk:** XLA graph caching may use stale shapes
- **Mitigation:** Clear TF session before tests or use subprocess isolation

---

## Phase C — Verify Non-XLA Path

### Checklist
- [ ] C1: **Check _translate_images_simple:** The empty tensor error at line 198-199 suggests translations is empty:
  ```python
  dx_expanded = tf.reshape(dx, [batch_size, 1, 1])  # Fails if dx is empty
  ```

- [ ] C2: **Verify translate_core Broadcast:** The broadcast logic at `tf_helper.py:779-794` should handle this:
  ```python
  translations_adjusted = tf.cond(
      tf.not_equal(images_batch, trans_batch),
      broadcast_translations,
      keep_translations
  )
  ```
  - **Issue:** If translations is empty (size 0), `images_batch // trans_batch` causes div-by-zero or produces empty result

- [ ] C3: **Add Empty Guard:** Before broadcast logic, check for empty tensors:
  ```python
  def guard_empty_translations():
      trans_size = tf.size(translations)
      return tf.cond(tf.equal(trans_size, 0),
          lambda: tf.zeros([images_batch, 2], dtype=translations.dtype),
          lambda: translations)
  ```

### Notes & Risks
- **Risk:** Empty translations may indicate upstream bug (offsets not properly passed)
- **Mitigation:** Add assertion to fail fast with actionable error message

---

## Phase D — Regression Testing

### Checklist
- [ ] D1: **Add Regression Test:** Create `tests/tf_helper/test_translation_gridsize.py`:
  ```python
  def test_translation_gridsize_2_xla():
      """Verify Translation layer works with gridsize=2 in XLA mode."""
      # Create batch with C=4 channels (gridsize=2)
      batch_size = 16
      N = 64
      C = 4  # gridsize^2
      images = tf.random.normal([batch_size, N, N, C])
      offsets = tf.random.normal([batch_size, 1, 2, C])

      # Flatten as done in reassembly
      imgs_flat = _channel_to_flat(images)  # (batch*C, N, N, 1)
      offsets_flat = flatten_offsets(offsets)  # (batch*C, 2)

      # Translate
      translated = Translation(use_xla=True)([imgs_flat, offsets_flat])

      assert translated.shape == imgs_flat.shape
  ```

- [ ] D2: **Run Existing Tests:** Ensure no regressions:
  ```bash
  pytest tests/test_model_factory.py -v
  pytest tests/tf_helper/test_translation_shape_guard.py -v
  ```

- [ ] D3: **End-to-End Test:** Run dose_response_study.py with gridsize=2:
  ```bash
  python scripts/studies/dose_response_study.py --nepochs 2 --output-dir tmp/test_fix
  ```

### Notes & Risks
- **Risk:** Fixing translation may expose other gridsize>1 issues downstream
- **Mitigation:** Start with minimal training (2 epochs) to verify shape compatibility

---

## Phase E — Documentation and Cleanup

### Checklist
- [ ] E1: **Update docs/findings.md:** Add new finding for gridsize>1 translation fix
- [ ] E2: **Update fix_plan.md:** Mark FIX-GRIDSIZE-TRANSLATE-BATCH-001 complete
- [ ] E3: **Unblock STUDY-SYNTH-DOSE-COMPARISON-001:** Remove from blocked status

---

## Open Questions

1. **Why is `offsets_flat` empty in non-XLA path?** Need to trace full call chain during training
2. **Is there a simpler architectural fix?** Perhaps ensure offsets are always flattened at the caller before passing to Translation
3. **XLA graph caching:** Does the fix need to handle cached graphs from prior runs?

## Artifacts Index

- Reports root: `plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/`
- Expected artifacts:
  - `shape_trace.log` — Debug output showing tensor shapes
  - `pytest_translation_gridsize.log` — Regression test results
  - `dose_study_test.log` — End-to-end verification
