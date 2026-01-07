Mode: Implementation
Focus: FIX-GRIDSIZE-TRANSLATE-BATCH-001 — Fix Translation Layer Batch Dimension Mismatch for gridsize>1
Branch: feature/torchapi-newprompt-2
Mapped tests: tests/tf_helper/test_translation_shape_guard.py (existing), tests/test_model_factory.py (regression)
Artifacts: plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/

## Summary

Fix the batch dimension mismatch in `translate_xla` that prevents gridsize>1 training. The root cause is that homography matrices are built with translations batch dimension, but images have a different (larger) batch dimension after channel flattening.

## Root Cause Analysis (CONFIRMED)

In `_reassemble_patches_position_real` (tf_helper.py:876-879):
```python
offsets_flat = flatten_offsets(offsets_xy)   # (B*C, 2) where C = gridsize^2
imgs_flat = _channel_to_flat(imgs)            # (B*C, N, N, 1)
Translation(...)([imgs_flat_padded, -offsets_flat])
```

Both `offsets_flat` and `imgs_flat` should have the same batch dimension `B*C`. However, the error suggests one of them doesn't.

**Key insight from error analysis:**
- Error: 389376 = 64 × 78 × 78 vs expected 24336 = 4 × 78 × 78
- 64 = batch_size (16) × gridsize² (4) — this is `B*C`
- 4 = gridsize² — this is just `C`
- The mask is being created with batch=64 but reshaped expecting batch=4

The bug is in `translate_xla` (projective_warp_xla.py:271):
```python
B = tf.shape(translations)[0]  # Gets B from translations
# ... builds M with shape [B, 3, 3]
```

Then `projective_warp_xla` at line 63:
```python
B = tf.shape(images)[0]  # Gets B from images (different!)
# ... tiles grid to [B, H, W, 3]
```

**The M matrix (from translations) and grid (from images) have mismatched batch dimensions!**

## Do Now

### Implement: `ptycho/projective_warp_xla.py::translate_xla`

**Phase B1: Add batch broadcast to translate_xla**

Add broadcast logic BEFORE building homography matrices (after line 268):

```python
def translate_xla(images: tf.Tensor, translations: tf.Tensor,
                  interpolation: str = 'bilinear',
                  use_jit: bool = True) -> tf.Tensor:
    # ... docstring and complex handling unchanged ...

    # Ensure translations has correct shape
    translations = tf.ensure_shape(translations, [None, 2])

    # === NEW: Broadcast translations to match images batch dimension ===
    images_batch = tf.shape(images)[0]
    trans_batch = tf.shape(translations)[0]

    # When gridsize > 1, images are flattened from (b, N, N, C) to (b*C, N, N, 1)
    # but translations may still be (b*C, 2) or (b, 2). Broadcast if needed.
    def broadcast_translations():
        # Compute repeat factor: images_batch = trans_batch * factor
        repeat_factor = images_batch // trans_batch
        # Tile translations: (trans_batch, 2) -> (images_batch, 2)
        return tf.repeat(translations, repeat_factor, axis=0)

    def keep_translations():
        return translations

    translations = tf.cond(
        tf.not_equal(images_batch, trans_batch),
        broadcast_translations,
        keep_translations
    )
    # === END NEW ===

    # Build translation-only homography matrices (now uses images_batch)
    B = tf.shape(translations)[0]  # Now matches images batch
    # ... rest unchanged ...
```

**File location:** ptycho/projective_warp_xla.py:267-271 (insert after line 268)

### Test: `tests/tf_helper/test_translation_shape_guard.py::test_translate_xla_gridsize_broadcast`

Add a new test to verify the broadcast works:

```python
def test_translate_xla_gridsize_broadcast():
    """Verify translate_xla broadcasts translations for gridsize>1 scenarios."""
    import tensorflow as tf
    from ptycho.projective_warp_xla import translate_xla

    # Simulate gridsize=2 scenario: images flattened, translations not
    batch_size = 16
    gridsize = 2
    C = gridsize * gridsize  # 4
    N = 64

    # Images after _channel_to_flat: (batch*C, N, N, 1)
    images_flat = tf.random.normal([batch_size * C, N, N, 1])

    # Translations after flatten_offsets: should also be (batch*C, 2)
    # But test the broadcast case where they're (batch, 2)
    translations_small = tf.random.normal([batch_size, 2])

    # This should work with broadcast
    result = translate_xla(images_flat, translations_small, use_jit=False)

    assert result.shape == images_flat.shape, \
        f"Expected shape {images_flat.shape}, got {result.shape}"

    # Also test the matching case (no broadcast needed)
    translations_full = tf.random.normal([batch_size * C, 2])
    result2 = translate_xla(images_flat, translations_full, use_jit=False)
    assert result2.shape == images_flat.shape
```

### Verify: Run regression tests

```bash
# Test the new fix
pytest tests/tf_helper/test_translation_shape_guard.py -v -k "test_translate_xla" 2>&1 | tee plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/pytest_translate_xla.log

# Verify existing tests still pass
pytest tests/test_model_factory.py -v 2>&1 | tee -a plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/pytest_regression.log

# Verify translation guard tests still pass
pytest tests/tf_helper/test_translation_shape_guard.py -v 2>&1 | tee -a plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/pytest_guard.log
```

### End-to-end verify (if tests pass):

```bash
# Quick test with dose_response_study.py
timeout 300 python scripts/studies/dose_response_study.py \
    --output-dir tmp/test_gridsize_fix \
    --nepochs 2 \
    2>&1 | tee plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/dose_study_test.log

# Check for success
grep -i "error\|exception\|shape.*mismatch" plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/dose_study_test.log || echo "No errors found"
```

## How-To Map

```bash
# Create artifacts dir
mkdir -p plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/

# Edit projective_warp_xla.py - add broadcast logic at line 268
# Edit test file - add test_translate_xla_gridsize_broadcast

# Run tests
pytest tests/tf_helper/test_translation_shape_guard.py -v
pytest tests/test_model_factory.py -v

# Quick e2e check
python scripts/studies/dose_response_study.py --nepochs 2 --output-dir tmp/test_fix
```

## Pitfalls To Avoid

1. **DO NOT** modify `projective_warp_xla` function — only modify `translate_xla` wrapper
2. **DO NOT** change the M matrix structure, only the input translations
3. **DO** ensure broadcast uses `tf.repeat` not `tf.tile` (repeat duplicates each element, tile repeats the whole tensor)
4. **DO** use `tf.cond` for graph-mode compatibility, not Python `if`
5. **DO** test with `use_jit=False` first to avoid XLA caching issues
6. **DO** handle the case where batches already match (no broadcast needed)
7. **DO NOT** add debug prints — they break XLA compilation
8. **Environment Freeze:** Do not install packages or modify environment

## If Blocked

1. If XLA still fails after fix: Test with `USE_XLA_TRANSLATE=0` to verify non-XLA path works
2. If broadcast produces wrong shapes: Log `images_batch`, `trans_batch`, `repeat_factor` with eager mode
3. If tests timeout: Reduce test scope or use smaller batch sizes
4. Record blocker in `plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/reports/2026-01-06T140000Z/blocked_<timestamp>.md`

## Findings Applied

- **TF-NON-XLA-SHAPE-001:** Non-XLA path already has broadcast logic at tf_helper.py:779-794; XLA path needs same fix
- **CONFIG-001:** `update_legacy_dict` must be called before legacy modules (handled in dose_response_study.py)
- **BUG-TF-001:** `params.cfg['gridsize']` must be set before data generation

## Pointers

- Implementation plan: `plans/active/FIX-GRIDSIZE-TRANSLATE-BATCH-001/implementation.md`
- XLA translate wrapper: `ptycho/projective_warp_xla.py:242-300`
- Non-XLA broadcast fix reference: `ptycho/tf_helper.py:779-794`
- Reassembly call site: `ptycho/tf_helper.py:876-879`
- fix_plan: `docs/fix_plan.md` (FIX-GRIDSIZE-TRANSLATE-BATCH-001 entry)

## Next Up

- If fix works: Unblock STUDY-SYNTH-DOSE-COMPARISON-001 and run dose response study
- Update docs/findings.md with new finding for XLA batch broadcast fix
