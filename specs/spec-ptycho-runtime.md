# spec-ptycho-runtime.md — Runtime and Execution (Normative)

Overview (Normative)
- Purpose: Define TensorFlow runtime guardrails to ensure performance, differentiability, and numerical stability across devices; describe XLA and translation paths; define dtype/device rules.

Runtime Guardrails (Normative)
- Dtypes:
  - Complex fields SHALL be tf.complex64; real tensors SHALL be tf.float32.
  - Implementations SHALL cast inputs explicitly before mixed‑type operations to avoid silent up/downcasts.
- Device initialization:
  - If GPU present, memory growth SHOULD be enabled; otherwise run on CPU. See `ptycho/tf_helper.py` device initialization.
- Vectorization:
  - Translation and reassembly SHALL use vectorized, batched paths; the streaming fallback MUST be available to avoid OOM (uses tf.while_loop). See `shift_and_sum`.
- Translation operator:
  - Default path SHOULD use XLA‑friendly implementation (`translate_xla`) when enabled (`USE_XLA_TRANSLATE=1` or `params['use_xla_translate']=True`).
  - Fallback path MAY use `tf.raw_ops.ImageProjectiveTransformV3` for speed when XLA disabled; final fallback is pure‑TF bilinear/nearest using gather_nd.
  - Interpolation modes: ‘bilinear’ default; ‘nearest’ allowed for debug.
- XLA compile:
  - Model compilation via `jit_compile` MAY be enabled (`USE_XLA_COMPILE=1` or `params['use_xla_compile']=True`). Shapes that affect compiled graphs SHOULD be kept stable across iterations.
- Graph hygiene:
  - No `.numpy()` or `.item()` on differentiable values inside forward/loss paths; persistence and visualization SHALL use detached copies after training steps.
- Seeds/determinism:
  - When Poisson sampling is enabled, stochasticity SHALL be expected. Deterministic tests SHALL either disable Poisson draws or rely on statistical thresholds.

Shape‑Change Policy (Normative)
- Reuse layers when `(N, gridsize, offset)` are constant; changing these SHALL rebuild dependent layers.
- Padded canvas size depends on `params.get_padded_size()`; callers SHALL re‑read current params before padding/cropping.

Environment (Normative)
- `USE_XLA_TRANSLATE`: 1/true/yes enables XLA‑friendly translation (default True via params).
- `USE_XLA_COMPILE`: 1/true/yes enables model `jit_compile`.

Error Conditions (Normative)
- Negative or NaN intensities: predicted intensities fed to `log` SHALL be strictly positive; use small epsilon guards if needed.
- Shape mismatches for masks/arrays/coordinates SHALL raise.
- Unsupported `N` values SHALL raise (supported: 64, 128, 256). 

