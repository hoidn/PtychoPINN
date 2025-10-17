# Phase D.D1 Evidence — Baseline Parity Test Design

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** D — params.cfg baseline comparison & override matrix
**Focus:** D1 baseline comparison test (`test_params_cfg_matches_baseline`)
**Timestamp:** 2025-10-17T061152Z (UTC)

---

## Objective
Validate that the PyTorch → TensorFlow config bridge produces **exactly** the legacy
`params.cfg` state captured from the canonical TensorFlow configs
(`baseline_params.json`). This will provide the green-phase guardrail before extending
to the override matrix (Phase D2) and warning coverage (Phase D3).

---

## Key Artifacts
- Baseline snapshot: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/baseline_params.json`
- Canonical fixtures (reference only): `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/fixtures.py`
- Adapter under test: `ptycho_torch/config_bridge.py`
- Existing parity suite scaffold: `tests/torch/test_config_bridge.py`

**Relevant findings:** CONFIG-001 (params.cfg must be initialized via `update_legacy_dict`).

---

## Test Blueprint
1. Start from a clean `params.cfg` snapshot (reuse `params_cfg_snapshot` fixture).
2. Instantiate PyTorch singleton configs with canonical values that exercise every spec
   field (see tables below).
3. Call adapter functions with explicit overrides to mirror the canonical TensorFlow
   baseline.
4. Populate `params.cfg` via two `update_legacy_dict` calls (training first, inference second).
5. Normalize `params.cfg` into a deterministic dictionary (Path → str, primitives passthrough).
6. Load the JSON baseline and assert key/value equality.
7. On failure, diff dictionaries to surface mismatched keys.

---

## Canonical PyTorch Configuration Inputs

### Shared setup
```python
from pathlib import Path
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig
from ptycho_torch import config_bridge
from ptycho.config.config import update_legacy_dict
import ptycho.params as params
```

### DataConfig (feeds Model/Training/Inference)
| Field | Value | Rationale |
|-------|-------|-----------|
| `N` | 128 | Matches baseline N |
| `grid_size` | (3, 3) | Produces `gridsize=3` (spec §5.1:2) |
| `K` | 6 | Produces `neighbor_count=6` (final inference value) |
| `nphotons` | 5e8 | Avoids default divergence; matches baseline |
| `probe_scale` | 2.0 | Matches baseline probe_scale |

### ModelConfig overrides
```python
pt_model = ModelConfig(
    mode='Unsupervised',
    n_filters_scale=2,
    amp_activation='silu',
    object_big=False,
    probe_big=False,
    intensity_scale_trainable=True,
)
model_overrides = dict(
    probe_mask=True,                # TORCH unavailable ⇒ must force boolean True
    pad_object=False,
    gaussian_smoothing_sigma=0.5,
)
```

### TrainingConfig
```python
pt_train = TrainingConfig(
    epochs=100,
    batch_size=32,
    nll=True,
)
training_overrides = dict(
    train_data_file=Path('/canonical/baseline/train_data.npz'),
    test_data_file=Path('/canonical/baseline/test_data.npz'),
    n_groups=1024,
    n_subsample=2048,
    subsample_seed=42,
    output_dir=Path('/canonical/baseline/training_outputs'),
    mae_weight=0.3,
    realspace_mae_weight=0.05,
    realspace_weight=0.1,
    positions_provided=False,
    probe_trainable=True,
    sequential_sampling=True,
)
```

### InferenceConfig
```python
pt_infer = InferenceConfig()
inference_overrides = dict(
    model_path=Path('/canonical/baseline/model_directory'),
    test_data_file=Path('/canonical/baseline/inference_data.npz'),
    n_groups=512,
    n_subsample=1024,
    subsample_seed=99,
    output_dir=Path('/canonical/baseline/inference_outputs'),
    debug=True,
)
```

### Adapter Calls
```python
tf_model = config_bridge.to_model_config(pt_data, pt_model, overrides=model_overrides)
tf_train = config_bridge.to_training_config(tf_model, pt_data, pt_model, pt_train, overrides=training_overrides)
tf_infer = config_bridge.to_inference_config(tf_model, pt_data, pt_infer, overrides=inference_overrides)

params.cfg.clear()
update_legacy_dict(params.cfg, tf_train)
update_legacy_dict(params.cfg, tf_infer)
```

---

## Assertion Strategy
```python
from json import load
from collections import OrderedDict

def canonicalize_params(cfg):
    normalized = OrderedDict()
    for key, value in sorted(cfg.items()):
        if isinstance(value, Path):
            normalized[key] = str(value)
        elif isinstance(value, (int, float, str, bool)):
            normalized[key] = value
        elif value is None:
            normalized[key] = None
        else:
            normalized[key] = str(value)
    return normalized

baseline_path = PROJECT_ROOT / 'plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/baseline_params.json'
with baseline_path.open() as fp:
    baseline = json.load(fp)

actual = canonicalize_params(params.cfg)
assert actual == baseline
```
- On failure, capture the diff into the report directory (e.g., `params_diff.json`).
- Keep comparison scope to baseline keys; log any extra keys for follow-up (Phase D2).

---

## Risks & Follow-ups
- **Path drift:** If canonical fixtures move, update the baseline path reference in the test.
- **KEY_MAPPINGS changes:** Any future spec change will require refreshing baseline JSON and this test.
- **Override matrix (Phase D2):** Test should log that it only checks equality; missing overrides and warning scenarios remain Future Work.

---

## Next Steps for Ralph
1. Implement `canonicalize_params` helper + baseline loader inside `tests/torch/test_config_bridge.py`.
2. Add new pytest test (e.g., `test_params_cfg_matches_baseline`) under `TestConfigBridgeParity` using the blueprint above.
3. Run targeted selector (author failing test first if needed) then make adapter adjustments only if the comparison fails.
4. Store pytest logs under this attempt directory and update docs/fix_plan.md Attempts.

**No code changes made in this supervisor loop; analysis only.**
