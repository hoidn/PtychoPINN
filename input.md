Mode: Implementation
Focus: REFACTOR-MODEL-SINGLETON-001 — Phase C1-C4 (Remove XLA Workarounds)
Branch: feature/torchapi-newprompt-2
Selector: tests/test_model_factory.py -vv
Artifacts: plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T060000Z/

## Summary

Remove Phase A XLA workarounds now that the Phase C spike test confirmed lazy loading fixes the multi-N XLA shape mismatch bug.

## Goal

Clean up temporary workarounds from Phase A (environment variables and eager execution). The spike test proved these are no longer needed:
- `USE_XLA_TRANSLATE=0` environment variable
- `TF_XLA_FLAGS=--tf_xla_auto_jit=0` environment variable
- `tf.config.run_functions_eagerly(True)` call

## Tasks

### C1: Remove XLA workarounds from dose_response_study.py

**File:** `scripts/studies/dose_response_study.py`

Delete lines 27-38 (the workaround block):

```python
# DELETE THIS BLOCK:
# CRITICAL: Disable XLA translation BEFORE any ptycho imports to avoid shape caching
# issues when creating models with different N values. See MODULE-SINGLETON-001.
# This must be at the very top, before any other imports that might trigger ptycho.
import os
os.environ['USE_XLA_TRANSLATE'] = '0'
# Also disable TensorFlow's XLA JIT to prevent compile-time constant errors
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# Force eager execution to avoid Keras 3.x XLA graph compilation issues
# with dynamic batch dimensions in the non-XLA translation path.
import tensorflow as tf
tf.config.run_functions_eagerly(True)
```

The module should start with just the docstring and normal imports:
```python
#!/usr/bin/env python3
"""
dose_response_study.py - Synthetic Dose Response & Loss Comparison Study
...docstring continues...
"""
import argparse
import logging
import sys
...
```

### C2: Remove XLA workarounds from tests/test_model_factory.py

**File:** `tests/test_model_factory.py`

1. Delete lines 11-27 (the workaround block at module level):
```python
# DELETE THIS BLOCK:
# CRITICAL: Set environment BEFORE any ptycho imports to avoid XLA trace caching
# See docs/findings.md MODULE-SINGLETON-001 and TF-NON-XLA-SHAPE-001
os.environ['USE_XLA_TRANSLATE'] = '0'

# Also disable TensorFlow's XLA JIT to prevent compile-time constant errors
# in tf.repeat during graph execution (the non-XLA translate path)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import tensorflow as tf
import numpy as np

# Force eager execution to avoid XLA compilation of the graph
# This is necessary because Keras 3.x uses XLA JIT by default for model.predict()
tf.config.run_functions_eagerly(True)
```

2. Replace with clean imports:
```python
import os
import pytest
import subprocess
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np
```

3. Update the `test_multi_n_model_creation` docstring to reflect the new reality:
   - Remove references to the XLA workaround
   - Update to say "lazy loading prevents import-time model creation, avoiding XLA trace conflicts"

4. Update `test_import_no_side_effects` subprocess code:
   - Remove the env var lines (`os.environ['USE_XLA_TRANSLATE'] = '0'` etc.)
   - Remove `tf.config.run_functions_eagerly(True)`
   - Keep the rest of the test logic

5. Keep `TestXLAReenablement::test_multi_n_with_xla_enabled` as-is — this is the permanent regression test for XLA mode.

### C3: Run all tests to verify no regressions

```bash
# Create artifacts directory
mkdir -p plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T060000Z/

# Run all model factory tests
pytest tests/test_model_factory.py -vv 2>&1 | tee plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T060000Z/pytest_phase_c_final.log

# Expected: 3 passed (test_multi_n_model_creation, test_import_no_side_effects, test_multi_n_with_xla_enabled)
```

### C4: Update docs/findings.md

Update the `MODULE-SINGLETON-001` entry to mark it fully resolved:

1. Find the line with status "Resolved" (near end of entry)
2. Update the synopsis to include the lazy loading fix:
   - Add note that Phase B lazy loading (`__getattr__` in model.py) eliminated the need for XLA workarounds
3. Update evidence pointer to include test file:
   - Add `tests/test_model_factory.py` reference

## Implement

- `scripts/studies/dose_response_study.py::` — remove XLA workaround block (lines 27-38)
- `tests/test_model_factory.py::` — remove XLA workarounds from module level and subprocess tests
- `docs/findings.md::MODULE-SINGLETON-001` — update status text

## How-To Map

```bash
# After making changes, verify with:
pytest tests/test_model_factory.py -vv 2>&1 | tee plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T060000Z/pytest_phase_c_final.log

# Parse results
grep -E "^(PASSED|FAILED|ERROR|tests/)" plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T060000Z/pytest_phase_c_final.log
```

## Pitfalls To Avoid

1. **DO NOT** remove the XLA spike test (`test_multi_n_with_xla_enabled`) — it's the permanent regression test
2. **DO NOT** change ptycho/model.py — the lazy loading code is already correct
3. **DO** update the test docstrings to reflect the new reality (lazy loading, not env vars)
4. **DO** keep subprocess isolation in tests — tests run in subprocess for clean Python state
5. **DO** verify all 3 tests pass before committing

## If Blocked

If tests fail after removing workarounds:
1. Check if the test is running in subprocess (subprocess tests should work because they start fresh)
2. Check if there's module-level import order issue in the test file itself
3. Re-read the spike test to understand what worked there
4. Log the specific error and stack trace

## Findings Applied

- **MODULE-SINGLETON-001**: Lazy loading fix verified by spike test; workarounds can be removed
- **CONFIG-001**: Params must still be set before model creation (unchanged)
- **ANTIPATTERN-001**: Import side effects eliminated by Phase B lazy loading

## Pointers

- Spike test evidence: `plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T050000Z/pytest_phase_c_spike_verbose.log`
- Implementation plan: `plans/active/REFACTOR-MODEL-SINGLETON-001/implementation.md` (Phase C checklist)
- Lazy loading: `ptycho/model.py:867-890` (`__getattr__`)
- Fix plan: `docs/fix_plan.md` (REFACTOR-MODEL-SINGLETON-001 entry)
