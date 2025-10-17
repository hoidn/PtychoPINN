"""
Capture baseline params.cfg snapshot from canonical TensorFlow configs.

This script instantiates the canonical TensorFlow configuration baseline
(defined in fixtures.py) and populates params.cfg via update_legacy_dict(),
then serializes the resulting params.cfg state to JSON for comparison testing.

Purpose: Phase C.C1 of B.B4 parity test plan
Output: baseline_params.json (serialized params.cfg state)
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

# Import config utilities first
from ptycho.config.config import update_legacy_dict
import ptycho.params as params

# Import canonical fixtures from local directory
# (import as module to avoid Path/relative import issues)
fixtures_path = Path(__file__).parent / 'fixtures.py'
import importlib.util
spec = importlib.util.spec_from_file_location("fixtures", fixtures_path)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def capture_baseline_params():
    """
    Capture params.cfg state after populating with canonical TensorFlow configs.

    Returns:
        dict: Serialized params.cfg state (all values converted to JSON-compatible types)
    """
    # Clear params.cfg to ensure clean baseline
    params.cfg.clear()

    # Get canonical TensorFlow configs
    model, training, inference = fixtures.get_all_canonical_configs()

    # Populate params.cfg via standard bridge (TrainingConfig and InferenceConfig)
    # Note: TrainingConfig includes nested ModelConfig, so this populates all model fields
    update_legacy_dict(params.cfg, training)
    update_legacy_dict(params.cfg, inference)

    # Serialize params.cfg to JSON-compatible format
    # Convert Path objects to strings, handle other non-serializable types
    baseline = {}
    for key, value in sorted(params.cfg.items()):
        if isinstance(value, Path):
            baseline[key] = str(value)
        elif isinstance(value, (int, float, str, bool, type(None))):
            baseline[key] = value
        else:
            # For complex types, convert to string representation
            baseline[key] = str(value)

    return baseline


def main():
    """Main entry point: capture baseline and save to JSON."""
    baseline = capture_baseline_params()

    # Save to JSON file
    output_path = Path(__file__).parent / 'baseline_params.json'
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2, sort_keys=True)

    print(f"✓ Baseline params.cfg captured to: {output_path}")
    print(f"  Total keys: {len(baseline)}")
    print(f"\nSample keys (first 10):")
    for i, (key, value) in enumerate(sorted(baseline.items())[:10]):
        print(f"    {key}: {value} ({type(value).__name__})")


if __name__ == '__main__':
    main()
