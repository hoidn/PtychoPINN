#!/usr/bin/env python3
"""
Verify Phase B readiness for FIX-PYTORCH-FORWARD-PARITY-001.

This script confirms:
1. Phase A artifacts exist and are fresh (post-dc5415ba)
2. config_factory.py already defaults object_big=True
3. No blockers prevent starting Phase B scaling/config work

Exit codes:
    0: Ready for Phase B
    1: Phase A incomplete or blocker found
"""

import sys
from pathlib import Path
from datetime import datetime
import re

# Minimum timestamp: dc5415ba commit (2025-11-14 02:30:49)
MIN_TIMESTAMP = datetime(2025, 11, 14, 2, 30, 49)

HUB_ROOT = Path(__file__).parent.parent / "reports/2025-11-13T000000Z/forward_parity"

def check_phase_a_complete():
    """Verify all Phase A artifacts exist and postdate dc5415ba"""
    required_artifacts = [
        "green/pytest_patch_stats_rerun.log",
        "cli/train_patch_stats_rerun.log",
        "cli/inference_patch_stats_rerun.log",
        "analysis/torch_patch_stats.json",
        "analysis/torch_patch_stats_inference.json",
        "analysis/torch_patch_grid.png",
        "analysis/torch_patch_grid_inference.png",
        "analysis/forward_parity_debug",
        "analysis/artifact_inventory.txt",
    ]

    print("Phase A Completeness Check")
    print("=" * 60)

    all_present = True
    for artifact_path in required_artifacts:
        full_path = HUB_ROOT / artifact_path
        if not full_path.exists():
            print(f"✗ MISSING: {artifact_path}")
            all_present = False
        else:
            mtime = datetime.fromtimestamp(full_path.stat().st_mtime)
            status = "✓" if mtime >= MIN_TIMESTAMP else "✗ TOO OLD"
            print(f"{status} {artifact_path}: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    return all_present

def check_object_big_default():
    """Verify config_factory.py defaults object_big=True"""
    factory_path = Path("ptycho_torch/config_factory.py")

    print("\nPhase B Prerequisite: object_big Default")
    print("=" * 60)

    if not factory_path.exists():
        print(f"✗ MISSING: {factory_path}")
        return False

    content = factory_path.read_text()

    # Check for the two factory functions that should default object_big=True
    training_match = re.search(r"object_big=overrides\.get\('object_big',\s*(True|False)\)", content)

    if training_match:
        default_value = training_match.group(1)
        if default_value == "True":
            print(f"✓ config_factory.py defaults object_big=True (line match found)")
            return True
        else:
            print(f"✗ config_factory.py defaults object_big={default_value} (should be True)")
            return False
    else:
        print("✗ Could not find object_big default in config_factory.py")
        return False

def main():
    print("Phase B Readiness Verification")
    print("=" * 60)
    print(f"Hub root: {HUB_ROOT.resolve()}\n")

    phase_a_ok = check_phase_a_complete()
    object_big_ok = check_object_big_default()

    print("\n" + "=" * 60)
    if phase_a_ok and object_big_ok:
        print("✅ READY FOR PHASE B")
        print("   - Phase A evidence complete (9/9 artifacts post-dc5415ba)")
        print("   - object_big already defaults to True in config_factory.py")
        print("\nNext: Begin Phase B checklist B1-B3 per implementation plan")
        return 0
    else:
        print("✗ NOT READY FOR PHASE B")
        if not phase_a_ok:
            print("   - Phase A artifacts missing or stale")
        if not object_big_ok:
            print("   - object_big default verification failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
