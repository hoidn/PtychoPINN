#!/usr/bin/env python3
"""
Verification nucleus: Prove Phase A evidence exists and postdates dc5415ba.

This script validates that all Phase A artifacts exist in the Reports Hub
with timestamps after the TrainingPayload fix (dc5415ba, 2025-11-14 02:30:49).

Exit code 0 if Phase A is complete, 1 if evidence is missing/stale.
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Timestamps
DC5415BA_EPOCH = 1763116249  # 2025-11-14 02:30:49 PST
# Navigate from bin/ -> FIX-PYTORCH-FORWARD-PARITY-001/ -> reports/
HUB_ROOT = Path(__file__).parent.parent / "reports/2025-11-13T000000Z/forward_parity"

def check_file_timestamp(path: Path, name: str, min_epoch: int) -> bool:
    """Check if file exists and postdates min_epoch."""
    if not path.exists():
        print(f"❌ MISSING: {name} ({path})")
        return False

    mtime = path.stat().st_mtime
    if mtime < min_epoch:
        dt = datetime.fromtimestamp(mtime)
        print(f"❌ STALE: {name} ({dt.strftime('%Y-%m-%d %H:%M:%S')}, expected >{min_epoch})")
        return False

    dt = datetime.fromtimestamp(mtime)
    print(f"✓ {name}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    return True

def main():
    print("Phase A Evidence Verification")
    print("=" * 60)
    print(f"Minimum timestamp: 2025-11-14 02:30:49 (dc5415ba)")
    print(f"Hub root: {HUB_ROOT}")
    print()

    checks = []

    # Check pytest log
    checks.append(check_file_timestamp(
        HUB_ROOT / "green/pytest_patch_stats_rerun.log",
        "pytest selector log",
        DC5415BA_EPOCH
    ))

    # Check training CLI log
    checks.append(check_file_timestamp(
        HUB_ROOT / "cli/train_patch_stats_rerun.log",
        "training CLI log",
        DC5415BA_EPOCH
    ))

    # Check inference CLI log
    checks.append(check_file_timestamp(
        HUB_ROOT / "cli/inference_patch_stats_rerun.log",
        "inference CLI log",
        DC5415BA_EPOCH
    ))

    # Check JSON artifacts
    checks.append(check_file_timestamp(
        HUB_ROOT / "analysis/torch_patch_stats.json",
        "training patch stats JSON",
        DC5415BA_EPOCH
    ))

    checks.append(check_file_timestamp(
        HUB_ROOT / "analysis/torch_patch_stats_inference.json",
        "inference patch stats JSON",
        DC5415BA_EPOCH
    ))

    # Check PNG artifacts
    checks.append(check_file_timestamp(
        HUB_ROOT / "analysis/torch_patch_grid.png",
        "training patch grid PNG",
        DC5415BA_EPOCH
    ))

    checks.append(check_file_timestamp(
        HUB_ROOT / "analysis/torch_patch_grid_inference.png",
        "inference patch grid PNG",
        DC5415BA_EPOCH
    ))

    # Check debug dump dir
    debug_dir = HUB_ROOT / "analysis/forward_parity_debug"
    if not debug_dir.is_dir():
        print(f"❌ MISSING: debug dump directory ({debug_dir})")
        checks.append(False)
    else:
        print(f"✓ debug dump directory exists")
        checks.append(True)

    # Check artifact inventory
    checks.append(check_file_timestamp(
        HUB_ROOT / "analysis/artifact_inventory.txt",
        "artifact inventory",
        DC5415BA_EPOCH
    ))

    print()
    print("=" * 60)

    if all(checks):
        print("✅ Phase A COMPLETE: All evidence postdates dc5415ba")
        print("   Ready for Phase B (scaling/config alignment)")
        return 0
    else:
        failed_count = sum(1 for c in checks if not c)
        print(f"❌ Phase A INCOMPLETE: {failed_count}/{len(checks)} checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
