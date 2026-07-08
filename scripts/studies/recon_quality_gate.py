"""Recon-quality gate for the ablation matrix (amendment #13).

Guardrail that catches degenerate arms BEFORE their VarPro/probe metrics are
trusted. Uses the HONEST reconstruction measure established during the Task 1.6
root-cause work: direct overlay of the model's object patches in the true
object-pixel frame (one coordinate source, single global complex gauge, NO
template-match), scored as |amp| NCC against the ground-truth object.

For every arm subdir under a matrix output root it loads the best checkpoint,
runs forward_predict on the arm's held-out test set, assembles a direct-overlay
canvas, and reports:
  - per-patch |amp| NCC at coords_global (patch fidelity)
  - assembled-canvas |amp| NCC, single global gauge (reassembly fidelity)
An arm whose canvas NCC falls below THRESHOLD is flagged FAIL.

Usage: python recon_quality_gate.py <matrix_root> [threshold]
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/ollie/Documents/PtychoPINN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/studies"))
from varpro_probe_ablation_runner import build_configs, build_test_dataset, ARM_TABLE
from diagnose_placement import gauge, ncc, trim, MIDDLE
from ptycho_torch.dataloader import Collate_Lightning
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.lightning_utils import load_checkpoint_with_configs

THRESHOLD = 0.6
SC = Path("/tmp/claude-1000/-home-ollie-Documents-PtychoPINN/9ff76e13-9c97-4cfe-a574-8d5fff1cd235/scratchpad/gate")


DATASET_STEM = "lines"  # override with argv[3]


def arm_test_npz(arm_name: str) -> Path:
    """Map arm -> its held-out test set by the arm's N and the dataset stem."""
    n = ARM_TABLE[arm_name].get("N", 64)
    return REPO / f".artifacts/varpro_ablation/datasets/{DATASET_STEM}_N{n}_test.npz"


def measure_arm(arm_name: str, ckpt: Path) -> dict:
    dc, mc, tc, ic, gc = build_configs(ARM_TABLE[arm_name], 16, 25)
    test_npz = arm_test_npz(arm_name)
    # fresh scratch: build_test_dataset REUSES an existing memory map in the dir,
    # keyed by path not content -> a stale map silently measures the wrong data.
    shutil.rmtree(SC / arm_name, ignore_errors=True)
    ds = build_test_dataset(test_npz, mc, dc, tc, SC / arm_name)
    n = len(ds)
    b = Collate_Lightning(False)(ds[list(range(n))]); td = b[0]
    model, _ = load_checkpoint_with_configs(str(ckpt), PtychoPINN_Lightning)
    model = model.cpu().eval()
    with torch.no_grad():
        patches = model.forward_predict(td["images"], td["coords_relative"], b[1], td["rms_scaling_constant"])
    # (n_groups, C, H, W) with C=gridsize**2; flatten groups+channels so gs1 (C=1)
    # and gs2 (C=4) are handled uniformly. coords_global is (n_groups, C, 1, 2).
    patches = trim(patches).numpy().reshape(-1, MIDDLE, MIDDLE)
    gc2 = td["coords_global"].squeeze(2).numpy().reshape(-1, 2)  # col0=x, col1=y
    assert patches.shape[0] == gc2.shape[0], f"patch/coord count mismatch {patches.shape[0]} vs {gc2.shape[0]}"
    obj = np.load(test_npz, allow_pickle=True)["objectGuess"]
    R = obj.shape[0]; h = MIDDLE // 2
    cx = np.round(gc2[:, 0]).astype(int); cy = np.round(gc2[:, 1]).astype(int)
    n = patches.shape[0]

    per = []
    canvas = np.zeros((R, R), np.complex128); wt = np.zeros((R, R), np.float64)
    for i in range(n):
        y0, x0 = cy[i] - h, cx[i] - h
        if y0 < 0 or x0 < 0 or y0 + MIDDLE > R or x0 + MIDDLE > R:
            continue
        truth = obj[y0:y0 + MIDDLE, x0:x0 + MIDDLE]
        rp = gauge(patches[i], truth)
        per.append(ncc(np.abs(rp), np.abs(truth)))
        canvas[y0:y0 + MIDDLE, x0:x0 + MIDDLE] += patches[i]
        wt[y0:y0 + MIDDLE, x0:x0 + MIDDLE] += 1.0
    m = wt > 0
    canvas[m] /= wt[m]
    g = gauge(canvas[m], obj[m])
    return {"per_patch": float(np.mean(per)) if per else 0.0,
            "canvas": float(ncc(np.abs(g), np.abs(obj[m]))), "n": len(per)}


def main() -> int:
    global DATASET_STEM
    root = Path(sys.argv[1])
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else THRESHOLD
    if len(sys.argv) > 3:
        DATASET_STEM = sys.argv[3]
    rows, failed = [], []
    for arm_name in ARM_TABLE:
        arm_dir = root / arm_name
        ckpts = list(arm_dir.glob("**/best-checkpoint.ckpt"))
        if not ckpts:
            rows.append((arm_name, None)); failed.append(arm_name); continue
        r = measure_arm(arm_name, ckpts[0])
        rows.append((arm_name, r))
        if r["canvas"] < thr:
            failed.append(arm_name)

    print(f"\nRECON-QUALITY GATE (threshold canvas |amp| NCC > {thr})")
    print(f"{'arm':26s} {'per_patch':>10s} {'canvas':>8s} {'n':>5s}  verdict")
    print("-" * 62)
    for arm_name, r in rows:
        if r is None:
            print(f"{arm_name:26s} {'--':>10s} {'--':>8s} {'--':>5s}  NO CKPT")
        else:
            v = "PASS" if r["canvas"] >= thr else "FAIL"
            print(f"{arm_name:26s} {r['per_patch']:10.3f} {r['canvas']:8.3f} {r['n']:5d}  {v}")
    print("-" * 62)
    print("GATE PASS" if not failed else f"GATE FAIL: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
