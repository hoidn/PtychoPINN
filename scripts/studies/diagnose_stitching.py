"""Isolate the reassembly-stage quality loss (user recipe Step 3).

Per-patch object recon is good (|amp| NCC ~0.72) but the harness's complex,
probe-weighted reassembly gives canvas NCC ~0.38. This builds three reference
reassemblies from the SAME patches + coords_global and template-matches each
against the true object, to tell WHICH stitching factor destroys quality:

  A. complex_uniform  : uniform-weighted complex average (naive)
  B. complex_probe    : probe|^2-weighted complex average (== harness)
  C. amp_uniform      : uniform-weighted AMPLITUDE average (phase-blind)
  D. phase_aligned    : greedily phase-align each patch to the running canvas
                        (single global phase per patch) before complex-averaging

If C/D >> A/B: patches are phase-gauge-INCONSISTENT (averaging cancels them).
If A ~= C ~= D but all ~0.4: phase gauge is fine; loss is positional/trim/weight.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.feature import match_template

REPO = Path("/home/ollie/Documents/PtychoPINN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/studies"))
from varpro_probe_ablation_runner import build_configs, build_test_dataset, ARM_TABLE
from ptycho_torch.dataloader import Collate_Lightning
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.lightning_utils import load_checkpoint_with_configs

MIDDLE = 32


def trim(p):
    N = p.shape[-1]; s = N // 2 - MIDDLE // 2
    return p[..., s:s + MIDDLE, s:s + MIDDLE]


def place(canvas, wt, patch, weight, cy, cx):
    h = MIDDLE
    y0 = cy - h // 2; x0 = cx - h // 2
    canvas[y0:y0 + h, x0:x0 + h] += patch * weight
    wt[y0:y0 + h, x0:x0 + h] += weight


def ncc_vs(obj_amp, canvas):
    c = np.abs(canvas)
    nz = np.argwhere(c > 1e-8)
    if len(nz) == 0:
        return 0.0
    y0, x0 = nz.min(0); y1, x1 = nz.max(0) + 1
    cc = c[y0:y1, x0:x1]
    cc = cc[:obj_amp.shape[0] - 1, :obj_amp.shape[1] - 1]
    return float(match_template(obj_amp.astype(np.float64), cc.astype(np.float64)).max())


def main():
    # argv: [test_npz] [checkpoint] ; defaults to full-phase deadleaves_N64
    test_npz = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        REPO / ".artifacts/varpro_ablation/datasets/deadleaves_N64_test.npz"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else \
        str(REPO / ".artifacts/varpro_ablation/matrix_dl/gs1_frozen/training_outputs/Synthetic_Runs/train/checkpoints/best-checkpoint.ckpt")
    arm = ARM_TABLE["gs1_frozen"]
    dc, mc, tc, ic, gc = build_configs(arm, 16, 25)
    SC = Path("/tmp/claude-1000/-home-ollie-Documents-PtychoPINN/9ff76e13-9c97-4cfe-a574-8d5fff1cd235/scratchpad/stitch")
    ds = build_test_dataset(test_npz, mc, dc, tc, SC)
    n = len(ds)
    raw = ds[list(range(n))]
    b = Collate_Lightning(False)(raw); td = b[0]
    model, _ = load_checkpoint_with_configs(ckpt, PtychoPINN_Lightning)
    model = model.cpu().eval()
    with torch.no_grad():
        patches = model.forward_predict(td["images"], td["coords_relative"], b[1], td["rms_scaling_constant"])
    patches = trim(patches).numpy()[:, 0]           # (n, MIDDLE, MIDDLE) complex
    gc2 = td["coords_global"].squeeze(2).numpy().reshape(-1, 2)  # (n,2) (x,y)
    probe = b[1].numpy()[0, 0]                       # (P,H,W)
    pw = trim(np.sum(np.abs(probe) ** 2, axis=0))    # (MIDDLE,MIDDLE)

    x = gc2[:, 0]; y = gc2[:, 1]
    x0f, y0f = x.min(), y.min()
    pad = MIDDLE
    H = int(y.max() - y0f) + 2 * pad; W = int(x.max() - x0f) + 2 * pad
    cyx = [(int(round(y[i] - y0f)) + pad, int(round(x[i] - x0f)) + pad) for i in range(n)]

    objamp = np.abs(np.load(test_npz, allow_pickle=True)["objectGuess"])

    results = {}
    for tag, mode in [("A complex_uniform", "cu"), ("B complex_probe", "cp"),
                      ("C amp_uniform", "au"), ("D phase_aligned", "pa")]:
        canvas = np.zeros((H, W), np.complex128); wt = np.zeros((H, W), np.float64)
        for i in range(n):
            p = patches[i]; cy, cx = cyx[i]
            w = pw if mode == "cp" else np.ones_like(pw)
            if mode == "au":
                place(canvas, wt, np.abs(p).astype(np.complex128), w, cy, cx)
            elif mode == "pa":
                # global phase to align this patch to the current canvas overlap
                h = MIDDLE; y0 = cy - h // 2; x0 = cx - h // 2
                cur = canvas[y0:y0 + h, x0:x0 + h]; curw = wt[y0:y0 + h, x0:x0 + h]
                if curw.sum() > 0:
                    ref = cur / (curw + 1e-12)
                    inner = np.sum(p * np.conj(ref) * (curw > 0))
                    if np.abs(inner) > 0:
                        p = p * np.exp(-1j * np.angle(inner))
                place(canvas, wt, p, w, cy, cx)
            else:
                place(canvas, wt, p, w, cy, cx)
        canvas = canvas / (wt + 1e-12)
        results[tag] = ncc_vs(objamp, canvas)

    print(f"patches n={n}, MIDDLE={MIDDLE}")
    for tag, v in results.items():
        print(f"  {tag:20s} canvas |amp| NCC vs object: {v:.3f}")


if __name__ == "__main__":
    main()
