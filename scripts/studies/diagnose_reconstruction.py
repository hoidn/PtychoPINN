"""Bottom-up reconstruction diagnosis (user's recipe, 2026-07-01).

Step 1b: per-image diffraction self-consistency at gs1 -- does the trained
model's PREDICTED diffraction match the MEASURED diffraction? This is exactly
what the unsupervised Poisson loss optimizes; if it fails, nothing downstream
can work.

Step 2: per-patch object fidelity -- does the model's reconstructed object
patch match the ground-truth patch (ground_truth_patches), up to the global
complex-scalar gauge?

Loads a trained gs1 checkpoint, runs compute_loss's exact forward call on a
batch of the TRAIN set (in-distribution), and writes a diagnostic PNG plus
prints correlation/relative-error numbers.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/ollie/Documents/PtychoPINN")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/studies"))

from varpro_probe_ablation_runner import build_configs, build_test_dataset, ARM_TABLE
from ptycho_torch.dataloader import Collate_Lightning
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.lightning_utils import load_checkpoint_with_configs

SCRATCH = Path("/tmp/claude-1000/-home-ollie-Documents-PtychoPINN/9ff76e13-9c97-4cfe-a574-8d5fff1cd235/scratchpad/diag")
OUT = REPO / "tmp"


def gauge(r, t):
    a = np.sum(np.conj(r) * t) / (np.sum(np.abs(r) ** 2) + 1e-30)
    return a * r, a


def corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(np.sum(a * b) / (np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-30))


def main(arm_name, train_npz, ckpt):
    arm = ARM_TABLE[arm_name]
    dc, mc, tc, ic, gc = build_configs(arm, batch_size=16, epochs=25)
    ds = build_test_dataset(Path(train_npz), mc, dc, tc, SCRATCH)
    raw = ds[list(range(16))]
    batch = Collate_Lightning(pin_memory_if_cuda=False)(raw)
    td = batch[0]
    x = td["images"]; probe = batch[1]; probe_scaling = batch[2]
    rms = td["rms_scaling_constant"]; phys = td["physics_scaling_constant"]
    mos = torch.sqrt(1 / (probe_scaling**2 * phys + 1e-9))

    model, _ = load_checkpoint_with_configs(ckpt, PtychoPINN_Lightning)
    model = model.cpu().eval()
    with torch.no_grad():
        pred, real, imag = model(x, td["coords_relative"], probe,
                                 input_scale_factor=rms, output_scale_factor=mos,
                                 experiment_ids=td["experiment_id"], fine_tune=False)
    pred = pred.numpy(); xm = x.numpy()
    print(f"[{arm_name}] pred shape {pred.shape}, measured shape {xm.shape}")
    print(f"[{arm_name}] pred (predicted diffraction) mean {pred.mean():.4g} max {pred.max():.4g}")
    print(f"[{arm_name}] measured 'images' mean {xm.mean():.4g} max {xm.max():.4g}")

    # Poisson compares pred (intensity rate) to measured. If measured is counts
    # and pred is intensity, they should match in scale AND structure.
    # Step 1b metric: per-image correlation of predicted vs measured diffraction.
    n = pred.shape[0]
    diff_corrs = [corr(pred[i, 0], xm[i, 0]) for i in range(n)]
    print(f"[{arm_name}] STEP 1b diffraction corr: mean {np.mean(diff_corrs):.3f} "
          f"min {np.min(diff_corrs):.3f} max {np.max(diff_corrs):.3f}")
    # scale check: ratio of total intensity
    ratio = pred.sum() / (xm.sum() + 1e-30)
    print(f"[{arm_name}] pred/measured total-intensity ratio: {ratio:.4g}")

    # Step 2: object patch vs ground truth patch
    step2 = None
    if "ground_truth_patches" in ds.__dict__ or hasattr(ds, "data_dict"):
        pass
    gt = np.load(train_npz, allow_pickle=True).get("ground_truth_patches", None)
    obj = (real.numpy() + 1j * imag.numpy())  # (B,C,H,W)
    if gt is not None:
        # gt (Npos, H, W, 1) -> match first 16 in dataset index order is not
        # guaranteed; use correlation on best-scale-aligned magnitude as a proxy
        gt0 = gt[:n, ..., 0] if gt.ndim == 4 else gt[:n]
        obj0 = obj[:, 0]
        pcorr = []
        for i in range(min(n, len(gt0))):
            og, _ = gauge(obj0[i], gt0[i])
            pcorr.append(corr(np.abs(og), np.abs(gt0[i])))
        step2 = float(np.mean(pcorr))
        print(f"[{arm_name}] STEP 2 object-patch |amp| corr (gauge-aligned, index-order): mean {step2:.3f}")

    # Diagnostic figure: measured vs predicted diffraction for 3 images
    fig, ax = plt.subplots(3, 4, figsize=(14, 10))
    for i in range(3):
        ax[i, 0].imshow(np.log1p(xm[i, 0]), cmap="viridis"); ax[i, 0].set_title(f"measured diff #{i} (log)")
        ax[i, 1].imshow(np.log1p(pred[i, 0]), cmap="viridis"); ax[i, 1].set_title(f"predicted diff #{i} (log)")
        ax[i, 2].imshow(np.abs(obj[i, 0]), cmap="gray"); ax[i, 2].set_title(f"recon |obj| #{i}")
        if gt is not None and i < len(gt0):
            ax[i, 3].imshow(np.abs(gt0[i]), cmap="gray"); ax[i, 3].set_title(f"truth |obj| #{i}")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle(f"{arm_name}: diffraction self-consistency (corr {np.mean(diff_corrs):.3f}) "
                 f"+ object patch (corr {step2})")
    fig.tight_layout()
    dest = OUT / f"diag_{arm_name}.png"
    fig.savefig(dest, dpi=90); plt.close(fig)
    print(f"[{arm_name}] wrote {dest}")


if __name__ == "__main__":
    main("gs1_frozen",
         str(REPO / ".artifacts/varpro_ablation/datasets/deadleaves_N64_train.npz"),
         str(REPO / ".artifacts/varpro_ablation/matrix_dl/gs1_frozen/training_outputs/Synthetic_Runs/train/checkpoints/best-checkpoint.ckpt"))
