"""Dump quick stats from a Stage A arm run directory.

Prints amplitude/phase mean/std from metrics.json, best val_loss from
history.json, and checks whether StablePtychoBlock.norm.weight stayed
near zero (for stable_hybrid runs with a saved model.pt).

Usage:
    python scripts/internal/stage_a_dump_stats.py \
        --run-dir outputs/grid_lines_stage_a/arm_stable/runs/pinn_stable_hybrid \
        --out-json plans/active/.../stage_a_arm_stable_stats.json
"""

import argparse
import json
import pathlib
import sys


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def compute_stats(run_dir: pathlib.Path) -> dict:
    stats = {}

    # History: training curves
    history_path = run_dir / "history.json"
    if history_path.exists():
        history = load_json(history_path)
        val_losses = history.get("val_loss", [])
        train_losses = history.get("train_loss", [])
        has_nan = any(v != v for v in val_losses)  # NaN check
        stats["val_loss_best"] = min(val_losses) if val_losses and not has_nan else None
        stats["val_loss_final"] = val_losses[-1] if val_losses else None
        stats["train_loss_final"] = train_losses[-1] if train_losses else None
        stats["has_nan"] = has_nan
        stats["n_epochs"] = len(val_losses)

        # Grad norm stats if present
        grad_norms = history.get("grad_norm", [])
        if grad_norms:
            valid = [g for g in grad_norms if g == g]  # exclude NaN
            stats["grad_norm_mean"] = sum(valid) / len(valid) if valid else None
            stats["grad_norm_max"] = max(valid) if valid else None
            stats["grad_norm_min"] = min(valid) if valid else None
    else:
        stats["error"] = "history.json not found"

    # Metrics: reconstruction quality
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        mae = metrics.get("mae", [None, None])
        ssim = metrics.get("ssim", [None, None])
        psnr = metrics.get("psnr", [None, None])
        stats["amp_mae"] = float(mae[0]) if mae[0] is not None else None
        stats["phase_mae"] = float(mae[1]) if mae[1] is not None else None
        stats["amp_ssim"] = float(ssim[0]) if ssim[0] is not None else None
        stats["phase_ssim"] = float(ssim[1]) if ssim[1] is not None else None
        stats["amp_psnr"] = float(psnr[0]) if psnr[0] is not None else None
        stats["phase_psnr"] = float(psnr[1]) if psnr[1] is not None else None

    # Model: check norm.weight for stable_hybrid
    model_path = run_dir / "model.pt"
    if model_path.exists():
        try:
            import torch
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            norm_keys = [k for k in state_dict if "norm.weight" in k]
            if norm_keys:
                norm_stats = {}
                for k in norm_keys:
                    w = state_dict[k]
                    norm_stats[k] = {
                        "mean": float(w.mean()),
                        "std": float(w.std()),
                        "abs_max": float(w.abs().max()),
                    }
                stats["norm_weights"] = norm_stats
                # Summary: did norm.weight stay near zero?
                all_near_zero = all(
                    abs(v["mean"]) < 0.1 for v in norm_stats.values()
                )
                stats["norm_weight_near_zero"] = all_near_zero
        except Exception as e:
            stats["model_load_error"] = str(e)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Dump Stage A arm stats")
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-json", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"ERROR: run dir not found: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    stats = compute_stats(args.run_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
