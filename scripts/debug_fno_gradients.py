import torch
from ptycho_torch.generators.fno import PtychoBlock


def debug_gradients():
    B, H, W, dim = 4, 64, 64, 32

    print(f"--- Debugging FNO Gradients (Hidden Dim={dim}) ---")
    block = PtychoBlock(channels=dim, modes=12)

    if hasattr(block.spectral, "weights"):
        spec_w = block.spectral.weights
        print(
            "Spectral Weights: mean={:.6f}, std={:.6f}, abs_mean={:.6f}".format(
                spec_w.mean().item(),
                spec_w.std().item(),
                spec_w.abs().mean().item(),
            )
        )

    local_w = block.local_conv.weight
    print(
        "Local Conv Weights: mean={:.6f}, std={:.6f}".format(
            local_w.mean().item(),
            local_w.std().item(),
        )
    )

    x = torch.randn(B, dim, H, W, requires_grad=True)
    y = block(x)
    loss = y.mean()
    loss.backward()

    print("\nGradients:")
    if hasattr(block.spectral, "weights"):
        spec_grad = block.spectral.weights.grad
        print(
            "Spectral Grad: mean={:.6f}, std={:.6f}, max={:.6f}".format(
                spec_grad.abs().mean().item(),
                spec_grad.std().item(),
                spec_grad.abs().max().item(),
            )
        )

    local_grad = block.local_conv.weight.grad
    print(
        "Local Conv Grad: mean={:.6f}, std={:.6f}, max={:.6f}".format(
            local_grad.abs().mean().item(),
            local_grad.std().item(),
            local_grad.abs().max().item(),
        )
    )

    if hasattr(block.spectral, "weights"):
        ratio = local_grad.abs().mean() / (spec_grad.abs().mean() + 1e-9)
        print(f"\nRatio (Local/Spectral Grad Magnitude): {ratio.item():.2f}")


if __name__ == "__main__":
    debug_gradients()
