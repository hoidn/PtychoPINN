import torch


def summarize_offsets(name, offsets: torch.Tensor) -> str:
    if offsets is None:
        return f"{name}: None"
    flat = offsets.reshape(-1, offsets.shape[-1]).detach().cpu()
    return (
        f"{name}: shape={tuple(offsets.shape)} "
        f"min={flat.min(0).values.tolist()} "
        f"max={flat.max(0).values.tolist()} "
        f"std={flat.std(0).tolist()} "
        f"unique={torch.unique(flat, dim=0).shape[0]}"
    )
