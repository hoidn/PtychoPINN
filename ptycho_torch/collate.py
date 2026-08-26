"""Collation, materialization, and the one native loader factory.

Owns the collate callables (``Collate`` / ``Collate_Lightning``), the expanded-
stride materialization helpers, and :func:`build_ptycho_loader`, the single
maintained ``DataLoader`` factory shared by the RAM and mmap datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensordict import TensorDict


def _materialize_expanded_tensor(value):
    if any(
        size > 1 and stride == 0
        for size, stride in zip(value.shape, value.stride())
    ):
        return value.clone()
    return value


def _materialize_expanded_tensordict(tensor_dict):
    return tensor_dict.apply(_materialize_expanded_tensor)

class TensorDictDataLoader(DataLoader):
    """Compatibility name for the native PyTorch DataLoader implementation."""


#Custom collation function which pins memory in order to transfer to gpu
#Taken from: https://pytorch.org/tensordict/stable/tutorials/tensorclass_imagenet.html
class Collate(nn.Module):
    """
    Classic data collation function that works with native pytorch training protocol.
    One gpu only.
    """
    def __init__(self, device = None):
        super().__init__()
        self.device = torch.device(device) if device is not None else None
    def __call__(self, x):
        '''
        Moves tensor to RAM, and then to GPU.

        Inputs
        -------
        x: TensorDict
        '''
        tensor_dict, probe, scaling = _coalesce_ptycho_batch(x)
        outputs = [
            _materialize_batch_fields(tensor_dict),
            _materialize_expanded_tensor(probe),
            _materialize_expanded_tensor(scaling),
        ]
        
        # Pin memory if using CUDA
        if self.device and self.device.type == 'cuda':
            outputs = [item.pin_memory() for item in outputs]
            
        # Move to device if specified
        if self.device:
            outputs = [item.to(self.device, non_blocking=False) for item in outputs]
            
        return tuple(outputs)

# Modified collate function for PyTorch lightning

class Collate_Lightning(nn.Module):
    """
    Modified data collation function that works specifically with pytorch lightning
    This is because pytorch lightning explicitly handles device transfers so we don't need to mention any devices in this function
    Otherwise, with multi GPU the device calls will return errors.
    """
    def __init__(self, pin_memory_if_cuda = True):
        super().__init__()
        self.pin_memory_if_cuda = pin_memory_if_cuda

    def __call__(self, x):
        """Prepare a CPU batch; DataLoader owns pinning and Lightning transfer."""
        tensor_dict, probe, scaling = _coalesce_ptycho_batch(x)
        outputs = [
            _materialize_batch_fields(tensor_dict),
            _materialize_expanded_tensor(probe),
            _materialize_expanded_tensor(scaling),
        ]
        return tuple(outputs)


def _materialize_batch_fields(fields):
    if isinstance(fields, TensorDict):
        return _materialize_expanded_tensordict(fields)
    return {
        name: (
            _materialize_expanded_tensor(value)
            if isinstance(value, torch.Tensor)
            else value
        )
        for name, value in fields.items()
    }


def _stack_batch_fields(fields):
    if isinstance(fields[0], TensorDict):
        return torch.stack(fields, dim=0)
    return {
        name: torch.stack([sample[name] for sample in fields], dim=0)
        for name in fields[0]
    }


def _coalesce_ptycho_batch(batch):
    """Accept native vectorized output or a list of scalar samples."""

    if isinstance(batch, list):
        if not batch:
            raise ValueError("cannot collate an empty ptychography batch")
        fields, probes, scalings = zip(*batch)
        return (
            _stack_batch_fields(fields),
            torch.stack(probes, dim=0),
            torch.stack(scalings, dim=0),
        )
    return batch


def build_ptycho_loader(
    dataset,
    *,
    batch_size,
    shuffle=False,
    sampler=None,
    seed=42,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=None,
    drop_last=False,
    collate_fn=None,
):
    """Construct the single native loader path used by RAM and mmap datasets."""

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return TensorDictDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle) and sampler is None,
        sampler=sampler,
        generator=generator,
        drop_last=drop_last,
        collate_fn=(
            Collate_Lightning(pin_memory_if_cuda=False)
            if collate_fn is None
            else collate_fn
        ),
        **kwargs,
    )
