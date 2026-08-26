'''Barycentric canvas accumulators extracted from reassembly.py (W4 split).

Owns the two vectorized barycentric accumulators used by
``reconstruct_image_barycentric``.  Pure torch, no cross-module references.
'''

from __future__ import annotations

import warnings
from typing import Tuple

import torch


class VectorizedBarycentricAccumulator:
    """
    Fully vectorized barycentric accumulation for ptychography reconstruction.
    Processes all patches simultaneously using scatter operations.
    """
    
    def __init__(self, canvas_shape: Tuple[int, int], device: torch.device):
        self.canvas_shape = canvas_shape
        self.device = device
    
    def accumulate_batch(self, 
                        canvas: torch.Tensor,
                        canvas_counts: torch.Tensor,
                        patches: torch.Tensor,
                        positions_px: torch.Tensor,
                        patch_size: int) -> None:
        """
        Vectorized accumulation of all patches at once.
        
        Args:
            canvas: (H, W) canvas tensor
            canvas_counts: (H, W) counts tensor
            patches: (N, patch_size, patch_size) patches to accumulate
            positions_px: (N, 2) positions in pixels
            patch_size: Size of each patch
        """
        N = patches.shape[0]
        half_size = patch_size / 2
        
        # Compute corners for all patches
        xmin = positions_px[:, 0] - half_size
        ymin = positions_px[:, 1] - half_size
        
        xmin_wh = xmin.floor().long()
        ymin_wh = ymin.floor().long()
        xmin_fr = xmin - xmin_wh.float()
        ymin_fr = ymin - ymin_wh.float()
        
        # Bounds checking
        valid_mask = (
            (xmin_wh >= 0) & (ymin_wh >= 0) &
            (xmin_wh + patch_size + 1 < self.canvas_shape[1]) &
            (ymin_wh + patch_size + 1 < self.canvas_shape[0])
        )
        
        if not valid_mask.all():
            valid_idx = torch.where(valid_mask)[0]
            if len(valid_idx) == 0:
                return
            patches = patches[valid_idx]
            xmin_wh, ymin_wh = xmin_wh[valid_idx], ymin_wh[valid_idx]
            xmin_fr, ymin_fr = xmin_fr[valid_idx], ymin_fr[valid_idx]
            N = len(valid_idx)
        
        # Bilinear interpolation weights
        xmin_fr_c = 1.0 - xmin_fr
        ymin_fr_c = 1.0 - ymin_fr
        
        w00 = (ymin_fr_c * xmin_fr_c).unsqueeze(-1).unsqueeze(-1)
        w01 = (ymin_fr_c * xmin_fr).unsqueeze(-1).unsqueeze(-1)
        w10 = (ymin_fr * xmin_fr_c).unsqueeze(-1).unsqueeze(-1)
        w11 = (ymin_fr * xmin_fr).unsqueeze(-1).unsqueeze(-1)
        
        # Create index tensors for vectorized operations
        patch_y, patch_x = torch.meshgrid(
            torch.arange(patch_size, device=self.device),
            torch.arange(patch_size, device=self.device),
            indexing='ij'
        )
        
        patch_y_exp = patch_y.unsqueeze(0).expand(N, -1, -1)
        patch_x_exp = patch_x.unsqueeze(0).expand(N, -1, -1)
        
        # Canvas coordinates for each patch pixel
        canvas_y_base = ymin_wh.unsqueeze(-1).unsqueeze(-1) + patch_y_exp
        canvas_x_base = xmin_wh.unsqueeze(-1).unsqueeze(-1) + patch_x_exp
        
        # Flatten for advanced indexing
        patches_flat = patches.reshape(N, -1)
        canvas_y_flat = canvas_y_base.reshape(N, -1)
        canvas_x_flat = canvas_x_base.reshape(N, -1)
        
        # Weighted patches
        w00_flat = w00.expand(-1, patch_size, patch_size).reshape(N, -1)
        w01_flat = w01.expand(-1, patch_size, patch_size).reshape(N, -1)
        w10_flat = w10.expand(-1, patch_size, patch_size).reshape(N, -1)
        w11_flat = w11.expand(-1, patch_size, patch_size).reshape(N, -1)
        
        weighted_patches_00 = (patches_flat * w00_flat).reshape(-1)
        weighted_patches_01 = (patches_flat * w01_flat).reshape(-1)
        weighted_patches_10 = (patches_flat * w10_flat).reshape(-1)
        weighted_patches_11 = (patches_flat * w11_flat).reshape(-1)
        
        # Canvas indices (flattened)
        idx_00 = (canvas_y_flat * self.canvas_shape[1] + canvas_x_flat).reshape(-1)
        idx_01 = (canvas_y_flat * self.canvas_shape[1] + canvas_x_flat + 1).reshape(-1)
        idx_10 = ((canvas_y_flat + 1) * self.canvas_shape[1] + canvas_x_flat).reshape(-1)
        idx_11 = ((canvas_y_flat + 1) * self.canvas_shape[1] + canvas_x_flat + 1).reshape(-1)
        
        # Accumulate on flattened canvas
        canvas_flat = canvas.reshape(-1)
        counts_flat = canvas_counts.reshape(-1)
        
        canvas_flat.scatter_add_(0, idx_00, weighted_patches_00)
        canvas_flat.scatter_add_(0, idx_01, weighted_patches_01)
        canvas_flat.scatter_add_(0, idx_10, weighted_patches_10)
        canvas_flat.scatter_add_(0, idx_11, weighted_patches_11)
        
        # Update counts
        counts_flat.scatter_add_(0, idx_00, w00_flat.reshape(-1))
        counts_flat.scatter_add_(0, idx_01, w01_flat.reshape(-1))
        counts_flat.scatter_add_(0, idx_10, w10_flat.reshape(-1))
        counts_flat.scatter_add_(0, idx_11, w11_flat.reshape(-1))



class VectorizedWeightedAccumulator:
    """
    Vectorized barycentric accumulation with probe-intensity confidence weighting.
    Identical to original implementation but scales contributions by |p|^2.
    """

    def __init__(self, canvas_shape: Tuple[int, int], device: torch.device):
        self.canvas_shape = canvas_shape
        self.device = device
        self.accepted_patches = 0
        self.total_patches = 0

    @property
    def patches_accepted(self) -> int:
        return self.accepted_patches

    @property
    def patches_total(self) -> int:
        return self.total_patches

    def accumulate_batch(self,
                        canvas: torch.Tensor,
                        canvas_weights: torch.Tensor,
                        patches: torch.Tensor,
                        positions_px: torch.Tensor,
                        probe_mag_sq: torch.Tensor,
                        patch_size: int,
                        uniform_weighting: bool = False) -> None:
        """
        Args:
            canvas: (H, W) Complex canvas tensor
            canvas_weights: (H, W) Float weights tensor (replaces counts)
            patches: (N, patch_size, patch_size) Complex texture patches
            positions_px: (N, 2) Sub-pixel global coordinates
            probe_mag_sq: (patch_size, patch_size) Intensity profile |p|^2
            patch_size: Size of each patch
        """
        N, H, W = patches.shape
        self.total_patches += int(N)
        half_size = patch_size / 2

        # 1. Coordinate and Bounds Logic (Identical to original)
        xmin = positions_px[:, 0] - half_size
        ymin = positions_px[:, 1] - half_size

        xmin_wh, ymin_wh = xmin.floor().long(), ymin.floor().long()
        xmin_fr, ymin_fr = xmin - xmin_wh.float(), ymin - ymin_wh.float()

        valid_mask = (
            (xmin_wh >= 0) & (ymin_wh >= 0) &
            (xmin_wh + patch_size + 1 < self.canvas_shape[1]) &
            (ymin_wh + patch_size + 1 < self.canvas_shape[0])
        )

        if not valid_mask.all():
            # A genuinely out-of-bounds patch (coordinate beyond the canvas
            # margin `reconstruct_image_barycentric` sized for) must not be
            # silently dropped -- warn loudly so callers notice a coverage
            # gap instead of discovering it downstream as an unexplained
            # metric regression (B4 report Sec 4: 2/59 patches were dropped
            # silently before this fix).
            n_dropped = int((~valid_mask).sum().item())
            warnings.warn(
                f"VectorizedWeightedAccumulator.accumulate_batch: dropping "
                f"{n_dropped}/{N} out-of-bounds patch(es) (canvas_shape="
                f"{self.canvas_shape}, patch_size={patch_size}) -- caller's "
                f"canvas was not sized to cover these coordinates.",
                stacklevel=2,
            )
            valid_idx = torch.where(valid_mask)[0]
            self.accepted_patches += int(len(valid_idx))
            if len(valid_idx) == 0:
                return
            patches = patches[valid_idx]
            xmin_wh, ymin_wh = xmin_wh[valid_idx], ymin_wh[valid_idx]
            xmin_fr, ymin_fr = xmin_fr[valid_idx], ymin_fr[valid_idx]
            N = len(valid_idx)
        else:
            self.accepted_patches += int(N)

        # 2. Bilinear Weights (Identical to original)
        xmin_fr_c, ymin_fr_c = 1.0 - xmin_fr, 1.0 - ymin_fr
        w00 = (ymin_fr_c * xmin_fr_c).view(N, 1, 1)
        w01 = (ymin_fr_c * xmin_fr).view(N, 1, 1)
        w10 = (ymin_fr * xmin_fr_c).view(N, 1, 1)
        w11 = (ymin_fr * xmin_fr).view(N, 1, 1)

        # 3. Apply Probe Intensity Weighting (|p|^2)
        if uniform_weighting:
            p_weight = torch.ones((H,W), device = probe_mag_sq.device)
        else:
            p_weight = probe_mag_sq.unsqueeze(0)
        weighted_patches = patches * p_weight

        # 4. Prepare Flattened Indices (Identical to original)
        patch_y, patch_x = torch.meshgrid(
            torch.arange(patch_size, device=self.device),
            torch.arange(patch_size, device=self.device),
            indexing='ij'
        )
        canvas_y_base = ymin_wh.view(N, 1, 1) + patch_y
        canvas_x_base = xmin_wh.view(N, 1, 1) + patch_x

        canvas_y_flat = canvas_y_base.reshape(N, -1)
        canvas_x_flat = canvas_x_base.reshape(N, -1)

        idx_00 = (canvas_y_flat * self.canvas_shape[1] + canvas_x_flat).reshape(-1)
        idx_01 = (canvas_y_flat * self.canvas_shape[1] + canvas_x_flat + 1).reshape(-1)
        idx_10 = ((canvas_y_flat + 1) * self.canvas_shape[1] + canvas_x_flat).reshape(-1)
        idx_11 = ((canvas_y_flat + 1) * self.canvas_shape[1] + canvas_x_flat + 1).reshape(-1)

        # 5. Scatter Accumulation
        canvas_flat = canvas.view(-1)
        weights_flat = canvas_weights.view(-1)

        def scatter_weighted(target, source_data, weight_coeff, indices):
            payload = (source_data * weight_coeff).reshape(-1)
            target.scatter_add_(0, indices, payload)

        # Data payloads
        wp_flat = weighted_patches.reshape(N, -1)
        pw_flat = p_weight.expand(N, -1, -1).reshape(N, -1)

        # Accumulate Complex Data
        scatter_weighted(canvas_flat, wp_flat, w00.reshape(N, 1), idx_00)
        scatter_weighted(canvas_flat, wp_flat, w01.reshape(N, 1), idx_01)
        scatter_weighted(canvas_flat, wp_flat, w10.reshape(N, 1), idx_10)
        scatter_weighted(canvas_flat, wp_flat, w11.reshape(N, 1), idx_11)

        # Accumulate Weights (Denominator)
        scatter_weighted(weights_flat, pw_flat, w00.reshape(N, 1), idx_00)
        scatter_weighted(weights_flat, pw_flat, w01.reshape(N, 1), idx_01)
        scatter_weighted(weights_flat, pw_flat, w10.reshape(N, 1), idx_10)
        scatter_weighted(weights_flat, pw_flat, w11.reshape(N, 1), idx_11)

