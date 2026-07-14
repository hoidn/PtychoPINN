#Type helpers
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple, Optional, Union, Any, List, Dict, Literal
from ptycho_torch.dataloader import PtychoDataset, TensorDictDataLoader

#Pytorch-related
import torch
from torch import nn
import torch.nn.functional as F
import ptycho_torch.helper as hh
from ptycho_torch.dataloader import Collate

#Other useful libraries
import time
import gc
import warnings

#Configurations
from ptycho_torch.config_params import ModelConfig, TrainingConfig, DataConfig, InferenceConfig
from ptycho_torch.probe_mask import resolve_probe_mask_torch
from ptycho_torch.reassembly_diagnostics import (
    FittedCountMetrics,
    NotApplicable,
    ReassemblyDiagnostics,
    VarProSufficientStatistics,
    array_digest,
    not_applicable,
    not_evaluated,
)
from ptycho_torch.scaling_contract import (
    CI_SCALE_CONTRACT,
    LEGACY_SCALE_CONTRACT,
    resolve_scale_contract,
)

#Default casting
torch.set_default_dtype(torch.float32)


InferencePrecision = Literal["32-true", "16-mixed", "bf16-mixed"]
_INFERENCE_PRECISIONS = {"32-true", "16-mixed", "bf16-mixed"}
ReassemblyReturn = Union[
    Tuple[torch.Tensor, Any, List[Any]],
    Tuple[torch.Tensor, Any, List[Any], torch.Tensor],
    Tuple[torch.Tensor, Any, ReassemblyDiagnostics, torch.Tensor],
]


def resolve_inference_precision(
    use_mixed_precision: Optional[bool] = None,
    precision: Optional[InferencePrecision] = None,
) -> InferencePrecision:
    """Resolve the legacy boolean alias and explicit inference precision."""
    if precision is not None and precision not in _INFERENCE_PRECISIONS:
        raise ValueError(
            "precision must be one of '32-true', '16-mixed', or 'bf16-mixed'"
        )
    legacy_precision: Optional[InferencePrecision] = None
    if use_mixed_precision is not None:
        if not isinstance(use_mixed_precision, bool):
            raise TypeError("use_mixed_precision must be bool or None")
        legacy_precision = "16-mixed" if use_mixed_precision else "32-true"
    if precision is not None and legacy_precision is not None:
        if precision != legacy_precision:
            raise ValueError(
                "Conflicting inference precision: "
                f"use_mixed_precision={use_mixed_precision!r} resolves to "
                f"{legacy_precision!r}, but precision={precision!r}"
            )
        return precision
    if precision is not None:
        return precision
    if legacy_precision is not None:
        return legacy_precision
    return "32-true"


def resolve_inference_precision_for_device(
    precision: InferencePrecision,
    device: Union[str, torch.device],
) -> InferencePrecision:
    """Apply the precision semantics used by Lightning for a device type."""
    resolved = resolve_inference_precision(None, precision)
    if torch.device(device).type == "cpu" and resolved == "16-mixed":
        return "bf16-mixed"
    return resolved


def _inference_autocast(
    device: Union[str, torch.device], precision: InferencePrecision
):
    if precision == "32-true":
        return nullcontext()
    dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def _forward_predict(
    model: nn.Module,
    intensity: torch.Tensor,
    positions: torch.Tensor,
    probe: torch.Tensor,
    input_scale: torch.Tensor,
    *,
    device: Union[str, torch.device],
    precision: InferencePrecision,
) -> torch.Tensor:
    with _inference_autocast(device, precision):
        if isinstance(model, nn.DataParallel):
            return model((intensity, positions, probe, input_scale))
        return model.forward_predict(intensity, positions, probe, input_scale)


def _synchronize_cuda_for_timing(device: Union[str, torch.device]) -> None:
    """Drain queued CUDA work on ``device`` before sampling wall time."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


#Currently adapted for multi-channel, but finding some problems in terms of reassembly
#Adapted from Oliver's tf_helper/shift_and_Sum
def reassemble_single_channel(im_tensor: torch.Tensor,
                              com: torch.Tensor,
                              max_offset: float,
                              global_coords: torch.Tensor,
                              data_config: DataConfig,
                              middle: int = 10) -> torch.Tensor:
    '''
    Takes a batch stack of object tensors, and then lays them out on a "canvas" and averages
    them based on their true positions.

    This is essentially the core behind reconstructing a full object from the independent "smaller" object
    patch predictions (through the neural net)

    Inputs
    -------
    im_tensor: torch.Tensor (B,C,H,W), output from neural net
    global_coords: torch.Tensor(B,C,1,2), global coordinates from ptycho scan
    com: torch.Tensor(2), center of mass coordinates
    data_config: DataConfig, contains all relevant data parameters used for NN
    middle: int, We want to exclude outer sections due to Nyquist sampling (outer border has bad signal)
            This parameter controls how much of the center section we grab

    Returns
    -------
    reassembled_image: sum of reassembled images on canvas
    reassembled_ones: sum of ones with same shape as reassmbled_images for norm
    '''
    
    N = data_config.N
    M = middle #Easier to read code
    

    #Squeezing global_coords
    #Need to modify shape to work with Translation properly, need singleton second dim
    global_coords = global_coords.float().squeeze()

    #Select the central part of the object tensor
    im_tensor = im_tensor[:,:,
                        N // 2 - M // 2:N // 2 + M // 2,
                        N // 2 - M // 2:N // 2 + M // 2].squeeze()
    #Get dynamic padding value
    dynamic_pad = torch.ceil(max_offset).int()

    #Subtracting COM to get relative coordinates (to center)
    if data_config.C == 1:
        #Recenter
        adjusted_offsets = global_coords - com[None,:]
        #Reshape offset
        adjusted_offsets = adjusted_offsets[:,None,:]
    elif data_config.C > 1: #Unused at the moment
        B, C, H, W = im_tensor.shape
        #Recenter
        adjusted_offsets = global_coords - com[None, None,:]
        #Reshape offset
        adjusted_offsets = adjusted_offsets.reshape(B*C,2).unsqueeze(1)
        #Reshape image tensor itself
        im_tensor = im_tensor.reshape(B*C, H, W)

    print(f'Padding Size: {dynamic_pad}')

    #Pad all tensors to the same size. Introduce a ones tensor for normalization
    # padded_images = hh.pad_patches(im_tensor, dynamic_pad)
        # padded_ones = hh.pad_patches(torch.ones_like(im_tensor),dynamic_pad)
    padded_images = F.pad(im_tensor, (dynamic_pad, dynamic_pad, dynamic_pad, dynamic_pad), "constant", 0)
    padded_ones = F.pad(torch.ones_like(im_tensor),  (dynamic_pad, dynamic_pad, dynamic_pad, dynamic_pad), "constant", 0)

    #Translate
    translated_images_4d = hh.Translation(padded_images, adjusted_offsets, jitter_amt=0.0)
    ones_4d = hh.Translation(padded_ones, adjusted_offsets, jitter_amt=0.0)

    if data_config.C == 1:
        #Squeeze channel dimension
        translated_images = translated_images_4d.squeeze(1)
        translated_ones = ones_4d.squeeze(1)
    else:
        _, _, padded_H, padded_W = translated_images_4d.shape
        translated_images = translated_images_4d.reshape(B,C,padded_H,padded_W)
        translated_ones = ones_4d.reshape(B,C,padded_H,padded_W)

    #Get output
    reassembled_image = torch.sum(translated_images, dim=0)
    reassembled_ones = torch.sum(translated_ones, dim = 0)

    return reassembled_image, reassembled_ones

def reassemble_multi_channel(im_tensor: torch.Tensor,
                              com: torch.Tensor,
                              max_offset: float,
                              relative_coords: torch.Tensor,
                              coord_centers: torch.Tensor,
                              data_config: DataConfig,
                              model_config: ModelConfig,
                              middle: int = 10) -> torch.Tensor:
    '''
    Takes a batch stack of object tensors, and then lays them out on a "canvas" and averages
    them based on their true positions.

    Specialized for mult-channel situation. We will assemble the solution region first, and then from there reassemble the canvas

    Inputs
    -------
    im_tensor: torch.Tensor (B,C,H,W), output from neural net
    relative_coords: torch.Tensor(B,C,1,2), relative coordinates for each sol'n region
    coord_centers: torch.Tensor(B,1,1,2), com coordinates for each sol'n region
    com: torch.Tensor(2), center of mass coordinates for global ptycho scan
    data_config: DataConfig, contains all relevant data parameters used for NN
    middle: int, We want to exclude outer sections due to Nyquist sampling (outer border has bad signal)
            This parameter controls how much of the center section we grab

    Returns
    -------
    reassembled_image: sum of reassembled images on canvas
    reassembled_ones: sum of ones with same shape as reassmbled_images for norm
    '''
    M = middle #Easier to read code

    #Perform the initial solution patch assembly and get the ones vector as well
    soln_patches, ones_mask, padded_size = hh.reassemble_patches_position_real(im_tensor,
                                                                  relative_coords,
                                                                  data_config, model_config,
                                                                  agg=True)

    #Select the central part of the object tensor
    soln_patches = soln_patches[:,
                        padded_size // 2 - M // 2:padded_size // 2 + M // 2,
                        padded_size // 2 - M // 2:padded_size // 2 + M // 2]
    
    ones_mask = ones_mask.float() #Convert from bool to float
    ones_mask = ones_mask[:,
                          padded_size // 2 - M // 2:padded_size // 2 + M // 2,
                          padded_size // 2 - M // 2:padded_size // 2 + M // 2]
    
    #Get dynamic padding value
    dynamic_pad = torch.ceil(max_offset).int()

    #Subtracting COM to get relative coordinates (to center)
    B, _, _ = soln_patches.shape
    #Recenter
    adjusted_offsets = coord_centers - com[None, None,:]
    #Reshape offset
    adjusted_offsets = adjusted_offsets.reshape(B,2).unsqueeze(1)

    #Pad all tensors to the same size. Introduce a ones tensor for normalization
    # padded_images = hh.pad_patches(im_tensor, dynamic_pad)
        # padded_ones = hh.pad_patches(torch.ones_like(im_tensor),dynamic_pad)
    padded_images = F.pad(soln_patches, (dynamic_pad, dynamic_pad, dynamic_pad, dynamic_pad), "constant", 0)
    padded_ones = F.pad(ones_mask,  (dynamic_pad, dynamic_pad, dynamic_pad, dynamic_pad), "constant", 0)

    #Translate (this was originally positive, setting to negative for now)
    translated_images = hh.Translation(padded_images, adjusted_offsets, jitter_amt=0.0)
    translated_ones= hh.Translation(padded_ones, adjusted_offsets, jitter_amt=0.0)

    #Get output
    reassembled_image = torch.sum(translated_images, dim=0)
    reassembled_ones = torch.sum(translated_ones, dim = 0)

    return reassembled_image, reassembled_ones


def reconstruct_image(model: nn.Module,
                      ptycho_dset: PtychoDataset,
                      training_config = TrainingConfig,
                      data_config = DataConfig,
                      model_config = ModelConfig,
                      inference_config = InferenceConfig):
    '''
    Reconstructs an image given a model as well as ptycho dataset. Assumes one image per dataset at the moment.

    Inputs
    ---------
    model: This must be a PtychoPINN Lightning module from train.py, assumed to be on GPU
    ptycho_dset: This must be the custom PtychoDataset 
    '''
    n_files = ptycho_dset.n_files
    experiment_number = inference_config.experiment_number

    #Get dataset subset
    if n_files > 1:
        ptycho_subset = ptycho_dset.get_experiment_dataset(experiment_number)
    else:
        ptycho_subset = ptycho_dset
    device = training_config.device

    infer_loader = TensorDictDataLoader(ptycho_subset, batch_size = inference_config.batch_size,
                                    collate_fn = Collate(device = training_config.device))

    

    #Get center of mass and max difference
    global_coords = ptycho_subset.mmap_ptycho['coords_global'].squeeze()

    #Dynamic center of mass that's channel agnostic. A stored data_dict['com']
    #is never read: its only production writer (dataloader.py::memory_map_data)
    #unconditionally overwrites data_dict['com'] per file, so on a multi-file
    #dataset it holds a stale last-file centroid rather than this subset's own.
    center_of_mass = torch.mean(global_coords,
                                dim = tuple(range(global_coords.dim()-1)))

    adjusted_offsets_float = global_coords - center_of_mass
    max_abs_offset = torch.ceil(torch.max(torch.abs(adjusted_offsets_float))).int()

    #Moving com and offset to device
    center_of_mass = center_of_mass.to(device)
    max_abs_offset = max_abs_offset.to(device)

    #Initialize accumulation tensors
    reassembled_image = None
    reassembled_ones = None

    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            start = time.time()

            # Unpack and transfer to device efficiently
            batch_data = batch[0]
            x = batch_data['images'].to(device, non_blocking=True)
            positions = batch_data['coords_relative'].to(device, non_blocking=True)
            probe = batch[1].to(device, non_blocking=True)
            in_scale = batch_data['rms_scaling_constant'].to(device, non_blocking=True)
            batch_coords_global = batch_data['coords_global'].to(device, non_blocking=True)
            
            # Model inference
            batch_output = model.forward_predict(x, positions, probe, in_scale)
            
            # Reassembly
            if data_config.C == 1:
                reassembled_batch_image, reassembled_batch_ones = reassemble_single_channel(
                    batch_output,           # im_tensor
                    center_of_mass,         # com
                    max_abs_offset,         # max_offset
                    batch_coords_global,    # global_coords
                    data_config,            # data_config
                    inference_config.middle_trim  # middle
                )
            else:
                batch_relative_center = batch_data['coords_center'].to(device, non_blocking=True)
                reassembled_batch_image, reassembled_batch_ones = reassemble_multi_channel(
                    batch_output,           # im_tensor
                    center_of_mass,         # com
                    max_abs_offset,         # max_offset
                    positions,              # relative_coords
                    batch_relative_center,  # coord_centers
                    data_config,            # data_config
                    model_config,           # model_config
                    inference_config.middle_trim  # middle
                )
                # Clean up multi-channel specific tensor
                del batch_relative_center

            if reassembled_image is not None and reassembled_ones is not None:
                reassembled_image += reassembled_batch_image
                reassembled_ones += reassembled_batch_ones
            else:
                reassembled_image = reassembled_batch_image
                reassembled_ones = reassembled_batch_ones

            print(f'Batch {i} completed in {time.time()-start} seconds')

    # Free all batch tensors
    del x, positions, probe, in_scale, batch_coords_global
    del batch_output, reassembled_batch_image, reassembled_batch_ones, batch
    torch.cuda.empty_cache()
    gc.collect()

    if reassembled_image is None or reassembled_ones is None:
        raise ValueError("Inference loader yielded no reconstruction batches")
    return reassembled_image/(reassembled_ones), ptycho_subset

def profile_memory():
    """Print current GPU memory usage."""
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")

# --- VarPro Scaler ---

class VarProScaler:
    """
    Solves for global scaling s1, s2 and background offset c4.
    Memory efficient: only stores 4x4 matrix and 4x1 vector.
    """
    def __init__(self, device):
        self.device = device
        self.ATA = torch.zeros((3, 3), device=device, dtype=torch.float64)
        self.ATb = torch.zeros((3, 1), device=device, dtype=torch.float64)

        # For autograd
        self.X1X1 = 0.0
        self.X2X2 = 0.0
        self.X3X3 = 0.0
        self.X1X2 = 0.0
        self.X1X3 = 0.0
        self.X2X3 = 0.0
        self.X1I = 0.0
        self.X2I = 0.0
        self.X3I = 0.0
        self.II = 0.0  # For computing residual
        self.n_pixels = 0

    @torch.no_grad()
    def accumulate_batch(self, I_raw, Psi_a, Psi_b,
    terms = 3):
        """
        I_raw: [B, H, W] or [B, C, H, W]
        Psi_a/Psi_b: detector waves matching I_raw, optionally with an
            additional incoherent-mode axis immediately before H, W.
        """
        X1 = torch.abs(Psi_a)**2
        X2 = torch.abs(Psi_b)**2
        X3 = 2 * torch.real(Psi_a * torch.conj(Psi_b))
        if X1.ndim == I_raw.ndim + 1:
            X1 = X1.sum(dim=-3)
            X2 = X2.sum(dim=-3)
            X3 = X3.sum(dim=-3)
        elif X1.ndim != I_raw.ndim:
            raise ValueError(
                "Psi_a/Psi_b must match I_raw dimensions or add one mode axis"
            )
        self.accumulate_batch_from_basis(I_raw, X1, X2, X3, terms=terms)

    @torch.no_grad()
    def accumulate_batch_from_basis(self, I_raw, X1, X2, X3, terms=3):
        """
        Accumulate from pre-computed mode-summed basis images.
        Used for incoherent multi-mode probes where:
            X1 = sum_p |F[p_p * a_tilde]|^2
            X2 = sum_p |F[j * p_p * b_tilde]|^2
            X3 = sum_p 2*Re(F[p_p * a_tilde] * conj(F[j * p_p * b_tilde]))

        I_raw: [B, H, W] or [B, C, H, W]
        X1, X2, X3: same shape as I_raw
        """
        intensity64 = I_raw.to(torch.float64)
        bases = [basis.to(torch.float64) for basis in (X1, X2, X3)]

        for i in range(terms):
            self.ATb[i] += torch.sum(bases[i] * intensity64).item()
            for j in range(i, terms):
                val = torch.sum(bases[i] * bases[j]).item()
                self.ATA[i, j] += val
                if i != j:
                    self.ATA[j, i] += val

        X1_64, X2_64, X3_64 = bases
        self.X1X1 += torch.sum(X1_64 * X1_64).item()
        self.X2X2 += torch.sum(X2_64 * X2_64).item()
        self.X3X3 += torch.sum(X3_64 * X3_64).item()
        self.X1X2 += torch.sum(X1_64 * X2_64).item()
        self.X1X3 += torch.sum(X1_64 * X3_64).item()
        self.X2X3 += torch.sum(X2_64 * X3_64).item()
        self.X1I += torch.sum(X1_64 * intensity64).item()
        self.X2I += torch.sum(X2_64 * intensity64).item()
        self.X3I += torch.sum(X3_64 * intensity64).item()
        self.II += torch.sum(intensity64 * intensity64).item()
        self.n_pixels += I_raw.numel()

    def swap_channels(self):
        """Transform accumulated statistics to correct for real/imag channel swap.

        Under channel swap, the basis images transform as:
        X1 <-> X2, X3 -> -X3. This applies the corresponding
        transformation T*ATA*T and T*ATb where T swaps indices 0,1
        and negates index 2.
        """
        # Transform ATA
        new_ATA = self.ATA.clone()
        new_ATA[0, 0] = self.ATA[1, 1]
        new_ATA[1, 1] = self.ATA[0, 0]
        # [0,1] and [1,0] unchanged
        new_ATA[0, 2] = -self.ATA[1, 2]
        new_ATA[2, 0] = -self.ATA[2, 1]
        new_ATA[1, 2] = -self.ATA[0, 2]
        new_ATA[2, 1] = -self.ATA[2, 0]
        self.ATA = new_ATA

        # Transform ATb
        new_ATb = self.ATb.clone()
        new_ATb[0] = self.ATb[1]
        new_ATb[1] = self.ATb[0]
        new_ATb[2] = -self.ATb[2]
        self.ATb = new_ATb

        # Transform autograd statistics
        X1X1_old, X1X3_old, X1I_old = self.X1X1, self.X1X3, self.X1I
        self.X1X1 = self.X2X2
        self.X2X2 = X1X1_old
        self.X1X3 = -self.X2X3
        self.X2X3 = -X1X3_old
        self.X1I = self.X2I
        self.X2I = X1I_old
        self.X3I = -self.X3I

    def get_condition_number(self):
        """Compute condition number of ATA matrix"""
        eigenvalues = torch.linalg.eigvalsh(self.ATA)
        cond_num = eigenvalues.max() / (eigenvalues.min() + 1e-15)
        return cond_num.item()

    def get_correlation_matrix(self):
        """Get correlation matrix to check channel correlation"""
        # Normalize to correlation matrix
        diag = torch.sqrt(torch.diag(self.ATA))
        corr = self.ATA / (diag.unsqueeze(1) * diag.unsqueeze(0) + 1e-15)
        return corr

    def sufficient_statistics(self) -> VarProSufficientStatistics:
        """Return a detached snapshot of all dataset-level fit evidence."""
        return VarProSufficientStatistics(
            ATA=self.ATA,
            ATb=self.ATb,
            sum_i2=self.II,
            n_pixels=self.n_pixels,
        )

    def solve(self, verbose = True):
        """Solves the system and returns (s1, s2, background)"""
        # Solve linear system c = (ATA^-1) * ATb
        # Add small epsilon to diagonal for numerical stability

        if verbose:
            cond_num = self.get_condition_number()
            corr = self.get_correlation_matrix()
            print(f"Condition number: {cond_num:.2e}")
            print(f"Correlation matrix:\n{corr}")

            if cond_num > 1e10:
                print("WARNING: Matrix is ill-conditioned!")

        reg = torch.eye(3, device=self.device) * (self.ATA.max() * 1e-9)
        c = torch.linalg.solve(self.ATA + reg, self.ATb).flatten() # [c1, c2, c3, c4]

        c1, c2, c3 = c[0], c[1], c[2]

        # Stage B: Eigen-projection for physical consistency (Object part)
        # Matrix C = [[c1, c3], [c3, c2]]
        disc = torch.sqrt((c1 - c2)**2 + 4 * c3**2)
        lambda_max = 0.5 * (c1 + c2 + disc)

        # Principal Eigenvector (v1, v2)
        v1 = torch.where(torch.abs(c3) > torch.abs(lambda_max - c1), c3, lambda_max - c2)
        v2 = torch.where(torch.abs(c3) > torch.abs(lambda_max - c1), lambda_max - c1, c3)
        norm = torch.sqrt(v1**2 + v2**2 + 1e-9)

        # Final Scale factors
        mag = torch.sqrt(torch.clamp(lambda_max, min=0))
        s1 = (v1 / norm) * mag
        s2 = (v2 / norm) * mag

        return s1.float(), s2.float()

    def solve_quadratic_direct(self, max_iter=50, verbose=True):
        """
        Directly solve for s1, s2 using Newton's method on the quadratic objective.
        Minimizes: ||s1^2*X1 + s2^2*X2 + s1*s2*X3 - I||^2
        """
        # Extract statistics from ATA and ATb
        X1X1 = self.ATA[0, 0].item()
        X2X2 = self.ATA[1, 1].item()
        X3X3 = self.ATA[2, 2].item()
        X1X2 = self.ATA[0, 1].item()
        X1X3 = self.ATA[0, 2].item()
        X2X3 = self.ATA[1, 2].item()
        X1I = self.ATb[0].item()
        X2I = self.ATb[1].item()
        X3I = self.ATb[2].item()

        # Initialize with positive square roots of diagonal solution
        s1 = torch.sqrt(torch.tensor(max(X1I / (X1X1 + 1e-10), 1e-6), device=self.device))
        s2 = torch.sqrt(torch.tensor(max(X2I / (X2X2 + 1e-10), 1e-6), device=self.device))


        if verbose:
            print(f"Initial guess: s1={s1:.4f}, s2={s2:.4f}")

        # Newton's method iterations
        for iter_num in range(max_iter):
            # Current objective value
            obj = (s1**4 * X1X1 + s2**4 * X2X2 + s1**2 * s2**2 * X3X3 +
                   2 * s1**2 * s2**2 * X1X2 + 2 * s1**3 * s2 * X1X3 + 2 * s1 * s2**3 * X2X3 -
                   2 * s1**2 * X1I - 2 * s2**2 * X2I - 2 * s1 * s2 * X3I)

            # Gradient components
            g1 = (4 * s1**3 * X1X1 + 2 * s1 * s2**2 * X3X3 + 4 * s1 * s2**2 * X1X2 +
                  6 * s1**2 * s2 * X1X3 + 2 * s2**3 * X2X3 -
                  4 * s1 * X1I - 2 * s2 * X3I)

            g2 = (4 * s2**3 * X2X2 + 2 * s1**2 * s2 * X3X3 + 4 * s1**2 * s2 * X1X2 +
                  2 * s1**3 * X1X3 + 6 * s1 * s2**2 * X2X3 -
                  4 * s2 * X2I - 2 * s1 * X3I)

            # Hessian components
            H11 = 12 * s1**2 * X1X1 + 2 * s2**2 * X3X3 + 4 * s2**2 * X1X2 + 12 * s1 * s2 * X1X3
            H22 = 12 * s2**2 * X2X2 + 2 * s1**2 * X3X3 + 4 * s1**2 * X1X2 + 12 * s1 * s2 * X2X3
            H12 = 4 * s1 * s2 * X3X3 + 8 * s1 * s2 * X1X2 + 6 * s1**2 * X1X3 + 6 * s2**2 * X2X3 - 2 * X3I

            # Check convergence
            grad_norm = torch.sqrt(g1**2 + g2**2)
            if verbose and iter_num % 5 == 0:
                print(f"Iter {iter_num}: obj={obj:.6e}, |grad|={grad_norm:.6e}, s1={s1:.4f}, s2={s2:.4f}")

            if grad_norm < 1e-8:
                if verbose:
                    print(f"Converged at iteration {iter_num}")
                break

            # Solve Newton system: H * delta = -g
            det = H11 * H22 - H12**2
            if torch.abs(det) < 1e-10:
                if verbose:
                    print(f"Warning: Near-singular Hessian at iteration {iter_num}, using gradient descent")
                # Fall back to gradient descent
                alpha = 0.01 / (grad_norm + 1e-10)
                delta_s1 = -alpha * g1
                delta_s2 = -alpha * g2
            else:
                delta_s1 = (-g1 * H22 + g2 * H12) / det
                delta_s2 = (g1 * H12 - g2 * H11) / det

            # Line search with positivity constraint
            alpha = 1.0
            for _ in range(10):
                s1_new = s1 + alpha * delta_s1
                s2_new = s2 + alpha * delta_s2

                if s1_new > 0 and s2_new > 0:
                    # Check if objective decreases
                    obj_new = (s1_new**4 * X1X1 + s2_new**4 * X2X2 + s1_new**2 * s2_new**2 * X3X3 +
                              2 * s1_new**2 * s2_new**2 * X1X2 + 2 * s1_new**3 * s2_new * X1X3 +
                              2 * s1_new * s2_new**3 * X2X3 - 2 * s1_new**2 * X1I -
                              2 * s2_new**2 * X2I - 2 * s1_new * s2_new * X3I)

                    if obj_new < obj:
                        break

                alpha *= 0.5
                if alpha < 1e-6:
                    if verbose:
                        print(f"Line search failed at iteration {iter_num}")
                    break

            s1 = s1 + alpha * delta_s1
            s2 = s2 + alpha * delta_s2

            # Ensure positivity
            s1 = torch.clamp(s1, min=1e-6)
            s2 = torch.clamp(s2, min=1e-6)

        return s1.float(), s2.float()

    def solve_autograd(self, max_iter=100, lr=0.1, verbose=True):
        """
        Solve for s1, s2 using PyTorch autograd with accumulated statistics.
        """
        # Convert statistics to tensors
        X1X1 = torch.tensor(self.X1X1, device=self.device, dtype=torch.float64)
        X2X2 = torch.tensor(self.X2X2, device=self.device, dtype=torch.float64)
        X3X3 = torch.tensor(self.X3X3, device=self.device, dtype=torch.float64)
        X1X2 = torch.tensor(self.X1X2, device=self.device, dtype=torch.float64)
        X1X3 = torch.tensor(self.X1X3, device=self.device, dtype=torch.float64)
        X2X3 = torch.tensor(self.X2X3, device=self.device, dtype=torch.float64)
        X1I = torch.tensor(self.X1I, device=self.device, dtype=torch.float64)
        X2I = torch.tensor(self.X2I, device=self.device, dtype=torch.float64)
        X3I = torch.tensor(self.X3I, device=self.device, dtype=torch.float64)
        II = torch.tensor(self.II, device=self.device, dtype=torch.float64)

        # Initialize parameters
        s1_init = torch.sqrt(torch.clamp(X1I / (X1X1 + 1e-10), min=1e-6))
        s2_init = torch.sqrt(torch.clamp(X2I / (X2X2 + 1e-10), min=1e-6))

        s1 = torch.nn.Parameter(s1_init)
        s2 = torch.nn.Parameter(s2_init)

        optimizer = torch.optim.Adam([s1, s2], lr=lr)

        if verbose:
            print(f"Initial: s1={s1.item():.4f}, s2={s2.item():.4f}")
            print(f"Total pixels accumulated: {self.n_pixels}")

        for iter_num in range(max_iter):
            optimizer.zero_grad()

            loss = (s1**4 * X1X1 +
                   s2**4 * X2X2 +
                   (s1*s2)**2 * X3X3 +
                   2 * s1**2 * s2**2 * X1X2 +
                   2 * s1**3 * s2 * X1X3 +
                   2 * s1 * s2**3 * X2X3 -
                   2 * s1**2 * X1I -
                   2 * s2**2 * X2I -
                   2 * s1 * s2 * X3I +
                   II)

            # Normalize by number of pixels for scale-invariant loss
            loss = loss / self.n_pixels

            loss.backward()
            optimizer.step()

            # Ensure positivity
            with torch.no_grad():
                s1.clamp_(min=1e-6)
                s2.clamp_(min=1e-6)

            if verbose and iter_num % 20 == 0:
                grad_norm = torch.sqrt(s1.grad**2 + s2.grad**2)
                print(f"Iter {iter_num}: loss={loss.item():.6e}, |grad|={grad_norm.item():.6e}, "
                      f"s1={s1.item():.4f}, s2={s2.item():.4f}")

            # Check convergence
            if s1.grad is not None and s2.grad is not None:
                grad_norm = torch.sqrt(s1.grad**2 + s2.grad**2)
                if grad_norm < 1e-8:
                    if verbose:
                        print(f"Converged at iteration {iter_num}")
                    break

        return s1.detach().float(), s2.detach().float()

    def solve_lbfgs(self, max_iter=50, verbose=True):
        """
        Solve using L-BFGS - often faster convergence for quadratic problems.
        """
        # Convert statistics to tensors
        X1X1 = torch.tensor(self.X1X1, device=self.device, dtype=torch.float64)
        X2X2 = torch.tensor(self.X2X2, device=self.device, dtype=torch.float64)
        X3X3 = torch.tensor(self.X3X3, device=self.device, dtype=torch.float64)
        X1X2 = torch.tensor(self.X1X2, device=self.device, dtype=torch.float64)
        X1X3 = torch.tensor(self.X1X3, device=self.device, dtype=torch.float64)
        X2X3 = torch.tensor(self.X2X3, device=self.device, dtype=torch.float64)
        X1I = torch.tensor(self.X1I, device=self.device, dtype=torch.float64)
        X2I = torch.tensor(self.X2I, device=self.device, dtype=torch.float64)
        X3I = torch.tensor(self.X3I, device=self.device, dtype=torch.float64)
        II = torch.tensor(self.II, device=self.device, dtype=torch.float64)

        # Initialize
        s1_init = torch.sqrt(torch.clamp(X1I / (X1X1 + 1e-10), min=1e-6))
        s2_init = torch.sqrt(torch.clamp(X2I / (X2X2 + 1e-10), min=1e-6))

        s1 = torch.nn.Parameter(s1_init)
        s2 = torch.nn.Parameter(s2_init)

        s1_pos, s2_pos = s1, s2

        optimizer = torch.optim.LBFGS([s1, s2], max_iter=max_iter, line_search_fn='strong_wolfe')

        if verbose:
            print(f"Initial: s1={s1.item():.4f}, s2={s2.item():.4f}")

        def closure():
            optimizer.zero_grad()

            loss = (s1_pos**4 * X1X1 +
                   s2_pos**4 * X2X2 +
                   (s1_pos*s2_pos)**2 * X3X3 +
                   2 * s1_pos**2 * s2_pos**2 * X1X2 +
                   2 * s1_pos**3 * s2_pos * X1X3 +
                   2 * s1_pos * s2_pos**3 * X2X3 -
                   2 * s1_pos**2 * X1I -
                   2 * s2_pos**2 * X2I -
                   2 * s1_pos * s2_pos * X3I +
                   II) / self.n_pixels

            loss.backward()
            return loss

        # Run optimization
        optimizer.step(closure)

        print(f"uncorrected scalars: {s1, s2}")
        if s1 < 0:
            s1_final, s2_final = -s1, -s2  # flip both, convention: s1 > 0
        else:
            s1_final, s2_final = s1, s2

        if verbose:
            final_loss = closure()
            print(f"Final: s1={s1_final.item():.4f}, s2={s2_final.item():.4f}, loss={final_loss.item():.6e}")

        return s1_final.detach().float(), s2_final.detach().float()


# --- Barycentric Interpolation ---


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


class PtychoDataParallelWrapper(nn.Module):
    """Wrapper to make ptychography models compatible with nn.DataParallel."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, inputs):
        x, positions, probe, in_scale = inputs
        return self.model.forward_predict(x, positions, probe, in_scale)


def setup_multi_gpu_model(model: nn.Module, 
                         gpu_ids: Optional[List[int]] = None) -> Tuple[nn.Module, List[int], torch.device]:
    """
    Setup model for multi-GPU using DataParallel.
    
    Returns:
        (model, gpu_ids_used, primary_device)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    total_gpus = torch.cuda.device_count()
    
    if gpu_ids is None:
        gpu_ids = list(range(total_gpus))
    
    if len(gpu_ids) == 1:
        primary_device = torch.device(f'cuda:{gpu_ids[0]}')
        model = model.to(primary_device)
        return model, gpu_ids, primary_device
    
    primary_device = torch.device(f'cuda:{gpu_ids[0]}')
    model = model.to(primary_device)
    
    wrapped_model = PtychoDataParallelWrapper(model)
    parallel_model = nn.DataParallel(wrapped_model, device_ids=gpu_ids)
    
    return parallel_model, gpu_ids, primary_device


def compute_varpro_basis(probe: torch.Tensor,
                          a_tilde: torch.Tensor,
                          b_tilde: torch.Tensor,
                          scale: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-mode VarPro exit-wave FFTs (Psi_a, Psi_b) and mode-summed basis
    images (X1, X2, X3) consumed by ``VarProScaler``.

    Uses ``norm='ortho'`` (energy-preserving / Parseval-exact), matching the
    convention used everywhere else this pattern appears -- this file's own
    ``detect_swap_probe_reference`` and ``ptycho_torch/model.py``'s
    ``Psi_a``/``Psi_b`` FFTs.

    Args:
        probe: (B,C,P,H,W) complex probe modes.
        a_tilde, b_tilde: (B,C,H,W) real/imag decoder textures.
        scale: optional output scale folded into the exit waves EXACTLY like
            the training forward (``RectangularScaledDiffraction.forward``,
            model.py:1395/1403: ``exit_wave = scale * probe * texture``), so
            the basis is in the same count units the training loss compared
            against (VARPRO-SOLVE-UNITS-001). A (B,1,1,1) tensor is broadcast
            over probe modes via ``unsqueeze(2)`` (model.py:1393 convention);
            ``None`` keeps the historical unscaled basis byte-for-byte. A
            dataset with ``physics_scaling_constant == 1.0`` (``normalize=
            'None'``, dataloader.py:736) yields ``output_scale`` ~= 1, i.e.
            no count-unit correction.

    Returns:
        (Psi_a, Psi_b, X1, X2, X3): Psi_a/Psi_b are (B,C,P,H,W) complex
        per-mode exit-wave FFTs; X1, X2, X3 are (B,C,H,W) real, mode-summed
        VarPro basis images.
    """
    # Unsqueeze texture for broadcasting with probe modes: (B,C,H,W) -> (B,C,1,H,W)
    a_5d = a_tilde.unsqueeze(2)
    b_5d = b_tilde.unsqueeze(2)

    # Per-mode exit waves: (B,C,P,H,W)
    exit_a = probe * a_5d
    exit_b = 1j * probe * b_5d
    if scale is not None:
        if torch.is_tensor(scale) and scale.dim() == 4:
            scale = scale.unsqueeze(2)  # (B,1,1,1) -> (B,1,1,1,1) over modes
        exit_a = scale * exit_a
        exit_b = scale * exit_b

    Psi_a = torch.fft.fftshift(torch.fft.fft2(exit_a, norm='ortho'), dim=(-2, -1))
    Psi_b = torch.fft.fftshift(torch.fft.fft2(exit_b, norm='ortho'), dim=(-2, -1))

    # Mode-summed basis images for VarPro: (B,C,P,H,W) -> (B,C,H,W)
    X1 = torch.sum(torch.abs(Psi_a)**2, dim=2)
    X2 = torch.sum(torch.abs(Psi_b)**2, dim=2)
    X3 = torch.sum(2 * torch.real(Psi_a * torch.conj(Psi_b)), dim=2)

    return Psi_a, Psi_b, X1, X2, X3


def _configured_probe_mask(
    reference: torch.Tensor,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> torch.Tensor:
    """Resolve the same effective probe mask used by training."""
    return resolve_probe_mask_torch(
        data_config.N,
        probe_mask=getattr(model_config, "probe_mask", False),
        probe_mask_tensor=getattr(model_config, "probe_mask_tensor", None),
        probe_mask_sigma=float(getattr(model_config, "probe_mask_sigma", 1.0)),
        probe_mask_diameter=getattr(model_config, "probe_mask_diameter", None),
        dtype=reference.real.dtype if reference.is_complex() else reference.dtype,
        device=reference.device,
    )


def _apply_configured_probe_mask(
    probe: torch.Tensor,
    reference: torch.Tensor,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> torch.Tensor:
    """Apply the same effective probe-mask resolver used by training."""
    mask = _configured_probe_mask(reference, data_config, model_config)
    return probe * mask.view(1, 1, 1, data_config.N, data_config.N)


@dataclass(frozen=True)
class _PreparedCIVarProBatch:
    measured_intensity: torch.Tensor
    positions: torch.Tensor
    probe_physical: torch.Tensor
    input_scale: torch.Tensor
    texture_raw: torch.Tensor
    effective_mask: torch.Tensor
    effective_probe: torch.Tensor
    psi_a: torch.Tensor
    psi_b: torch.Tensor
    x1: torch.Tensor
    x2: torch.Tensor
    x3: torch.Tensor
    inference_time: float
    assembly_start: float


def _prepare_ci_varpro_batch(
    model: nn.Module,
    batch_data: Any,
    data_config: DataConfig,
    model_config: ModelConfig,
    *,
    device: torch.device,
    precision: InferencePrecision,
    channels_swapped: bool,
    collect_timing: bool = False,
) -> _PreparedCIVarProBatch:
    """Prepare one CI batch identically for fitting and count evaluation."""
    measured_intensity = batch_data["measured_intensity"].to(
        device, non_blocking=True
    )
    positions = batch_data["coords_relative"].to(device, non_blocking=True)
    probe_physical = batch_data["probe_physical"].to(device, non_blocking=True)
    input_scale = batch_data["rms_input_scale"].to(device, non_blocking=True)

    if collect_timing:
        _synchronize_cuda_for_timing(device)
        inference_start = time.time()
    else:
        inference_start = 0.0
    texture_raw = _forward_predict(
        model,
        measured_intensity,
        positions,
        probe_physical,
        input_scale,
        device=device,
        precision=precision,
    ).to(torch.complex64)
    if collect_timing:
        _synchronize_cuda_for_timing(device)
        inference_time = time.time() - inference_start
    else:
        inference_time = 0.0

    if channels_swapped:
        texture_raw = torch.complex(texture_raw.imag, texture_raw.real)
    effective_mask = _configured_probe_mask(
        measured_intensity,
        data_config,
        model_config,
    )
    effective_probe = (
        probe_physical
        * effective_mask.view(1, 1, 1, data_config.N, data_config.N)
    ).to(torch.complex64)

    if collect_timing:
        _synchronize_cuda_for_timing(device)
        assembly_start = time.time()
    else:
        assembly_start = 0.0
    psi_a, psi_b, x1, x2, x3 = compute_varpro_basis(
        effective_probe,
        texture_raw.real.float(),
        texture_raw.imag.float(),
    )
    return _PreparedCIVarProBatch(
        measured_intensity=measured_intensity,
        positions=positions,
        probe_physical=probe_physical,
        input_scale=input_scale,
        texture_raw=texture_raw,
        effective_mask=effective_mask,
        effective_probe=effective_probe,
        psi_a=psi_a,
        psi_b=psi_b,
        x1=x1,
        x2=x2,
        x3=x3,
        inference_time=inference_time,
        assembly_start=assembly_start,
    )


def apply_varpro_canvas_scaling(
    texture_canvas: torch.Tensor,
    scaler: VarProScaler,
    *,
    enabled: bool = True,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply VarPro real/imag scaling to a stitched canvas, or return identity."""
    if not enabled:
        one = torch.tensor(1.0, device=texture_canvas.device, dtype=torch.float32)
        return texture_canvas, one, one

    s1, s2 = scaler.solve_lbfgs(verbose=verbose)
    scaled_canvas = torch.complex(s1 * texture_canvas.real, s2 * texture_canvas.imag)
    return scaled_canvas, s1, s2


def evaluate_fitted_count_metrics(
    model: nn.Module,
    infer_loader: Any,
    data_config: DataConfig,
    model_config: ModelConfig,
    *,
    s1: Any,
    s2: Any,
    device: Union[str, torch.device],
    scale_profile: str,
    precision: Optional[InferencePrecision] = None,
    channels_swapped: bool = False,
    local_to_source_ids: Any = None,
) -> Union[FittedCountMetrics, NotApplicable]:
    """Stream a deterministic fitted count-space pass over every batch."""
    if scale_profile == LEGACY_SCALE_CONTRACT:
        return not_applicable()
    if scale_profile != CI_SCALE_CONTRACT:
        raise ValueError(f"Unsupported count-metric scale profile: {scale_profile!r}")

    device = torch.device(device)
    effective_precision = resolve_inference_precision_for_device(precision, device)
    s1_value = torch.as_tensor(s1, dtype=torch.float32, device=device).reshape(())
    s2_value = torch.as_tensor(s2, dtype=torch.float32, device=device).reshape(())
    squared_error_sum = torch.zeros((), dtype=torch.float64, device=device)
    measured_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    poisson_nll_sum = torch.zeros((), dtype=torch.float64, device=device)
    n_samples = 0
    n_pixels = 0
    effective_mask_digest = None
    sample_id_batches: list[torch.Tensor] = []
    source_id_map = None
    if local_to_source_ids is not None:
        source_id_map = torch.as_tensor(
            local_to_source_ids, dtype=torch.int64, device=device
        ).reshape(-1)

    with torch.no_grad():
        for batch in infer_loader:
            batch_data = batch[0]
            raw_sample_ids = batch_data.get("nn_indices")
            if raw_sample_ids is None:
                batch_size = int(batch_data["measured_intensity"].shape[0])
                channels = int(batch_data["measured_intensity"].shape[1])
                start = sum(item.numel() for item in sample_id_batches)
                batch_sample_ids = torch.arange(
                    start, start + batch_size * channels,
                    dtype=torch.int64, device=device,
                )
            else:
                batch_sample_ids = raw_sample_ids.to(
                    device=device, dtype=torch.int64
                ).reshape(-1)
            if source_id_map is not None:
                if batch_sample_ids.numel() and (
                    bool(torch.any(batch_sample_ids < 0))
                    or bool(torch.any(batch_sample_ids >= source_id_map.numel()))
                ):
                    raise ValueError("Count-metric local sample id is out of range")
                batch_sample_ids = source_id_map[batch_sample_ids]
            sample_id_batches.append(batch_sample_ids)
            prepared = _prepare_ci_varpro_batch(
                model,
                batch_data,
                data_config,
                model_config,
                device=device,
                precision=effective_precision,
                channels_swapped=channels_swapped,
            )
            measured_intensity = prepared.measured_intensity
            batch_mask_digest = array_digest(prepared.effective_mask)
            if effective_mask_digest is None:
                effective_mask_digest = batch_mask_digest
            elif effective_mask_digest != batch_mask_digest:
                raise ValueError("Effective probe mask changed across count batches")
            prediction = (
                s1_value.square() * prepared.x1.float()
                + s2_value.square() * prepared.x2.float()
                + (s1_value * s2_value) * prepared.x3.float()
            )
            measured64 = measured_intensity.to(torch.float64)
            prediction64 = prediction.to(torch.float64)
            residual = prediction64 - measured64
            squared_error_sum += residual.square().sum()
            measured_square_sum += measured64.square().sum()
            poisson_nll_sum += (
                prediction64
                - measured64 * torch.log(torch.clamp(prediction64, min=1e-8))
            ).sum()
            n_samples += int(measured_intensity.shape[0] * measured_intensity.shape[1])
            n_pixels += int(measured_intensity.numel())

    if n_pixels == 0:
        raise ValueError("Count-metric loader yielded no detector pixels")
    if float(measured_square_sum.item()) <= 0.0:
        raise ValueError("Measured intensity has zero count-space energy")
    assert effective_mask_digest is not None
    sample_ids = tuple(
        int(item)
        for item in torch.cat(sample_id_batches).detach().cpu().tolist()
    )
    if len(sample_ids) != n_samples:
        raise ValueError("Count-metric sample identity does not match loader samples")
    relative_l2 = torch.sqrt(squared_error_sum / measured_square_sum)
    return FittedCountMetrics(
        relative_l2_intensity_error=float(relative_l2.item()),
        mean_raw_poisson_nll=float((poisson_nll_sum / n_pixels).item()),
        n_samples=n_samples,
        n_pixels=n_pixels,
        effective_mask_digest=effective_mask_digest,
        sample_ids=sample_ids,
    )


def _scan_identity_evidence(
    source_dataset: Any,
    subset_dataset: Any,
    experiment_number: int,
) -> tuple[
    tuple[int, ...], tuple[int, ...], bool, tuple[int, ...], tuple[int, ...]
]:
    """Return participants, centers, center availability, eligible, and source IDs."""
    mmap = getattr(subset_dataset, "mmap_ptycho", None)
    valid_per_file = getattr(source_dataset, "valid_indices_per_file", None)
    source_per_file = getattr(source_dataset, "source_indices_per_file", None)
    if (
        mmap is None
        or "nn_indices" not in mmap.keys()
        or valid_per_file is None
        or source_per_file is None
    ):
        return (), (), False, (), ()
    if not 0 <= experiment_number < len(valid_per_file) or not 0 <= experiment_number < len(source_per_file):
        return (), (), False, (), ()
    used = torch.as_tensor(mmap["nn_indices"]).detach().cpu().reshape(-1)
    filtered = torch.as_tensor(valid_per_file[experiment_number]).detach().cpu().reshape(-1)
    source = torch.as_tensor(source_per_file[experiment_number]).detach().cpu().reshape(-1)
    grouping_enabled = getattr(source_dataset, "group_coords_enabled", None)
    if callable(grouping_enabled):
        grouped = bool(grouping_enabled())
    else:
        grouped = bool(
            getattr(getattr(source_dataset, "model_config", None), "object_big", False)
        )
    if not grouped:
        if used.numel() and (
            bool(torch.any(used < 0)) or bool(torch.any(used >= filtered.numel()))
        ):
            raise ValueError("Ungrouped scan ids are outside the filtered split")
        used = filtered[used.to(torch.int64)]
        centers = used
        center_identity_available = True
    else:
        required = {"center_scan_id", "center_scan_id_available"}
        if not required <= set(mmap.keys()):
            raise ValueError("Grouped scan identity requires persisted center fields")
        raw_centers = torch.as_tensor(mmap["center_scan_id"]).detach().cpu().reshape(-1)
        availability = torch.as_tensor(
            mmap["center_scan_id_available"]
        ).detach().cpu().to(torch.bool).reshape(-1)
        if raw_centers.shape != availability.shape:
            raise ValueError("Grouped center identity fields have incompatible shapes")
        if bool(torch.any((~availability) & (raw_centers != -1))):
            raise ValueError("Unavailable grouped center ids must use sentinel -1")
        if bool(torch.any(availability & (raw_centers < 0))):
            raise ValueError("Available grouped center ids must be nonnegative")
        center_identity_available = bool(torch.all(availability))
        centers = raw_centers if center_identity_available else raw_centers[:0]
    used_ids = tuple(int(item) for item in torch.unique(used, sorted=True).tolist())
    center_ids = tuple(
        int(item) for item in torch.unique(centers, sorted=True).tolist()
    )
    filtered_ids = tuple(
        int(item) for item in torch.unique(filtered, sorted=True).tolist()
    )
    source_ids = tuple(int(item) for item in torch.unique(source, sorted=True).tolist())
    if not set(filtered_ids).issubset(source_ids):
        raise ValueError("Filtered scan ids are outside the source split")
    if not set(center_ids).issubset(filtered_ids):
        raise ValueError("Used center scan ids are outside the eligible center split")
    if not set(used_ids).issubset(source_ids):
        raise ValueError("Participating scan ids are outside the source split")
    return (
        used_ids,
        center_ids,
        center_identity_available,
        filtered_ids,
        source_ids,
    )


def padded_canvas_size(middle_trim: int, max_offset_y: int, max_offset_x: int) -> Tuple[int, int]:
    """Compact-canvas (H, W) sized with one extra ``middle_trim`` of margin
    per dimension beyond the tight ``middle_trim + 2*max_offset`` bound.

    The tight bound drops the extreme-offset patch: with ``canvas_size ==
    middle_trim + 2*max_offset`` exactly, a patch centered at the true max
    offset fails ``VectorizedWeightedAccumulator``'s bounds check
    (``xmin_wh + patch_size + 1 < canvas_shape[1]``) by exactly one pixel --
    2/59 patches were silently dropped on the B4 report's data (Sec 4). The
    extra margin is symmetric around the canvas center used by
    ``reconstruct_image_barycentric``'s placement math, so per-patch relative
    offsets are unchanged; it only enlarges the zero-padded border.
    """
    return (
        middle_trim + 2 * max_offset_y + middle_trim,
        middle_trim + 2 * max_offset_x + middle_trim,
    )


def build_canvas_anchor(center_of_mass: torch.Tensor, canvas_size: Tuple[int, int]) -> Dict[str, Any]:
    """Describe the compact canvas's placement (Task B4a anchor disclosure).

    The canvas is anchored at the scan center of mass, not the object
    center -- consumers that separately load a truth object and crop it
    around ITS OWN center (rather than this anchor) incur a several-pixel
    frame offset that is devastating on fine texture (B4 report Sec 1c/4:
    0.33 corr / 0.28 amp MAE damage on otherwise-perfect input).

    Args:
        center_of_mass: (2,) tensor, column 0 = x, column 1 = y (dataloader.py
            convention).
        canvas_size: (H, W) of the canvas ``center_of_mass`` is anchored in.

    Returns:
        ``{"scan_com": Tensor(2,) on CPU, "canvas_shape": (H, W),
        "canvas_origin_offset": (dx, dy)}``, where ``canvas_origin_offset``
        is the (x, y) pixel offset from the canvas's own center to the scan
        center of mass.
    """
    com_cpu = center_of_mass.detach().cpu()
    has_xy = com_cpu.numel() >= 2
    dx = canvas_size[1] // 2 - float(com_cpu[0].item()) if has_xy else None
    dy = canvas_size[0] // 2 - float(com_cpu[1].item()) if has_xy else None
    return {
        "scan_com": com_cpu,
        "canvas_shape": canvas_size,
        "canvas_origin_offset": (dx, dy),
    }


def reconstruct_image_barycentric(model: nn.Module,
                     ptycho_dset: PtychoDataset,
                     training_config: TrainingConfig,
                     data_config: DataConfig,
                     model_config: ModelConfig,
                     inference_config: InferenceConfig,
                     gpu_ids: Optional[List[int]] = None,
                     use_mixed_precision: Optional[bool] = None,
                     verbose: bool = True,
                     swap_detection: str = 'None',
                     return_diagnostics: bool = False,
                     structured_diagnostics: bool = False,
                     precision: Optional[InferencePrecision] = None,
                     compute_count_metrics: bool = True,
                     ) -> ReassemblyReturn:
    """
    Multi-GPU ptychography reconstruction using probe-weighted barycentric
    coordinate assembly with VarPro scaling.

    Args:
        model: Neural network model
        ptycho_dset: Ptychography dataset
        training_config: Training configuration
        data_config: Data configuration
        model_config: Model configuration
        inference_config: Inference configuration
        gpu_ids: List of GPU IDs to use (None for single GPU on training_config.device)
        use_mixed_precision: Legacy precision alias. True selects FP16 mixed,
            False selects FP32, and None defers to ``precision``.
        verbose: Whether to print progress
        swap_detection: Method for detecting real/imag channel swap.
            'None' - no swap detection
            'mean' - compare |mean(real)| vs |mean(imag)| on assembled canvas
            'probe' - pass probe-only diffraction through autoencoder and check
                      which channel dominates (transparent object should be real-dominated)
        return_diagnostics: If True, return 4-tuple with VarPro diagnostics;
            if False (default), return backward-compatible 3-tuple.
        structured_diagnostics: Return schema-v1 dataset-level diagnostics
            instead of the positional legacy diagnostics list.
        precision: Explicit inference precision. Supported values are
            ``32-true``, ``16-mixed``, and ``bf16-mixed``.
        compute_count_metrics: When structured CI diagnostics are requested,
            run the fitted count-space evaluation. Defaults to True for
            backwards compatibility. Canonical runtimes may defer this pass
            until after production checkpoint reload.

    Returns:
        If return_diagnostics is False:
            (scaled_canvas, dataset_subset,
             [inference_time, assembly_time, canvas_anchor])
        If return_diagnostics is True:
            (scaled_canvas, dataset_subset,
             [inference_time, assembly_time, Psi_a, Psi_b, s1, s2, canvas_anchor],
             modified_scaled_canvas)

        If structured_diagnostics is True:
            (scaled_canvas, dataset_subset, ReassemblyDiagnostics,
             prescale_canvas)

        ``canvas_anchor`` (Task B4a, always the LAST stats-list element --
        index into it positionally from the front, not via negative
        indices, since this list may grow again) is a dict describing the
        compact canvas's placement: ``{"scan_com": Tensor(2,),
        "canvas_shape": (H, W), "canvas_origin_offset": (dx, dy)}``, where
        ``canvas_origin_offset`` is the (x, y) pixel offset from the
        canvas's own center to the scan center of mass -- consumers cropping
        a separately-loaded truth object must anchor at ``scan_com``, not
        the object's own center, to avoid the frame-offset error documented
        in the B4 report (Sec 1c/4).
    """

    requested_precision = resolve_inference_precision(
        use_mixed_precision, precision
    )

    # Setup model (single or multi-GPU)
    if gpu_ids is None or len(gpu_ids) <= 1:
        # Single GPU mode
        if gpu_ids and len(gpu_ids) == 1:
            device = torch.device(f'cuda:{gpu_ids[0]}')
        else:
            device = training_config.device
        model = model.to(device)
        primary_device = device
        if verbose:
            print(f"Using single GPU: {device}")
    else:
        # Multi-GPU mode
        model, gpu_ids, primary_device = setup_multi_gpu_model(model, gpu_ids)
        if verbose:
            print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    uses_cuda = (
        torch.device(primary_device).type == "cuda" and torch.cuda.is_available()
    )
    effective_precision = resolve_inference_precision_for_device(
        requested_precision,
        primary_device,
    )

    # Get dataset subset
    n_files = ptycho_dset.n_files
    experiment_number = inference_config.experiment_number

    if n_files > 1:
        ptycho_subset = ptycho_dset.get_experiment_dataset(experiment_number)
    else:
        ptycho_subset = ptycho_dset
    (used_scan_ids, used_center_scan_ids, center_identity_available,
     filtered_eligible_scan_ids, expected_scan_ids) = _scan_identity_evidence(
         ptycho_dset, ptycho_subset, experiment_number
     )

    # Pre-compute constants
    global_coords = ptycho_subset.mmap_ptycho['coords_global'].squeeze()

    # A stored data_dict['com'] is never read: its only production writer
    # (dataloader.py::memory_map_data) unconditionally overwrites
    # data_dict['com'] per file, so on a multi-file dataset it holds a stale
    # last-file centroid rather than this subset's own.
    center_of_mass = torch.mean(global_coords,
                              dim=tuple(range(global_coords.dim()-1)))

    center_of_mass = center_of_mass.to(primary_device)

    # Determine canvas size (asymmetric for rectangular scans). See
    # padded_canvas_size's docstring for why one extra middle_trim of margin
    # is added per dimension (report Sec 4 recommendation: prevents the
    # accumulator's bounds check from silently dropping the extreme-offset
    # patch). The extra margin is symmetric around the existing (unchanged)
    # canvas center, so per-patch relative offsets are untouched.
    adjusted_coords = global_coords - center_of_mass.cpu()
    print(f"global coords shape: {global_coords.shape}")
    max_offset_x = torch.ceil(torch.max(torch.abs(adjusted_coords[..., 0]))).int().item()
    max_offset_y = torch.ceil(torch.max(torch.abs(adjusted_coords[..., 1]))).int().item()
    canvas_size = padded_canvas_size(inference_config.middle_trim, max_offset_y, max_offset_x)

    if verbose:
        print(f"Canvas size: {canvas_size}, Max offsets: {max_offset_x, max_offset_y}")

    # Initialize canvas
    canvas = torch.zeros(canvas_size, device=primary_device, dtype=torch.complex64)
    canvas_weights = torch.zeros(canvas_size, device=primary_device, dtype=torch.float32)

    # Create dataloader
    infer_loader = TensorDictDataLoader(
        ptycho_subset,
        batch_size=inference_config.batch_size,
        num_workers=training_config.num_workers,
        collate_fn=Collate(device=primary_device),
        pin_memory = True,
        persistent_workers=training_config.num_workers > 0
    )

    #Other setup
    model.eval()
    total_inference_time = 0.0
    total_assembly_time = 0.0

    #Setting up scaler/accumulators
    scaler = VarProScaler(primary_device)
    accumulator = VectorizedWeightedAccumulator(canvas_size, primary_device)

    #Allow for uniform object weighting
    patch_weighting = getattr(inference_config, 'patch_weighting', 'probe')
    if patch_weighting not in {'uniform', 'probe'}:
        raise ValueError("patch_weighting must be 'uniform' or 'probe'")
    uniform_weighting = (patch_weighting == 'uniform')
    varpro_scaling = getattr(inference_config, 'varpro_scaling', True)

    # Profiles only govern rectangular_scaled. Amplitude mode retains its
    # historical unscaled-probe behavior even though DataConfig defaults to CI.
    physics_forward_mode = getattr(model_config, 'physics_forward_mode', 'amplitude') \
        if model_config is not None else 'amplitude'
    rectangular_scaled_mode = (physics_forward_mode == 'rectangular_scaled')
    ci_varpro_mode = False
    scale_profile = LEGACY_SCALE_CONTRACT
    if rectangular_scaled_mode:
        scale_contract = resolve_scale_contract(
            getattr(data_config, "scale_contract_version", None),
            getattr(data_config, "measurement_domain", None),
        )
        scale_profile = scale_contract.version
        ci_varpro_mode = scale_profile == CI_SCALE_CONTRACT

    effective_probe_mask = torch.ones(
        (data_config.N, data_config.N),
        dtype=torch.float32,
        device=primary_device,
    )

    # Save a reference probe for probe-based swap detection
    saved_probe_single = None

    # Guard the verbose "Scalars solved" print below against an unbound local
    # if infer_loader ever yields zero batches (output_scale is otherwise only
    # assigned inside the loop body).
    output_scale = None
    track_decoder_saturation = (
        getattr(model_config, "architecture", "cnn") == "cnn"
        and physics_forward_mode == "rectangular_scaled"
    )
    decoder_saturation_counts = torch.zeros(
        4, dtype=torch.int64, device=primary_device
    )
    decoder_value_count = 0

    #Actual loop
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            batch_start_time = time.time()

            # Prepare data
            batch_data = batch[0]
            batch_global_coords = batch_data['coords_global'].to(primary_device, non_blocking=True)
            if ci_varpro_mode:
                prepared = _prepare_ci_varpro_batch(
                    model,
                    batch_data,
                    data_config,
                    model_config,
                    device=torch.device(primary_device),
                    precision=effective_precision,
                    channels_swapped=False,
                    collect_timing=True,
                )
                I_raw = prepared.measured_intensity
                positions = prepared.positions
                probe = prepared.probe_physical
                in_scale = prepared.input_scale
                texture_raw = prepared.texture_raw
                effective_probe_mask = prepared.effective_mask
                effective_probe = prepared.effective_probe
                Psi_a, Psi_b = prepared.psi_a, prepared.psi_b
                X1, X2, X3 = prepared.x1, prepared.x2, prepared.x3
                inference_time = prepared.inference_time
                assembly_start = prepared.assembly_start
                output_scale = None
            else:
                positions = batch_data['coords_relative'].to(
                    primary_device, non_blocking=True
                )
                I_raw = batch_data['images'].to(primary_device, non_blocking=True)
                probe = batch[1].to(
                    primary_device, non_blocking=True
                )  # (B, C, P, H, W)
                in_scale = batch_data['rms_scaling_constant'].to(
                    primary_device, non_blocking=True
                )
                effective_probe = probe
                _synchronize_cuda_for_timing(primary_device)
                inference_start = time.time()
                texture_raw = _forward_predict(
                    model,
                    I_raw,
                    positions,
                    probe,
                    in_scale,
                    device=primary_device,
                    precision=effective_precision,
                ).to(torch.complex64)
                _synchronize_cuda_for_timing(primary_device)
                inference_time = time.time() - inference_start
                _synchronize_cuda_for_timing(primary_device)
                assembly_start = time.time()
                if rectangular_scaled_mode:
                    physics_scale = batch_data['physics_scaling_constant'].to(primary_device, non_blocking=True)
                    probe_scaling = batch[2].to(primary_device, non_blocking=True)
                    output_scale = torch.sqrt(1.0 / (probe_scaling ** 2 * physics_scale + 1e-9))
                    Psi_a, Psi_b, X1, X2, X3 = compute_varpro_basis(
                        effective_probe,
                        texture_raw.real,
                        texture_raw.imag,
                        scale=output_scale,
                    )
                else:  # amplitude: preserve the historical unscaled basis
                    output_scale = None
                    Psi_a, Psi_b, X1, X2, X3 = compute_varpro_basis(
                        effective_probe,
                        texture_raw.real,
                        texture_raw.imag,
                    )

            # Save uncropped probe from first batch for probe-based swap detection
            if saved_probe_single is None:
                saved_probe_single = effective_probe[0, 0, 0].clone()

            total_inference_time += inference_time
            a_tilde = texture_raw.real
            b_tilde = texture_raw.imag

            # VarPro always accumulates on the full detector frame. CI uses the
            # calibrated, masked physical probe directly; explicit legacy
            # rectangular mode retains its historical output-scale fold.
            scaler.accumulate_batch_from_basis(I_raw, X1, X2, X3)

            # Center crop (stitching only -- VarPro above uses the full frame)
            N = data_config.N
            middle = inference_config.middle_trim
            center_start = N // 2 - middle // 2
            center_end = N // 2 + middle // 2

            I_raw = I_raw[:,:,center_start:center_end, center_start:center_end]
            a_tilde = a_tilde[:, :, center_start:center_end, center_start:center_end]
            b_tilde = b_tilde[:, :, center_start:center_end, center_start:center_end]
            if track_decoder_saturation:
                real_tolerance = (1.2 - (-0.8)) * 1e-3
                imag_tolerance = (1.2 - (-1.2)) * 1e-3
                decoder_saturation_counts += torch.stack(
                    (
                        torch.count_nonzero(a_tilde <= -0.8 + real_tolerance),
                        torch.count_nonzero(a_tilde >= 1.2 - real_tolerance),
                        torch.count_nonzero(b_tilde <= -1.2 + imag_tolerance),
                        torch.count_nonzero(b_tilde >= 1.2 - imag_tolerance),
                    )
                )
                decoder_value_count += int(a_tilde.numel())

            # Also crop the probe to match (B, C, P, H, W)
            effective_probe = effective_probe[
                :, :, :, center_start:center_end, center_start:center_end
            ]

            # --- Weighted Stitching ---
            B,C,H,W= a_tilde.shape
            global_coords_2d = batch_global_coords.squeeze(2).view(B * C, 2)
            relative_positions = global_coords_2d - center_of_mass.unsqueeze(0)
            canvas_center = torch.tensor([canvas_size[1] // 2, canvas_size[0] // 2],
                                       device=primary_device, dtype=torch.float32)
            canvas_positions = relative_positions + canvas_center.unsqueeze(0)

            # Total probe intensity: sum |P_p|^2 over all incoherent modes
            probe_mag_sq = torch.sum(
                torch.abs(effective_probe[0, 0, :, :, :]) ** 2, dim=0
            )  # (P,H,W) -> (H,W)

            # Change texture_raw to complex
            O_tilde = torch.complex(a_tilde, b_tilde)
            O_tilde = O_tilde.view(B*C,middle,middle)

            # Canvas assembly
            accumulator.accumulate_batch(canvas, canvas_weights, O_tilde,
                                        canvas_positions, probe_mag_sq,
                                        patch_size = inference_config.middle_trim,
                                        uniform_weighting = uniform_weighting)

            _synchronize_cuda_for_timing(primary_device)
            assembly_time = time.time() - assembly_start
            total_assembly_time += assembly_time

            # Memory cleanup
            del I_raw, positions, probe, effective_probe, in_scale, batch_global_coords
            del texture_raw, canvas_positions

            if i % 5 == 0:
                if uses_cuda:
                    torch.cuda.empty_cache()
                gc.collect()
            # Logging
            if verbose:
                batch_time = time.time() - batch_start_time
                efficiency = inference_time / assembly_time if assembly_time > 0 else float('inf')
                print(f'Batch {i+1}/{len(infer_loader)}: {batch_time:.3f}s '
                      f'(inference: {inference_time:.3f}s, assembly: {assembly_time:.3f}s, '
                      f'efficiency: {efficiency:.1f}x)')

    # 2. Finalize texture canvas (before solving, to check for swap)
    texture_canvas = canvas / (canvas_weights + 1e-12)

    # 3. Detect channel swap
    if swap_detection == 'probe':
        channels_swapped = detect_swap_probe_reference(
            model, saved_probe_single, data_config, model_config,
            primary_device, verbose=verbose, precision=effective_precision)
    elif swap_detection == 'mean':
        real_mean = texture_canvas.real.mean().item()
        imag_mean = texture_canvas.imag.mean().item()
        channels_swapped = abs(imag_mean) > abs(real_mean)
        if verbose:
            print(f"Mean swap check: |real_mean|={abs(real_mean):.4f}, "
                  f"|imag_mean|={abs(imag_mean):.4f}")
    else:
        channels_swapped = False

    if channels_swapped and verbose:
        print("Channel swap detected — correcting accumulated statistics and swapping canvas channels...")

    if channels_swapped:
        texture_canvas = torch.complex(texture_canvas.imag, texture_canvas.real)
        scaler.swap_channels()

    # 4. Solve for constants (using corrected statistics if swapped)
    scaler_solve_time_start = time.time()
    scaled_canvas, s1, s2 = apply_varpro_canvas_scaling(
        texture_canvas,
        scaler,
        enabled=varpro_scaling,
        verbose=verbose,
    )
    scaler_solve_time_end = time.time() - scaler_solve_time_start

    if verbose:
        print(f"Scalars solved: S1 = {s1}, S2 = {s2} (effective output_scale = {output_scale})")
    if channels_swapped:
        print("(Solved after channel-swap correction)")

    if verbose:
        avg_inference_time = total_inference_time / len(infer_loader)
        avg_assembly_time = total_assembly_time / len(infer_loader)
        efficiency = avg_inference_time / avg_assembly_time if avg_assembly_time > 0 else float('inf')

        print("\nPerformance Summary:")
        print(f"  Average inference time per batch: {avg_inference_time:.3f}s")
        print(f"  Average assembly time per batch: {avg_assembly_time:.3f}s")
        print(f"  Parallel efficiency ratio: {efficiency:.1f}x")
        print(f"  Total reconstruction time: {total_inference_time + total_assembly_time:.2f}s")
        print(f"  Total constant solve time: {scaler_solve_time_end:.2f}s")

    # Final cleanup
    if uses_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    # Anchor disclosure (Task B4a, build_canvas_anchor's docstring). Recorded
    # here as an extra stats-list element (backward-compatible: existing
    # positional/front-indexed consumers of the stats list are unaffected;
    # do not read via negative indices).
    canvas_anchor = build_canvas_anchor(center_of_mass, canvas_size)
    (decoder_real_lower_saturated, decoder_real_upper_saturated,
     decoder_imag_lower_saturated, decoder_imag_upper_saturated) = (
        decoder_saturation_counts.detach().cpu().tolist()
    )
    decoder_real_saturated = (
        decoder_real_lower_saturated + decoder_real_upper_saturated
    )
    decoder_imag_saturated = (
        decoder_imag_lower_saturated + decoder_imag_upper_saturated
    )
    decoder_real_saturation_fraction = (
        decoder_real_saturated / decoder_value_count
        if track_decoder_saturation and decoder_value_count
        else None
    )
    decoder_imag_saturation_fraction = (
        decoder_imag_saturated / decoder_value_count
        if track_decoder_saturation and decoder_value_count
        else None
    )
    def saturation_fraction(count: int) -> Optional[float]:
        return count / decoder_value_count if track_decoder_saturation and decoder_value_count else None

    decoder_real_lower_saturation_fraction = saturation_fraction(decoder_real_lower_saturated)
    decoder_real_upper_saturation_fraction = saturation_fraction(decoder_real_upper_saturated)
    decoder_imag_lower_saturation_fraction = saturation_fraction(decoder_imag_lower_saturated)
    decoder_imag_upper_saturation_fraction = saturation_fraction(decoder_imag_upper_saturated)

    if structured_diagnostics:
        prescale_canvas = torch.complex(texture_canvas.real, texture_canvas.imag)
        if ci_varpro_mode:
            if compute_count_metrics:
                count_metrics = evaluate_fitted_count_metrics(
                    model,
                    infer_loader,
                    data_config,
                    model_config,
                    s1=s1,
                    s2=s2,
                    device=primary_device,
                    scale_profile=scale_profile,
                    precision=effective_precision,
                    channels_swapped=channels_swapped,
                )
            else:
                count_metrics = not_evaluated()
            diagnostics = ReassemblyDiagnostics.from_statistics(
                scaler.sufficient_statistics(),
                inference_time=total_inference_time,
                assembly_time=total_assembly_time,
                solve_time=scaler_solve_time_end,
                s1=s1,
                s2=s2,
                scale_profile=scale_profile,
                effective_probe_mask=effective_probe_mask,
                canvas_anchor=canvas_anchor,
                canvas_weights=canvas_weights,
                accepted_patches=accumulator.accepted_patches,
                total_patches=accumulator.total_patches,
                count_metrics=count_metrics,
                effective_precision=effective_precision,
                used_scan_ids=used_scan_ids,
                used_center_scan_ids=used_center_scan_ids,
                center_identity_available=center_identity_available,
                expected_scan_ids=expected_scan_ids,
                filtered_eligible_scan_ids=filtered_eligible_scan_ids,
                decoder_real_saturation_fraction=decoder_real_saturation_fraction,
                decoder_imag_saturation_fraction=decoder_imag_saturation_fraction,
                decoder_real_lower_saturation_fraction=decoder_real_lower_saturation_fraction,
                decoder_real_upper_saturation_fraction=decoder_real_upper_saturation_fraction,
                decoder_imag_lower_saturation_fraction=decoder_imag_lower_saturation_fraction,
                decoder_imag_upper_saturation_fraction=decoder_imag_upper_saturation_fraction,
            )
        else:
            count_metrics = not_applicable()
            diagnostics = ReassemblyDiagnostics.legacy_not_applicable(
                inference_time=total_inference_time,
                assembly_time=total_assembly_time,
                solve_time=scaler_solve_time_end,
                s1=s1,
                s2=s2,
                scale_profile=scale_profile,
                effective_probe_mask=effective_probe_mask,
                canvas_anchor=canvas_anchor,
                canvas_weights=canvas_weights,
                accepted_patches=accumulator.accepted_patches,
                total_patches=accumulator.total_patches,
                count_metrics=count_metrics,
                effective_precision=effective_precision,
                used_scan_ids=used_scan_ids,
                used_center_scan_ids=used_center_scan_ids,
                center_identity_available=center_identity_available,
                expected_scan_ids=expected_scan_ids,
                filtered_eligible_scan_ids=filtered_eligible_scan_ids,
                decoder_real_saturation_fraction=decoder_real_saturation_fraction,
                decoder_imag_saturation_fraction=decoder_imag_saturation_fraction,
                decoder_real_lower_saturation_fraction=decoder_real_lower_saturation_fraction,
                decoder_real_upper_saturation_fraction=decoder_real_upper_saturation_fraction,
                decoder_imag_lower_saturation_fraction=decoder_imag_lower_saturation_fraction,
                decoder_imag_upper_saturation_fraction=decoder_imag_upper_saturation_fraction,
            )
        return scaled_canvas, ptycho_subset, diagnostics, prescale_canvas

    if return_diagnostics:
        modified_scaled_canvas = texture_canvas.real + 1j * texture_canvas.imag
        # Psi_a/Psi_b are full-frame. They are physical-probe/unscaled for CI
        # and output-scale-folded for explicit legacy rectangular mode; only
        # s1/s2 (indices 4/5) are contract-stable.
        return (
            scaled_canvas, ptycho_subset,
            [total_inference_time, total_assembly_time, Psi_a, Psi_b, s1, s2, canvas_anchor],
            modified_scaled_canvas,
        )

    return scaled_canvas, ptycho_subset, [total_inference_time, total_assembly_time, canvas_anchor]


reconstruct_image_barycentric_weighted = reconstruct_image_barycentric


def detect_swap_probe_reference(model: nn.Module,
                                probe_single: torch.Tensor,
                                data_config: DataConfig,
                                model_config: ModelConfig,
                                device: torch.device,
                                verbose: bool = True,
                                precision: Optional[InferencePrecision] = None,
                                ) -> bool:
    """Detect channel swap by passing probe-only diffraction through the model.

    For a transparent object (O=1+0j), the detector intensity is |FFT(P)|^2.
    The model should produce output dominated by the real channel;
    if the imaginary channel dominates, the channels are swapped.

    Args:
        model: Neural network model (Lightning, PtychoPINN, or DataParallel wrapped)
        probe_single: Single complex probe, shape (N, N)
        data_config: DataConfig instance
        model_config: ModelConfig instance
        device: torch device
        verbose: Print diagnostic info

    Returns:
        bool: True if channels are swapped
    """
    C_in = model_config.C_model if model_config.object_big else 1

    with torch.no_grad():
        # |FFT(probe)|^2 — diffraction pattern of a transparent object
        I_ref = torch.abs(
            torch.fft.fftshift(torch.fft.fft2(probe_single, norm='ortho'))
        ) ** 2  # (N, N) real

        # Expand to model input shape: (1, C_in, N, N)
        I_ref = I_ref.unsqueeze(0).unsqueeze(0).expand(1, C_in, -1, -1).contiguous()

        # RMS-normalize (baked into input so in_scale=1)
        rms = torch.sqrt(torch.mean(I_ref ** 2)) + 1e-8
        I_ref_normed = I_ref / rms

        # Dummy inputs — forward_predict only uses x and in_scale through
        # the autoencoder; positions and probe are unused
        dummy_positions = torch.zeros(1, C_in, 1, 2, device=device)
        dummy_probe = probe_single.unsqueeze(0).unsqueeze(0).expand(1, C_in, -1, -1)
        in_scale = torch.ones(1, device=device)

        # Call model with the same precision contract as the inference loop.
        texture_raw = _forward_predict(
            model,
            I_ref_normed,
            dummy_positions,
            dummy_probe,
            in_scale,
            device=device,
            precision=resolve_inference_precision(None, precision),
        ).to(torch.complex64)

        # texture_raw is a complex tensor (B, C, H, W)
        ref_real = texture_raw.real
        ref_imag = texture_raw.imag

        real_energy = (ref_real ** 2).mean().item()
        imag_energy = (ref_imag ** 2).mean().item()

    swapped = imag_energy > real_energy

    if verbose:
        print(f"Probe reference swap check: real_energy={real_energy:.6f}, "
              f"imag_energy={imag_energy:.6f}")
        if swapped:
            print("  -> Imaginary channel dominates: channels ARE swapped")
        else:
            print("  -> Real channel dominates: channels NOT swapped")

    return swapped


def equalize_by_ratio(real, imag):
    """
    Scale one component by the ratio of standard deviations.
    Preserves the absolute scale of one component.
    """
    real_mean = real.abs().mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
    imag_mean = imag.abs().mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

    ratio = real_mean / (imag_mean + 1e-8)  # [B, C, 1, 1]

    normalized_imag = imag * ratio  # Broadcasting: [B, C, H, W] * [B, C, 1, 1]

    return real, normalized_imag
