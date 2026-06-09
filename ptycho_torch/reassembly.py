#Type helpers
from typing import Tuple, Optional, Union, Callable, Any, List
from ptycho_torch.dataloader import PtychoDataset, TensorDictDataLoader
import torch.nn as nn

#Pytorch-related
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import ptycho_torch.helper as hh
from ptycho_torch.dataloader import Collate

#Other useful libraries
import time
import gc

#Configurations
from ptycho_torch.config_params import ModelConfig, TrainingConfig, DataConfig, InferenceConfig

#Default casting
torch.set_default_dtype(torch.float32)
 
#Currently adapted for multi-channel, but finding some problems in terms of reassembly
#Adapted from Oliver's tf_helper/shift_and_Sum
def reassemble_single_channel(im_tensor: torch.Tensor,
                              com: float,
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
                              com: float,
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

    #Dynamic center of mass that's channel agnostic
    if 'com' in ptycho_subset.data_dict:
        center_of_mass = torch.mean(global_coords,
                                    dim = tuple(range(global_coords.dim()-1)))
    else:
        center_of_mass = ptycho_subset.data_dict['com']
  
    adjusted_offsets_float = global_coords - center_of_mass
    max_abs_offset = torch.ceil(torch.max(torch.abs(adjusted_offsets_float))).int()

    #Moving com and offset to device
    center_of_mass = center_of_mass.to(device)
    max_abs_offset = max_abs_offset.to(device)

    #Reassembly function
    if data_config.C == 1:
        reassemble_fn = reassemble_single_channel  # Your original function
    else:
        reassemble_fn = reassemble_multi_channel   # Your original function
    
    #Initialize accumulation tensors
    reassembled_image = None

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
                reassembled_batch_image, reassembled_batch_ones = reassemble_fn(
                    batch_output,           # im_tensor
                    center_of_mass,         # com
                    max_abs_offset,         # max_offset
                    batch_coords_global,    # global_coords
                    data_config,            # data_config
                    inference_config.middle_trim  # middle
                )
            else:
                batch_relative_center = batch_data['coords_center'].to(device, non_blocking=True)
                reassembled_batch_image, reassembled_batch_ones = reassemble_fn(
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

            if reassembled_image is not None:
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
        I_raw: [B, H, W]
        Psi_a: [B, H, W] Complex Basis (F[p * a_tilde])
        Psi_b: [B, H, W] Complex Basis (F[j * p * b_tilde])
        """
        # 1. Compute basis images
        X1 = torch.abs(Psi_a)**2
        X2 = torch.abs(Psi_b)**2
        X3 = 2 * torch.real(Psi_a * torch.conj(Psi_b))

        bases = [X1, X2, X3]

        # 2. Update ATA (4x4) and ATb (4x1)
        # We flatten pixels to treat them as observations
        for i in range(terms):
            # Fill ATb
            self.ATb[i] += torch.sum(bases[i] * I_raw).item()
            for j in range(i, terms):
                # Fill ATA (Symmetric)
                val = torch.sum(bases[i] * bases[j]).item()
                self.ATA[i, j] += val
                if i != j:
                    self.ATA[j, i] += val

        # For autograd

        self.X1X1 += torch.sum(X1 * X1).item()
        self.X2X2 += torch.sum(X2 * X2).item()
        self.X3X3 += torch.sum(X3 * X3).item()
        self.X1X2 += torch.sum(X1 * X2).item()
        self.X1X3 += torch.sum(X1 * X3).item()
        self.X2X3 += torch.sum(X2 * X3).item()
        self.X1I += torch.sum(X1 * I_raw).item()
        self.X2I += torch.sum(X2 * I_raw).item()
        self.X3I += torch.sum(X3 * I_raw).item()
        self.II += torch.sum(I_raw * I_raw).item()
        self.n_pixels += I_raw.numel()

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
        bases = [X1, X2, X3]

        for i in range(terms):
            self.ATb[i] += torch.sum(bases[i] * I_raw).item()
            for j in range(i, terms):
                val = torch.sum(bases[i] * bases[j]).item()
                self.ATA[i, j] += val
                if i != j:
                    self.ATA[j, i] += val

        self.X1X1 += torch.sum(X1 * X1).item()
        self.X2X2 += torch.sum(X2 * X2).item()
        self.X3X3 += torch.sum(X3 * X3).item()
        self.X1X2 += torch.sum(X1 * X2).item()
        self.X1X3 += torch.sum(X1 * X3).item()
        self.X2X3 += torch.sum(X2 * X3).item()
        self.X1I += torch.sum(X1 * I_raw).item()
        self.X2I += torch.sum(X2 * I_raw).item()
        self.X3I += torch.sum(X3 * I_raw).item()
        self.II += torch.sum(I_raw * I_raw).item()
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
            valid_idx = torch.where(valid_mask)[0]
            if len(valid_idx) == 0: return
            patches = patches[valid_idx]
            xmin_wh, ymin_wh = xmin_wh[valid_idx], ymin_wh[valid_idx]
            xmin_fr, ymin_fr = xmin_fr[valid_idx], ymin_fr[valid_idx]
            N = len(valid_idx)

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
        x, positions, probe, in_scale, probe_intensity = inputs
        return self.model.forward_predict(x, positions, probe, in_scale,
                                          probe_intensity=probe_intensity)


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


def reconstruct_image_barycentric(model: nn.Module,
                     ptycho_dset: PtychoDataset,
                     training_config: TrainingConfig,
                     data_config: DataConfig,
                     model_config: ModelConfig,
                     inference_config: InferenceConfig,
                     gpu_ids: Optional[List[int]] = None,
                     use_mixed_precision: bool = True,
                     verbose: bool = True,
                     swap_detection: str = 'None',
                     return_diagnostics: bool = False) -> Tuple[torch.Tensor, Any]:
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
        use_mixed_precision: Whether to use FP16 mixed precision
        verbose: Whether to print progress
        swap_detection: Method for detecting real/imag channel swap.
            'None' - no swap detection
            'mean' - compare |mean(real)| vs |mean(imag)| on assembled canvas
            'probe' - pass probe-only diffraction through autoencoder and check
                      which channel dominates (transparent object should be real-dominated)
        return_diagnostics: If True, return 4-tuple with VarPro diagnostics;
            if False (default), return backward-compatible 3-tuple.

    Returns:
        If return_diagnostics is False:
            (scaled_canvas, dataset_subset, [inference_time, assembly_time])
        If return_diagnostics is True:
            (scaled_canvas, dataset_subset,
             [inference_time, assembly_time, Psi_a, Psi_b, s1, s2],
             modified_scaled_canvas)
    """

    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

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

    # Get dataset subset
    n_files = ptycho_dset.n_files
    experiment_number = inference_config.experiment_number

    if n_files > 1:
        ptycho_subset = ptycho_dset.get_experiment_dataset(experiment_number)
    else:
        ptycho_subset = ptycho_dset

    # Pre-compute constants
    global_coords = ptycho_subset.mmap_ptycho['coords_global'].squeeze()

    if 'com' in ptycho_subset.data_dict:
        center_of_mass = torch.mean(global_coords,
                                  dim=tuple(range(global_coords.dim()-1)))
    else:
        center_of_mass = ptycho_subset.data_dict['com']

    center_of_mass = center_of_mass.to(primary_device)

    # Determine canvas size (asymmetric for rectangular scans)
    adjusted_coords = global_coords - center_of_mass.cpu()
    print(f"global coords shape: {global_coords.shape}")
    max_offset_x = torch.ceil(torch.max(torch.abs(adjusted_coords[..., 0]))).int().item()
    max_offset_y = torch.ceil(torch.max(torch.abs(adjusted_coords[..., 1]))).int().item()
    N = data_config.N
    canvas_size = (N + 2 * max_offset_y,
                   N + 2 * max_offset_x)

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
        persistent_workers=True
    )

    #Other setup
    model.eval()
    total_inference_time = 0
    total_assembly_time = 0

    #Setting up scaler/accumulators
    scaler = VarProScaler(primary_device)
    accumulator = VectorizedWeightedAccumulator(canvas_size, primary_device)

    #Allow for uniform object weighting
    patch_weighting = getattr(inference_config, 'patch_weighting', 'probe')
    uniform_weighting = (patch_weighting == 'uniform')

    # Save a reference probe for probe-based swap detection
    saved_probe_single = None

    #Actual loop
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            batch_start_time = time.time()

            # Prepare data
            batch_data = batch[0]
            I_raw = batch_data['images'].to(primary_device, non_blocking=True)
            positions = batch_data['coords_relative'].to(primary_device, non_blocking=True)
            probe = batch[1].to(primary_device, non_blocking=True)  # (B, C, P, H, W)
            in_scale = batch_data['rms_scaling_constant'].to(primary_device, non_blocking=True)
            batch_global_coords = batch_data['coords_global'].to(primary_device, non_blocking=True)
            probe_intensity = batch[3].to(primary_device, non_blocking=True) if len(batch) > 3 else None

            # Save uncropped probe from first batch for probe-based swap detection
            if saved_probe_single is None:
                saved_probe_single = probe[0, 0, 0].clone()  # (N, N) complex, first mode

            # Model inference
            inference_start = time.time()

            if isinstance(model, nn.DataParallel):
                inputs = (I_raw, positions, probe, in_scale, probe_intensity)
                texture_raw = model(inputs)
            else:
                texture_raw = model.forward_predict(I_raw, positions, probe, in_scale,
                                                    probe_intensity=probe_intensity)

            torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            # Getting the real and imaginary parts
            a_tilde = texture_raw[:,:,:,:,0]
            b_tilde = texture_raw[:,:,:,:,1]

            # --- VarPro Accumulation (incoherent multi-mode) ---
            assembly_start = time.time()
            B, C, H, W = a_tilde.shape
            P = probe.shape[2]
            varpro_chunk = max(1, 800 // (C * P))
            for vi in range(0, B, varpro_chunk):
                vj = min(vi + varpro_chunk, B)
                a_chunk = a_tilde[vi:vj].unsqueeze(2)
                b_chunk = b_tilde[vi:vj].unsqueeze(2)
                p_chunk = probe[vi:vj]
                Psi_a_c = torch.fft.fftshift(torch.fft.fft2(p_chunk * a_chunk), dim=(-2,-1))
                Psi_b_c = torch.fft.fftshift(torch.fft.fft2(1j * p_chunk * b_chunk), dim=(-2,-1))
                X1_c = torch.sum(torch.abs(Psi_a_c)**2, dim=2)
                X2_c = torch.sum(torch.abs(Psi_b_c)**2, dim=2)
                X3_c = torch.sum(2 * torch.real(Psi_a_c * torch.conj(Psi_b_c)), dim=2)
                scaler.accumulate_batch_from_basis(I_raw[vi:vj], X1_c, X2_c, X3_c)
                del a_chunk, b_chunk, p_chunk, Psi_a_c, Psi_b_c, X1_c, X2_c, X3_c

            # --- Weighted Stitching ---
            B,C,H,W= a_tilde.shape
            global_coords_2d = batch_global_coords.squeeze(2).view(B * C, 2)
            relative_positions = global_coords_2d - center_of_mass.unsqueeze(0)
            canvas_center = torch.tensor([canvas_size[1] // 2, canvas_size[0] // 2],
                                       device=primary_device, dtype=torch.float32)
            canvas_positions = relative_positions + canvas_center.unsqueeze(0)

            # Total probe intensity: sum |P_p|^2 over all incoherent modes
            probe_mag_sq = torch.sum(torch.abs(probe[0, 0, :, :, :]) ** 2, dim=0)  # (P,H,W) -> (H,W)

            # Change texture_raw to complex
            O_tilde = torch.complex(a_tilde, b_tilde)
            O_tilde = O_tilde.view(B*C, N, N)

            # Canvas assembly
            accumulator.accumulate_batch(canvas, canvas_weights, O_tilde,
                                        canvas_positions, probe_mag_sq,
                                        patch_size=N,
                                        uniform_weighting=uniform_weighting)

            assembly_time = time.time() - assembly_start
            total_assembly_time += assembly_time

            # Memory cleanup
            del I_raw, positions, probe, in_scale, batch_global_coords
            del texture_raw, canvas_positions

            if i % 5 == 0:
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
            primary_device, verbose=verbose)
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
    s1, s2 = scaler.solve_lbfgs()
    scaler_solve_time_end = time.time() - scaler_solve_time_start

    print(f"Scalars solved: S1 = {s1}, S2 = {s2}")
    if channels_swapped:
        print("(Solved after channel-swap correction)")

    # 5. Apply scaling
    scaled_canvas = (s1* texture_canvas.real) + 1j * (s2 * texture_canvas.imag)

    if verbose:
        avg_inference_time = total_inference_time / len(infer_loader)
        avg_assembly_time = total_assembly_time / len(infer_loader)
        efficiency = avg_inference_time / avg_assembly_time if avg_assembly_time > 0 else float('inf')

        print(f"\nPerformance Summary:")
        print(f"  Average inference time per batch: {avg_inference_time:.3f}s")
        print(f"  Average assembly time per batch: {avg_assembly_time:.3f}s")
        print(f"  Parallel efficiency ratio: {efficiency:.1f}x")
        print(f"  Total reconstruction time: {total_inference_time + total_assembly_time:.2f}s")
        print(f"  Total constant solve time: {scaler_solve_time_end:.2f}s")

    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()

    if return_diagnostics:
        modified_s1 = torch.sqrt((s1**2 + s2**2)/2 * 0.52)
        modified_s2 = torch.sqrt((s1**2 + s2**2)/2 * 0.48)
        modified_scaled_canvas = (modified_s1 * texture_canvas.real) + 1j * (modified_s2 * texture_canvas.imag)
        modified_scaled_canvas = texture_canvas.real + 1j * texture_canvas.imag
        return scaled_canvas, ptycho_subset, [total_inference_time, total_assembly_time, Psi_a, Psi_b, s1, s2], modified_scaled_canvas

    return scaled_canvas, ptycho_subset, [total_inference_time, total_assembly_time]


reconstruct_image_barycentric_weighted = reconstruct_image_barycentric


def detect_swap_probe_reference(model: nn.Module,
                                probe_single: torch.Tensor,
                                data_config: DataConfig,
                                model_config: ModelConfig,
                                device: torch.device,
                                verbose: bool = True) -> bool:
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
    N = data_config.N

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

        # Call model the same way as the inference loop
        if isinstance(model, nn.DataParallel):
            texture_raw = model((I_ref_normed, dummy_positions, dummy_probe, in_scale))
        else:
            texture_raw = model.forward_predict(I_ref_normed, dummy_positions,
                                                dummy_probe, in_scale)

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
