#Type helpers
from contextlib import nullcontext
from typing import Tuple, Optional, Union, Any, List, Dict, Literal
from ptycho_torch.dataloader import PtychoDataset, TensorDictDataLoader

#Pytorch-related
import torch
from torch import nn
import torch.nn.functional as F
import ptycho_torch.helper as hh
from ptycho_torch.dataloader import Collate
from ptycho.reconstruction_policy import CalibrationSpec
from ptycho_torch.reconstruction_ports import calibrate_reconstruction_canvas
from ptycho_torch.object_compatibility import resolve_model_object_compatibility

#Other useful libraries
import time
import gc

#Configurations
from ptycho_torch.config_params import ModelConfig, TrainingConfig, DataConfig, InferenceConfig
from ptycho_torch.reassembly_diagnostics import (
    ReassemblyDiagnostics,
    not_applicable,
    not_evaluated,
)
from ptycho_torch.scaling_contract import (
    CI_SCALE_CONTRACT,
    LEGACY_SCALE_CONTRACT,
    resolve_scale_contract,
)
from ptycho_torch.varpro import (
    VarProScaler,
    apply_varpro_canvas_scaling,
    compute_varpro_basis,
    evaluate_fitted_count_metrics,
    _prepare_ci_varpro_batch,
)
from ptycho_torch.reassembly_accumulators import (
    VectorizedBarycentricAccumulator,
    VectorizedWeightedAccumulator,
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
    if data_config.gridsize == 1:

        #Recenter
        adjusted_offsets = global_coords - com[None,:]
        #Reshape offset
        adjusted_offsets = adjusted_offsets[:,None,:]
    elif data_config.gridsize > 1: #Unused at the moment

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

    if data_config.gridsize == 1:
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

    infer_loader = TensorDictDataLoader(
        ptycho_subset,
        batch_size=inference_config.batch_size,
        collate_fn=Collate(),
        pin_memory=torch.device(device).type == "cuda",
    )

    

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
            if data_config.gridsize == 1:

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
        compatibility = getattr(source_dataset, "object_compatibility", None)
        if compatibility is None:
            source_model_config = getattr(source_dataset, "model_config", None)
            compatibility = (
                resolve_model_object_compatibility(source_model_config)
                if source_model_config is not None
                else None
            )
        grouped = bool(
            compatibility is not None
            and compatibility.layout == "grouped_patch_components_v1"
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
        collate_fn=Collate(),
        pin_memory=uses_cuda,
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
                probe_scaling = batch[2].to(
                    primary_device, non_blocking=True
                )
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
                    output_scale = torch.sqrt(1.0 / (probe_scaling ** 2 * physics_scale + 1e-9))
                    Psi_a, Psi_b, X1, X2, X3 = compute_varpro_basis(
                        effective_probe,
                        texture_raw.real,
                        texture_raw.imag,
                        scale=output_scale,
                    )
                else:
                    if torch.any(
                        ~torch.isfinite(probe_scaling) | (probe_scaling <= 0)
                    ):
                        raise ValueError(
                            "legacy amplitude VarPro requires finite positive "
                            "probe scaling"
                        )
                    # The model consumes the normalized training probe
                    # P_training = probe_scaling * P_physical.  VarPro returns
                    # a field compared with the acquisition object, so form its
                    # detector basis with P_physical and keep the object in that
                    # gauge.  The singleton append broadcasts over probe modes.
                    varpro_probe = effective_probe / probe_scaling.unsqueeze(-1)
                    output_scale = None
                    Psi_a, Psi_b, X1, X2, X3 = compute_varpro_basis(
                        varpro_probe,
                        texture_raw.real,
                        texture_raw.imag,
                    )

            # Save uncropped probe from first batch for probe-based swap detection
            if saved_probe_single is None:
                saved_probe_single = effective_probe[0, 0, 0].clone()

            total_inference_time += inference_time
            a_tilde = texture_raw.real
            b_tilde = texture_raw.imag

            # VarPro always fits detector intensity on the full frame. CI
            # supplies physical count intensity directly. Legacy ``images``
            # carry normalized diffraction amplitude, so square them exactly
            # once before fitting the intensity basis.
            varpro_observation = I_raw if ci_varpro_mode else I_raw.square()
            scaler.accumulate_batch_from_basis(
                varpro_observation, X1, X2, X3
            )

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
    calibration_spec = CalibrationSpec(
        method="varpro_s1s2_v1" if varpro_scaling else "identity_v1"
    )
    scaled_canvas, s1, s2 = calibrate_reconstruction_canvas(
        texture_canvas,
        calibration_spec,
        varpro_calibrator=lambda canvas: apply_varpro_canvas_scaling(
            canvas,
            scaler,
            enabled=True,
            verbose=verbose,
        ),
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
    compatibility = resolve_model_object_compatibility(model_config)
    C_in = (
        data_config.gridsize * data_config.gridsize
        if compatibility.layout == "grouped_patch_components_v1"
        else 1
    )

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

