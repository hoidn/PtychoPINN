#Utility
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from collections.abc import Mapping
import time
import os
import shutil
import warnings
from dataclasses import replace

#Torch
from torch.utils.data import Dataset
import torch
import torch.distributed as dist

#Memory mapping
from tensordict import MemoryMappedTensor, TensorDict

from ptycho.acquisition import decode_acquisition, inspect_acquisition

#Patch generation
from ptycho_torch.patch_generator import get_relative_coords

#Grouping plans
from ptycho.grouping import (
    CENTERED_NEAREST_GROUPING_CONTRACT,
    GroupingPlan,
    plan_nearest_groups,
)

#Parameters
from ptycho_torch.config_params import TrainingConfig, DataConfig, ModelConfig
from ptycho_torch.object_compatibility import resolve_model_object_compatibility

#Helper methods
import ptycho_torch.helper as hh
from ptycho_torch.scaling_contract import (
    CI_SCALE_CONTRACT,
    COUNT_INTENSITY,
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
    ci_scaling_active,
    resolve_scale_contract,
)

#Batch emission + collation (split out of this module in W4)
from ptycho_torch.batch_emission import (
    _as_tensor,
    _canonical_bank_scalars,
    _canonical_probe_bank,
    _emit_ptycho_batch,
)
from ptycho_torch.collate import (
    Collate,
    Collate_Lightning,
    TensorDictDataLoader,
    build_ptycho_loader,
)

# --- Helper functions for the dataloader ---
_MMAP_SCHEMA_NAME = "ptycho_torch_mmap"
_MMAP_SCHEMA_VERSION = 5
_CI_STATISTICS_CHUNK_SIZE = 256
_COMMON_MMAP_FIELDS = {
    "images",
    "coords_global",
    "coords_center",
    "coords_relative",
    "coords_start_center",
    "coords_start_relative",
    "nn_indices",
    "center_scan_id",
    "experiment_id",
    "object_index",
    "label_amp",
    "label_phase",
}
_CI_MMAP_FIELDS = _COMMON_MMAP_FIELDS | {"measured_intensity"}
_LEGACY_MMAP_FIELDS = _COMMON_MMAP_FIELDS | {
    "rms_scaling_constant",
    "physics_scaling_constant",
}


def _ci_profile_active(model_config, data_config):
    if not ci_scaling_active(model_config):
        return False
    profile = resolve_scale_contract(
        getattr(data_config, "scale_contract_version", None),
        getattr(data_config, "measurement_domain", None),
    )
    return profile.version == CI_SCALE_CONTRACT


def npz_headers(npz):
    """Return canonical shape and mmap-compatible aligned coordinates."""

    header = inspect_acquisition(npz, coordinate_policy="trailing")
    return header.diffraction_shape, header.xcoords, header.ycoords


def _validate_writer_inputs(npz_file, tensor_shape, model_config, data_config):
    """Reject writer-required NPZ inputs before memory-map allocation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        header = inspect_acquisition(npz_file, coordinate_policy="trailing")
    missing_keys = []
    if header.object_shape is None:
        missing_keys.append("objectGuess")
    if model_config.mode == "Supervised" and header.label_shape is None:
        missing_keys.append("label")
    if missing_keys:
        raise ValueError(
            f"{npz_file}: missing required key(s): {', '.join(missing_keys)}."
        )

    probe_shape = header.probe_shape
    object_shape = header.object_shape
    label_shape = header.label_shape if model_config.mode == "Supervised" else None

    if len(object_shape) != 2:
        raise ValueError(
            f"{npz_file}: objectGuess must be 2D; got shape {object_shape}."
        )

    if label_shape is not None and label_shape != tensor_shape:
        raise ValueError(
            f"{npz_file}: label shape mismatch. Expected {tensor_shape}, "
            f"got {label_shape}."
        )

    diffraction_shape = tensor_shape[1:]
    config_shape = (data_config.N, data_config.N)
    if probe_shape == diffraction_shape:
        probe_spatial_shape = probe_shape
    elif len(probe_shape) == 3 and probe_shape[-1] == 1:
        probe_spatial_shape = probe_shape[:2]
    elif len(probe_shape) == 3:
        probe_spatial_shape = probe_shape[1:]
    else:
        raise ValueError(
            f"{npz_file}: probeGuess shape {probe_shape} must be (N, N), "
            "(N, N, 1), or (P, N, N)."
        )

    if probe_spatial_shape != diffraction_shape or probe_spatial_shape != config_shape:
        raise ValueError(
            f"{npz_file}: probeGuess spatial shape {probe_spatial_shape} must match "
            f"diffraction shape {diffraction_shape} and data_config.N shape {config_shape}."
        )


def _get_diffraction_stack(npz_file):
    """Load the canonical diffraction stack through the shared NPZ decoder."""

    return decode_acquisition(
        npz_file, coordinate_policy="trailing"
    ).diff3d


# --- Tensordict patcher function ---
def fix_tensordict_memmap_state(tensordict, prefix):
    """
    Fix TensorDict memory map state - handles both manual fix and loaded TensorDicts
    Memory map state is not properly updated when calling memmap_like
    Memmap_like is necessary to create pre-allocated empty memory map which we can gradually fill
    with multiple experimental datasets.
    
    Args:
        tensordict: TensorDict to fix
        prefix: memmap prefix path
    
    Returns:
        Fixed TensorDict with proper memmap state
    """
    if not tensordict._is_memmap:
        tensordict._is_memmap = True
    
    if tensordict._memmap_prefix is None:
        tensordict._memmap_prefix = prefix #This prefix should be filled but is not automatically done by memmap_like
    
    return tensordict

# --- DDP Helper Functions ---
def is_ddp_initialized_and_active():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def get_current_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class _PtychoContainerDataset(Dataset):
    """Vectorized RAM adapter for the same batch emitter as mmap datasets."""

    _ptycho_vectorized_batch = True

    def __init__(
        self,
        container,
        *,
        model_config=None,
        data_config=None,
        ci_active=False,
    ):
        self.container = container
        self.model_config = model_config
        self.ci_active = bool(ci_active)

        def get(name, default=None):
            value = (
                container.get(name, default)
                if isinstance(container, Mapping)
                else getattr(container, name, default)
            )
            return _as_tensor(value)

        images = get("X")
        if images is None or images.ndim != 4:
            raise ValueError("Container X must have shape (B,H,W,C)")
        self.length = int(images.shape[0])
        compatibility = (
            resolve_model_object_compatibility(model_config)
            if model_config is not None
            else None
        )
        coords = get("coords_relative")
        if coords is None:
            if (
                compatibility is not None
                and compatibility.layout == "grouped_patch_components_v1"
            ):
                raise ValueError(
                    "coords_relative is required for grouped patch components "
                    "(legacy object_big=True)."
                )
            coords = get("coords_nominal")
        if coords is None:
            coords = torch.zeros(
                self.length, 1, 2, images.shape[-1], dtype=torch.float32
            )

        self.fields = {
            "images": images,
            "coords_relative": coords,
            "experiment_id": get(
                "experiment_id", torch.zeros(self.length, dtype=torch.long)
            ),
            "object_index": get(
                "object_index", torch.zeros(self.length, dtype=torch.long)
            ),
        }
        nn_indices = get("nn_indices")
        if nn_indices is not None:
            self.fields["nn_indices"] = nn_indices
        for name in (
            "observed_images",
            "measured_intensity",
            "rms_scaling_constant",
            "physics_scaling_constant",
            "label_amp",
            "label_phase",
        ):
            value = get(name)
            if value is not None:
                if name in {
                    "rms_scaling_constant",
                    "physics_scaling_constant",
                }:
                    if value.numel() == 1:
                        value = value.reshape(1, 1, 1, 1).expand(
                            self.length, -1, -1, -1
                        )
                    elif value.shape[0] == 1:
                        value = value.expand(self.length, *value.shape[1:])
                self.fields[name] = value

        for name, value in self.fields.items():
            if value.shape[0] != self.length:
                raise ValueError(
                    f"Container field {name!r} must align with X; got "
                    f"{tuple(value.shape)} for {self.length} rows."
                )
        if (
            model_config is not None
            and (
                getattr(model_config, "mode", None) == "Supervised"
                or getattr(model_config, "model_type", None) == "supervised"
            )
            and not {"label_amp", "label_phase"}.issubset(self.fields)
        ):
            raise ValueError(
                "Supervised training requires label_amp and label_phase on "
                "the container."
            )

        probe = get("probe_bank")
        self.probes = get("probe") if probe is None else probe
        if self.probes is None:
            self.probes = torch.ones(
                images.shape[1], images.shape[2], dtype=torch.complex64
            )
        self.probes_physical = get("probe_physical") if self.ci_active else None
        if self.ci_active:
            training_probe = get("probe_training")
            if training_probe is not None:
                self.probes = training_probe
        scaling = get("probe_scaling")
        if scaling is None:
            scaling = get("probe_normalization")
        if scaling is None:
            scaling = get("scaling_constant")
        if (
            scaling is None
            and not self.ci_active
            and bool(getattr(data_config, "probe_normalize", False))
        ):
            normalized, scaling = hh.normalize_probe_like_tf(
                self.probes.detach().cpu().numpy(),
                probe_scale=data_config.probe_scale,
                probe_mask=getattr(model_config, "probe_mask", False),
                probe_mask_tensor=getattr(model_config, "probe_mask_tensor", None),
                probe_mask_sigma=getattr(model_config, "probe_mask_sigma", 1.0),
                probe_mask_diameter=getattr(
                    model_config, "probe_mask_diameter", None
                ),
            )
            self.probes = torch.from_numpy(normalized).to(torch.complex64)
        probe_count = int(_canonical_probe_bank(self.probes).shape[0])
        self.probe_scaling = _canonical_bank_scalars(
            scaling, probe_count, name="probe_scaling"
        )
        self.ci_statistics = None
        if self.ci_active:
            self.ci_statistics = {
                "rms_input_scale": get("rms_input_scale"),
                "mean_measured_intensity": get("mean_measured_intensity"),
            }
            missing = [
                name
                for name, value in self.ci_statistics.items()
                if value is None
            ]
            if self.probes_physical is None or missing:
                raise ValueError(
                    "CI container is missing named fields: "
                    + ", ".join(
                        (["probe_physical"] if self.probes_physical is None else [])
                        + missing
                    )
                )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        selected = {name: value[index] for name, value in self.fields.items()}
        return _emit_ptycho_batch(
            selected,
            probes=self.probes,
            probe_scaling=self.probe_scaling,
            probes_physical=self.probes_physical,
            ci_statistics=self.ci_statistics,
            channel_last=True,
        )

    def __getitems__(self, indices):
        return self[torch.as_tensor(indices, dtype=torch.long)]

# --- Actual Dataset Class ---

class PtychoDataset(Dataset):
    """
    Ptychography Dataset for PtychoPINN

    Important: Some data is memory-mapped in order to provide fast loading for dynamic data
    #Memory-mapped data: Diffraction images, coordinates, scan_index
    #Non-memory-mapped data: Probe, Object,

    The layout of the data will be such that the index is always the image #. If you have multiple experiments
    from multiple .npz files being loaded into the same memory map, the scan number continues in a linear sequence
    (i.e. no hierarchy). There are ways of finding out which indices correspond to which .npz files, if you take a look at
    the indexing code in the memory_map section. That typically isn't needed.

    Currently can handle multiple gpus (i.e. ddp), which adds a bit of bloat to the __init__ call.

    Inputs
    -------
    ptycho_dir: Directory containing individual ptychography scans as npz files. If non-npz, expected to be normalized or
    rewritten via a data adapting software such as Ptychodus
    model_config: ModelConfig instance.
    data_config: DataConfig instance, expected to have attributes like x_bounds, y_bounds, C, N, etc.
    data_dir: Directory for memory map files.
    remake_map: Boolean, if True, recreate the memory map.

    """
    _ptycho_vectorized_batch = True

    def __init__(self, ptycho_dir: str, model_config: 'ModelConfig', data_config: 'DataConfig',
                 training_config: 'TrainingConfig' = None,
                 data_dir: str = 'data/memmap', remake_map: bool = False,
                 defer_ci_statistics: bool = False,
                 rescale_to_nphotons: float | None = None,
                 groups_per_center: int = 1):
        
        # --- Initial loading ---
        self.model_config = model_config
        self.data_config = data_config
        self.groups_per_center = groups_per_center
        self.object_compatibility = resolve_model_object_compatibility(model_config)
        self.ci_contract_active = _ci_profile_active(model_config, data_config)
        self.defer_ci_statistics = defer_ci_statistics
        self.rescale_to_nphotons = rescale_to_nphotons
        self.is_ddp_active = is_ddp_initialized_and_active()
        self.current_rank = get_current_rank()
        self.data_dict = {} #Includes important tensors that don't need to be memory mapped

        # --- File paths and initial attribute setup ---
        self.ptycho_dir = ptycho_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok = True)
        self.data_dir = data_dir # Storing the string if needed, otherwise data_dir_path is primary
        self.data_dir_path = Path(data_dir)
        data_prefix_path = self.data_dir_path.parent
        self.state_path = data_prefix_path / 'state_files.npz' # State files contain data_dict from Rank 0 (see below)
        self.manifest_path = data_prefix_path / "mmap_manifest.json"
        
        # Find npz files, try except because of distributed data parallel hang-up
        try:
            self.file_list = sorted(list(Path(self.ptycho_dir).glob('*.npz')))
            self.n_files = len(self.file_list)
            if self.n_files == 0 and self.current_rank == 0:
                raise FileNotFoundError(f"[Rank 0] No NPZ files found in directory: {self.ptycho_dir}. Cannot proceed.")
        except Exception as e:
            if self.current_rank == 0: # Only rank 0 should make the decision to halt all processes
                print(f"[Rank 0] ERROR during NPZ file listing: {e}")
                raise

        # Calculate length of total memory map, with try/except for ddp
        try:
            (self.length, self.im_shape, self.cum_length,
             self.valid_indices_per_file, self.source_indices_per_file,
             self.grouping_per_file) = self.calculate_length()
            if self.length == 0:
                 raise ValueError(
                     f"[Rank {self.current_rank}] calculate_length() resulted in 0 items. "
                     "Cannot proceed."
                 )
            # Exact centered plans emit one row per valid center per repeat, so
            # every rank agrees on self.length. Only rank 0 writes the map; the
            # rest can drop cached plans rather than hold them for the process
            # lifetime.
            if self.current_rank != 0:
                self.grouping_per_file = [None] * len(self.grouping_per_file)
        except Exception as e:
            if self.current_rank == 0:
                print(f"[Rank 0] ERROR in calculate_length(): {e}")
            raise

        #Backwards compatibility
        if not training_config:
            training_config = TrainingConfig()
            training_config.orchestrator = 'Mlflow'

        # --- Coordinated Memory Map Creation/Loading (Multi-GPU, Rank 0 orchestrates) ---
        # This is set up so the memory map is ONLY created from Rank 0 and isn't duplicated. All ranks 
        # (i.e. GPUs) will access the same memory map that was initialized by Rank 0.

        #Old Mlflow setup
        if training_config.orchestrator == 'Mlflow':
            if self.current_rank == 0:
                create_the_map_on_rank_0 = False
                map_files_exist = self.data_dir_path.exists() and any(self.data_dir_path.iterdir())
                state_file_exists = self.state_path.exists()

                if remake_map or not map_files_exist or not state_file_exists:
                    create_the_map_on_rank_0 = True
                
                if create_the_map_on_rank_0: #Creates memory map only at Rank 0. All other ranks wait at barrier
                    try:
                        data_prefix_path.mkdir(parents=True, exist_ok=True)
                        self.data_dir_path.mkdir(parents=True, exist_ok=True)
                        self.memory_map_data(self.file_list)
                        self._write_mmap_manifest()
                        np.savez(self.state_path, data_dict=self.data_dict)
                    except Exception as e:
                        print(f"[Rank 0] FATAL ERROR during map creation/saving: {e}")
                        raise # This will halt rank 0; other ranks will time out at barrier.

            # --- Barrier for DDP synchronization ---
            if self.is_ddp_active:
                dist.barrier()

            # --- Load map and state for ALL ranks ---
            # All ranks must execute this to get handles to the memory map.
            try:
                if not self.data_dir_path.exists() or not any(self.data_dir_path.iterdir()) or not self.state_path.exists():
                    # This indicates rank 0 failed to create the files, or they were deleted.
                    raise FileNotFoundError(f"[Rank {self.current_rank}] Critical map/state files missing after barrier. "
                                            f"Map dir: {self.data_dir_path} (exists: {self.data_dir_path.exists()}), "
                                            f"State file: {self.state_path} (exists: {self.state_path.exists()})")
                self._validate_mmap_manifest()
                self.mmap_ptycho = TensorDict.load_memmap(str(self.data_dir_path)) # Load memory map that was initialized by Rank 0
                self._validate_loaded_mmap_fields()
                loaded_state = np.load(self.state_path, allow_pickle=True)
                self.data_dict = loaded_state['data_dict'].item()

            except Exception as e:
                print(f"[Rank {self.current_rank}] FATAL ERROR loading map files or state AFTER barrier: {e}")
                raise
        
        #Lightning-only setup
        elif training_config.orchestrator == 'Lightning':
            print("Lightning")
            if remake_map:
                # Rank 0 will enter here via prepare_data
                print(f"Creating memory mapped tensor dictionary...")
                self.memory_map_data(self.file_list)
                self._write_mmap_manifest()
                np.savez(self.state_path, data_dict=self.data_dict)
            else:
                # All ranks will enter here via setup
                print(f"Loading existing dataset on rank {self.current_rank}")
                if not self.state_path.exists():
                    raise FileNotFoundError(f"Map files missing. prepare_data should have created them.")
                self._validate_mmap_manifest()
                self.mmap_ptycho = TensorDict.load_memmap(str(self.data_dir_path))
                self._validate_loaded_mmap_fields()
                sample_sum = self.mmap_ptycho["images"][:10].sum()
                if sample_sum == 0:
                    print(f"[Rank {self.current_rank}] WARNING: Loaded memory map contains only zeros!")
                    # If Rank 1 sees zeros, it means the OS sync hasn't propagated.
                    # In a real DDP scenario, you might want to raise an error here
                    # so the process restarts, rather than training on garbage.
                    raise RuntimeError(f"Rank {self.current_rank} loaded empty memory map data.")
                loaded_state = np.load(self.state_path, allow_pickle=True)
                self.data_dict = loaded_state['data_dict'].item()

                # 1. Check a sample from the END of the file (Validation data area)
                end_sample = self.mmap_ptycho["images"][-10:].sum()
                
                # 2. Check the scaling constants (If these are 0, loss collapses)
                rms_sample = None
                if not self.ci_contract_active:
                    rms_sample = self.mmap_ptycho[
                        "rms_scaling_constant"
                    ][:10].sum()
                
                if end_sample == 0 or (
                    rms_sample is not None and rms_sample == 0
                ):
                    print(f"[Rank {self.current_rank}] CRITICAL: Metadata or End-of-file data is ZERO.")
                    print(f"  End images sum: {end_sample}")
                    if rms_sample is not None:
                        print(f"  RMS constant sum: {rms_sample}")
                    raise RuntimeError(f"Rank {self.current_rank} loaded corrupted data.")
        
        
        # Minimal success log, good for confirming init completion on all ranks
        if self.current_rank == 0:
             print(f"[PtychoDataset Rank 0] Initialization successful. Dataset length: {self.length}.")

    def calculate_length(self):
        """
        The purpose of this function is to get the total number of diffraction patterns from all provided datasets
        that will exist in the memory map. This length is needed to pre-allocate the size of the total memory map.

        Calculates length from series of npz files, accounting for coordinate bounds.
        Uses stored model_config and data_config (esp. x_bounds, y_bounds).
        Also calculates cumulative length for linear indexing based on *filtered* counts.
        Stores the valid indices per file for reuse in memory_map_data.

        When coordinate grouping applies, the groups are planned here rather
        than estimated: centered-nearest plans exactly one group per bounded
        center per ``groups_per_center`` repeat, and planning once and caching
        the ``GroupingPlan`` keeps the allocation, cum_length, and the tensors
        written by memory_map_data exactly consistent -- and means the grouping
        is not recomputed with different random draws on the write pass.
        """
        total_length = 0
        cumulative_length = [0]
        first_im_shape = None
        valid_indices_per_file = [] # Store valid indices for each file
        source_indices_per_file = [] # Every source diffraction index before bounds
        grouping_per_file = [] # GroupingPlan per file when grouping applies, else None

        group_coordinates = self.grouping_enabled()
        print("Calculating dataset length with coordinate bounds...")
        # Make sure bounds are valid
        if not (0.0 <= self.data_config.x_bounds[0] < self.data_config.x_bounds[1] <= 1.0):
            raise ValueError(f"Invalid x_bounds: {self.data_config.x_bounds}. Must be [min_pct, max_pct] between 0.0 and 1.0.")
        if not (0.0 <= self.data_config.y_bounds[0] < self.data_config.y_bounds[1] <= 1.0):
             raise ValueError(f"Invalid y_bounds: {self.data_config.y_bounds}. Must be [min_pct, max_pct] between 0.0 and 1.0.")

        for i, npz_file in enumerate(self.file_list): # Use ordered list
            header = inspect_acquisition(npz_file, coordinate_policy="trailing")
            tensor_shape = header.diffraction_shape
            xcoords = header.xcoords
            ycoords = header.ycoords

            if i == 0:
                first_im_shape = tensor_shape[1:] # Get H, W from the first file
            elif tensor_shape[1:] != first_im_shape:
                raise ValueError(
                    f"{npz_file}: image shape mismatch. Expected {first_im_shape}, "
                    f"got {tensor_shape[1:]}."
                )

            _validate_writer_inputs(
                npz_file, tensor_shape, self.model_config, self.data_config
            )

            # --- Apply Coordinate Bounding ---
            # Cannot pick points that don't have full probe coverage
            xmin, xmax = np.min(xcoords), np.max(xcoords)
            ymin, ymax = np.min(ycoords), np.max(ycoords)

            print(f'For file {npz_file}, maximum x_range is {xmin, xmax}, yrange is {ymin, ymax}')

            # Handle cases where min == max to avoid division by zero or zero range
            x_range = xmax - xmin if xmax > xmin else 1.0
            y_range = ymax - ymin if ymax > ymin else 1.0

            # Apply further bounding if we don't trust the edges
            x_lower = xmin + self.data_config.x_bounds[0] * x_range
            x_upper = xmin + self.data_config.x_bounds[1] * x_range
            y_lower = ymin + self.data_config.y_bounds[0] * y_range
            y_upper = ymin + self.data_config.y_bounds[1] * y_range

            # Ensure upper bound is at least the lower bound if range was zero
            if xmax <= xmin: x_upper = x_lower
            if ymax <= ymin: y_upper = y_lower

            mask = (xcoords >= x_lower) & (xcoords <= x_upper) & \
                   (ycoords >= y_lower) & (ycoords <= y_upper)

            valid_indices = np.where(mask)[0]
            n_valid_points = len(valid_indices)
            # Stores indices of points whose coordinates lie within specified bounds
            # We want to skip image edges because predictions may be unstable there
            valid_indices_per_file.append(valid_indices)
            source_indices_per_file.append(np.arange(tensor_shape[0], dtype=np.int64))

            if n_valid_points == 0:
                print(f"Warning: No points found within bounds for file {npz_file}")
            # ---------------------------------

            # Plan the coordinate groups now so the length is the true group
            # count: one centered-nearest row per bounded center per repeat.
            # The plan is cached and reused verbatim by the write pass.
            if group_coordinates and n_valid_points > 0:
                local_rng = np.random.default_rng(
                    0 if self.data_config.subsample_seed is None
                    else self.data_config.subsample_seed
                )
                plan = plan_nearest_groups(
                    xcoords,
                    ycoords,
                    center_indices=valid_indices,
                    candidate_indices=valid_indices,
                    group_size=(
                        self.data_config.gridsize * self.data_config.gridsize
                    ),
                    neighbor_count=self.data_config.neighbor_count,
                    repeats=self.groups_per_center,
                    object_index=header.object_index,
                    experiment_id=i,
                    source_indices=np.arange(
                        tensor_shape[0], dtype=np.int64
                    ),
                    rng=local_rng,
                )
                expected_rows = n_valid_points * self.groups_per_center
                if len(plan.neighbor_indices) != expected_rows:
                    raise ValueError(
                        f"{npz_file}: cached grouping plan produced "
                        f"{len(plan.neighbor_indices)} rows, expected "
                        f"{expected_rows} ({n_valid_points} valid points x "
                        f"{self.groups_per_center} groups per center)."
                    )
                grouping_per_file.append(plan)
                length_contribution = len(plan.neighbor_indices)
            else:
                grouping_per_file.append(None)
                length_contribution = n_valid_points

            total_length += length_contribution
            cumulative_length.append(total_length)

        if first_im_shape is None:
             raise ValueError("Could not determine image shape from any NPZ file.")

        return (total_length, first_im_shape, cumulative_length,
                valid_indices_per_file, source_indices_per_file, grouping_per_file)

    def grouping_enabled(self):
        """Whether memory_map_data groups coordinates into solution regions.

        Mirrors the branch condition in memory_map_data exactly: calculate_length
        must size the memory map for the same branch that writes it.
        """
        return (
            self.model_config.mode == 'Unsupervised'
            and self.object_compatibility.layout
            == 'grouped_patch_components_v1'
        )

    def _expected_mmap_manifest(self):
        if self.ci_contract_active:
            contract_version = CI_SCALE_CONTRACT
            measurement_domain = COUNT_INTENSITY
            required_fields = _CI_MMAP_FIELDS
        else:
            contract_version = LEGACY_SCALE_CONTRACT
            measurement_domain = NORMALIZED_AMPLITUDE
            required_fields = _LEGACY_MMAP_FIELDS
        return {
            "schema_name": _MMAP_SCHEMA_NAME,
            "schema_version": _MMAP_SCHEMA_VERSION,
            "scale_contract_version": contract_version,
            "measurement_domain": measurement_domain,
            "required_fields": sorted(required_fields),
            "grouping_contract": CENTERED_NEAREST_GROUPING_CONTRACT,
        }

    def _mmap_rebuild_error(self, reason):
        return ValueError(
            f"Incompatible memory map: {reason}. Rebuild it with remake_map=True."
        )

    def _write_mmap_manifest(self):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self._expected_mmap_manifest(), indent=2, sort_keys=True)
        )

    def _validate_mmap_manifest(self):
        if not self.manifest_path.exists():
            raise self._mmap_rebuild_error(
                f"missing schema manifest at {self.manifest_path}"
            )
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise self._mmap_rebuild_error("unreadable schema manifest") from exc

        expected = self._expected_mmap_manifest()
        for field in (
            "schema_name",
            "schema_version",
            "scale_contract_version",
            "measurement_domain",
            "required_fields",
            "grouping_contract",
        ):
            observed = manifest.get(field)
            if observed != expected[field]:
                raise self._mmap_rebuild_error(
                    f"manifest {field}={observed!r}, "
                    f"expected {expected[field]!r}"
                )

    def _validate_loaded_mmap_fields(self):
        required = set(self._expected_mmap_manifest()["required_fields"])
        missing = sorted(required - set(self.mmap_ptycho.keys()))
        if missing:
            raise self._mmap_rebuild_error(
                f"stored TensorDict is missing required fields {missing}"
            )
    
    @classmethod
    def from_existing_map(
        cls,
        map_path,
        model_config,
        data_config,
        current_rank=0,
        is_ddp_active=False,
    ):
        """
        Creates data instance from existing memory map. Do NOT run without a memory map!

        Assumes:
        1. Memory map already exists at map_path
        2. State files exist
        3. No rank coordination
        4. No file operations
        """

        instance = cls.__new__(cls)

        #Set basic attributes
        instance.model_config = model_config
        instance.data_config = data_config
        instance.object_compatibility = resolve_model_object_compatibility(
            model_config
        )
        instance.ci_contract_active = _ci_profile_active(
            model_config,
            data_config,
        )
        instance.defer_ci_statistics = False
        instance.current_rank = current_rank
        instance.is_ddp_active = is_ddp_active

        #Set paths
        instance.data_dir = str(map_path)
        instance.data_dir_path = Path(map_path)
        data_prefix_path = instance.data_dir_path.parent
        instance.state_path = data_prefix_path / 'state_files.npz'
        instance.manifest_path = data_prefix_path / "mmap_manifest.json"

        #Load existing map
        try:
            instance._validate_mmap_manifest()
            instance.mmap_ptycho = TensorDict.load_memmap(str(instance.data_dir_path))
            instance._validate_loaded_mmap_fields()
            instance.length = len(instance.mmap_ptycho)

            #Load state data
            loaded_state = np.load(instance.state_path, allow_pickle = True)
            instance.data_dict = loaded_state['data_dict'].item()
            instance.n_files = int(instance.data_dict["probes"].shape[0])
            
            print(f"[PtychoDataset Rank {current_rank}] Loaded existing memory map: {instance.length} samples")

        except Exception as e:
            raise RuntimeError(
                f"[Rank {current_rank}] Failed to load existing memory map from {map_path}. "
                f"Ensure prepare_memory_mapped_data() was called first. Error: {e}"
            )
        
        return instance

    # Methods for diffraction data mapping
    def memory_map_data(self, image_paths):
        """
        Creates memory mapped tensor dictionary containg diffraction images and relevant coordinate information.
        Great care needs to be taken to track the indices corresponding to each unique dataset. This is because we pre-allocate
        the memory of the memory map and batch fill it.
        1.  Reuses the cached GroupingPlan built by calculate_length
        2.  Writes to respective memory maps. The diffraction map is populated in batches, while the other maps
        are populated in full for every individual dataset
            - "images" - (N x C x H x W), N = # of patterns, C = # of images per soln patch, H = height, W = width
            - "coords_offsets" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
            - "coords_relative" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
            - "coords_start_offsets" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
            - "coords_start_relative" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
            - "nn_indices" - (N, C) , N = # of patterns, C = # of images per soln patch, gives indices of each coord group
            - "experiment_id" - N, N = # of patterns, gives association to specific npz/experiment file
            - "object_index" - N, source-object identity for each emitted row

        Note: Probe/object stored in the data_dict, not in the memory map.
        ---
        Args:
            image_paths - list of paths to independent experiment npz files
            grid_size - tuple of image grid size (e.g. 2 x 2 is most used)

        """
        #Config grabbing/setting using stored configs
        if self.object_compatibility.layout == 'grouped_patch_components_v1':
            n_channels = self.data_config.gridsize * self.data_config.gridsize

        else:
            n_channels = 1

        N = self.data_config.N
        #Create memory map for every tensor. We'll be populating the diffraction image in batches, and the
        #other coordinate tensors in full for every individual dataset

        mmap_length = self.length

        #Time creation of tensordict with printed messages
        print("Creating memory mapped tensor dictionary...")
        print("Memory map length: {}".format(mmap_length))

        #Start timer
        start = time.time()

        mmap_fields = {
                "images": MemoryMappedTensor.empty(
                    (mmap_length, n_channels, *self.im_shape),
                    dtype=torch.float32,
                ),
                "coords_global": MemoryMappedTensor.empty(
                    (mmap_length, n_channels, 1, 2),
                    dtype=torch.float32,
                ),
                "coords_center": MemoryMappedTensor.empty(
                    (mmap_length, 1, 1, 2),
                    dtype=torch.float32,
                ),
                "coords_relative": MemoryMappedTensor.empty(
                    (mmap_length, n_channels, 1, 2),
                    dtype=torch.float32,
                ),
                "coords_start_center": MemoryMappedTensor.empty(
                    (mmap_length, 1, 1, 2),
                    dtype=torch.float32,
                ),
                "coords_start_relative": MemoryMappedTensor.empty(
                    (mmap_length, n_channels, 1, 2),
                    dtype=torch.float32,
                ),
                "nn_indices": MemoryMappedTensor.empty(
                    (mmap_length, n_channels),
                    dtype=torch.int64
                ),
                "center_scan_id": MemoryMappedTensor.empty(
                    (mmap_length), dtype=torch.int64
                ),
                "experiment_id": MemoryMappedTensor.empty(
                    (mmap_length),
                    dtype=torch.int32
                ),
                "object_index": MemoryMappedTensor.empty(
                    (mmap_length),
                    dtype=torch.int64,
                ),
                # Optional: Empty if self-supervised. Meant to be a complex tensor
                "label_amp": MemoryMappedTensor.empty(
                    (mmap_length, n_channels, *self.im_shape),
                    dtype=torch.float32
                ),
                "label_phase": MemoryMappedTensor.empty(
                    (mmap_length, n_channels, *self.im_shape),
                    dtype=torch.float32
                ),
        }
        if self.ci_contract_active:
            mmap_fields.update({
                "measured_intensity": MemoryMappedTensor.empty(
                    (mmap_length, n_channels, *self.im_shape),
                    dtype=torch.float32,
                ),
            })
        else:
            mmap_fields.update({
                "rms_scaling_constant": MemoryMappedTensor.empty(
                    (mmap_length,1,1,1),
                    dtype=torch.float32
                ),
                "physics_scaling_constant": MemoryMappedTensor.empty(
                    (mmap_length,1,1,1),
                    dtype=torch.float32
                ),
            })
        mmap_ptycho = TensorDict(mmap_fields, batch_size=mmap_length)
        #End timer
        end = time.time()
        print("Memory map creation time: {}".format(end - start))

        #Lock memory map, ensure proper pathing
        mmap_ptycho = mmap_ptycho.memmap_like(prefix=self.data_dir)
        mmap_ptycho = fix_tensordict_memmap_state(mmap_ptycho, self.data_dir)

        #Go through each npz file and populate mmap_diffraction
        batch_size = 3000 #Batch size for writing diffraction tensors to memory map
        #Keep track of memory map write indices
        global_from, global_to = 0, 0

        #Initialize probes and objects in datadict
        #Pre-scan probe files to determine max number of incoherent modes
        max_modes = 1
        for npz_file in image_paths:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p_shape = inspect_acquisition(
                    npz_file, coordinate_policy="trailing"
                ).probe_shape
            if len(p_shape) == 3 and p_shape[-1] != 1:
                max_modes = max(max_modes, p_shape[0])
        if max_modes > 1:
            print(f"Detected multi-mode probes: max {max_modes} modes")
        self.data_dict['probes'] = torch.zeros(size=(self.n_files, max_modes, N, N), dtype=torch.complex64)
        if self.ci_contract_active:
            self.data_dict['probes_physical'] = torch.zeros(
                size=(self.n_files, max_modes, N, N),
                dtype=torch.complex64,
            )
        self.data_dict['probe_scaling'] = torch.zeros(size=(self.n_files,), dtype = torch.float32)
        self.data_dict['objectGuess'] = []
        effective_batch_normalization = (
            self.data_config.normalize == 'Batch' or
            (self.data_config.normalize == 'Group' and self.data_config.gridsize == 1 and
             self.model_config.mode != 'Supervised')
        )
        if self.data_config.normalize == 'None' or effective_batch_normalization:
            # Legacy scaling constant needed for older model artifacts.
            self.data_dict["scaling_constant"] = torch.empty(
                self.n_files, dtype=torch.float32)

        #Supervised learning correction factor (PtychoNN-related)
        if self.model_config.mode == 'Supervised':
            self.data_dict['phase_correction'] = []

        # Iterate through all npz files in directory
        for i, npz_file in enumerate(image_paths):

            print("Populating memory map for dataset {}".format(i))
            #Calculating all non-diffraction related parameters/tensors
            #Assume: N = # of scans
            start, end = self.cum_length[i], self.cum_length[i+1]

            print(f"Start - end = {end- start}")
            #Writing to non-diffraction memory maps in one go:
            non_diff_timer_start = time.time()

            with warnings.catch_warnings():
                # calculate_length already warned about this file via npz_headers
                warnings.simplefilter("ignore", RuntimeWarning)
                acquisition = decode_acquisition(
                    npz_file,
                    coordinate_policy="trailing",
                    experiment_id=i,
                )
            diffraction = acquisition.diff3d
            probe_data = acquisition.probeGuess
            if self.rescale_to_nphotons is not None:
                from ptycho_torch.scaling_contract import (
                    rescale_amplitude_to_nphotons,
                )

                diffraction, probe_data, probe_simulated = (
                    rescale_amplitude_to_nphotons(
                        diffraction,
                        probe_data,
                        self.rescale_to_nphotons,
                        acquisition.probe_simulated,
                    )
                )
                if probe_simulated is not None:
                    self.data_dict.setdefault(
                        "probe_simulated", [None] * self.n_files
                    )[i] = torch.from_numpy(np.ascontiguousarray(probe_simulated))
            diff_stack = torch.from_numpy(diffraction).to(torch.float32)
            xcoords_full = acquisition.xcoords
            ycoords_full = acquisition.ycoords

            #Apply coordinate filter to remove edge points based on self.calculate_length
            xcoords = xcoords_full[self.valid_indices_per_file[i]]
            ycoords = ycoords_full[self.valid_indices_per_file[i]]
            self.data_dict['com'] = torch.from_numpy(np.array([xcoords.mean(), ycoords.mean()])) #Center of mass (see reassembly.py)

            #--- Coordinate patches/Supervised Labels ---
            # Note that object_big = True means we are enforcing ptychographic constraints and need to group coordinates
            if self.grouping_enabled(): # PtychoPINN/Ptychography Constraint
                #Reuse the plan built in calculate_length: regrouping here would
                #redraw the random candidate/subsample picks and desync from cum_length
                plan = self.grouping_per_file[i]
                if plan is None:
                    raise RuntimeError(
                        f"cached grouping plan for file {i} is missing; "
                        "cannot write the allocated mmap slice"
                    )
                if end - start != len(plan.neighbor_indices):
                    raise RuntimeError(
                        f"allocated mmap slice [{start}:{end}] has {end - start} "
                        f"rows but the cached grouping plan has "
                        f"{len(plan.neighbor_indices)}"
                    )

                # Map plan-local rows onto global source rows and persist them.
                nn_indices = plan.source_indices[plan.neighbor_indices]
                center_indices = plan.source_indices[plan.center_indices]
                coords_nn = np.stack(
                    [
                        xcoords_full[nn_indices],
                        ycoords_full[nn_indices],
                    ],
                    axis=2,
                )[:, :, None, :]

                #Get relative and center of mass coordinates for each coordinate group
                coords_com, coords_relative = get_relative_coords(coords_nn)
                mmap_ptycho["coords_center"][start:end] = torch.from_numpy(coords_com)
                mmap_ptycho["coords_relative"][start:end] = torch.from_numpy(coords_relative)
                mmap_ptycho["nn_indices"][start:end] = torch.from_numpy(
                    np.asarray(nn_indices, dtype=np.int64)
                )
                mmap_ptycho["center_scan_id"][start:end] = torch.from_numpy(
                    np.asarray(center_indices, dtype=np.int64)
                )

                #Coordinates just outside the "valid range" are still allowed to be used to create coordinate
                #groupings. These will be used for solution region translation
                regular_global_coords = torch.from_numpy(np.stack([xcoords_full,
                                                        ycoords_full],axis=1)).to(torch.float32)
                
                mmap_ptycho["coords_global"][start:end] = regular_global_coords[nn_indices].unsqueeze(2)

                #Grouping arrays are large; release the cached copy once written
                self.grouping_per_file[i] = None

            else: #Unsupervised CDI or supervised learning

                #Otherwise, the indices are just an arange from 0 to N-1
                nn_indices = self.valid_indices_per_file[i]
                index_range = np.arange(end-start, dtype=np.int64)
                mmap_ptycho["nn_indices"][start:end] = torch.from_numpy(index_range)[:,None]
                mmap_ptycho["center_scan_id"][start:end] = torch.from_numpy(nn_indices)
                mmap_ptycho["coords_global"][start:end] = torch.from_numpy(
                                                            np.stack([xcoords,
                                                            ycoords],axis=1)[:, None, None, :]).to(torch.float32)
                
                #Add labels if supervised model is selected
                if self.model_config.mode == 'Supervised':
                    print("Assigning labels...")
                    #Only grab valid labels which were calculated before. Validity based on coordinates
                    valid_labels = acquisition.label[nn_indices][:,None,:,:] # Channel dimension added for consistency, size = 1
                    
                    #Do phase correction based on prior PtychoNN conventions
                    objectGuess = acquisition.objectGuess
                    obj_phase = np.angle(objectGuess)
                    phase_corr_factor = obj_phase[int(obj_phase.shape[0] / 3.):int(obj_phase.shape[0] * 2 / 3.),
                                                  int(obj_phase.shape[1] / 3.):int(obj_phase.shape[1] * 2 / 3.)].mean()
                    self.data_dict['phase_correction'].append(phase_corr_factor)
                    valid_label_phase, valid_label_amp = np.angle(valid_labels), np.abs(valid_labels)
                    if self.data_config.phase_subtraction:
                        valid_label_phase -= phase_corr_factor
                    valid_label_phase = np.angle(np.exp(1j*valid_label_phase)) #Phase wrap back to [-pi,pi]

                    #Write rescaled labels to memory map, complex not supported by MemoryMappedTensor.
                    mmap_ptycho["label_amp"][start:end] = torch.from_numpy(valid_label_amp)
                    mmap_ptycho["label_phase"][start:end] = torch.from_numpy(valid_label_phase)

            selected_object_index = acquisition.object_index[nn_indices]
            if selected_object_index.ndim == 2:
                if not np.all(
                    selected_object_index == selected_object_index[:, :1]
                ):
                    raise ValueError(
                        "grouped rows must remain within one object_index partition"
                    )
                selected_object_index = selected_object_index[:, 0]
            mmap_ptycho["object_index"][start:end] = torch.from_numpy(
                np.asarray(selected_object_index, dtype=np.int64)
            )

            # Mapping experiment IDs.
            mmap_ptycho["experiment_id"][start:end] = torch.tensor(i)

            #Mapping probes
            if probe_data.ndim == 3 and probe_data.shape[-1] == 1:
                probe_data = probe_data[..., 0]  # Canonicalize (N, N, 1) -> (N, N)
            probe_physical = np.ascontiguousarray(
                probe_data[None] if probe_data.ndim == 2 else probe_data
            )
            #Optional: normalize probe for forward model to be photon agnostic. We almost always normalize.
            #Handles single-mode (N, N) and incoherent multi-mode (P, N, N) probes.
            if self.data_config.probe_normalize:
                probe_data, scaling_factor = hh.normalize_probe_like_tf(
                    probe_data,
                    probe_scale=self.data_config.probe_scale,
                    probe_mask=getattr(self.model_config, "probe_mask", False),
                    probe_mask_tensor=getattr(self.model_config, "probe_mask_tensor", None),
                    probe_mask_sigma=getattr(self.model_config, "probe_mask_sigma", 1.0),
                    probe_mask_diameter=getattr(self.model_config, "probe_mask_diameter", None),
                )
                self.data_dict['probe_scaling'][i] = float(scaling_factor)
            else:
                #Save a scaling constant, it's just 1 though
                self.data_dict['probe_scaling'][i] = float(1)
            if probe_data.ndim == 2:
                probe_data = np.expand_dims(probe_data, axis=0)
            n_modes = probe_data.shape[0]
            self.data_dict['probes'][i,:n_modes] = torch.from_numpy(probe_data).to(torch.complex64)
            if self.ci_contract_active:
                self.data_dict['probes_physical'][i, :n_modes] = torch.from_numpy(
                    probe_physical
                ).to(torch.complex64)

            #Object
            objectGuess = acquisition.objectGuess
            if int(objectGuess.sum().real) != (objectGuess.shape[0] * objectGuess.shape[1]): #Check if matrix of ones
                self.data_dict['objectGuess'].append(objectGuess)
            
            non_diff_time = time.time() - non_diff_timer_start
            print("Non-diffraction memory map write time: {}".format(non_diff_time))

            #--- DIFFRACTION IMAGE MAPPING/NORMALIZATION ---
            diff_timer_start = time.time()
            curr_nn_index_length = len(nn_indices)

            #Load diffraction images (standalone 'diff3d' key with 'diffraction' compatibility alias)
            # NOTE: no .round() here. docs/specs/spec-ptycho-core.md and
            # docs/DATA_NORMALIZATION_GUIDE.md mandate this array is
            # normalized amplitude (typically max < 1.0), with nphotons carried only as a
            # separate config-time physics-scaling parameter -- never baked into the data.
            # Rounding a normalized-amplitude array to the nearest integer zeros it out
            # entirely (confirmed empirically: a real fly64_p1e9 fixture, all values < 0.03,
            # rounds to all-zero, which then makes get_rms_scaling_factor divide by zero
            # and return inf). Found while running Task 1.5's Step 0 smoke gate.
            #Inserting dummy channel dimension when nn_indices is flat (M,) rather
            #than grouped (M, C): keyed on the same branch that produced nn_indices
            if not self.grouping_enabled():
                diff_stack = diff_stack[:,None]

            # Normalizing diffraction images for explicit legacy/amplitude paths.
            if not self.ci_contract_active:
                print("Getting normalization coefficients...")
            # A configured C=1 Group is effectively Batch normalization, but
            # the helper must see the Batch config because diff_stack is 3D.
            B = end - start #Batch size
            if self.ci_contract_active:
                pass
            elif self.data_config.normalize == 'None':
                norm_factor = torch.ones(size=(B,1,1,1))
                mmap_ptycho["rms_scaling_constant"][start:end] = norm_factor
                mmap_ptycho["physics_scaling_constant"][start:end] = norm_factor
                self.data_dict["scaling_constant"][i] = 1.0
            elif effective_batch_normalization:
                factor_config = (
                    self.data_config if self.data_config.normalize == 'Batch'
                    else replace(self.data_config, normalize='Batch')
                )
                # Calculate rms normalization factor (used in publication)
                norm_rms_factor = hh.get_rms_scaling_factor(diff_stack, factor_config)
                print("Batch rms factor is", norm_rms_factor)
                mmap_ptycho["rms_scaling_constant"][start:end] = norm_rms_factor.expand(B,1,1,1)
                # Calculate physics normalization factor
                norm_physics_factor = hh.get_physics_scaling_factor(diff_stack, factor_config)
                mmap_ptycho["physics_scaling_constant"][start:end] = norm_physics_factor.expand(B,1,1,1)
                # Legacy scaling constant
                self.data_dict["scaling_constant"][i] = norm_rms_factor

            #Write to memory mapped tensor in batches to avoid huge memory overhead
            for j in range(0, curr_nn_index_length, batch_size): #Write all diffraction images for current experiment
                #Calculate end index (to not exceed length of list)
                local_to = min(j + batch_size, curr_nn_index_length)
                global_to += local_to - j
                
                #NN_indices gives us our coordinate groups of diffraction patterns
                mmap_ptycho["images"][global_from:global_to] = diff_stack[nn_indices[j:local_to]]
                if self.ci_contract_active:
                    mmap_ptycho["measured_intensity"][global_from:global_to] = (
                        diff_stack[nn_indices[j:local_to]]
                    )

                #Calculate group normalization if specified
                if (not self.ci_contract_active and
                        self.data_config.normalize == 'Group' and self.data_config.gridsize > 1):
                    # RMS normalization
                    norm_rms_factor = hh.get_rms_scaling_factor(diff_stack[nn_indices[j:local_to]], self.data_config)
                    mmap_ptycho["rms_scaling_constant"][global_from:global_to] = norm_rms_factor
                    #Physics normalization
                    norm_physics_factor = hh.get_physics_scaling_factor(diff_stack[nn_indices[j:local_to]], self.data_config)
                    mmap_ptycho["physics_scaling_constant"][global_from:global_to] = norm_physics_factor

                #Update global
                global_from += global_to - global_from

            diff_time = time.time() - diff_timer_start
            print("Diffraction memory map write time: {}".format(diff_time))

        #Every allocated row must have been written by exactly one experiment.
        if global_to != mmap_length:
            raise RuntimeError(
                f"memory map write cursor {global_to} does not match the "
                f"allocated length {mmap_length}"
            )

        #Assign memory map to class attribute
        self.mmap_ptycho = mmap_ptycho
        if self.ci_contract_active and not self.defer_ci_statistics:
            self.set_ci_statistics_from_indices(torch.arange(self.length))

        return

    def set_ci_statistics_from_indices(self, indices):
        """Freeze per-experiment CI statistics from the selected samples."""
        if not self.ci_contract_active:
            return None

        if isinstance(indices, torch.Tensor):
            flattened_indices = indices.reshape(-1)
            index_count = flattened_indices.numel()

            def get_index_chunk(start, stop):
                return flattened_indices[start:stop].to(dtype=torch.long)
        else:
            index_count = len(indices)

            def get_index_chunk(start, stop):
                return torch.as_tensor(
                    indices[start:stop],
                    dtype=torch.long,
                ).reshape(-1)

        if index_count == 0:
            raise ValueError("CI training indices must not be empty.")
        sum_squares = torch.zeros(self.n_files, dtype=torch.float64)
        intensity_sums = torch.zeros(self.n_files, dtype=torch.float64)
        sample_channel_counts = torch.zeros(self.n_files, dtype=torch.int64)
        element_counts = torch.zeros(self.n_files, dtype=torch.int64)
        measured_dtype = None

        for start in range(0, index_count, _CI_STATISTICS_CHUNK_SIZE):
            chunk_indices = get_index_chunk(
                start,
                min(start + _CI_STATISTICS_CHUNK_SIZE, index_count),
            )
            experiment_ids = torch.as_tensor(
                self.mmap_ptycho["experiment_id"][chunk_indices],
                dtype=torch.long,
            )
            measured = torch.as_tensor(
                self.mmap_ptycho["measured_intensity"][chunk_indices]
            )
            if measured.ndim != 4:
                raise ValueError(
                    "measured_intensity must have shape (B, C, H, W)."
                )
            if not torch.is_floating_point(measured) or torch.is_complex(measured):
                raise TypeError(
                    "measured_intensity must be a real floating-point tensor."
                )
            if not bool(torch.isfinite(measured).all()):
                raise ValueError(
                    "measured_intensity must contain only finite values."
                )
            if bool((measured < 0).any()):
                raise ValueError(
                    "measured_intensity must contain nonnegative counts."
                )
            measured_dtype = measured.dtype

            for experiment_id in experiment_ids.unique().tolist():
                selected = measured[experiment_ids == experiment_id]
                selected_float64 = selected.to(torch.float64)
                sum_squares[experiment_id] += selected_float64.square().sum()
                intensity_sums[experiment_id] += selected_float64.sum()
                sample_channel_counts[experiment_id] += (
                    selected.shape[0] * selected.shape[1]
                )
                element_counts[experiment_id] += selected.numel()

        missing_experiments = torch.where(sample_channel_counts == 0)[0]
        if missing_experiments.numel():
            missing = ", ".join(str(int(value)) for value in missing_experiments)
            raise ValueError(
                "The finalized CI training split contains no samples for "
                f"experiment(s) {missing}."
            )

        mean_squared_energy = sum_squares / sample_channel_counts.to(torch.float64)
        mean_measured_intensity = intensity_sums / element_counts.to(torch.float64)
        target_energy = (float(self.data_config.N) / 2.0) ** 2
        rms_input_scale = torch.sqrt(target_energy / mean_squared_energy)
        if not bool(torch.isfinite(rms_input_scale).all()) or not bool(
            (rms_input_scale > 0).all()
        ):
            raise ValueError("rms_input_scale must be positive and finite.")
        if not bool(torch.isfinite(mean_measured_intensity).all()) or not bool(
            (mean_measured_intensity > 0).all()
        ):
            raise ValueError(
                "mean_measured_intensity must be positive and finite."
            )

        rms_values = rms_input_scale.to(dtype=measured_dtype)
        mean_values = mean_measured_intensity.to(dtype=measured_dtype)

        self.data_dict["ci_statistics"] = {
            "rms_input_scale": rms_values,
            "mean_measured_intensity": mean_values,
        }
        return self.get_ci_statistics()

    def get_ci_statistics(self):
        if not self.ci_contract_active:
            return None
        return {
            name: value.detach().clone()
            for name, value in self.data_dict["ci_statistics"].items()
        }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return _emit_ptycho_batch(
            self.mmap_ptycho[idx],
            probes=self.data_dict["probes"],
            probe_scaling=self.data_dict["probe_scaling"],
            probes_physical=(
                self.data_dict["probes_physical"]
                if self.ci_contract_active
                else None
            ),
            ci_statistics=(
                self.data_dict["ci_statistics"]
                if self.ci_contract_active
                else None
            ),
        )

    def __getitems__(self, indices):
        """Let native DataLoader workers perform one vectorized mmap read."""

        return self[torch.as_tensor(indices, dtype=torch.long)]

    
    def get_experiment_dataset(self, experiment_idx):
        """
        Returns a new PtychoDataset instance containing only data from the specified experiment.
        This is used by reassembly.py to reconstruct a specific experiment from a dataloader whose memory map
        has multiple experiments saved to it.

        E.g. I have 3 experiments that I've loaded into the dataloader and want to reconstruct experiment 2 ONLY.
             Then reassembly.py will call get_experiment_dataset(2) to return a subset of the data.
        
        Parameters:
        -----------
        experiment_idx: int
            The experiment index to filter by
            
        Returns:
        --------
        PtychoDataset
            A new dataset instance with only the data from the specified experiment
        """
        # Create a shallow copy of the current dataset 
        import copy
        subset_dataset = copy.copy(self)
        
        # Find indices corresponding to the requested experiment
        mask = self.mmap_ptycho["experiment_id"][:] == experiment_idx
        indices = torch.where(mask)[0]
        
        if len(indices) == 0:
            raise ValueError(f"No data found for experiment_idx {experiment_idx}")
        
        # Create a filtered view of the memory-mapped TensorDict
        subset_dataset.mmap_ptycho = self.mmap_ptycho[indices]
        
        # Update length and cumulative length
        subset_dataset.length = len(indices)
        subset_dataset.cum_length = [0, subset_dataset.length]
        
        # Filter file list to only include the specified experiment
        subset_dataset.file_list = [self.file_list[experiment_idx]]
        subset_dataset.n_files = 1
        
        # Update data_dict to only include data for this experiment
        subset_dataset.data_dict = {
            "probes": self.data_dict["probes"][experiment_idx:experiment_idx+1],
            "probe_scaling": self.data_dict["probe_scaling"][experiment_idx:experiment_idx+1],
        }
        if self.ci_contract_active:
            subset_dataset.data_dict["probes_physical"] = self.data_dict[
                "probes_physical"
            ][experiment_idx:experiment_idx+1]
            subset_dataset.data_dict["ci_statistics"] = {
                name: value[experiment_idx:experiment_idx+1]
                for name, value in self.data_dict["ci_statistics"].items()
            }
        if "scaling_constant" in self.data_dict:
            subset_dataset.data_dict["scaling_constant"] = (
                self.data_dict["scaling_constant"][experiment_idx:experiment_idx+1])
        
        # Handle objectGuess if available
        if len(self.data_dict.get('objectGuess', [])) > experiment_idx:
            subset_dataset.data_dict["objectGuess"] = [self.data_dict["objectGuess"][experiment_idx]]
        else:
            subset_dataset.data_dict["objectGuess"] = []
        
        # Copy center of mass if available
        if "com" in self.data_dict:
            subset_dataset.data_dict["com"] = self.data_dict["com"]
        
        return subset_dataset
