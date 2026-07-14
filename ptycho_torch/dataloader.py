#Utility
import numpy as np
from pathlib import Path
import json
import zipfile
from collections import defaultdict
import time
import os
import shutil
import warnings
from dataclasses import replace

#Torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.distributed as dist

#Memory mapping
from tensordict import MemoryMappedTensor, TensorDict

#Patch generation
from ptycho_torch.patch_generator import group_coords, get_relative_coords, get_neighbor_indices, get_neighbors_indices_within_bounds

#Parameters
from ptycho_torch.config_params import TrainingConfig, DataConfig, ModelConfig
from ptycho_torch.npz_utils import read_npy_shape

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

# --- Helper functions for the dataloader ---
_DIFFRACTION_KEYS = ('diffraction', 'diff3d')
_MMAP_SCHEMA_NAME = "ptycho_torch_mmap"
_MMAP_SCHEMA_VERSION = 3
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
    "center_scan_id_available",
    "experiment_id",
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


def _select_diffraction_key(available_keys):
    """Select the first exact diffraction key in loader priority order."""
    available_keys = set(available_keys)
    return next((key for key in _DIFFRACTION_KEYS if key in available_keys), None)


def _resolve_neighbor_function(data_config):
    """Map the configured neighbor_function name onto its implementation.

    '4_quadrant' is passed through as a sentinel string: group_coords dispatches
    on the config value rather than calling it.
    """
    if data_config.neighbor_function == 'Nearest':
        return get_neighbor_indices
    if data_config.neighbor_function == 'Min_dist':
        return get_neighbors_indices_within_bounds
    return '4_quadrant' #This is the one used in PtychoPINNv2


def _normalize_legacy_grouping_records(file_list, valid_per_file, grouping_per_file):
    """Upgrade grouping records without inventing unavailable legacy centers."""
    if len(grouping_per_file) != len(valid_per_file):
        raise ValueError("legacy grouping records must align with valid-index files")
    normalized = []
    for index, record in enumerate(grouping_per_file):
        if record is None or len(record) == 4:
            normalized.append(record)
            continue
        if len(record) == 3:
            nn_indices, coords_nn, centers = record
            available = np.ones(len(nn_indices), dtype=np.bool_)
        elif len(record) == 2 and index < len(file_list):
            nn_indices, coords_nn = record
            centers = np.full(len(nn_indices), -1, dtype=np.int64)
            available = np.zeros(len(nn_indices), dtype=np.bool_)
        else:
            raise ValueError("legacy grouping record must contain two or three elements")
        normalized.append((nn_indices, coords_nn, centers, available))
    return normalized


def _align_coords_to_diffraction(xcoords, ycoords, n_diff, source):
    """Reconcile the scan-position count with the diffraction stack length.

    Some datasets carry trailing coordinate entries with no matching pattern.
    Those indices would run off the end of the diffraction stack once
    group_coords maps them back to global indices, so drop them. The reverse
    (fewer positions than patterns) means patterns with no position, which we
    refuse to guess at.
    """
    if np.ndim(xcoords) != 1 or np.ndim(ycoords) != 1:
        raise ValueError(
            f"{source}: xcoords shape {np.shape(xcoords)} and ycoords shape "
            f"{np.shape(ycoords)} must be one-dimensional."
        )

    xcoords_len = len(xcoords)
    ycoords_len = len(ycoords)
    if xcoords_len != ycoords_len:
        raise ValueError(
            f"{source}: xcoords={xcoords_len} and ycoords={ycoords_len} "
            "must have equal lengths."
        )

    n_coords = xcoords_len
    if n_coords == n_diff:
        return xcoords, ycoords
    if n_coords < n_diff:
        raise ValueError(
            f"{source}: {n_coords} scan positions for {n_diff} diffraction patterns. "
            f"Every pattern needs a position."
        )
    warnings.warn(
        f"{source}: {n_coords} scan positions for {n_diff} diffraction patterns; "
        f"dropping the trailing {n_coords - n_diff} positions.",
        RuntimeWarning, stacklevel=2,
    )
    return xcoords[:n_diff], ycoords[:n_diff]


def _canonical_diffraction_layout(shape, n_coords, source):
    """Return canonical (N, H, W) shape and whether the source needs transposing."""
    if len(shape) != 3:
        raise ValueError(
            f"{source}: diffraction data must be 3D (N, H, W) or legacy "
            f"(H, W, N); got shape {shape}."
        )

    canonical_square = shape[1] == shape[2]
    legacy_square = shape[0] == shape[1]
    canonical_match = n_coords is not None and shape[0] == n_coords
    legacy_match = n_coords is not None and shape[2] == n_coords
    if canonical_square != legacy_square:
        legacy_hwn = legacy_square
    elif canonical_match != legacy_match:
        legacy_hwn = legacy_match
    elif n_coords is not None:
        # Coordinates may include trailing entries with no matching pattern.
        canonical_compatible = shape[0] < n_coords
        legacy_compatible = shape[2] < n_coords
        if canonical_compatible != legacy_compatible:
            legacy_hwn = legacy_compatible
        else:
            legacy_hwn = shape[2] > max(shape[0], shape[1])
    else:
        legacy_hwn = shape[2] > max(shape[0], shape[1])

    if legacy_hwn:
        return (shape[2], shape[0], shape[1]), True
    return tuple(shape), False


def npz_headers(npz):
    """
    Takes a path to an .npz file, which is a Zip archive of .npy files.
    We can use this to determine shape of the scan tensor in the npz file without loading it
    This will be useful in the __len__ method for the dataset

    Checks the 'diffraction' key first (accepted alias here; canonically the H5
    /raw_data dataset name, specs/data_contracts.md), falling back to 'diff3d',
    the standalone-NPZ canonical key (docs/specs/spec-ptycho-core.md).

    Taken from: https://stackoverflow.com/questions/68224572/how-to-determine-the-shape-size-of-npz-file
    Modified to quickly grab dimension we care about
    """
    with zipfile.ZipFile(npz) as archive:
        xcoords = None
        ycoords = None

        archive_keys = (
            name[:-4] for name in archive.namelist() if name.endswith('.npy'))
        diffraction_key = _select_diffraction_key(archive_keys)
        if diffraction_key is None:
            raise ValueError(
                f"Could not find diffraction data in {npz}. "
                f"Expected standalone-NPZ 'diff3d' key or compatibility alias 'diffraction'. "
                f"See docs/specs/spec-ptycho-core.md for required NPZ format."
            )
        with archive.open(f'{diffraction_key}.npy') as npy:
            diffraction_shape = read_npy_shape(npy)

        # Second pass for coordinates (load them) - needed for filtering
        with np.load(npz) as data:
            if 'xcoords' in data and 'ycoords' in data:
                xcoords = data['xcoords']
                ycoords = data['ycoords']
            else:
                raise ValueError(f"Could not find 'xcoords' or 'ycoords' in {npz}")

        n_coords = (
            len(xcoords)
            if np.ndim(xcoords) == np.ndim(ycoords) == 1 and len(xcoords) == len(ycoords)
            else None
        )
        diffraction_shape, _ = _canonical_diffraction_layout(
            diffraction_shape, n_coords, npz)

        # The bounds mask and the memory-map allocation are both derived from these
        # coordinates, so they must agree with the diffraction stack length here,
        # not just at the later indexing site in memory_map_data.
        xcoords, ycoords = _align_coords_to_diffraction(
            xcoords, ycoords, diffraction_shape[0], f"{npz}")

        return diffraction_shape, xcoords, ycoords


def _validate_writer_inputs(npz_file, tensor_shape, model_config, data_config):
    """Reject writer-required NPZ inputs before memory-map allocation."""
    required_keys = ["probeGuess", "objectGuess"]
    if model_config.mode == "Supervised":
        required_keys.append("label")

    with np.load(npz_file) as data:
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(
                f"{npz_file}: missing required key(s): {', '.join(missing_keys)}."
            )

        probe_shape = data["probeGuess"].shape
        object_shape = data["objectGuess"].shape
        label_shape = data["label"].shape if model_config.mode == "Supervised" else None

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
    """
    Helper to load diffraction stack from NPZ, accepting either key name,
    with automatic legacy format handling.

    Checks 'diffraction' first (accepted alias here; canonically the H5 /raw_data
    dataset name, specs/data_contracts.md), falling back to 'diff3d', the
    standalone-NPZ canonical key (docs/specs/spec-ptycho-core.md). Automatically
    detects and transposes legacy (H, W, N) format to the compliant (N, H, W) format.

    Args:
        npz_file: Path to NPZ file

    Returns:
        numpy.ndarray: Diffraction patterns (amplitude, float32) in shape (N, H, W)

    Raises:
        ValueError: If neither key exists
    """
    with np.load(npz_file) as data:
        diffraction_key = _select_diffraction_key(data.files)
        if diffraction_key is None:
            raise ValueError(
                f"Could not find diffraction data in {npz_file}. "
                f"Expected standalone-NPZ 'diff3d' key or compatibility alias 'diffraction'. "
                f"See docs/specs/spec-ptycho-core.md for required NPZ format."
            )
        diff_array = data[diffraction_key]

        n_coords = None
        if 'xcoords' in data and 'ycoords' in data:
            xcoords = data['xcoords']
            ycoords = data['ycoords']
            if (np.ndim(xcoords) == np.ndim(ycoords) == 1 and
                    len(xcoords) == len(ycoords)):
                n_coords = len(xcoords)

        _, legacy_hwn = _canonical_diffraction_layout(
            diff_array.shape, n_coords, npz_file)
        if legacy_hwn:
            print(
                f"⚠ Legacy format {diff_array.shape} detected in {npz_file}, "
                f"transposing to DATA-001 compliant (N, H, W)"
            )
            diff_array = np.transpose(diff_array, [2, 0, 1])

        return diff_array


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
    def __init__(self, ptycho_dir: str, model_config: 'ModelConfig', data_config: 'DataConfig',
                 training_config: 'TrainingConfig' = None,
                 data_dir: str = 'data/memmap', remake_map: bool = False,
                 defer_ci_statistics: bool = False):
        
        # --- Initial loading ---
        self.model_config = model_config
        self.data_config = data_config
        self.ci_contract_active = _ci_profile_active(model_config, data_config)
        self.defer_ci_statistics = defer_ci_statistics
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
            length_result = self.calculate_length()
            if len(length_result) == 5:
                (self.length, self.im_shape, self.cum_length,
                 self.valid_indices_per_file,
                 self.grouping_per_file) = length_result
                self.source_indices_per_file = [
                    np.arange(npz_headers(path)[0][0], dtype=np.int64)
                    for path in self.file_list
                ]
                self.grouping_per_file = _normalize_legacy_grouping_records(
                    self.file_list,
                    self.valid_indices_per_file,
                    self.grouping_per_file,
                )
            else:
                (self.length, self.im_shape, self.cum_length,
                 self.valid_indices_per_file, self.source_indices_per_file,
                 self.grouping_per_file) = length_result
            if self.length == 0:
                 raise ValueError(
                     f"[Rank {self.current_rank}] calculate_length() resulted in 0 items. "
                     "Cannot proceed."
                 )
            # Group counts are deterministic (RNG only picks which neighbor fills a
            # quadrant, never how many groups survive), so every rank agrees on
            # self.length. Only rank 0 writes the map, so the rest can drop the
            # cached grouping arrays rather than hold them for the process lifetime.
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

        When coordinate grouping applies, the groups are built here rather than
        estimated: group_coords can return fewer groups than there are valid
        points (a '4_quadrant' center whose quadrants are not all populated is
        discarded), so `n_valid_points * n_subsample` overcounts. Grouping once
        and caching it keeps the allocation, cum_length, and the tensors written
        by memory_map_data exactly consistent -- and means the grouping is not
        recomputed with different random draws on the write pass.
        """
        total_length = 0
        cumulative_length = [0]
        first_im_shape = None
        valid_indices_per_file = [] # Store valid indices for each file
        source_indices_per_file = [] # Every source diffraction index before bounds
        grouping_per_file = [] # (nn_indices, coords_nn) per file when grouping applies, else None

        group_coordinates = self.group_coords_enabled()
        neighbor_function = _resolve_neighbor_function(self.data_config)

        print("Calculating dataset length with coordinate bounds...")
        # Make sure bounds are valid
        if not (0.0 <= self.data_config.x_bounds[0] < self.data_config.x_bounds[1] <= 1.0):
            raise ValueError(f"Invalid x_bounds: {self.data_config.x_bounds}. Must be [min_pct, max_pct] between 0.0 and 1.0.")
        if not (0.0 <= self.data_config.y_bounds[0] < self.data_config.y_bounds[1] <= 1.0):
             raise ValueError(f"Invalid y_bounds: {self.data_config.y_bounds}. Must be [min_pct, max_pct] between 0.0 and 1.0.")

        for i, npz_file in enumerate(self.file_list): # Use ordered list
            tensor_shape, xcoords, ycoords = npz_headers(npz_file)

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

            # Build the coordinate groups now so the length is the true group count.
            # n_subsample is applied inside group_coords, so it is not multiplied in here.
            if group_coordinates and n_valid_points > 0:
                nn_indices, coords_nn, center_indices = group_coords(
                    xcoords, ycoords,
                    xcoords[valid_indices], ycoords[valid_indices],
                    neighbor_function,
                    valid_indices,
                    self.data_config, C=self.data_config.C,
                    return_center_indices=True)
                nn_indices = nn_indices.astype(np.int64)
                grouping_per_file.append(
                    (
                        nn_indices,
                        coords_nn,
                        center_indices,
                        np.ones(len(nn_indices), dtype=np.bool_),
                    )
                )
                length_contribution = len(nn_indices)
                if length_contribution != n_valid_points * self.data_config.n_subsample:
                    print(f"  {npz_file}: grouping kept {length_contribution} of "
                          f"{n_valid_points * self.data_config.n_subsample} candidate groups "
                          f"({n_valid_points} valid points x {self.data_config.n_subsample} subsamples).")
            else:
                grouping_per_file.append(None)
                length_contribution = n_valid_points

            total_length += length_contribution
            cumulative_length.append(total_length)

        if first_im_shape is None:
             raise ValueError("Could not determine image shape from any NPZ file.")

        return (total_length, first_im_shape, cumulative_length,
                valid_indices_per_file, source_indices_per_file, grouping_per_file)

    def group_coords_enabled(self):
        """Whether memory_map_data groups coordinates into solution regions.

        Mirrors the branch condition in memory_map_data exactly: calculate_length
        must size the memory map for the same branch that writes it.
        """
        return self.model_config.mode == 'Unsupervised' and self.model_config.object_big

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
        ):
            if manifest.get(field) != expected[field]:
                raise self._mmap_rebuild_error(
                    f"manifest {field}={manifest.get(field)!r}, "
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
    def from_existing_map(cls, map_path, model_config, data_config, current_rank = 0, is_ddp_active = False):
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
        1.  Solves for solution patch indices using group_coords method
        2.  Writes to respective memory maps. The diffraction map is populated in batches, while the other maps
        are populated in full for every individual dataset
            - "images" - (N x C x H x W), N = # of patterns, C = # of images per soln patch, H = height, W = width
            - "coords_offsets" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
            - "coords_relative" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
            - "coords_start_offsets" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
            - "coords_start_relative" - (N x C x 1 x 2), N = # of patterns, C = # of images per soln patch, 2 = x,y
            - "nn_indices" - (N, C) , N = # of patterns, C = # of images per soln patch, gives indices of each coord group
            - "experiment_id" - N, N = # of patterns, gives association to specific npz/experiment file

        Note: Probe/object stored in the data_dict, not in the memory map.
        ---
        Args:
            image_paths - list of paths to independent experiment npz files
            grid_size - tuple of image grid size (e.g. 2 x 2 is most used)

        """
        #Config grabbing/setting using stored configs
        if self.model_config.object_big:
            n_channels = self.data_config.grid_size[0] * self.data_config.grid_size[1]
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
                "center_scan_id_available": MemoryMappedTensor.empty(
                    (mmap_length), dtype=torch.bool
                ),
                "experiment_id": MemoryMappedTensor.empty(
                    (mmap_length),
                    dtype=torch.int32
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
            p_shape = np.load(npz_file)['probeGuess'].shape
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
            (self.data_config.normalize == 'Group' and self.data_config.C == 1 and
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

            # Load the canonical stack before coordinating alignment so legacy
            # (H, W, N) datasets use their pattern count rather than raw axis 0.
            diff_stack = torch.from_numpy(_get_diffraction_stack(npz_file)).to(torch.float32)
            n_diff = diff_stack.shape[0]
            with np.load(npz_file) as npz_data:
                xcoords_full = npz_data['xcoords']
                ycoords_full = npz_data['ycoords']
            with warnings.catch_warnings():
                # calculate_length already warned about this file via npz_headers
                warnings.simplefilter("ignore", RuntimeWarning)
                xcoords_full, ycoords_full = _align_coords_to_diffraction(
                    xcoords_full, ycoords_full, n_diff, f"{npz_file}")

            #Apply coordinate filter to remove edge points based on self.calculate_length
            xcoords = xcoords_full[self.valid_indices_per_file[i]]
            ycoords = ycoords_full[self.valid_indices_per_file[i]]
            self.data_dict['com'] = torch.from_numpy(np.array([xcoords.mean(), ycoords.mean()])) #Center of mass (see reassembly.py)

            #--- Coordinate patches/Supervised Labels ---
            # Note that object_big = True means we are enforcing ptychographic constraints and need to group coordinates
            if self.group_coords_enabled(): # PtychoPINN/Ptychography Constraint
                #Reuse the grouping built in calculate_length: regrouping here would
                #redraw the random candidate/subsample picks and desync from cum_length
                (nn_indices, coords_nn, center_indices,
                 center_indices_available) = self.grouping_per_file[i]

                #Get relative and center of mass coordinates for each coordinate group
                coords_com, coords_relative = get_relative_coords(coords_nn)
                mmap_ptycho["coords_center"][start:end] = torch.from_numpy(coords_com)
                mmap_ptycho["coords_relative"][start:end] = torch.from_numpy(coords_relative)
                mmap_ptycho["nn_indices"][start:end] = torch.from_numpy(nn_indices)
                mmap_ptycho["center_scan_id"][start:end] = torch.from_numpy(
                    center_indices
                )
                mmap_ptycho["center_scan_id_available"][start:end] = torch.from_numpy(
                    center_indices_available
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
                mmap_ptycho["center_scan_id_available"][start:end] = True
                mmap_ptycho["coords_global"][start:end] = torch.from_numpy(
                                                            np.stack([xcoords,
                                                            ycoords],axis=1)[:, None, None, :]).to(torch.float32)
                
                #Add labels if supervised model is selected
                if self.model_config.mode == 'Supervised':
                    print("Assigning labels...")
                    #Only grab valid labels which were calculated before. Validity based on coordinates
                    valid_labels = np.load(npz_file)['label'][nn_indices][:,None,:,:] # Channel dimension added for consistency, size = 1
                    
                    #Do phase correction based on prior PtychoNN conventions
                    objectGuess = np.load(npz_file)['objectGuess']
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

            #Mapping experiment Ids
            mmap_ptycho["experiment_id"][start:end] = torch.tensor(i)

            #Mapping probes
            probe_data = np.load(npz_file)['probeGuess']
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
            objectGuess = np.load(npz_file)['objectGuess']
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
            if not self.group_coords_enabled():
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
                        self.data_config.normalize == 'Group' and self.data_config.C > 1):
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

    @classmethod
    def from_np(cls,
                diff_patterns: np.ndarray,
                probe: np.ndarray,
                positions: np.ndarray,
                model_config: 'ModelConfig',
                data_config: 'DataConfig',
                scaling_constant: float = None) -> 'PtychoDataset':
        """
        Create a PtychoDataset directly from in-memory numpy arrays, bypassing
        NPZ file I/O and the on-disk memory map. Intended for inference
        workflows where data arrives as numpy arrays. Unsupervised mode only
        (no label arrays are built).

        Parameters
        ----------
        diff_patterns : np.ndarray, shape (N, H, W), float amplitude.
        probe : np.ndarray, shape (H, W) single mode or (P, H, W) multi-mode.
        positions : np.ndarray, shape (N, 2), [ypix, xpix] per row.
        model_config : ModelConfig
        data_config : DataConfig
        scaling_constant : float, optional override for RMS normalization constant.

        Returns
        -------
        PtychoDataset ready-to-use with an in-memory TensorDict.
        """
        import warnings

        if model_config.mode == 'Supervised':
            raise ValueError("from_np supports Unsupervised mode only: it builds "
                             "no label_amp/label_phase arrays.")
        if diff_patterns.ndim != 3:
            raise ValueError(f"diff_patterns must be 3D (N, H, W), got shape {diff_patterns.shape}")
        if probe.ndim not in (2, 3):
            raise ValueError(f"probe must be 2D (H, W) or 3D (P, H, W), got shape {probe.shape}")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"positions must be (N, 2), got shape {positions.shape}")

        N_total, H, W = diff_patterns.shape
        if positions.shape[0] != N_total:
            raise ValueError(f"positions length {positions.shape[0]} != diff_patterns length {N_total}")
        if probe.ndim == 2 and probe.shape != (H, W):
            raise ValueError(f"probe shape {probe.shape} != pattern shape ({H}, {W})")
        if probe.ndim == 3 and probe.shape[1:] != (H, W):
            raise ValueError(f"probe spatial shape {probe.shape[1:]} != pattern shape ({H}, {W})")
        if data_config.N != H or data_config.N != W:
            warnings.warn(f"data_config.N={data_config.N} does not match pattern size ({H}, {W})")

        # Create instance, set attributes
        dataset = cls.__new__(cls)
        dataset.model_config = model_config
        dataset.data_config = data_config
        dataset.ci_contract_active = _ci_profile_active(model_config, data_config)
        dataset.is_ddp_active = False
        dataset.current_rank = 0
        dataset.ptycho_dir = None
        dataset.file_list = [None]
        dataset.n_files = 1
        dataset.data_dict = {}
        dataset.im_shape = (H, W)

        # Extract coordinates and apply bounds filtering
        xcoords_full = positions[:, 1].astype(np.float32)
        ycoords_full = positions[:, 0].astype(np.float32)

        xmin, xmax = xcoords_full.min(), xcoords_full.max()
        ymin, ymax = ycoords_full.min(), ycoords_full.max()
        x_range = (xmax - xmin) if xmax > xmin else 1.0
        y_range = (ymax - ymin) if ymax > ymin else 1.0

        x_lower = xmin + data_config.x_bounds[0] * x_range
        x_upper = xmin + data_config.x_bounds[1] * x_range
        y_lower = ymin + data_config.y_bounds[0] * y_range
        y_upper = ymin + data_config.y_bounds[1] * y_range

        if xmax <= xmin:
            x_upper = x_lower
        if ymax <= ymin:
            y_upper = y_lower

        mask = ((xcoords_full >= x_lower) & (xcoords_full <= x_upper) &
                (ycoords_full >= y_lower) & (ycoords_full <= y_upper))
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            raise ValueError("No positions remain after bounds filtering. Check x_bounds/y_bounds.")

        xcoords = xcoords_full[valid_indices]
        ycoords = ycoords_full[valid_indices]

        dataset.data_dict['com'] = torch.from_numpy(
            np.array([xcoords.mean(), ycoords.mean()]))

        # Coordinate grouping (get_relative_coords carries the TF-parity
        # local_offset_sign=-1 convention at the source)
        if model_config.object_big:
            n_channels = data_config.C

            nn_indices, coords_nn, center_indices = group_coords(
                xcoords_full, ycoords_full,
                xcoords, ycoords,
                _resolve_neighbor_function(data_config),
                valid_indices,
                data_config, C=data_config.C, return_center_indices=True)
            nn_indices = nn_indices.astype(np.int64)

            coords_com, coords_relative = get_relative_coords(coords_nn)

            regular_global_coords = torch.from_numpy(
                np.stack([xcoords_full, ycoords_full], axis=1)).to(torch.float32)
            coords_global = regular_global_coords[nn_indices].unsqueeze(2)

            N_groups = len(nn_indices)
        else:
            n_channels = 1
            nn_indices = valid_indices
            center_indices = valid_indices
            N_groups = len(valid_indices)

            coords_global = torch.from_numpy(
                np.stack([xcoords, ycoords], axis=1)[:, None, None, :]).to(torch.float32)
            coords_com = np.zeros((N_groups, 1, 1, 2), dtype=np.float32)
            coords_relative = np.zeros((N_groups, n_channels, 1, 2), dtype=np.float32)

        dataset.length = N_groups
        dataset.cum_length = [0, N_groups]
        dataset.valid_indices_per_file = [valid_indices]
        dataset.source_indices_per_file = [
            np.arange(len(diff_patterns), dtype=np.int64)
        ]

        # Process probe: single-mode (H, W) or incoherent multi-mode (P, H, W)
        probe_data = probe.copy()
        if probe_data.ndim == 3 and probe_data.shape[-1] == 1:
            probe_data = probe_data[..., 0]
        probe_physical = np.ascontiguousarray(
            probe_data[None] if probe_data.ndim == 2 else probe_data
        )

        if data_config.probe_normalize:
            probe_data, probe_sf = hh.normalize_probe_like_tf(
                probe_data,
                probe_scale=data_config.probe_scale,
                probe_mask=getattr(model_config, "probe_mask", False),
                probe_mask_tensor=getattr(model_config, "probe_mask_tensor", None),
                probe_mask_sigma=getattr(model_config, "probe_mask_sigma", 1.0),
                probe_mask_diameter=getattr(model_config, "probe_mask_diameter", None),
            )
            probe_sf = float(probe_sf)
        else:
            probe_sf = 1.0

        if probe_data.ndim == 2:
            probe_data = np.expand_dims(probe_data, axis=0)

        dataset.data_dict['probes'] = torch.from_numpy(np.ascontiguousarray(probe_data)).to(torch.complex64).unsqueeze(0)
        if dataset.ci_contract_active:
            dataset.data_dict['probes_physical'] = torch.from_numpy(
                probe_physical
            ).to(torch.complex64).unsqueeze(0)
        dataset.data_dict['probe_scaling'] = torch.tensor([probe_sf], dtype=torch.float32)
        dataset.data_dict['objectGuess'] = []

        # Construct the grouped images before deriving Group factors.
        diff_tensor = torch.from_numpy(diff_patterns).to(torch.float32)
        if model_config.object_big:
            images = diff_tensor[nn_indices]
            nn_indices_tensor = torch.from_numpy(nn_indices)
        else:
            images = diff_tensor[nn_indices][:, None]
            nn_indices_tensor = torch.arange(N_groups, dtype=torch.int64)[:, None]

        effective_batch_normalization = (
            data_config.normalize == 'Batch' or
            (data_config.normalize == 'Group' and data_config.C == 1 and
             model_config.mode != 'Supervised')
        )
        if dataset.ci_contract_active:
            rms_factors = None
            physics_factors = None
        elif data_config.normalize == 'None':
            rms_factors = torch.ones(N_groups, 1, 1, 1, dtype=torch.float32)
            physics_factors = torch.ones(N_groups, 1, 1, 1, dtype=torch.float32)
        elif effective_batch_normalization:
            factor_config = (
                data_config if data_config.normalize == 'Batch'
                else replace(data_config, normalize='Batch')
            )
            rms_factors = hh.get_rms_scaling_factor(diff_tensor, factor_config).expand(
                N_groups, 1, 1, 1).clone()
            physics_factors = hh.get_physics_scaling_factor(diff_tensor, factor_config).expand(
                N_groups, 1, 1, 1).clone()
        elif data_config.normalize == 'Group' and data_config.C > 1:
            rms_factors = hh.get_rms_scaling_factor(images, data_config)
            physics_factors = hh.get_physics_scaling_factor(images, data_config)
        else:
            raise ValueError(f"Unsupported normalization mode: {data_config.normalize}")

        if dataset.ci_contract_active and scaling_constant is not None:
            raise ValueError(
                "CI batches use rms_input_scale and do not accept the legacy "
                "scaling_constant override."
            )
        if not dataset.ci_contract_active:
            if scaling_constant is not None:
                rms_factors = torch.full_like(
                    rms_factors,
                    float(scaling_constant),
                )
                dataset.data_dict['scaling_constant'] = torch.tensor(
                    [scaling_constant], dtype=torch.float32)
            elif data_config.normalize == 'None':
                dataset.data_dict['scaling_constant'] = torch.tensor(
                    [1.0], dtype=torch.float32)
            elif effective_batch_normalization:
                dataset.data_dict['scaling_constant'] = (
                    rms_factors[0].reshape(1).clone()
                )

        td_fields = {
            "images": images,
            "coords_global": coords_global,
            "coords_center": torch.from_numpy(coords_com).to(torch.float32),
            "coords_relative": torch.from_numpy(coords_relative).to(torch.float32),
            "coords_start_center": torch.zeros(N_groups, 1, 1, 2, dtype=torch.float32),
            "coords_start_relative": torch.zeros(N_groups, n_channels, 1, 2, dtype=torch.float32),
            "nn_indices": nn_indices_tensor,
            "center_scan_id": torch.from_numpy(
                np.asarray(center_indices, dtype=np.int64)
            ),
            "center_scan_id_available": torch.ones(
                N_groups, dtype=torch.bool
            ),
            "experiment_id": torch.zeros(N_groups, dtype=torch.int32),
        }
        if dataset.ci_contract_active:
            td_fields.update({
                "measured_intensity": images.clone(),
            })
        else:
            td_fields.update({
                "rms_scaling_constant": rms_factors,
                "physics_scaling_constant": physics_factors,
            })
        td = TensorDict(td_fields, batch_size=N_groups)

        dataset.mmap_ptycho = td
        if dataset.ci_contract_active:
            dataset.set_ci_statistics_from_indices(torch.arange(dataset.length))

        print(f"[PtychoDataset.from_np] Created in-memory dataset with {N_groups} groups, "
              f"{n_channels} channels, image shape {(H, W)}")

        return dataset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns memory mapped tensordict, alongside probe. Written to be batched
        so you can return multiple instances.

        Probe dimensionality is expanded to match the data channels. This is so multiplication operations are broadcast correctly.
        
        Output
        -------
        self.mmap_ptych[idx] - Batched TensorDict containing all relevant information for training. See
            function memory_map_data for further details. Length is batch size
        probes_indexed - (B,C,P,N,N) tensor, where B is batch size, C is number of channels (probe duplicated
            across channels via unsqueeze+expand), P is the number of probe modes, and N,N are the height and
            width of the diffraction pattern. The dimensionality should be exactly the same as the output of the autoencoder.
        scaling_constant - (N) tensor, scaling constants required for each diffraction image
        
        """
        #Experimental index is used to find the probe corresponding to the right experiment
        #We can then get the correct probe tensor organized according to diffraction patterns
        exp_idx = torch.as_tensor(self.mmap_ptycho['experiment_id'][idx])
        is_scalar = exp_idx.ndim == 0
        exp_idx_batch = exp_idx.reshape(-1).to(dtype=torch.long)

        if self.model_config.object_big: # Use stored config
            channels = self.data_config.C # Use stored config
        else:
            channels = 1

        if self.n_files > 1:
            get_idx = exp_idx_batch
        else:
            get_idx = torch.zeros_like(exp_idx_batch)
        # Expand probe to match number of channels for data.
        probes_indexed = self.data_dict['probes'][get_idx].unsqueeze(1).expand(
            -1, channels, -1, -1, -1)
        probe_scaling = self.data_dict['probe_scaling'][get_idx].view(-1, 1, 1, 1)

        tensor_dict = self.mmap_ptycho[idx]
        if self.ci_contract_active:
            probes_physical = self.data_dict['probes_physical'][get_idx].unsqueeze(
                1
            ).expand(-1, channels, -1, -1, -1)
            probe_normalization = probe_scaling.unsqueeze(-1)
            statistics = self.data_dict["ci_statistics"]
            rms_input_scale = statistics["rms_input_scale"][get_idx].view(
                -1, 1, 1, 1
            )
            mean_measured_intensity = statistics[
                "mean_measured_intensity"
            ][get_idx].view(-1, 1, 1, 1)
            tensor_dict = TensorDict(
                {key: value for key, value in tensor_dict.items()},
                batch_size=tensor_dict.batch_size,
            )
            tensor_dict["probe_training"] = probes_indexed
            tensor_dict["probe_physical"] = probes_physical
            tensor_dict["probe_normalization"] = probe_normalization
            if is_scalar:
                tensor_dict["rms_input_scale"] = rms_input_scale[0]
                tensor_dict["mean_measured_intensity"] = (
                    mean_measured_intensity[0]
                )
            else:
                tensor_dict["rms_input_scale"] = rms_input_scale
                tensor_dict["mean_measured_intensity"] = (
                    mean_measured_intensity
                )

        if is_scalar:
            probes_indexed = probes_indexed[0]
            probe_scaling = probe_scaling[0]

        return tensor_dict, probes_indexed, probe_scaling

    
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
        
#Collation

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
    '''
    Modifiers dataloader class that allows for batch sampling exploiting the structure of TensorDicts
    Given a set of indices, we can directly index all of them simultaneously from the TensorDict instead of calling
    yield on a single index at a time.

    This allows us to return a TensorDict object which already has indexing built in.
    '''
    def __iter__(self):
        #Iterator over sampler
        batch_sampler = self.batch_sampler
        dataset = self.dataset
        collate_fn = self.collate_fn

        for batch_indices in batch_sampler:
            batch = dataset[batch_indices]
            if collate_fn is not None:
                batch = collate_fn(batch)
            yield batch


#Custom collation function which pins memory in order to transfer to gpu
#Taken from: https://pytorch.org/tensordict/stable/tutorials/tensorclass_imagenet.html
class Collate(nn.Module):
    """
    Classic data collation function that works with native pytorch training protocol.
    One gpu only.
    """
    def __init__(self, device = None):
        super().__init__()
        self.device = torch.device(device)
    def __call__(self, x):
        '''
        Moves tensor to RAM, and then to GPU.

        Inputs
        -------
        x: TensorDict
        '''
        tensor_dict, probe, scaling = x
        outputs = [
            _materialize_expanded_tensordict(tensor_dict),
            _materialize_expanded_tensor(probe),
            _materialize_expanded_tensor(scaling),
        ]
        
        # Pin memory if using CUDA
        if self.device and self.device.type == 'cuda':
            outputs = [item.pin_memory() for item in outputs]
            
        # Move to device if specified
        if self.device:
            outputs = [item.to(self.device) for item in outputs]
            
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
        """
        Prep batch. Lightning calls the device transfer
        """
        tensor_dict, probe, scaling = x
        outputs = [tensor_dict, probe.clone(), scaling.clone()]

        if self.pin_memory_if_cuda and torch.cuda.is_available():
            try:
                if hasattr(outputs[0], 'pin_memory'):
                    outputs[0] = outputs[0].pin_memory() #Try calling tensordict native method
                else:
                    for key in enumerate(outputs[0].keys()):
                        if isinstance(outputs[0][key], torch.Tensor):
                            outputs[0][key] = outputs[0][key].pin_memory()
                outputs[1] = outputs[1].pin_memory()
                outputs[2] = outputs[2].pin_memory()
            except Exception as e:
                print(f"Warning: Collate failed to pin memory: {e}")


        return tuple(outputs)
