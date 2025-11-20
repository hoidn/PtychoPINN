#Utility
import numpy as np
from pathlib import Path
import zipfile
from collections import defaultdict
import time
import os
import shutil

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

#Helper methods
import ptycho_torch.helper as hh

# --- Helper functions for the dataloader ---
def npz_headers(npz):
    """
    Takes a path to an .npz file, which is a Zip archive of .npy files.
    We can use this to determine shape of the scan tensor in the npz file without loading it
    This will be useful in the __len__ method for the dataset

    Prefers canonical 'diffraction' key per DATA-001 spec (specs/data_contracts.md),
    with graceful fallback to legacy 'diff3d' key for backward compatibility.

    Taken from: https://stackoverflow.com/questions/68224572/how-to-determine-the-shape-size-of-npz-file
    Modified to quickly grab dimension we care about
    """
    with zipfile.ZipFile(npz) as archive:
        npy_header_found = False
        diffraction_shape = None
        xcoords = None
        ycoords = None

        # First pass: try canonical 'diffraction' key (DATA-001 spec)
        for name in archive.namelist():
            if name.startswith('diffraction') and name.endswith('.npy'):
                npy = archive.open(name)
                version = np.lib.format.read_magic(npy)
                shape, _, _ = np.lib.format._read_array_header(npy, version)
                diffraction_shape = shape
                npy_header_found = True
                break

        # Fallback: try legacy 'diff3d' key for backward compatibility
        if not npy_header_found:
            for name in archive.namelist():
                if name.startswith('diff3d') and name.endswith('.npy'):
                    npy = archive.open(name)
                    version = np.lib.format.read_magic(npy)
                    shape, _, _ = np.lib.format._read_array_header(npy, version)
                    diffraction_shape = shape
                    npy_header_found = True
                    break

        if not npy_header_found:
             raise ValueError(
                 f"Could not find diffraction data in {npz}. "
                 f"Expected canonical 'diffraction' key or legacy 'diff3d' key. "
                 f"See specs/data_contracts.md for required NPZ format."
             )

        # Auto-detect and fix legacy (H, W, N) format
        # This MUST match the transpose logic in _get_diffraction_stack()
        # so memory maps are allocated with correct dimensions
        if len(diffraction_shape) == 3:
            if diffraction_shape[2] > max(diffraction_shape[0], diffraction_shape[1]):
                # Legacy format detected: transpose (H, W, N) → (N, H, W)
                diffraction_shape = (diffraction_shape[2], diffraction_shape[0], diffraction_shape[1])

        # Second pass for coordinates (load them) - needed for filtering
        with np.load(npz) as data:
            if 'xcoords' in data and 'ycoords' in data:
                xcoords = data['xcoords']
                ycoords = data['ycoords']
            else:
                raise ValueError(f"Could not find 'xcoords' or 'ycoords' in {npz}")

        return diffraction_shape, xcoords, ycoords


def _get_diffraction_stack(npz_file):
    """
    Helper to load diffraction stack from NPZ with canonical key preference
    and automatic legacy format handling.

    Prefers canonical 'diffraction' key per DATA-001 spec, with fallback to
    legacy 'diff3d' key for backward compatibility. Automatically detects and
    transposes legacy (H, W, N) format to DATA-001 compliant (N, H, W) format.

    Args:
        npz_file: Path to NPZ file

    Returns:
        numpy.ndarray: Diffraction patterns (amplitude, float32) in shape (N, H, W)

    Raises:
        ValueError: If neither canonical nor legacy key exists
    """
    with np.load(npz_file) as data:
        # Try canonical key first (DATA-001 spec)
        if 'diffraction' in data:
            diff_array = data['diffraction']
        # Fallback to legacy key
        elif 'diff3d' in data:
            diff_array = data['diff3d']
        else:
            raise ValueError(
                f"Could not find diffraction data in {npz_file}. "
                f"Expected canonical 'diffraction' key or legacy 'diff3d' key. "
                f"See specs/data_contracts.md for required NPZ format."
            )

        # Auto-detect and fix legacy (H, W, N) format
        # DATA-001 requires (N, H, W) where N is typically >> H,W (e.g., 1087 vs 64)
        if len(diff_array.shape) == 3:
            # Heuristic: if last dim is much larger than first two, assume legacy format
            if diff_array.shape[2] > max(diff_array.shape[0], diff_array.shape[1]):
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
                 data_dir: str = 'data/memmap', remake_map: bool = False):
        
        # --- Initial loading ---
        self.model_config = model_config
        self.data_config = data_config
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
            self.length, self.im_shape, self.cum_length, self.valid_indices_per_file = self.calculate_length()
            if self.length == 0 and self.current_rank == 0:
                 raise ValueError(f"[Rank 0] calculate_length() resulted in 0 items. Cannot proceed.")
        except Exception as e:
            if self.current_rank == 0:
                print(f"[Rank 0] ERROR in calculate_length(): {e}")
                raise

        # --- Coordinated Memory Map Creation/Loading (Multi-GPU, Rank 0 orchestrates) ---
        # This is set up so the memory map is ONLY created from Rank 0 and isn't duplicated. All ranks 
        # (i.e. GPUs) will access the same memory map that was initialized by Rank 0.
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
            self.mmap_ptycho = TensorDict.load_memmap(str(self.data_dir_path)) # Load memory map that was initialized by Rank 0
            loaded_state = np.load(self.state_path, allow_pickle=True)
            self.data_dict = loaded_state['data_dict'].item()

        except Exception as e:
            print(f"[Rank {self.current_rank}] FATAL ERROR loading map files or state AFTER barrier: {e}")
            raise
        
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
        """
        total_length = 0
        cumulative_length = [0]
        first_im_shape = None
        valid_indices_per_file = [] # Store valid indices for each file

        print("Calculating dataset length with coordinate bounds...")
        # Make sure bounds are valid
        if not (0.0 <= self.data_config.x_bounds[0] < self.data_config.x_bounds[1] <= 1.0):
            raise ValueError(f"Invalid x_bounds: {self.data_config.x_bounds}. Must be [min_pct, max_pct] between 0.0 and 1.0.")
        if not (0.0 <= self.data_config.y_bounds[0] < self.data_config.y_bounds[1] <= 1.0):
             raise ValueError(f"Invalid y_bounds: {self.data_config.y_bounds}. Must be [min_pct, max_pct] between 0.0 and 1.0.")

        for i, npz_file in enumerate(self.file_list): # Use ordered list
            try:
                tensor_shape, xcoords, ycoords = npz_headers(npz_file)
            except Exception as e:
                print(f"Error processing headers/coords for {npz_file}: {e}")
                continue # Skip problematic files or raise error

            if i == 0:
                first_im_shape = tensor_shape[1:] # Get H, W from the first file
            elif tensor_shape[1:] != first_im_shape:
                print(f"Warning: Image shape mismatch in {npz_file}. Expected {first_im_shape}, got {tensor_shape[1:]}. Skipping file.")
                # Decide whether to raise an error or just skip
                # cumulative_length.append(total_length) # Keep cumulative length consistent even if skipping
                continue

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

            if n_valid_points == 0:
                print(f"Warning: No points found within bounds for file {npz_file}")
            # ---------------------------------

            # Check whether subsampling is enabled, which will multiply the size of the dataset
            if self.model_config.object_big:
                multiplier = self.data_config.n_subsample
            else:
                multiplier = 1

            length_contribution = n_valid_points * multiplier
            total_length += length_contribution
            cumulative_length.append(total_length)

        if first_im_shape is None:
             raise ValueError("Could not determine image shape from any NPZ file.")

        return total_length, first_im_shape, cumulative_length, valid_indices_per_file
    
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
        instance.current_rank = current_rank
        instance.is_ddp_active = is_ddp_active

        #Set paths
        instance.data_dir = str(map_path)
        instance.data_dir_path = Path(map_path)
        data_prefix_path = instance.data_dir_path.parent
        instance.state_path = data_prefix_path / 'state_files.npz'

        #Load existing map
        try:
            instance.mmap_ptycho = TensorDict.load_memmap(str(instance.data_dir_path))
            instance.length = len(instance.mmap_ptycho)

            #Load state data
            loaded_state = np.load(instance.state_path, allow_pickle = True)
            instance.data_dict = loaded_state['data_dict'].item()
            
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

        mmap_ptycho = TensorDict(
            {   "images": MemoryMappedTensor.empty(
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
                ), #Scaling constant on a per patch basis
                "rms_scaling_constant": MemoryMappedTensor.empty(
                    (mmap_length,1,1,1),
                    dtype=torch.float32
                ),
                "physics_scaling_constant": MemoryMappedTensor.empty(
                    (mmap_length,1,1,1),
                    dtype=torch.float32
                )
                },
                
            batch_size = mmap_length,
        )
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
        #Currently works for a single probe mode, but can technically support more.
        self.data_dict['probes'] = torch.zeros(size=(self.n_files,1, N, N), dtype=torch.complex64)
        self.data_dict['probe_scaling'] = torch.zeros(size=(self.n_files,), dtype = torch.float32)
        self.data_dict['objectGuess'] = []
        #Legacy scaling constant needed for older model artifacts
        self.data_dict["scaling_constant"] = torch.empty(self.n_files,
                                                        dtype = torch.float32)

        #Supervised learning correction factor (PtychoNN-related)
        if self.model_config.mode == 'Supervised':
            self.data_dict['phase_correction'] = []
        
        #Define neighbor grouping function when C > 1
        if self.data_config.neighbor_function == 'Nearest':
            neighbor_function = get_neighbor_indices
        elif self.data_config.neighbor_function == 'Min_dist':
            neighbor_function = get_neighbors_indices_within_bounds
        else:
            neighbor_function = '4_quadrant' #This is the one used in PtychoPINNv2
        
        # Iterate through all npz files in directory
        for i, npz_file in enumerate(image_paths):

            print("Populating memory map for dataset {}".format(i))
            #Calculating all non-diffraction related parameters/tensors
            #Assume: N = # of scans
            start, end = self.cum_length[i], self.cum_length[i+1]

            print(f"Start - end = {end- start}")
            #Writing to non-diffraction memory maps in one go:
            non_diff_timer_start = time.time()

            #Load coordinates
            xcoords_full = np.load(npz_file)['xcoords']
            ycoords_full = np.load(npz_file)['ycoords']

            #Apply coordinate filter to remove edge points based on self.calculate_length
            xcoords = xcoords_full[self.valid_indices_per_file[i]]
            ycoords = ycoords_full[self.valid_indices_per_file[i]]
            self.data_dict['com'] = torch.from_numpy(np.array([xcoords.mean(), ycoords.mean()])) #Center of mass (see reassembly.py)
    
            #--- Coordinate patches/Supervised Labels ---
            # Note that object_big = True means we are enforcing ptychographic constraints and need to group coordinates
            if self.model_config.mode == 'Unsupervised' and self.model_config.object_big: # PtychoPINN/Ptychography Constraint
                #Get indices for coordinate groups using defined neighbor function
                nn_indices, coords_nn = group_coords(xcoords_full, ycoords_full,
                                                    xcoords, ycoords,
                                                    neighbor_function,
                                                    self.valid_indices_per_file[i],
                                                    self.data_config, C=self.data_config.C) # Use stored config
                nn_indices = nn_indices.astype(np.int64)
                
                #Get relative and center of mass coordinates for each coordinate group
                coords_com, coords_relative = get_relative_coords(coords_nn)
                mmap_ptycho["coords_center"][start:end] = torch.from_numpy(coords_com)
                mmap_ptycho["coords_relative"][start:end] = torch.from_numpy(coords_relative)
                mmap_ptycho["nn_indices"][start:end] = torch.from_numpy(nn_indices)

                #Coordinates just outside the "valid range" are still allowed to be used to create coordinate
                #groupings. These will be used for solution region translation
                regular_global_coords = torch.from_numpy(np.stack([xcoords_full,
                                                        ycoords_full],axis=1)).to(torch.float32)
                
                mmap_ptycho["coords_global"][start:end] = regular_global_coords[nn_indices].unsqueeze(2)
            
            else: #Unsupervised CDI or supervised learning

                #Otherwise, the indices are just an arange from 0 to N-1
                nn_indices = self.valid_indices_per_file[i]
                index_range = np.arange(end-start, dtype=np.int64)
                mmap_ptycho["nn_indices"][start:end] = torch.from_numpy(index_range)[:,None]
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
            #Optional: normalize probe for forward model to be photon agnostic. We almost always normalize.
            if len(probe_data.shape) == 2:
                if self.data_config.probe_normalize:
                    probe_data, scaling_factor = hh.normalize_probe(probe_data)
                    self.data_dict['probe_scaling'][i] = float(scaling_factor)
                else:
                    #Save a scaling constant, it's just 1 though
                    self.data_dict['probe_scaling'][i] = float(1)
                probe_data = np.expand_dims(probe_data, axis = 0) # Add number of modes dimension
                
            n_modes = probe_data.shape[0]
            self.data_dict['probes'][i,:n_modes] = torch.from_numpy(probe_data).to(torch.complex64)

            #Object
            objectGuess = np.load(npz_file)['objectGuess']
            if int(objectGuess.sum().real) != (objectGuess.shape[0] * objectGuess.shape[1]): #Check if matrix of ones
                self.data_dict['objectGuess'].append(objectGuess)
            
            non_diff_time = time.time() - non_diff_timer_start
            print("Non-diffraction memory map write time: {}".format(non_diff_time))

            #--- DIFFRACTION IMAGE MAPPING/NORMALIZATION ---
            diff_timer_start = time.time()
            curr_nn_index_length = len(nn_indices)

            #Load diffraction images (canonical 'diffraction' key with 'diff3d' fallback)
            diff_stack = torch.from_numpy(_get_diffraction_stack(npz_file)).round().to(torch.float32) #Round for non-photon detectors

            #Inserting dummy channel dimension if channel number = 1
            if not self.model_config.object_big: # Use stored config
                diff_stack = diff_stack[:,None]

            #Normalizing diffraction images
            print("Getting normalization coefficients...")
            #Batch normalization if specified or you aren't using overlaps
            B = end - start #Batch size
            if self.data_config.normalize == 'Batch' or (self.data_config.C == 1 and self.model_config.mode != 'Supervised'): #
                # Calculate rms normalization factor (used in publication)
                norm_rms_factor = hh.get_rms_scaling_factor(diff_stack, self.data_config)
                print("Batch rms factor is", norm_rms_factor)
                mmap_ptycho["rms_scaling_constant"][start:end] = norm_rms_factor.expand(B,1,1,1)
                # Calculate physics normalization factor
                norm_physics_factor = hh.get_physics_scaling_factor(diff_stack, self.data_config)
                mmap_ptycho["physics_scaling_constant"][start:end] = norm_physics_factor.expand(B,1,1,1)
                # Legacy scaling constant
                self.data_dict["scaling_constant"][i] = norm_rms_factor
            elif self.data_config.normalize == 'None': # Use raw diffraction values
                norm_factor = torch.ones(size=(B,1,1,1))
                mmap_ptycho["rms_scaling_constant"][start:end] = norm_factor
                mmap_ptycho["physics_scaling_constant"][start:end] = norm_factor
                #Legacy scaling constant
                self.data_dict["scaling_constant"][i] = norm_factor

            #Write to memory mapped tensor in batches to avoid huge memory overhead
            for j in range(0, curr_nn_index_length, batch_size): #Write all diffraction images for current experiment
                #Calculate end index (to not exceed length of list)
                local_to = min(j + batch_size, curr_nn_index_length)
                global_to += local_to - j
                
                #NN_indices gives us our coordinate groups of diffraction patterns
                mmap_ptycho["images"][global_from:global_to] = diff_stack[nn_indices[j:local_to]]

                #Calculate group normalization if specified
                if self.data_config.normalize == 'Group' and self.data_config.C > 1:
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

        return 


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
        probes_indexed - (N,C,H,W) tensor, where N is batch size. C is number of channels, where the probe is duplicated
            H and W are height and width of diffraction pattern. The dimensionality should be exactly the same as the output of the autoencoder.
        scaling_constant - (N) tensor, scaling constants required for each diffraction image
        
        """
        #Experimental index is used to find the probe corresponding to the right experiment
        #We can then get the correct probe tensor organized according to diffraction patterns
        exp_idx = self.mmap_ptycho['experiment_id'][idx]

        if self.model_config.object_big: # Use stored config
            channels = self.data_config.C # Use stored config
        else:
            channels = 1

        if self.n_files > 1:
            get_idx = exp_idx
        else:
            get_idx = torch.zeros_like(exp_idx)
        #Expand probe to match number of channels for data.
        if isinstance(exp_idx, int): #If single index access e.g. dataloader[6]
            probes_indexed = self.data_dict['probes'][get_idx]
            probe_scaling = self.data_dict['probe_scaling'][get_idx]
        else: #If tensor/list of indices e.g. dataloader[1:5]
            probes_indexed = self.data_dict['probes'][get_idx].unsqueeze(1).expand(-1,channels,-1,-1,-1)
            probe_scaling = self.data_dict['probe_scaling'][get_idx].view(-1,1,1,1)

            scaling_const = self.data_dict['scaling_constant'][get_idx]
            scaling_const = scaling_const.view(-1,1,1,1)

        return self.mmap_ptycho[idx], probes_indexed, scaling_const

    
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
            "scaling_constant": self.data_dict["scaling_constant"][experiment_idx:experiment_idx+1]
        }
        
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
        outputs = [tensor_dict, probe.clone(), scaling]  # Clone probe to avoid memory sharing issues
        
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