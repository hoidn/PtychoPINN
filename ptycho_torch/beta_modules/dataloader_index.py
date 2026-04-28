"""Index-only ptychography dataloader with overlap-based neighbor sampling.

Stores each diffraction pattern once in a memory-mapped tensor store, and
gathers C-pattern groups at __getitem__ time via index lookups. Neighbor
groupings can be resampled each epoch from a cached candidate pool.

Drop-in replacement for PtychoDataset — returns the same 3-tuple
(TensorDict, probe, probe_scaling) consumed by PtychoPINN_Lightning.
"""

import copy
import math
import os
import time
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset
from tensordict import MemoryMappedTensor, TensorDict

from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
from ptycho_torch.dataloader import (
    npz_headers,
    _get_diffraction_stack,
    fix_tensordict_memmap_state,
    is_ddp_initialized_and_active,
    get_current_rank,
)
from ptycho_torch.patch_generator import get_relative_coords
import ptycho_torch.helper as hh


# ---------------------------------------------------------------------------
# Overlap-based neighbor sampling
# ---------------------------------------------------------------------------

def overlap_fraction(coord_a: np.ndarray, coord_b: np.ndarray, N: int) -> float:
    """Fraction of shared pixels between two N×N patches at given coordinates.

    Args:
        coord_a: (2,) array [x, y] of patch center
        coord_b: (2,) array [x, y] of patch center
        N: patch side length in pixels

    Returns:
        Overlap area / N² in [0, 1].
    """
    dx = abs(coord_a[0] - coord_b[0])
    dy = abs(coord_a[1] - coord_b[1])
    overlap_x = max(0.0, N - dx)
    overlap_y = max(0.0, N - dy)
    return (overlap_x * overlap_y) / (N * N)


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = scores - scores.max()
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum()


def _group_bbox_aspect(coords: np.ndarray, group_indices, candidate_idx):
    """Aspect ratio (x_span / y_span) of the bbox enclosing group + candidate."""
    all_idx = list(group_indices) + [candidate_idx]
    pts = coords[all_idx]
    x_span = pts[:, 0].max() - pts[:, 0].min()
    y_span = pts[:, 1].max() - pts[:, 1].min()
    eps = 1e-8
    if min(x_span, y_span) < eps:
        return float("inf")
    return x_span / y_span


def sample_overlap_group(
    center_idx: int,
    candidate_indices: np.ndarray,
    coords: np.ndarray,
    N: int,
    C: int,
    min_overlap_frac: float = 0.25,
    temperature: float = 0.5,
    aspect_range: tuple = None,
    rng: np.random.Generator = None,
) -> Optional[np.ndarray]:
    """Greedy group selection maximizing minimum pairwise overlap.

    Builds a group of C patterns starting from center_idx, greedily adding
    the candidate that maximizes its minimum overlap with all existing
    group members. Stochastic softmax weighting provides epoch-level
    diversity.

    Args:
        center_idx: Global index of the center pattern.
        candidate_indices: (K,) global indices of neighbor candidates.
        coords: (N_total, 2) full coordinate array.
        N: Patch side length.
        C: Target group size.
        min_overlap_frac: Hard floor — reject candidates below this overlap.
        temperature: Softmax temperature (lower = more deterministic).
        aspect_range: (lo, hi) acceptable range for bbox x_span/y_span.
            None disables the constraint.
        rng: numpy random Generator for reproducibility.

    Returns:
        (C,) int array of global indices, or None if a valid group can't
        be formed.
    """
    if rng is None:
        rng = np.random.default_rng()

    group = [center_idx]
    remaining = [c for c in candidate_indices if c != center_idx]

    for _ in range(C - 1):
        if not remaining:
            return None

        scores = np.empty(len(remaining))
        for k, r in enumerate(remaining):
            min_ov = min(
                overlap_fraction(coords[r], coords[g], N) for g in group
            )
            scores[k] = min_ov

        valid_mask = scores >= min_overlap_frac

        if aspect_range is not None and len(group) >= 2:
            ar_lo, ar_hi = aspect_range
            for k, r in enumerate(remaining):
                if valid_mask[k]:
                    ar = _group_bbox_aspect(coords, group, r)
                    if ar < ar_lo or ar > ar_hi:
                        valid_mask[k] = False

        if not valid_mask.any():
            return None

        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_indices]

        weights = _softmax(valid_scores / max(temperature, 1e-8))
        chosen_local = rng.choice(valid_indices, p=weights)
        chosen_global = remaining[chosen_local]

        group.append(chosen_global)
        remaining.pop(chosen_local)

    return np.array(group, dtype=np.int64)


# ---------------------------------------------------------------------------
# Neighbor graph with cached candidate pool
# ---------------------------------------------------------------------------

class NeighborGraph:
    """Cached neighbor candidate pool for overlap-based epoch-level resampling.

    Builds a KDTree once at initialization and caches K nearest candidates
    per bounded point. The sample_groups() method draws C-size groups using
    overlap-weighted greedy selection with stochastic relaxation.
    """

    def __init__(
        self,
        xcoords_full: np.ndarray,
        ycoords_full: np.ndarray,
        valid_indices: np.ndarray,
        data_config: DataConfig,
        K_candidates: int = 30,
    ):
        self.coords_full = np.stack([xcoords_full, ycoords_full], axis=1).astype(
            np.float64
        )
        self.valid_indices = valid_indices
        self.N = data_config.N
        self.n_subsample = data_config.n_subsample

        coords_bounded = self.coords_full[valid_indices]
        tree = cKDTree(self.coords_full)

        # Query K_candidates nearest neighbors for each bounded point
        # (query against full set so neighbors outside bounds are available)
        _, nn_raw = tree.query(coords_bounded, k=min(K_candidates + 1, len(self.coords_full)))

        # nn_raw: (N_bounded, K+1) — includes self at distance 0
        self.candidate_pool = nn_raw  # global indices into coords_full

    def sample_groups(
        self,
        C: int,
        n_subsample: int = None,
        min_overlap_frac: float = 0.25,
        temperature: float = 0.5,
        seed: Optional[int] = None,
        aspect_range: tuple = None,
        n_relaxation_steps: int = 3,
    ) -> tuple:
        """Sample neighbor groups from the cached candidate pool.

        Args:
            C: Group size (number of patterns per group).
            n_subsample: Number of distinct groups per center point.
            min_overlap_frac: Minimum pairwise overlap fraction.
            temperature: Softmax temperature for stochastic selection.
            seed: Random seed for reproducibility.
            aspect_range: (lo, hi) target range for group bbox x/y ratio.
                None disables the constraint.
            n_relaxation_steps: Number of times to widen aspect_range
                (by 1.5x each step) before dropping it entirely.

        Returns:
            nn_indices: (M, C) int64 array of global pattern indices.
            coords_nn: (M, C, 1, 2) float64 coordinate array.
        """
        if n_subsample is None:
            n_subsample = self.n_subsample

        rng = np.random.default_rng(seed)

        relaxation_schedule = [aspect_range]
        if aspect_range is not None:
            lo, hi = aspect_range
            for _ in range(n_relaxation_steps):
                lo, hi = lo / 1.5, hi * 1.5
                relaxation_schedule.append((lo, hi))
            relaxation_schedule.append(None)

        n_relaxed = 0
        all_groups = []
        for i, center_global in enumerate(self.valid_indices):
            candidates = self.candidate_pool[i]
            n_collected = 0

            for stage, ar in enumerate(relaxation_schedule):
                if n_collected >= n_subsample:
                    break
                attempts = 0
                max_attempts = n_subsample * 3

                while n_collected < n_subsample and attempts < max_attempts:
                    group = sample_overlap_group(
                        center_idx=center_global,
                        candidate_indices=candidates,
                        coords=self.coords_full,
                        N=self.N,
                        C=C,
                        min_overlap_frac=min_overlap_frac,
                        temperature=temperature,
                        aspect_range=ar,
                        rng=rng,
                    )
                    if group is not None:
                        all_groups.append(group)
                        n_collected += 1
                        if stage > 0:
                            n_relaxed += 1
                    attempts += 1

        if aspect_range is not None and n_relaxed > 0:
            total = len(all_groups)
            print(
                f"  [NeighborGraph] {n_relaxed}/{total} groups "
                f"({100*n_relaxed/max(total,1):.1f}%) needed aspect relaxation"
            )

        if not all_groups:
            raise ValueError(
                "No valid groups could be formed. Check scan density "
                "relative to min_overlap_frac and C."
            )

        nn_indices = np.stack(all_groups)  # (M, C)
        coords_nn = np.stack(
            [self.coords_full[nn_indices, 0], self.coords_full[nn_indices, 1]], axis=2
        )[:, :, None, :]  # (M, C, 1, 2)

        return nn_indices, coords_nn


# ---------------------------------------------------------------------------
# Index-only dataset
# ---------------------------------------------------------------------------

class PtychoDatasetIndexed(Dataset):
    """Index-only ptychography dataset.

    Stores each diffraction pattern once in a memory-mapped tensor store.
    Groups are defined by lightweight index arrays and gathered at
    __getitem__ time.  Neighbor groupings can be resampled each epoch
    via resample_groups().

    The __getitem__ return signature is identical to PtychoDataset:
        (TensorDict, probes_indexed, probe_scaling)
    """

    def __init__(
        self,
        ptycho_dir: str,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig = None,
        data_dir: str = "data/memmap_indexed",
        remake_map: bool = False,
        min_overlap_frac: float = 0.25,
        temperature: float = 0.5,
        K_candidates: int = 30,
        aspect_range: tuple = (0.7, 1.3),
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.is_ddp_active = is_ddp_initialized_and_active()
        self.current_rank = get_current_rank()
        self.data_dict = {}
        self.min_overlap_frac = min_overlap_frac
        self.temperature = temperature
        self.K_candidates = K_candidates
        self.aspect_range = aspect_range
        self.neighbor_graph = None  # set during memory_map_data

        self.ptycho_dir = ptycho_dir
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.data_dir_path = Path(data_dir)
        data_prefix_path = self.data_dir_path.parent
        self.state_path = data_prefix_path / "state_files_indexed.npz"

        self.file_list = sorted(list(Path(self.ptycho_dir).glob("*.npz")))
        self.n_files = len(self.file_list)
        if self.n_files == 0 and self.current_rank == 0:
            raise FileNotFoundError(
                f"No NPZ files found in directory: {self.ptycho_dir}"
            )

        # Calculate per-file valid indices and total pattern count
        self._calculate_pattern_counts()

        if not training_config:
            training_config = TrainingConfig()
            training_config.orchestrator = "Mlflow"

        # Memory map creation / loading
        if training_config.orchestrator == "Lightning":
            if remake_map:
                print("Creating indexed memory mapped tensor dictionary...")
                self.memory_map_data(self.file_list)
                np.savez(self.state_path, data_dict=self.data_dict)
            else:
                print(f"Loading existing indexed dataset on rank {self.current_rank}")
                if not self.state_path.exists():
                    raise FileNotFoundError(
                        "Map files missing. prepare_data should have created them."
                    )
                self.mmap_patterns = TensorDict.load_memmap(str(self.data_dir_path))
                loaded_state = np.load(self.state_path, allow_pickle=True)
                self.data_dict = loaded_state["data_dict"].item()
                self._rebuild_groups_from_state()
        else:
            # MLflow orchestrator
            if self.current_rank == 0:
                should_create = remake_map or not self.data_dir_path.exists() or \
                    not any(self.data_dir_path.iterdir()) or not self.state_path.exists()
                if should_create:
                    self.data_dir_path.mkdir(parents=True, exist_ok=True)
                    self.memory_map_data(self.file_list)
                    np.savez(self.state_path, data_dict=self.data_dict)

            if self.is_ddp_active:
                import torch.distributed as dist
                dist.barrier()

            if not hasattr(self, "mmap_patterns"):
                self.mmap_patterns = TensorDict.load_memmap(str(self.data_dir_path))
                loaded_state = np.load(self.state_path, allow_pickle=True)
                self.data_dict = loaded_state["data_dict"].item()
                self._rebuild_groups_from_state()

        if self.current_rank == 0:
            print(
                f"[PtychoDatasetIndexed Rank 0] Init done. "
                f"{self.n_patterns} patterns, {self.length} groups."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_pattern_counts(self):
        """Count total patterns and valid indices per file (for mmap allocation)."""
        self.n_patterns = 0
        self.cum_pattern_count = [0]
        self.valid_indices_per_file = []
        self.im_shape = None

        for i, npz_file in enumerate(self.file_list):
            tensor_shape, xcoords, ycoords = npz_headers(npz_file)
            if i == 0:
                self.im_shape = tensor_shape[1:]
            n_total = tensor_shape[0]

            xmin, xmax = xcoords.min(), xcoords.max()
            ymin, ymax = ycoords.min(), ycoords.max()
            x_range = (xmax - xmin) if xmax > xmin else 1.0
            y_range = (ymax - ymin) if ymax > ymin else 1.0

            x_lo = xmin + self.data_config.x_bounds[0] * x_range
            x_hi = xmin + self.data_config.x_bounds[1] * x_range
            y_lo = ymin + self.data_config.y_bounds[0] * y_range
            y_hi = ymin + self.data_config.y_bounds[1] * y_range

            mask = (xcoords >= x_lo) & (xcoords <= x_hi) & \
                   (ycoords >= y_lo) & (ycoords <= y_hi)
            valid = np.where(mask)[0]
            self.valid_indices_per_file.append(valid)

            self.n_patterns += n_total
            self.cum_pattern_count.append(self.n_patterns)

    def _rebuild_groups_from_state(self):
        """Rebuild in-memory group arrays from saved state after loading."""
        self.nn_indices = torch.from_numpy(
            self.data_dict["nn_indices_np"]
        ).to(torch.int64)
        self.coords_relative = torch.from_numpy(
            self.data_dict["coords_relative_np"]
        ).to(torch.float32)
        self.coords_center = torch.from_numpy(
            self.data_dict["coords_center_np"]
        ).to(torch.float32)
        self.coords_global_group = torch.from_numpy(
            self.data_dict["coords_global_group_np"]
        ).to(torch.float32)
        self.group_experiment_id = torch.from_numpy(
            self.data_dict["group_experiment_id_np"]
        ).to(torch.int32)
        self.rms_scaling = torch.from_numpy(
            self.data_dict["rms_scaling_np"]
        ).to(torch.float32)
        self.physics_scaling = torch.from_numpy(
            self.data_dict["physics_scaling_np"]
        ).to(torch.float32)
        self.length = len(self.nn_indices)

        # Rebuild neighbor graph for resampling
        coords_full_np = self.data_dict.get("coords_full_np")
        valid_indices_all = self.data_dict.get("valid_indices_all_np")
        if coords_full_np is not None and valid_indices_all is not None:
            self.neighbor_graph = NeighborGraph(
                coords_full_np[:, 0],
                coords_full_np[:, 1],
                valid_indices_all,
                self.data_config,
                K_candidates=self.K_candidates,
            )

    def _build_groups(self, xcoords_full, ycoords_full, valid_indices,
                      experiment_id, rms_factor, physics_factor):
        """Build neighbor groups for one experiment using overlap sampling.

        Returns:
            nn_indices: (M, C) int64
            coords_nn: (M, C, 1, 2) float64
            experiment_ids: (M,) int32
            rms: (M, 1, 1, 1) float32
            physics: (M, 1, 1, 1) float32
        """
        C = self.data_config.C

        if C == 1 or not self.model_config.object_big:
            n_sub = self.data_config.n_subsample
            nn_indices = np.repeat(valid_indices[:, None], n_sub, axis=0).astype(np.int64)
            coords_nn = np.stack(
                [xcoords_full[nn_indices], ycoords_full[nn_indices]], axis=2
            )[:, :, None, :]
        else:
            graph = NeighborGraph(
                xcoords_full, ycoords_full, valid_indices,
                self.data_config, K_candidates=self.K_candidates,
            )
            nn_indices, coords_nn = graph.sample_groups(
                C=C,
                n_subsample=self.data_config.n_subsample,
                min_overlap_frac=self.min_overlap_frac,
                temperature=self.temperature,
                aspect_range=self.aspect_range,
            )
            self.neighbor_graph = graph

        M = len(nn_indices)
        exp_ids = np.full(M, experiment_id, dtype=np.int32)

        rms_arr = rms_factor.expand(M, 1, 1, 1).numpy()
        phys_arr = physics_factor.expand(M, 1, 1, 1).numpy()

        if self.data_config.normalize == "Group" and C > 1:
            diff_stack = self.mmap_patterns["patterns"]
            for j in range(M):
                group_patterns = diff_stack[nn_indices[j]]  # (C, H, W)
                rms_arr[j] = hh.get_rms_scaling_factor(
                    group_patterns.unsqueeze(0), self.data_config
                ).numpy()
                phys_arr[j] = hh.get_physics_scaling_factor(
                    group_patterns.unsqueeze(0), self.data_config
                ).numpy()

        return nn_indices, coords_nn, exp_ids, rms_arr, phys_arr

    # ------------------------------------------------------------------
    # Memory map creation
    # ------------------------------------------------------------------

    def memory_map_data(self, image_paths):
        """Create the Level-1 pattern store and build initial groups.

        The mmap stores each pattern exactly once. Group definitions
        (nn_indices, coords, scaling) are kept in-memory tensors.
        """
        N = self.data_config.N
        n_total = self.n_patterns

        print(f"Allocating pattern store for {n_total} patterns...")
        start_t = time.time()

        has_labels = self.model_config.mode == "Supervised"
        mmap_dict = {
            "patterns": MemoryMappedTensor.empty(
                (n_total, *self.im_shape), dtype=torch.float32
            ),
        }
        if has_labels:
            mmap_dict["label_amp"] = MemoryMappedTensor.empty(
                (n_total, *self.im_shape), dtype=torch.float32
            )
            mmap_dict["label_phase"] = MemoryMappedTensor.empty(
                (n_total, *self.im_shape), dtype=torch.float32
            )

        mmap_patterns = TensorDict(mmap_dict, batch_size=n_total)
        mmap_patterns = mmap_patterns.memmap_like(prefix=self.data_dir)
        mmap_patterns = fix_tensordict_memmap_state(mmap_patterns, self.data_dir)
        print(f"Pattern store allocated in {time.time() - start_t:.1f}s")

        # Probe / object storage
        max_modes = 1
        for npz_file in image_paths:
            p = np.load(npz_file)["probeGuess"]
            if p.ndim == 3:
                max_modes = max(max_modes, p.shape[0])
        if max_modes > 1:
            print(f"Detected multi-mode probes: max {max_modes} modes")

        self.data_dict["probes"] = torch.zeros(
            (self.n_files, max_modes, N, N), dtype=torch.complex64
        )
        self.data_dict["probe_scaling"] = torch.zeros(self.n_files, dtype=torch.float32)
        self.data_dict["objectGuess"] = []
        self.data_dict["scaling_constant"] = torch.empty(self.n_files, dtype=torch.float32)

        if has_labels:
            self.data_dict["phase_correction"] = []

        # Aggregate coordinates across all experiments for multi-experiment graphs
        all_nn_indices = []
        all_coords_nn = []
        all_exp_ids = []
        all_rms = []
        all_physics = []
        all_coords_full = []
        all_valid_indices = []

        batch_write = 3000

        for i, npz_file in enumerate(image_paths):
            print(f"Processing experiment {i}: {npz_file}")
            pat_start = self.cum_pattern_count[i]
            pat_end = self.cum_pattern_count[i + 1]

            # Load coordinates
            npz_data = np.load(npz_file)
            xcoords_full = npz_data["xcoords"].astype(np.float64)
            ycoords_full = npz_data["ycoords"].astype(np.float64)
            n_diff_key = next(k for k in ("diffraction", "diff3d") if k in npz_data)
            n_diff = len(npz_data[n_diff_key])
            if len(xcoords_full) > n_diff:
                xcoords_full = xcoords_full[:n_diff]
                ycoords_full = ycoords_full[:n_diff]

            valid_indices = self.valid_indices_per_file[i]

            self.data_dict["com"] = torch.from_numpy(
                np.array([
                    xcoords_full[valid_indices].mean(),
                    ycoords_full[valid_indices].mean(),
                ])
            )

            # ---- Write patterns to mmap (once per pattern, no duplication) ----
            diff_stack = torch.from_numpy(
                _get_diffraction_stack(npz_file)
            ).round().to(torch.float32)

            for j in range(0, len(diff_stack), batch_write):
                end_j = min(j + batch_write, len(diff_stack))
                mmap_patterns["patterns"][pat_start + j : pat_start + end_j] = \
                    diff_stack[j:end_j]

            # ---- Supervised labels ----
            if has_labels:
                labels = np.load(npz_file)["label"]
                objectGuess = np.load(npz_file)["objectGuess"]
                obj_phase = np.angle(objectGuess)
                mid = obj_phase.shape[0]
                phase_corr = obj_phase[
                    int(mid / 3.0) : int(mid * 2 / 3.0),
                    int(mid / 3.0) : int(mid * 2 / 3.0),
                ].mean()
                self.data_dict["phase_correction"].append(phase_corr)

                valid_labels = labels[valid_indices][:, None, :, :]
                label_phase = np.angle(valid_labels)
                if self.data_config.phase_subtraction:
                    label_phase -= phase_corr
                label_phase = np.angle(np.exp(1j * label_phase))
                label_amp = np.abs(valid_labels)

                for j in range(0, len(valid_indices), batch_write):
                    end_j = min(j + batch_write, len(valid_indices))
                    gl = pat_start + valid_indices[j:end_j]
                    mmap_patterns["label_amp"][gl] = torch.from_numpy(
                        label_amp[j:end_j, 0]
                    )
                    mmap_patterns["label_phase"][gl] = torch.from_numpy(
                        label_phase[j:end_j, 0]
                    )

            # ---- Probe ----
            probe_data = np.load(npz_file)["probeGuess"]
            if self.data_config.probe_ramp_removal:
                if probe_data.ndim == 3:
                    probe_data = np.stack(
                        [hh.standardize_probe(probe_data[p])
                         for p in range(probe_data.shape[0])]
                    )
                else:
                    probe_data = hh.standardize_probe(probe_data)

            if probe_data.ndim == 2:
                if self.data_config.probe_normalize:
                    probe_data, sf = hh.normalize_probe(probe_data)
                    self.data_dict["probe_scaling"][i] = float(sf)
                else:
                    self.data_dict["probe_scaling"][i] = 1.0
                probe_data = np.expand_dims(probe_data, axis=0)
            elif probe_data.ndim == 3:
                if self.data_config.probe_normalize:
                    probe_data, sf = hh.normalize_probe(probe_data)
                    self.data_dict["probe_scaling"][i] = float(sf)
                else:
                    self.data_dict["probe_scaling"][i] = 1.0

            n_modes = probe_data.shape[0]
            self.data_dict["probes"][i, :n_modes] = torch.from_numpy(
                probe_data
            ).to(torch.complex64)

            # Object
            objectGuess = np.load(npz_file)["objectGuess"]
            if int(objectGuess.sum().real) != (objectGuess.shape[0] * objectGuess.shape[1]):
                self.data_dict["objectGuess"].append(objectGuess)

            # ---- Batch/experiment-level normalization ----
            rms_factor = hh.get_rms_scaling_factor(diff_stack, self.data_config)
            physics_factor = hh.get_physics_scaling_factor(diff_stack, self.data_config)
            self.data_dict["scaling_constant"][i] = rms_factor.item()

            # ---- Build groups ----
            # Offset valid_indices to global pattern indices in the mmap
            valid_global = valid_indices + pat_start
            nn_indices_exp, coords_nn_exp, exp_ids, rms, physics = self._build_groups(
                xcoords_full, ycoords_full, valid_indices,
                experiment_id=i,
                rms_factor=rms_factor,
                physics_factor=physics_factor,
            )

            # Offset nn_indices to global pattern space
            nn_indices_exp = nn_indices_exp + pat_start

            all_nn_indices.append(nn_indices_exp)
            all_coords_nn.append(coords_nn_exp)
            all_exp_ids.append(exp_ids)
            all_rms.append(rms)
            all_physics.append(physics)

            # Cache for resampling
            coords_full_exp = np.stack([xcoords_full, ycoords_full], axis=1)
            all_coords_full.append(coords_full_exp)
            all_valid_indices.append(valid_indices)

        # ---- Concatenate all experiments ----
        nn_indices_all = np.concatenate(all_nn_indices)
        coords_nn_all = np.concatenate(all_coords_nn)
        exp_ids_all = np.concatenate(all_exp_ids)
        rms_all = np.concatenate(all_rms)
        physics_all = np.concatenate(all_physics)

        # Compute relative coordinates
        coords_center_all, coords_relative_all = get_relative_coords(coords_nn_all)

        # Build global coordinate array
        coords_full_concat = np.stack(
            [np.concatenate([c[:, 0] for c in all_coords_full]),
             np.concatenate([c[:, 1] for c in all_coords_full])],
            axis=1,
        )

        # Build coords_global_group: (M, C, 1, 2)
        coords_global_group = coords_nn_all  # already (M, C, 1, 2)

        # Store in-memory group tensors
        self.nn_indices = torch.from_numpy(nn_indices_all).to(torch.int64)
        self.coords_relative = torch.from_numpy(coords_relative_all).to(torch.float32)
        self.coords_center = torch.from_numpy(coords_center_all).to(torch.float32)
        self.coords_global_group = torch.from_numpy(coords_global_group).to(torch.float32)
        self.group_experiment_id = torch.from_numpy(exp_ids_all).to(torch.int32)
        self.rms_scaling = torch.from_numpy(rms_all).to(torch.float32)
        self.physics_scaling = torch.from_numpy(physics_all).to(torch.float32)
        self.length = len(self.nn_indices)

        # Save group state for reload
        self.data_dict["nn_indices_np"] = nn_indices_all
        self.data_dict["coords_relative_np"] = coords_relative_all
        self.data_dict["coords_center_np"] = coords_center_all
        self.data_dict["coords_global_group_np"] = coords_global_group.astype(np.float32)
        self.data_dict["group_experiment_id_np"] = exp_ids_all
        self.data_dict["rms_scaling_np"] = rms_all
        self.data_dict["physics_scaling_np"] = physics_all
        self.data_dict["coords_full_np"] = coords_full_concat
        # Concatenate valid indices with offsets for global pattern space
        valid_offset = []
        for i, vi in enumerate(all_valid_indices):
            valid_offset.append(vi + self.cum_pattern_count[i])
        self.data_dict["valid_indices_all_np"] = np.concatenate(valid_offset)

        self.mmap_patterns = mmap_patterns

        print(
            f"Index-only dataset built: {self.n_patterns} patterns, "
            f"{self.length} groups, C={self.data_config.C}"
        )

    # ------------------------------------------------------------------
    # Epoch resampling
    # ------------------------------------------------------------------

    def resample_groups(self, seed: Optional[int] = None):
        """Resample neighbor groupings from the cached candidate pool.

        Call at epoch boundaries to provide diversity in group composition
        while reusing the same deduplicated pattern store.
        """
        if self.neighbor_graph is None:
            return
        if not self.model_config.object_big or self.data_config.C <= 1:
            return

        C = self.data_config.C
        nn_indices, coords_nn = self.neighbor_graph.sample_groups(
            C=C,
            n_subsample=self.data_config.n_subsample,
            min_overlap_frac=self.min_overlap_frac,
            temperature=self.temperature,
            seed=seed,
            aspect_range=self.aspect_range,
        )

        coords_center, coords_relative = get_relative_coords(coords_nn)

        # Re-offset to global pattern indices
        # (neighbor_graph uses experiment-local indices — need offset)
        # For single-experiment case this is the common path
        pat_offset = 0
        if len(self.cum_pattern_count) > 2:
            # Multi-experiment: neighbor graph was built on last experiment
            # For full multi-experiment resampling, would need per-experiment graphs
            pat_offset = self.cum_pattern_count[-2]

        nn_indices_global = nn_indices + pat_offset

        self.nn_indices = torch.from_numpy(nn_indices_global).to(torch.int64)
        self.coords_relative = torch.from_numpy(coords_relative).to(torch.float32)
        self.coords_center = torch.from_numpy(coords_center).to(torch.float32)

        coords_full = self.neighbor_graph.coords_full
        coords_global = np.stack(
            [coords_full[nn_indices, 0], coords_full[nn_indices, 1]], axis=2
        )[:, :, None, :]
        self.coords_global_group = torch.from_numpy(
            coords_global.astype(np.float32)
        )

        M = len(nn_indices)
        # Recompute group normalization if needed
        if self.data_config.normalize == "Group" and C > 1:
            rms_arr = np.empty((M, 1, 1, 1), dtype=np.float32)
            phys_arr = np.empty((M, 1, 1, 1), dtype=np.float32)
            for j in range(M):
                group_pats = self.mmap_patterns["patterns"][
                    self.nn_indices[j]
                ]  # (C, H, W)
                rms_arr[j] = hh.get_rms_scaling_factor(
                    group_pats.unsqueeze(0), self.data_config
                ).numpy()
                phys_arr[j] = hh.get_physics_scaling_factor(
                    group_pats.unsqueeze(0), self.data_config
                ).numpy()
            self.rms_scaling = torch.from_numpy(rms_arr)
            self.physics_scaling = torch.from_numpy(phys_arr)
        else:
            # Batch normalization — constant per experiment
            exp_id = self.group_experiment_id[0].item() if len(self.group_experiment_id) > 0 else 0
            rms_val = self.data_dict["scaling_constant"][exp_id].item()
            self.rms_scaling = torch.full((M, 1, 1, 1), rms_val, dtype=torch.float32)
            phys_val = self.physics_scaling[0].item() if len(self.physics_scaling) > 0 else 1.0
            self.physics_scaling = torch.full((M, 1, 1, 1), phys_val, dtype=torch.float32)

        self.group_experiment_id = torch.zeros(M, dtype=torch.int32)
        self.length = M

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Gather patterns at runtime and return the standard 3-tuple.

        Returns:
            (TensorDict, probes_indexed, probe_scaling)
        """
        group_indices = self.nn_indices[idx]  # (C,) or (B, C)
        images = self.mmap_patterns["patterns"][group_indices]  # (C, H, W) or (B, C, H, W)

        exp_idx = self.group_experiment_id[idx]

        # Build TensorDict matching PtychoDataset's field layout
        td = TensorDict(
            {
                "images": images,
                "coords_relative": self.coords_relative[idx],
                "coords_global": self.coords_global_group[idx],
                "coords_center": self.coords_center[idx],
                "nn_indices": group_indices,
                "experiment_id": exp_idx,
                "rms_scaling_constant": self.rms_scaling[idx],
                "physics_scaling_constant": self.physics_scaling[idx],
            },
            batch_size=[],
        )

        # Supervised labels
        if self.model_config.mode == "Supervised":
            td["label_amp"] = self.mmap_patterns["label_amp"][group_indices]
            td["label_phase"] = self.mmap_patterns["label_phase"][group_indices]

        # Probe indexing (same logic as PtychoDataset)
        if self.model_config.object_big:
            channels = self.data_config.C
        else:
            channels = 1

        if self.n_files > 1:
            get_idx = exp_idx
        else:
            get_idx = torch.zeros_like(exp_idx)

        if isinstance(idx, int):
            probes_indexed = self.data_dict["probes"][get_idx]
            probe_scaling = self.data_dict["probe_scaling"][get_idx]
        else:
            probes_indexed = (
                self.data_dict["probes"][get_idx]
                .unsqueeze(1)
                .expand(-1, channels, -1, -1, -1)
            )
            probe_scaling = self.data_dict["probe_scaling"][get_idx].view(-1, 1, 1, 1)

        return td, probes_indexed, probe_scaling

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_existing_map(
        cls, map_path, model_config, data_config, current_rank=0, is_ddp_active=False
    ):
        """Load from an existing indexed memory map."""
        instance = cls.__new__(cls)
        instance.model_config = model_config
        instance.data_config = data_config
        instance.current_rank = current_rank
        instance.is_ddp_active = is_ddp_active
        instance.min_overlap_frac = 0.25
        instance.temperature = 0.5
        instance.K_candidates = 30
        instance.aspect_range = (0.7, 1.3)

        instance.data_dir = str(map_path)
        instance.data_dir_path = Path(map_path)
        data_prefix_path = instance.data_dir_path.parent
        instance.state_path = data_prefix_path / "state_files_indexed.npz"

        instance.mmap_patterns = TensorDict.load_memmap(str(instance.data_dir_path))
        instance.n_patterns = len(instance.mmap_patterns)
        instance.n_files = 1
        instance.cum_pattern_count = [0, instance.n_patterns]

        loaded_state = np.load(instance.state_path, allow_pickle=True)
        instance.data_dict = loaded_state["data_dict"].item()
        instance._rebuild_groups_from_state()

        print(
            f"[PtychoDatasetIndexed] Loaded existing map: "
            f"{instance.n_patterns} patterns, {instance.length} groups"
        )
        return instance

    @classmethod
    def from_np(
        cls,
        diff_patterns: np.ndarray,
        probe: np.ndarray,
        positions: np.ndarray,
        model_config: ModelConfig,
        data_config: DataConfig,
        scaling_constant: float = None,
        min_overlap_frac: float = 0.25,
        temperature: float = 0.5,
        K_candidates: int = 30,
        aspect_range: tuple = (0.7, 1.3),
    ) -> "PtychoDatasetIndexed":
        """Create from in-memory numpy arrays (for inference pipelines)."""
        if diff_patterns.ndim != 3:
            raise ValueError(f"diff_patterns must be 3D (N, H, W), got {diff_patterns.shape}")
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"positions must be (N, 2), got {positions.shape}")

        N_total, H, W = diff_patterns.shape

        dataset = cls.__new__(cls)
        dataset.model_config = model_config
        dataset.data_config = data_config
        dataset.is_ddp_active = False
        dataset.current_rank = 0
        dataset.n_files = 1
        dataset.n_patterns = N_total
        dataset.cum_pattern_count = [0, N_total]
        dataset.data_dict = {}
        dataset.im_shape = (H, W)
        dataset.min_overlap_frac = min_overlap_frac
        dataset.temperature = temperature
        dataset.K_candidates = K_candidates
        dataset.aspect_range = aspect_range
        dataset.neighbor_graph = None

        # Coordinate filtering
        xcoords_full = positions[:, 1].astype(np.float64)
        ycoords_full = positions[:, 0].astype(np.float64)

        xmin, xmax = xcoords_full.min(), xcoords_full.max()
        ymin, ymax = ycoords_full.min(), ycoords_full.max()
        x_range = (xmax - xmin) if xmax > xmin else 1.0
        y_range = (ymax - ymin) if ymax > ymin else 1.0

        x_lo = xmin + data_config.x_bounds[0] * x_range
        x_hi = xmin + data_config.x_bounds[1] * x_range
        y_lo = ymin + data_config.y_bounds[0] * y_range
        y_hi = ymin + data_config.y_bounds[1] * y_range

        mask = (xcoords_full >= x_lo) & (xcoords_full <= x_hi) & \
               (ycoords_full >= y_lo) & (ycoords_full <= y_hi)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            raise ValueError("No positions remain after bounds filtering.")

        dataset.data_dict["com"] = torch.from_numpy(
            np.array([xcoords_full[valid_indices].mean(),
                      ycoords_full[valid_indices].mean()])
        )

        # Process probe
        probe_data = probe.copy()
        if data_config.probe_ramp_removal:
            if probe_data.ndim == 3:
                probe_data = np.stack(
                    [hh.standardize_probe(probe_data[p])
                     for p in range(probe_data.shape[0])]
                )
            else:
                probe_data = hh.standardize_probe(probe_data)

        if data_config.probe_normalize:
            probe_data, probe_sf = hh.normalize_probe(probe_data)
            probe_sf = float(probe_sf)
        else:
            probe_sf = 1.0

        if probe_data.ndim == 2:
            probe_data = np.expand_dims(probe_data, axis=0)

        dataset.data_dict["probes"] = (
            torch.from_numpy(probe_data).to(torch.complex64).unsqueeze(0)
        )
        dataset.data_dict["probe_scaling"] = torch.tensor([probe_sf], dtype=torch.float32)
        dataset.data_dict["objectGuess"] = []

        # Normalization
        diff_tensor = torch.from_numpy(diff_patterns).to(torch.float32)

        if scaling_constant is not None:
            rms_factor = torch.tensor(scaling_constant, dtype=torch.float32).view(1, 1, 1, 1)
        else:
            rms_factor = hh.get_rms_scaling_factor(diff_tensor, data_config)
        physics_factor = hh.get_physics_scaling_factor(diff_tensor, data_config)
        dataset.data_dict["scaling_constant"] = rms_factor.view(1)

        # Build in-memory pattern store (no mmap for from_np)
        dataset.mmap_patterns = TensorDict(
            {"patterns": diff_tensor}, batch_size=N_total
        )

        # Build groups
        C = data_config.C
        if model_config.object_big and C > 1:
            graph = NeighborGraph(
                xcoords_full, ycoords_full, valid_indices,
                data_config, K_candidates=K_candidates,
            )
            nn_indices, coords_nn = graph.sample_groups(
                C=C, n_subsample=data_config.n_subsample,
                min_overlap_frac=min_overlap_frac,
                temperature=temperature,
                aspect_range=aspect_range,
            )
            dataset.neighbor_graph = graph
        else:
            n_sub = data_config.n_subsample if model_config.object_big else 1
            nn_indices = np.repeat(valid_indices[:, None], n_sub, axis=0).astype(np.int64)
            coords_nn = np.stack(
                [xcoords_full[nn_indices], ycoords_full[nn_indices]], axis=2
            )[:, :, None, :]

        coords_center, coords_relative = get_relative_coords(coords_nn)
        M = len(nn_indices)

        dataset.nn_indices = torch.from_numpy(nn_indices).to(torch.int64)
        dataset.coords_relative = torch.from_numpy(coords_relative).to(torch.float32)
        dataset.coords_center = torch.from_numpy(coords_center).to(torch.float32)
        dataset.coords_global_group = torch.from_numpy(
            coords_nn.astype(np.float32)
        )
        dataset.group_experiment_id = torch.zeros(M, dtype=torch.int32)
        dataset.rms_scaling = rms_factor.expand(M, 1, 1, 1).clone()
        dataset.physics_scaling = physics_factor.expand(M, 1, 1, 1).clone()
        dataset.length = M

        dataset.valid_indices_per_file = [valid_indices]
        dataset.file_list = [None]

        print(
            f"[PtychoDatasetIndexed.from_np] {M} groups, "
            f"{C} channels, image shape {(H, W)}"
        )
        return dataset

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def get_experiment_dataset(self, experiment_idx):
        """Return a subset dataset for a single experiment (inference)."""
        subset = copy.copy(self)

        mask = self.group_experiment_id == experiment_idx
        indices = torch.where(mask)[0]

        if len(indices) == 0:
            raise ValueError(f"No data found for experiment_idx {experiment_idx}")

        subset.nn_indices = self.nn_indices[indices]
        subset.coords_relative = self.coords_relative[indices]
        subset.coords_center = self.coords_center[indices]
        subset.coords_global_group = self.coords_global_group[indices]
        subset.group_experiment_id = self.group_experiment_id[indices]
        subset.rms_scaling = self.rms_scaling[indices]
        subset.physics_scaling = self.physics_scaling[indices]
        subset.length = len(indices)

        subset.file_list = [self.file_list[experiment_idx]]
        subset.n_files = 1

        subset.data_dict = {
            "probes": self.data_dict["probes"][experiment_idx : experiment_idx + 1],
            "probe_scaling": self.data_dict["probe_scaling"][
                experiment_idx : experiment_idx + 1
            ],
            "scaling_constant": self.data_dict["scaling_constant"][
                experiment_idx : experiment_idx + 1
            ],
        }
        if len(self.data_dict.get("objectGuess", [])) > experiment_idx:
            subset.data_dict["objectGuess"] = [
                self.data_dict["objectGuess"][experiment_idx]
            ]
        else:
            subset.data_dict["objectGuess"] = []
        if "com" in self.data_dict:
            subset.data_dict["com"] = self.data_dict["com"]

        # Build a synthetic mmap_ptycho view for inference code that accesses it directly
        subset.mmap_ptycho = TensorDict(
            {
                "coords_global": subset.coords_global_group,
                "coords_center": subset.coords_center,
                "coords_relative": subset.coords_relative,
                "experiment_id": subset.group_experiment_id,
                "nn_indices": subset.nn_indices,
                "rms_scaling_constant": subset.rms_scaling,
                "physics_scaling_constant": subset.physics_scaling,
            },
            batch_size=subset.length,
        )

        return subset
