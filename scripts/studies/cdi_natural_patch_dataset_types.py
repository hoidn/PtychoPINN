"""Frozen contract dataclasses + module-level constants shared by the
natural-patch fixed-probe CDI dataset builder and its I/O helpers.
Defining them here keeps the orchestration module
(``scripts/studies/cdi_natural_patch_dataset.py``) and the I/O module
(``scripts/studies/cdi_natural_patch_dataset_io.py``) decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Tuple
import math

import numpy as np


DEFAULT_DATASET_ID = "natural_patches128_fixedprobe_v1"
DEFAULT_PATCH_SIZE = 128
DEFAULT_TOTAL_CAP = 10_000
DEFAULT_PROBE_SOURCE = "datasets/Run1084_recon3_postPC_shrunk_3.npz"
DEFAULT_PROBE_SMOOTHING_SIGMA = 0.5
DEFAULT_PROBE_SCALE_MODE = "pad_extrapolate"
DEFAULT_SPLIT_SEED = 1337
DEFAULT_CROP_SEED = 4242

DEFAULT_SKIMAGE_SOURCE_NAMES: Tuple[str, ...] = tuple(
    sorted(
        (
            "astronaut",
            "brick",
            "camera",
            "cat",
            "chelsea",
            "clock",
            "coffee",
            "coins",
            "eagle",
            "grass",
            "gravel",
            "hubble_deep_field",
            "moon",
            "page",
            "retina",
            "rocket",
        )
    )
)

DEFAULT_SPLIT_COUNTS: Mapping[str, int] = {"train": 8_000, "val": 1_000, "test": 1_000}
DEFAULT_SPLIT_SOURCE_COUNTS: Mapping[str, int] = {"train": 12, "val": 2, "test": 2}
SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class NaturalImageRecord:
    image_id: str
    pixels: np.ndarray  # 2D float32 in [0, 1]
    height: int
    width: int

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


@dataclass(frozen=True)
class ProbeBundle:
    probe: np.ndarray  # (N, N) complex64
    source_path: str
    source_shape: Tuple[int, int]
    target_N: int
    smoothing_sigma: float
    scale_mode: str
    pipeline_spec: str


@dataclass(frozen=True)
class ObjectEncodingContract:
    grayscale: str = "rec_709_luminance"
    normalization: str = "uint8_to_unit"
    amplitude_min: float = 0.5
    amplitude_max: float = 1.0
    phase_min_rad: float = -math.pi / 2.0
    phase_max_rad: float = math.pi / 2.0
    description: str = (
        "amplitude = 0.5 + 0.5 * x; phase = pi * (x - 0.5); "
        "object = amplitude * exp(1j * phase) where x in [0, 1]"
    )


@dataclass(frozen=True)
class SimulationContract:
    forward_model: str = "single_shot_cdi_fraunhofer"
    formula: str = "diffraction = abs(fftshift(fft2(probe * object)) / sqrt(N**2))"
    dtype_object: str = "complex64"
    dtype_diffraction: str = "float32"


@dataclass
class BuildResult:
    dataset_root: Path
    split_counts: Dict[str, int]
    source_split_membership: Dict[str, List[str]]
    artifact_paths: Dict[str, Path] = field(default_factory=dict)
