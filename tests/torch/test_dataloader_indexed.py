#!/usr/bin/env python
"""Validation test for PtychoDatasetIndexed: overlap quality, batch shapes, mmap I/O."""

import os
import shutil
import sys
import tempfile
import time
from itertools import combinations

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ptycho_torch.beta_modules.dataloader_index import (
    PtychoDatasetIndexed,
    overlap_fraction,
)
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
from ptycho_torch.dataloader import Collate_Lightning, TensorDictDataLoader

PTYCHO_DIR = "/local/PtychoPINN/data/pinn_velo_ic_2"
C = 8
N = 64
N_PATTERNS_EXPECTED = 9436


def section1_dataset_creation(tmp_dir):
    """Create dataset and validate basic shapes."""
    print("=" * 60)
    print("SECTION 1: Dataset Creation & Basic Validation")
    print("=" * 60)

    data_config = DataConfig(C=C, N=N, n_subsample=7)
    model_config = ModelConfig(object_big=True, mode="Unsupervised")
    training_config = TrainingConfig()
    training_config.orchestrator = "Mlflow"

    mmap_dir = os.path.join(tmp_dir, "memmap_indexed")

    t0 = time.time()
    dataset = PtychoDatasetIndexed(
        ptycho_dir=PTYCHO_DIR,
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        data_dir=mmap_dir,
        remake_map=True,
    )
    create_time = time.time() - t0

    M = dataset.length
    print(f"  Memory map creation time: {create_time:.1f}s")
    print(f"  Total patterns in store:  {dataset.n_patterns}")
    print(f"  Total groups (M):         {M}")
    print(f"  C (channels per group):   {C}")

    assert dataset.n_patterns == N_PATTERNS_EXPECTED, (
        f"Expected {N_PATTERNS_EXPECTED} patterns, got {dataset.n_patterns}"
    )
    assert M > 0, "Dataset produced zero groups"
    assert dataset.nn_indices.shape == (M, C), (
        f"nn_indices shape {dataset.nn_indices.shape}, expected ({M}, {C})"
    )
    assert dataset.coords_relative.shape == (M, C, 1, 2)
    assert dataset.coords_center.shape == (M, 1, 1, 2)
    assert dataset.coords_global_group.shape == (M, C, 1, 2)
    assert dataset.mmap_patterns["patterns"].shape == (N_PATTERNS_EXPECTED, N, N)

    probes = dataset.data_dict["probes"]
    print(f"  Probes shape:             {probes.shape}")
    print(f"  nn_indices shape:         {dataset.nn_indices.shape}")
    print(f"  Pattern store shape:      {dataset.mmap_patterns['patterns'].shape}")

    assert dataset.nn_indices.min() >= 0
    assert dataset.nn_indices.max() < dataset.n_patterns

    print("  All basic validations PASSED.")
    return dataset, create_time


def section2_overlap_quality(dataset, n_samples=100):
    """Analyze overlap fractions and bounding box aspect ratios."""
    print()
    print("=" * 60)
    print("SECTION 2: Overlap Quality Analysis")
    print("=" * 60)

    rng = np.random.default_rng(42)
    sample_idx = rng.integers(0, len(dataset), size=n_samples)

    group_min_overlaps = []
    group_mean_overlaps = []
    aspect_ratios = []
    violations = []

    for g in sample_idx:
        coords = dataset.coords_global_group[g].numpy().squeeze(1)  # (C, 2)

        pairwise = []
        for i, j in combinations(range(C), 2):
            ov = overlap_fraction(coords[i], coords[j], N)
            pairwise.append(ov)
        pairwise = np.array(pairwise)

        group_min_overlaps.append(pairwise.min())
        group_mean_overlaps.append(pairwise.mean())

        x_span = coords[:, 0].max() - coords[:, 0].min()
        y_span = coords[:, 1].max() - coords[:, 1].min()
        eps = 1e-8
        if min(x_span, y_span) < eps:
            ar = float("inf")
        else:
            ar = x_span / y_span
        aspect_ratios.append(ar)

        if ar < 0.7 or ar > 1.3:
            violations.append((int(g), ar, x_span, y_span))

    min_ov = np.array(group_min_overlaps)
    mean_ov = np.array(group_mean_overlaps)
    ar_arr = np.array([a for a in aspect_ratios if np.isfinite(a)])

    print(f"  Analyzed {n_samples} random groups:")
    print(f"  Per-group MIN overlap:  min={min_ov.min():.3f}  mean={min_ov.mean():.3f}  max={min_ov.max():.3f}")
    print(f"  Per-group MEAN overlap: min={mean_ov.min():.3f}  mean={mean_ov.mean():.3f}  max={mean_ov.max():.3f}")
    if len(ar_arr) > 0:
        print(f"  Aspect ratio (x/y):     min={ar_arr.min():.3f}  mean={ar_arr.mean():.3f}  max={ar_arr.max():.3f}")
    print(f"  Aspect ratio violations ([0.7, 1.3]): {len(violations)}/{n_samples}")

    if violations:
        print("  Violations (group_idx, ratio, x_span, y_span):")
        for v in violations[:10]:
            print(f"    group {v[0]:>6d}: ratio={v[1]:.3f}  x_span={v[2]:.1f}  y_span={v[3]:.1f}")
        if len(violations) > 10:
            print(f"    ... and {len(violations) - 10} more")

    assert min_ov.min() >= 0.20, (
        f"Some groups have critically low overlap: {min_ov.min():.3f} < 0.20"
    )
    print("  Overlap quality checks PASSED.")
    return {
        "min_overlaps": min_ov,
        "mean_overlaps": mean_ov,
        "aspect_ratios": ar_arr,
        "violations": violations,
    }


def section3_batch_inspection(dataset, batch_size=16):
    """Fetch a batch via DataLoader and verify shapes and values."""
    print()
    print("=" * 60)
    print("SECTION 3: Batch Inspection")
    print("=" * 60)

    loader = TensorDictDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=Collate_Lightning(pin_memory_if_cuda=False),
    )

    batch = next(iter(loader))
    td, probes, probe_scaling = batch

    B = batch_size
    print(f"  Batch size: {B}")
    print(f"  images:                  {td['images'].shape}  dtype={td['images'].dtype}")
    print(f"  coords_relative:         {td['coords_relative'].shape}")
    print(f"  coords_global:           {td['coords_global'].shape}")
    print(f"  coords_center:           {td['coords_center'].shape}")
    print(f"  nn_indices:              {td['nn_indices'].shape}")
    print(f"  experiment_id:           {td['experiment_id'].shape}")
    print(f"  rms_scaling_constant:    {td['rms_scaling_constant'].shape}")
    print(f"  physics_scaling_constant:{td['physics_scaling_constant'].shape}")
    print(f"  probes:                  {probes.shape}  dtype={probes.dtype}")
    print(f"  probe_scaling:           {probe_scaling.shape}")

    assert td["images"].shape == (B, C, N, N), f"images shape {td['images'].shape}"
    assert td["coords_relative"].shape == (B, C, 1, 2)
    assert td["coords_global"].shape == (B, C, 1, 2)
    assert td["coords_center"].shape == (B, 1, 1, 2)
    assert td["nn_indices"].shape == (B, C)
    assert td["experiment_id"].shape == (B,)
    assert td["rms_scaling_constant"].shape == (B, 1, 1, 1)
    assert probes.dtype == torch.complex64
    assert probes.shape[1] == C, f"probes channels {probes.shape[1]}, expected {C}"

    img_sum = td["images"].sum().item()
    assert img_sum > 0, "Images are all zeros — mmap read may have failed"
    print(f"  Images sum: {img_sum:.2e} (non-zero: OK)")

    assert td["images"].min() >= 0, "Diffraction patterns should be non-negative"

    rel_mean = td["coords_relative"].mean(dim=1)  # (B, 1, 2)
    max_dev = rel_mean.abs().max().item()
    print(f"  Relative coords mean deviation: {max_dev:.4f}")
    assert max_dev < 5.0, f"Relative coords should be near-zero mean, got {max_dev}"

    idx_max = td["nn_indices"].max().item()
    idx_min = td["nn_indices"].min().item()
    assert idx_min >= 0 and idx_max < dataset.n_patterns, (
        f"nn_indices range [{idx_min}, {idx_max}] outside [0, {dataset.n_patterns})"
    )

    assert probes.abs().sum() > 0, "Probes are all zeros"

    print("  All batch inspections PASSED.")


def section4_mmap_stats(dataset, create_time):
    """Benchmark memory-mapped read performance."""
    print()
    print("=" * 60)
    print("SECTION 4: Memory Map Read/Write Statistics")
    print("=" * 60)

    bytes_per_group = C * N * N * 4  # float32
    print(f"  Memory map creation time: {create_time:.1f}s")
    print(f"  Bytes per group gather:   {bytes_per_group / 1e6:.2f} MB")
    print()

    # Sequential read
    N_bench = 500
    t0 = time.time()
    for i in range(N_bench):
        _ = dataset[i]
    seq_time = time.time() - t0
    seq_rate = N_bench / seq_time
    seq_mb = N_bench * bytes_per_group / seq_time / 1e6
    print(f"  Sequential read ({N_bench} items): {seq_time:.2f}s  ({seq_rate:.0f} groups/sec, {seq_mb:.1f} MB/sec)")

    # Random read
    rng = np.random.default_rng(42)
    rand_idx = rng.integers(0, len(dataset), size=N_bench)
    t0 = time.time()
    for i in rand_idx:
        _ = dataset[int(i)]
    rand_time = time.time() - t0
    rand_rate = N_bench / rand_time
    rand_mb = N_bench * bytes_per_group / rand_time / 1e6
    print(f"  Random read    ({N_bench} items): {rand_time:.2f}s  ({rand_rate:.0f} groups/sec, {rand_mb:.1f} MB/sec)")

    # Full epoch via DataLoader
    loader = TensorDictDataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        collate_fn=Collate_Lightning(pin_memory_if_cuda=False),
    )
    t0 = time.time()
    n_batches = 0
    for _ in loader:
        n_batches += 1
    epoch_time = time.time() - t0
    epoch_rate = len(dataset) / epoch_time
    print(f"  Full epoch (bs=64):       {epoch_time:.2f}s  ({n_batches} batches, {epoch_rate:.0f} groups/sec)")

    # Indexed gather vs contiguous read
    n_gather = 100
    sample_groups = dataset.nn_indices[:n_gather]
    t0 = time.time()
    for g in sample_groups:
        _ = dataset.mmap_patterns["patterns"][g]
    gather_time = time.time() - t0

    n_contig = n_gather * C
    t0 = time.time()
    _ = dataset.mmap_patterns["patterns"][:n_contig]
    contig_time = time.time() - t0

    ratio = gather_time / max(contig_time, 1e-9)
    print(f"  Indexed gather ({n_gather} groups): {gather_time:.4f}s")
    print(f"  Contiguous read ({n_contig} patterns): {contig_time:.4f}s")
    print(f"  Gather / contiguous ratio: {ratio:.1f}x")

    # Disk usage
    mmap_dir = dataset.data_dir
    total_bytes = 0
    for root, dirs, files in os.walk(mmap_dir):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))
    print(f"  Memory map disk usage: {total_bytes / 1e6:.1f} MB")
    naive_bytes = len(dataset) * C * N * N * 4
    print(f"  Naive materialized size (M*C*H*W*4): {naive_bytes / 1e6:.1f} MB")
    print(f"  Storage savings: {naive_bytes / max(total_bytes, 1):.1f}x")

    print("  Memory map statistics COMPLETE.")


def main():
    tmp_dir = tempfile.mkdtemp(prefix="ptychotest_indexed_")
    print(f"Temporary directory: {tmp_dir}")
    try:
        dataset, create_time = section1_dataset_creation(tmp_dir)
        stats = section2_overlap_quality(dataset)
        section3_batch_inspection(dataset)
        section4_mmap_stats(dataset, create_time)
        print()
        print("=" * 60)
        print("ALL SECTIONS COMPLETE")
        print("=" * 60)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up {tmp_dir}")


if __name__ == "__main__":
    main()
