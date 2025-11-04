"""
Phase C dataset generation orchestration for fly64 dose/overlap study.

This module automates the pipeline:
  simulate → canonicalize → patch generation → train/test split → validate

for each dose in StudyDesign, producing DATA-001 compliant train/test NPZ pairs
with reproducible metadata for downstream overlap and training phases.

References:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md §Phase C
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T032018Z/phase_c_dataset_generation/plan.md
- specs/data_contracts.md §2 (canonical NPZ format)
- docs/GRIDSIZE_N_GROUPS_GUIDE.md §Inter-Group Overlap Control
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Import study design
from studies.fly64_dose_overlap.design import get_study_design
from studies.fly64_dose_overlap.validation import validate_dataset_contract

# Import ptycho config and tools
from ptycho.config.config import TrainingConfig, ModelConfig

# Import tool functions
from scripts.simulation.simulate_and_save import simulate_and_save
from scripts.tools.transpose_rename_convert_tool import transpose_rename_convert
from scripts.tools.generate_patches_tool import generate_patches
from scripts.tools.split_dataset_tool import split_dataset


@dataclass
class SimulationPlan:
    """Configuration plan for simulating one dose level."""
    dose: float
    training_config: TrainingConfig
    model_config: ModelConfig
    n_images: int
    seed: int
    patch_size: int


def build_simulation_plan(
    dose: float,
    base_npz_path: Path,
    design_params: Dict[str, Any],
) -> SimulationPlan:
    """
    Construct TrainingConfig and ModelConfig for a single dose.

    Args:
        dose: Photon dose (nphotons per exposure)
        base_npz_path: Path to base dataset with object/probe
        design_params: StudyDesign parameters (as dict)

    Returns:
        SimulationPlan with dose-specific configs and derived counts

    References:
        - docs/GRIDSIZE_N_GROUPS_GUIDE.md:141-151 (spacing context)
        - specs/data_contracts.md:207 (canonical format requirements)
    """
    # Load base dataset to determine n_images
    with np.load(base_npz_path) as data:
        n_images = len(data['xcoords'])

    # Extract design constants
    simulation_seed = design_params['rng_seeds']['simulation']
    patch_size = design_params['patch_size_pixels']

    # Build configs (using gridsize=1 for initial simulation)
    # Grouping will happen in Phase D
    model_config = ModelConfig(
        gridsize=1,
        N=patch_size,
    )

    training_config = TrainingConfig(
        model=model_config,
        train_data_file=str(base_npz_path),
        n_groups=n_images,  # gs=1 initially (one group per scan position)
        n_images=int(n_images),  # Required for legacy simulator coordinate array sizing
        nphotons=int(dose),
        # Use defaults for other params; they'll be overridden in phase E training
    )

    return SimulationPlan(
        dose=dose,
        training_config=training_config,
        model_config=model_config,
        n_images=n_images,
        seed=simulation_seed,
        patch_size=patch_size,
    )


def generate_dataset_for_dose(
    dose: float,
    base_npz_path: Path,
    output_root: Path,
    design_params: Dict[str, Any],
    *,
    simulate_fn=None,
    canonicalize_fn=None,
    patch_gen_fn=None,
    split_fn=None,
    validate_fn=None,
) -> Dict[str, Path]:
    """
    Orchestrate simulation → canonicalization → patch generation → split → validation.

    This function wires together the dataset generation pipeline for a single dose,
    ensuring outputs satisfy DATA-001 via validator and preserving y-axis separation.

    Args:
        dose: Photon dose level (nphotons per exposure)
        base_npz_path: Path to base dataset with object/probe
        output_root: Root directory for outputs (dose subdirs created here)
        design_params: StudyDesign parameters (as dict)
        simulate_fn: Injection point for simulation (default: simulate_and_save)
        canonicalize_fn: Injection point for canonicalization (default: transpose_rename_convert)
        patch_gen_fn: Injection point for patch generation (default: generate_patches)
        split_fn: Injection point for dataset split (default: split_dataset)
        validate_fn: Injection point for validation (default: validate_dataset_contract)

    Returns:
        Dictionary with paths: 'train', 'test', 'intermediate_dir'

    Raises:
        ValueError: If validation fails on train or test outputs

    References:
        - CONFIG-001: Ensure update_legacy_dict happens inside simulate_and_save
        - DATA-001: Validator enforces canonical keys/dtypes/layout
        - OVERSAMPLING-001: Preserve neighbor_count metadata for Phase D
    """
    # Default to production functions
    if simulate_fn is None:
        simulate_fn = simulate_and_save
    if canonicalize_fn is None:
        canonicalize_fn = transpose_rename_convert
    if patch_gen_fn is None:
        patch_gen_fn = generate_patches
    if split_fn is None:
        split_fn = split_dataset
    if validate_fn is None:
        validate_fn = validate_dataset_contract

    # Build simulation plan
    plan = build_simulation_plan(dose, base_npz_path, design_params)

    # Prepare output directories
    dose_dir = output_root / f"dose_{int(dose)}"
    dose_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating dataset for dose={dose:.0e} photons")
    print(f"Output directory: {dose_dir}")
    print(f"{'='*60}\n")

    # Stage 1: Simulate
    simulated_npz = dose_dir / "simulated_raw.npz"
    print(f"[Stage 1/5] Simulating diffraction patterns...")
    simulate_fn(
        config=plan.training_config,
        input_file_path=base_npz_path,
        output_file_path=simulated_npz,
        original_data_for_vis=None,
        seed=plan.seed,
        visualize=False,
    )
    print(f"  ✓ Simulation complete: {simulated_npz}\n")

    # Stage 2: Canonicalize (ensure NHW layout, rename diff3d→diffraction, etc.)
    canonical_npz = dose_dir / "canonical.npz"
    print(f"[Stage 2/5] Canonicalizing to DATA-001 format...")
    canonicalize_fn(
        in_file=simulated_npz,
        out_file=canonical_npz,
    )
    print(f"  ✓ Canonicalization complete: {canonical_npz}\n")

    # Stage 3: Generate patches (Y ground truth from objectGuess)
    patched_npz = dose_dir / "patched.npz"
    print(f"[Stage 3/5] Generating Y patches...")
    patch_gen_fn(
        input_path=canonical_npz,
        output_path=patched_npz,
        patch_size=plan.patch_size,
        k_neighbors=design_params['neighbor_count'],
        nsamples=1,
    )
    print(f"  ✓ Patch generation complete: {patched_npz}\n")

    # Stage 4: Split into train/test on y-axis
    print(f"[Stage 4/5] Splitting train/test on y-axis...")
    split_fn(
        input_path=patched_npz,
        output_dir=dose_dir,
        split_fraction=0.5,
        split_axis=design_params['train_test_split_axis'],
    )
    train_npz = dose_dir / f"{patched_npz.stem}_train.npz"
    test_npz = dose_dir / f"{patched_npz.stem}_test.npz"
    print(f"  ✓ Split complete:")
    print(f"    Train: {train_npz}")
    print(f"    Test:  {test_npz}\n")

    # Stage 5: Validate both train and test
    print(f"[Stage 5/5] Validating DATA-001 compliance...")
    for split_name, split_path in [('train', train_npz), ('test', test_npz)]:
        print(f"  Validating {split_name}...")
        validate_fn(
            dataset_path=split_path,
            design_params=design_params,
            expected_dose=dose,
        )
        print(f"    ✓ {split_name} validation passed")

    print(f"\n{'='*60}")
    print(f"Dataset generation complete for dose={dose:.0e}")
    print(f"{'='*60}\n")

    return {
        'train': train_npz,
        'test': test_npz,
        'intermediate_dir': dose_dir,
    }


def main():
    """CLI entry point to generate datasets for all doses in StudyDesign."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic fly64 dose sweep datasets (Phase C)"
    )
    parser.add_argument(
        '--base-npz',
        type=Path,
        default=Path('tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz'),
        help='Path to base NPZ with object/probe (default: fly001 prepared)',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('data/studies/fly64_dose_overlap'),
        help='Root directory for generated datasets (default: data/studies/fly64_dose_overlap)',
    )
    args = parser.parse_args()

    # Load study design
    design = get_study_design()
    design_dict = design.to_dict()

    print("=" * 80)
    print("Phase C: fly64 Dose Sweep Dataset Generation")
    print("=" * 80)
    print(f"Base dataset: {args.base_npz}")
    print(f"Output root:  {args.output_root}")
    print(f"Dose levels:  {design.dose_list}")
    print("=" * 80)
    print()

    # Verify base dataset exists
    if not args.base_npz.exists():
        print(f"ERROR: Base dataset not found: {args.base_npz}", file=sys.stderr)
        sys.exit(1)

    # Generate datasets for each dose
    manifest = {}
    for dose in design.dose_list:
        try:
            paths = generate_dataset_for_dose(
                dose=dose,
                base_npz_path=args.base_npz,
                output_root=args.output_root,
                design_params=design_dict,
            )
            manifest[f"dose_{int(dose)}"] = {
                'dose': dose,
                'train': str(paths['train']),
                'test': str(paths['test']),
                'intermediate_dir': str(paths['intermediate_dir']),
            }
        except Exception as e:
            print(f"\nERROR generating dose={dose}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Write manifest
    manifest_path = args.output_root / 'run_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("All datasets generated successfully!")
    print(f"Manifest written to: {manifest_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
