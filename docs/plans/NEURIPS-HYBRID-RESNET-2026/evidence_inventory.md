# NeurIPS Hybrid ResNet Phase 0 Evidence Inventory

## Scope and Sources

This Phase 0 pass inventories reusable evidence for the NeurIPS Hybrid ResNet campaign. It does not select a PDE benchmark, launch CDI regeneration, run baselines, run `N=256` variants, or create `/home/ollie/Documents/neurips/` artifacts.

Documents and sources used:

- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-0-evidence-inventory/execution_plan.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-0-evidence-inventory/tranche-context.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/workflows/pytorch.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- `scripts/reconstruction/hio_cdi_benchmark.py`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- primary PDE benchmark sources recorded in `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/pde_candidate_inventory.json`

## Gate Status

| Gate | Status | Evidence |
| --- | --- | --- |
| Roadmap Phase 0 only | pass | This pass created inventory and planning artifacts only. No PDE training, CDI rerun, N=256 variant, or paper-facing artifact was launched. |
| Paper-grade versus decision-support classification | pass-with-gaps | Raw CDI JSON separates `paper-grade`, `decision-support`, `not-usable`, and `unknown` categories. No paper-grade CDI N=128 anchor was found. |
| Lost `128x128` anchor condition | pass-with-gaps | The local inventory found legacy `pinn_hybrid` artifacts but no current `pinn_hybrid_resnet` paper-grade anchor. |
| Regeneration note when no anchor recovered | pass | `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md` schedules a fresh Phase 3 regeneration path, and the documented wrapper command parses with `grid_lines_compare_wrapper.py`. |
| Neutral PDE candidate inventory | pass | Three PDE/forward-modeling candidates are listed with `not_a_selection=true`; Phase 1 chooses primary/fallback. |
| N=256 boundary | pass | Existing N=256 records are labeled secondary scaling context only. |
| Artifact hygiene | pass | Raw machine-readable inventory is under ignored `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/`. |

## CDI N=128 Anchor Candidates

Raw source: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_128_hybrid_candidates.json`.

Summary:

- paper-grade: none
- decision-support: six historical legacy `pinn_hybrid` candidates
- not-usable: one wrapper root with no Hybrid/Hybrid ResNet metric entry
- unknown: none left as gate-satisfying evidence

The inspected decision-support roots include:

- `outputs/grid_lines_gs1_n128_e50_phi_all_rerun1`
- `outputs/grid_lines_gs1_n128_e20_phi_all`
- `outputs/grid_lines_gs1_n128_e20_phi_cnn_hybrid`
- `outputs/grid_lines_gs1_n128_tf1_torch50_neuralop_clip1_hybrid`
- `outputs/grid_lines_gs1_n128_tf1_torch50_neuralop_clip0_hybrid_log1p`
- `outputs/grid_lines_gs1_n128_tf1_torch50_neuralop_clip2_hybrid`

The blocking gaps are consistent across these candidates: no current `pinn_hybrid_resnet` metrics key, missing invocation artifacts, missing recorded git commit, incomplete seed/config/scheduler provenance, and incomplete dataset/split proof. These artifacts are useful for triage only and must not be used as the paper-grade CDI anchor.

Anchor gate statement: no complete paper-grade CDI anchor identified; `cdi_anchor_regeneration_plan.md` schedules a fresh `128x128` Hybrid ResNet regeneration run for Roadmap Phase 3.

## CDI Baseline Inventory

Raw source: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_baseline_candidates.json`.

Local neural baseline context:

- TensorFlow `pinn`/CNN-style and wrapper baseline rows exist as decision-support in historical N=128 grid-lines wrapper roots.
- Torch `pinn_fno` rows exist in historical roots and can inform a compact Phase 3 baseline choice.
- Legacy `pinn_hybrid` rows are historical predecessor evidence only; they are not current Hybrid ResNet evidence.
- `pinn_fno_vanilla` and current `pinn_hybrid_resnet` reusable metrics were not found locally.
- Unavailable metric values in the raw baseline inventory are represented as JSON `null`, not non-standard `NaN`, so strict JSON parsers can consume the artifact.

Non-ML and revision-study context:

- PyNX HIO/ER single-shot CDI evidence exists under `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/`, but it is an N=64 Table-2-style revision-study comparator, not a same-protocol NeurIPS N=128 row.
- The same-split PtychoPINN N=64 comparator is revision-study context only.
- The repo-local `known_probe_object_hio_er` smoke row is not usable because the metric payload lacks trusted evaluation metrics.

Baseline gaps:

- No local baseline can be labeled protocol-compatible with the future regenerated Hybrid ResNet anchor until Phase 3 names the comparator contract and either reruns or recovers complete provenance on the same dataset/split.
- Cheap reruns are plausible for CNN/PINN, FNO, and FNO-vanilla style baselines, but Phase 0 did not run them.

## N=256 Secondary Scaling Inventory

Raw source: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_n256_candidates.json`.

All N=256 entries are secondary scaling context only:

- `docs/studies/lines_256_dataset.md`: defines the `custom_npz_pair_n256` / `lines_256` dataset contract, but the compatibility train/test NPZs are absent locally and no reusable metrics were found.
- `scripts/studies/run_lines_256_arch_experiment.py`: records fixed `N=256` Hybrid ResNet settings and higher-mode presets such as 24 and 48, but no output metrics were found.
- `scripts/studies/runbooks/run_nersc_scan807_cameraman_study_n256.py`: local NERSC scan807/cameraman HDF5 inputs exist, but no N=256 orchestration output metrics were found.
- `scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py`: contains N=256 promotion gates, mode-count axes, and runtime/model-size limits, but no local N=256 sweep summary was found.
- `docs/studies/lines_256_arch_improvement_loop.md`: operational guidance only; it requires a fresh baseline in its own loop and is not reusable result evidence.

Higher-mode selection remains a Phase 4 question. Phase 0 does not promote N=256 to the CDI headline and does not run higher-mode variants.

## PDE Candidate Inventory for Phase 1

Raw source: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/pde_candidate_inventory.json`.

Neutral candidates:

| Candidate | Bucket | Task type | Phase 1 focus |
| --- | --- | --- | --- |
| PDEBench 2D incompressible Navier-Stokes or compressible fluid task | fluids / operator learning | `forward_prediction` | Pick a disk-feasible subset, pin rollout metrics, and identify local FNO/U-Net baselines. |
| PDEArena Maxwell-3D | wave propagation / Maxwell | `wave_propagation` | Decide whether a 3D or reduced task is feasible on the RTX 3090 without expanding implementation scope. |
| OpenFWI 2D acoustic full waveform inversion | inverse wave / seismic FWI | `inverse_reconstruction` | Choose a small 2D shard, decide whether official InversionNet can run locally, and pin metrics/splits. |

This is not a selection. Phase 0 does not select a primary or fallback benchmark. Phase 1 must score candidates for fit, maturity, metric clarity, data size, install burden, RTX 3090 feasibility, local baselines, published-SOTA availability, and paper-story fit.

## Environment Constraints

Raw source: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/environment_probe.json`.

Current local constraints:

- Python: `3.11.13`
- PyTorch: `2.9.1+cu128`
- CUDA-visible GPU: one NVIDIA GeForce RTX 3090 with 24576 MiB VRAM
- Packages available: `torch`, `lightning`, `neuralop`, `pynx`
- Package unavailable: `pdearena`
- Disk: `df -h` reported about 31 GB free on the root filesystem, so full benchmark dataset downloads are a serious Phase 1 risk.

## Raw Artifact Links

Ignored raw artifacts for this pass:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/environment_probe.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/local_metrics_index.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_128_hybrid_candidates.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_baseline_candidates.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_n256_candidates.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/pde_candidate_inventory.json`

These files are intentionally ignored and should not be committed.
They are still part of the workflow handoff contract and must pass strict JSON parsing before later phases consume them.

## CDI Anchor Regeneration Note

Required because no paper-grade anchor was recovered:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md`

The note records the current Hybrid ResNet baseline, `docs/studies/index.md`, `tests/torch/test_grid_lines_hybrid_resnet_integration.py`, a wrapper/runner command source, seed/config/provenance requirements, metric contract, qualitative output plan, runtime guardrails, and the later-phase boundary.

## Carry-Forward Notes

- Existing raw CDI, baseline, N=256, metrics, and environment JSON artifacts parsed and were reused where they matched the current Phase 0 scope.
- PDE candidate inventory was newly written in this pass.
- `docs/index.md` was updated for discoverability of the durable inventory and regeneration note.
- The current checkout was dirty before this pass; unrelated files were left alone.

## Non-Goals Confirmed

This pass did not:

- run the `128x128` Hybrid ResNet regeneration
- run compact CDI baselines or ablations
- run or install PDE benchmarks
- choose a PDE primary/fallback
- run N=256 scaling variants
- create `/home/ollie/Documents/neurips/` artifacts
- modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`
- create a worktree
