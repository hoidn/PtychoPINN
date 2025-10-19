Summary: Capture Phase B fixture scope telemetry for PyTorch regression.
Mode: Perf
Focus: [TEST-PYTORCH-001] Author PyTorch integration workflow regression — Phase B1 fixture requirements
Branch: feature/torchapi
Mapped tests: none — measurement loop
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/{fixture_scope.md,logs/}

Do Now:
1. TEST-PYTORCH-001 B1.A @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Profile Run1084 dataset axes/dtypes/counts; record results in fixture_scope.md; tests: none.
2. TEST-PYTORCH-001 B1.B @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Run paired training dry-runs (epochs/n_images sweep) to measure runtime/output size; save terminals to logs/ and summarize in fixture_scope.md; tests: none.
3. TEST-PYTORCH-001 B1.C @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Draft acceptance criteria bullets (target runtime, subset size, dtype conversions) in fixture_scope.md; tests: none.

If Blocked: Document unresolved questions plus partial measurements in fixture_scope.md, leave B1 rows `[P]`, and note blockers in docs/fix_plan.md Attempts before ending the loop.

Priorities & Rationale:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Phase B1 tasks define scope before generator work.
- specs/data_contracts.md:16 — Canonical dtype/shape expectations for new fixture.
- docs/workflows/pytorch.md:120 — CONFIG-001 enforcement and CLI knobs referenced during runtime sweeps.
- plans/pytorch_integration_test_plan.md:15 — Original fixture intent requiring deterministic subset.

How-To Map:
- mkdir -p plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/logs
- python - <<'PY' > plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/dataset_probe.txt
from pathlib import Path
import numpy as np
path = Path('datasets/Run1084_recon3_postPC_shrunk_3.npz')
with np.load(path) as data:
    keys = sorted(data.files)
    print('keys:', keys)
    arr = data['diffraction'] if 'diffraction' in data else data.get('diff3d')
    print('diff_shape', arr.shape, 'dtype', arr.dtype, 'min', float(arr.min()), 'max', float(arr.max()))
    if 'Y' in data:
        print('Y_shape', data['Y'].shape, 'dtype', data['Y'].dtype)
    print('objectGuess', data['objectGuess'].shape, data['objectGuess'].dtype)
    print('probeGuess', data['probeGuess'].shape, data['probeGuess'].dtype)
    for key in ('xcoords','ycoords'):
        if key in data:
            vals = data[key]
            print(key, vals.dtype, 'count', vals.shape[0], 'min', float(vals.min()), 'max', float(vals.max()))
PY
- For each runtime sample, run:
  - `CUDA_VISIBLE_DEVICES="" time python -m ptycho_torch.train --train_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz --test_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz --output_dir tmp/phase_b_fixture/run_ep2_n64 --max_epochs 2 --n_images 64 --gridsize 1 --batch_size 4 --device cpu --disable_mlflow`
  - `CUDA_VISIBLE_DEVICES="" time python -m ptycho_torch.train --train_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz --test_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz --output_dir tmp/phase_b_fixture/run_ep1_n16 --max_epochs 1 --n_images 16 --gridsize 1 --batch_size 2 --device cpu --disable_mlflow`
  Capture stdout/stderr with `tee` into `plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/logs/train_ep*_*.log`, then remove `tmp/phase_b_fixture/run_*` directories after collecting artifact sizes via `du -sh` (record numbers in fixture_scope.md).
- Summarize dataset probe + timing deltas inside fixture_scope.md (include runtime, exit status, artifact sizes, and observations about dtype/orientation).

Pitfalls To Avoid:
- Do not modify tests or generator scripts yet; this loop is evidence-only.
- Keep CUDA disabled (CPU-only) for both runs to match existing runtime profile.
- Clean up tmp directories after measurements to avoid polluting the repo.
- Record exact commands/outputs; no paraphrasing in fixture_scope.md.
- If runtime exceeds 2 minutes, stop and flag in fixture_scope.md instead of retrying endlessly.

Pointers:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md
- plans/active/TEST-PYTORCH-001/implementation.md:38
- specs/data_contracts.md:12
- docs/workflows/pytorch.md:210
- docs/findings.md:8

Next Up: 1. TEST-PYTORCH-001 Phase B2 generator design + TDD stub once scope doc finalized.
