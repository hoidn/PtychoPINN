# Implementation Plan: Grid-Based Simulation for Dose Study

## Initiative
- **ID:** ALIGN-DOSE-STUDY-GRID-001
- **Title:** Add Grid-Based Simulation Mode to Dose Study
- **Owner:** Ralph
- **Status:** pending
- **Parent:** STUDY-SYNTH-DOSE-COMPARISON-001

## Goal
Add a `--grid-mode` flag to `dose_response_study.py` that generates data using the legacy grid-based `mk_simdata()` path, matching `notebooks/dose_dependence.ipynb` behavior.

## Why
The current script uses random coordinate sampling (nongrid). The notebook uses fixed grid extraction. Results can't be directly compared. Grid mode enables reproducibility validation.

## Exit Criteria
1. `python scripts/studies/dose_response_study.py --grid-mode` runs without error
2. Generated data has grid-aligned coordinates (visual check)
3. Existing nongrid behavior unchanged (no `--grid-mode` flag = current behavior)

## Compliance
- `CONFIG-001`: Set `params.cfg` before `mk_simdata()` call
- `CONVENTION-001`: Grid mode explicitly uses legacy system

---

## Phase A — Implementation

### A1: Add grid-mode simulation function

```python
def simulate_datasets_grid_mode(nphotons, probeGuess):
    """Grid-based simulation matching notebooks/dose_dependence.ipynb."""
    from ptycho.params import params as p
    from ptycho.diffsim import mk_simdata

    # Match notebook's dose.py::init()
    p.set('N', 128)
    p.set('gridsize', 2)
    p.set('offset', 4)
    p.set('outer_offset_train', 8)
    p.set('outer_offset_test', 20)
    p.set('nphotons', nphotons)
    p.set('size', 392)
    p.set('data_source', 'lines')
    p.set('max_position_jitter', 3)
    p.set('sim_jitter_scale', 0.0)

    X_train, Y_I_train, Y_phi_train, _ = mk_simdata(
        nimgs=2, size=392, probe=probeGuess, outer_offset=8
    )
    X_test, Y_I_test, Y_phi_test, _ = mk_simdata(
        nimgs=2, size=392, probe=probeGuess, outer_offset=20
    )

    return (X_train, Y_I_train, Y_phi_train), (X_test, Y_I_test, Y_phi_test)
```

### A2: Add CLI flag

```python
parser.add_argument('--grid-mode', action='store_true',
                    help='Use legacy grid-based simulation (notebook-compatible)')
```

### A3: Route in main()

```python
if args.grid_mode:
    train_data, test_data = simulate_datasets_grid_mode(nphotons, probeGuess)
else:
    train_data, test_data = simulate_datasets(...)  # existing path
```

### A4: Create PtychoDataContainer directly (no RawData bridge needed)

`mk_simdata()` outputs can construct `PtychoDataContainer` directly:

```python
from ptycho.loader import PtychoDataContainer
from ptycho.diffsim import scale_nphotons

X, Y_I, Y_phi, coords = mk_simdata(...)

container = PtychoDataContainer(
    X=X,
    Y_I=Y_I,
    Y_phi=Y_phi,
    norm_Y_I=scale_nphotons(tf.convert_to_tensor(X)),
    YY_full=None,
    coords_nominal=coords,
    coords_true=coords,
    nn_indices=None,
    global_offsets=None,
    local_offsets=None,
    probeGuess=probeGuess
)
```

No reshape or bridge function required.

---

## Phase B — Validation

### B1: Smoke test
Run with `--grid-mode` and verify no crash.

### B2: Visual check
Plot coordinates to confirm grid pattern (not random scatter).

### B3: Regression check
Run without `--grid-mode` and verify identical to pre-change behavior.

---

## Key Differences: Grid vs Nongrid

| Parameter | Nongrid (current) | Grid (notebook) |
|-----------|-------------------|-----------------|
| N | 64 | 128 |
| Object size | 128 (2×N) | 392 |
| Coordinates | Random uniform | Fixed grid |
| Grouping | KDTree post-hoc | Built-in |
| Amplitude shift | [0.5, 1.5] | None |
| nimgs default | 2000/128 | 2/2 |

---

## Open Question
Should grid mode also adjust training to match notebook (N=128, different epochs, etc.), or just data generation? Start with data generation only; extend if needed.

## Resolved Questions
- **RawData bridge needed?** No. `mk_simdata()` outputs feed directly into `PtychoDataContainer` constructor. See A4.

---

## Phase C — Documentation

### C1: Update fix_plan.md ledger
Add entry to `docs/fix_plan.md`:
```markdown
### [ALIGN-DOSE-STUDY-GRID-001] Add Grid-Based Simulation Mode to Dose Study
- Depends on: STUDY-SYNTH-DOSE-COMPARISON-001
- Priority: Low (Reproducibility)
- Status: <status>
- Working Plan: `plans/active/ALIGN-DOSE-STUDY-GRID-001/implementation.md`
```

### C2: Update parent initiative
Add note to `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/implementation.md` under Open Questions:
- Grid mode available via `--grid-mode` flag for notebook-compatible data generation

### C3: Update script docstring
Add `--grid-mode` usage to `scripts/studies/dose_response_study.py` module docstring.

---

## Artifacts
- `plans/active/ALIGN-DOSE-STUDY-GRID-001/reports/` — validation outputs
