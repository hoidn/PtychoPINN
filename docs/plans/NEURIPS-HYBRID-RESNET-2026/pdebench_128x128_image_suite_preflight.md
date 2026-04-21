# PDEBench 128x128 Image-Suite Preflight

- Created: `2026-04-21T03:55:53.171504+00:00`
- Data root: `/home/ollie/Documents/pdebench-data`
- Ready tasks: `3` / `3`
- All tasks ready: `True`
- Missing listed data volume: `0.0` GB
- Data-root available bytes: `18885656576`

Meaningful benchmark rows must train on the full available training split after validation/test holdout. Capped, subsampled, smoke, and pilot runs are decision-support only.

| Task | Status | Native shape | Available supervision units | Blocker | Message |
| --- | --- | ---: | ---: | --- | --- |
| `swe` | `ready` | `128x128` | 100000 |  |  |
| `darcy` | `ready` | `128x128` | 10000 |  |  |
| `2d_cfd_cns` | `ready` | `128x128` | 190000 |  |  |

## Dataset Schema Details

- `swe`: dynamic state `*/data`, shape `[1000, 101, 128, 128, 1]`, axis `NTHWC`
- `darcy`: static input `nu`, shape `[10000, 128, 128]`, axis `NHW`; target `tensor`, shape `[10000, 1, 128, 128]`, axis `NCHW`
- `2d_cfd_cns`: dynamic multi-field state fields `density`, `Vx`, `Vy`, `pressure`, shape `[10000, 21, 128, 128]`, axis `NTHW`

## Next Action

If any task is `missing_file`, stage the official PDEBench file outside git or on an approved external data root before launching benchmark training.
