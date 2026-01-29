# Stage A Shootout — Shared Parameters

- **Seed:** 20260128
- **N:** 64
- **gridsize:** 1
- **nimgs_train:** 2
- **nimgs_test:** 2
- **nphotons:** 1e9
- **nepochs:** 50
- **fno_blocks:** 4
- **torch_loss_mode:** mae
- **torch_infer_batch_size:** 8

## Arms

| Arm | Architecture | Clip Algorithm | Clip Value |
|-----|-------------|---------------|------------|
| control | hybrid | norm | 1.0 |
| stable | stable_hybrid | norm | 0.0 (disabled) |
| agc | hybrid | agc | 0.01 |
