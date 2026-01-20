# 2026-01-20T103000Z — Intensity scaler inspection

- Implemented `bin/inspect_intensity_scaler.py` to load a scenario's `wts.h5.zip`, read the saved `params.dill`, and report the trained IntensityScaler/IntensityScaler_inv gains alongside the recorded `intensity_scale`.
- `gs1_ideal` and `gs2_ideal` both report `exp(log_scale)=988.211666` while the archived `intensity_scale` is `988.211669921875` (delta `-3.9e-06`, ratio `0.999999996`), ruling out a scale mismatch between scenarios.
- Artifacts:
  - `intensity_scaler_gs1_ideal.json` / `.md`
  - `intensity_scaler_gs2_ideal.json` / `.md`
