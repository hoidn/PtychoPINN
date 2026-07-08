### Turn Summary
Added a CLI+JSON emitting `debug_fno_gradients.py` and smoke test, and verified collection and test pass.
Ran loss-unit tests and the activation monitor on grid-lines data; loss units pass, lifter tail ratio is ~3.21 (<5), and low_freq_ratio ~62 indicates strong DC dominance.
The gradient report shows spectral/local gradient ratio ~0.051 (<0.1), supporting the spectral gradient collapse hypothesis.
Next: prioritize spectral weight scaling investigation before any input-transform adjustments.
Artifacts: .artifacts/fno_hybrid_convergence/ (pytest_debug_fno_gradients_red.log, pytest_debug_fno_gradients_green.log, pytest_loss_units.log, pytest_debug_fno_gradients_collect.log, debug_fno_activations.log, debug_fno_gradients.log, ruff.log), .artifacts/debug_fno_gradients/gradient_report.json, .artifacts/debug_fno_activations/activation_report.json
