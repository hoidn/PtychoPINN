### Turn Summary
Added integration test `test_lazy_container_inference_integration` to verify lazy container works in the inference path via `create_ptycho_data_container()`.
Test confirms: container stores NumPy internally, tensor cache empty initially, lazy conversion on `.X` access, caching works, `coords_nominal` conversion for model.predict([X, coords]).
All tests pass: lazy loading suite 14/14 (1 intentional skip), model factory regression 3/3, integration test 2/2.
Next: Consider STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 G-scaled verification complete; focus can shift to remaining blockers (BASELINE-CHUNKED-001/002).
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2026-01-08T210000Z/ (pytest_integration.log, pytest_lazy_loading.log, pytest_model_factory.log)
