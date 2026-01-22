# Troubleshooting

## Shape Mismatch Errors

- Ensure the dataset `diff3d` shape matches `N` (square patterns).
- Confirm `gridsize` matches the model and grouping configuration.

## Model Load Failures

- Check that `<output>/wts.h5_<model_name>` exists.
- Ensure `params.dill` and `custom_objects.dill` are present in the model directory.
