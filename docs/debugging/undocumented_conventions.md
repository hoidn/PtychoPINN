# Undocumented Conventions

- `params.cfg` is treated as a global singleton and mutated at runtime.
- Some scripts use legacy CLI args (`train_data_file_path`) and map them to
  newer names (`train_data_file`).
