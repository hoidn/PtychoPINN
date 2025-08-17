# Data Contracts for the PtychoPINN Pipeline

This document defines the official format for key data artifacts used in this project. All tools that generate or consume these files MUST adhere to these contracts.

---

## 1. Canonical Ptychography Dataset (`.npz` format)

This contract applies to any dataset that is considered "ready for training" or is the final output of a preparation pipeline (e.g., from `prepare.sh`).

**File Naming Convention:** `*_train.npz`, `*_test.npz`, `*_prepared.npz`

| Key Name      | Shape                 | Data Type      | Description                                                              | Notes                                                              |
| :------------ | :-------------------- | :------------- | :----------------------------------------------------------------------- | :----------------------------------------------------------------- |
| `diffraction` | `(n_images, H, W)`    | `float32`      | The stack of measured diffraction patterns (amplitude, not intensity).   | **Required.** Formerly `diff3d`. Must be 3D.                       |
| `Y`           | `(n_images, H, W)`    | `complex64`    | The stack of ground truth real-space object patches.                     | **Required for supervised training.** **MUST be 3D.** Squeeze any channel dimension. |
| `objectGuess` | `(M, M)`              | `complex64`    | The full, un-patched ground truth object.                                | **Required.**                                                      |
| `probeGuess`  | `(H, W)`              | `complex64`    | The ground truth probe.                                                  | **Required.**                                                      |
| `xcoords`     | `(n_images,)`         | `float64`      | The x-coordinates of each scan position.                                 | **Required.**                                                      |
| `ycoords`     | `(n_images,)`         | `float64`      | The y-coordinates of each scan position.                                 | **Required.**                                                      |
| `scan_index`  | `(n_images,)`         | `int`          | The index of the scan point for each diffraction pattern.                | Optional, but recommended.                                         |

---

## 2. Experimental and Raw Dataset Formats

Some datasets may not initially conform to the canonical format above and require preprocessing before use with PtychoPINN. These are typically raw experimental datasets or legacy formats.

### Raw Dataset Format (requires preprocessing)

Raw experimental datasets often use legacy naming conventions and data types that require conversion:

| Key Name      | Shape                 | Data Type      | Description                                                              | Action Required                                                    |
| :------------ | :-------------------- | :------------- | :----------------------------------------------------------------------- | :----------------------------------------------------------------- |
| `diff3d`      | `(n_images, H, W)`    | `uint16`       | Legacy diffraction patterns as intensity data.                          | **Convert to `diffraction` with float32 amplitude using <code-ref type="tool">scripts/tools/transpose_rename_convert_tool.py</code-ref>** |
| Missing `Y`   | N/A                   | N/A            | Ground truth patches not pre-computed.                                  | **Generate using <code-ref type="tool">scripts/tools/generate_patches_tool.py</code-ref>** |

### Preprocessing Requirements

Raw datasets must undergo format conversion to ensure PtychoPINN compatibility:

1. **Data Type Conversion:** `uint16` intensity → `float32` amplitude
2. **Key Renaming:** `diff3d` → `diffraction`
3. **Array Reshaping:** Ensure Y arrays are 3D (squeeze any singleton dimensions)

**Essential preprocessing command:**
```bash
python scripts/tools/transpose_rename_convert_tool.py raw_dataset.npz converted_dataset.npz
```

### Experimental Dataset Documentation

For detailed preprocessing workflows for specific experimental datasets, see:
- <doc-ref type="guide">docs/FLY64_DATASET_GUIDE.md</doc-ref> - FLY64 experimental dataset guide

---

## 3. AI Context Maskset (`.maskset` format)

This contract applies to any file used to specify inclusion patterns for AI context-generation tools, particularly the `/generate-doc-context` command.

**File Naming Convention:** `*.maskset`  
**Purpose:** To define a subset of the codebase for specific analysis tasks, typically by creating a "documentation-only" view for AI context priming or focused analysis.

### File Format

- **Type:** Plain text
- **Encoding:** UTF-8  
- **Structure:** A list of glob patterns, one per line

### Rules

1. **One Pattern Per Line:** Each line in the file represents a single glob pattern that will be used to select files.
2. **Recursive Globs:** The `**` pattern is supported for recursive directory matching (e.g., `ptycho/**/*.py`).
3. **Comments:** Lines starting with a `#` character are treated as comments and are ignored by the parser.
4. **Empty Lines:** Empty lines are ignored.
5. **Inclusion Only:** The patterns are used for inclusion only. There is no syntax for exclusion within the maskset file itself; filtering should be done by crafting more specific inclusion patterns.
6. **Order Matters:** Files are included in the order they match patterns, with earlier patterns taking precedence for organization.

### Example Maskset File (`doc_context.maskset`)

```
# Maskset for generating high-level architectural context.
# This focuses on the core library and workflow components.

# Include all modules in the main ptycho library
ptycho/**/*.py

# Include key workflow scripts
scripts/workflows/*.py
scripts/studies/run_complete_generalization_study.sh

# Include high-level documentation
docs/DEVELOPER_GUIDE.md
docs/architecture.md
```

### Common Usage Patterns

#### Architecture Overview Maskset
```
# Core library architecture
ptycho/config/*.py
ptycho/model.py
ptycho/diffsim.py
ptycho/loader.py

# High-level workflows
scripts/workflows/*.py

# Documentation
docs/DEVELOPER_GUIDE.md
docs/data_contracts.md
```

#### Data Pipeline Maskset
```
# Data loading and processing
ptycho/loader.py
ptycho/raw_data.py
ptycho/image/**/*.py

# Data preparation tools
scripts/tools/*_tool.py

# Data contracts
docs/data_contracts.md
```

#### Study-Specific Maskset
```
# Study orchestration
scripts/studies/run_*.sh
scripts/studies/aggregate_*.py

# Analysis workflows
scripts/analysis/*.py

# Study documentation
docs/studies/*.md
```

### Integration with Commands

The primary consumer of maskset files is the `/generate-doc-context` command:

```bash
# Generate context using a maskset file
/generate-doc-context --maskset architecture.maskset

# Generate context with output to file
/generate-doc-context --maskset data_pipeline.maskset --output context.md
```

### Best Practices

1. **Name Descriptively:** Use names that describe the focus area (e.g., `architecture.maskset`, `data_pipeline.maskset`)
2. **Document Purpose:** Always include a comment header explaining the maskset's intended use
3. **Start Broad, Then Narrow:** Begin with general patterns and add specific files as needed
4. **Test Coverage:** Run the command with `--dry-run` to verify which files will be included
5. **Version Control:** Commit useful masksets to the repository for team reuse

---