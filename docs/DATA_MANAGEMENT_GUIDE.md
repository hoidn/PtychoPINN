# Data Management Guide

## ⚠️ CRITICAL: Never Commit Data Files to Git

This guide establishes mandatory practices for handling data files in the PtychoPINN repository.

## The Golden Rule

**NEVER commit data files (NPZ, HDF5, etc.) directly to the Git repository.**

Data files belong in:
1. Cloud storage (Google Drive, Dropbox, etc.)
2. Shared network drives
3. Local directories excluded from Git
4. External data repositories (Zenodo, FigShare, etc.)

## Why This Matters

- **Repository Size**: Data files bloat the repository, making clones slow and expensive
- **Git History**: Once committed, files remain in history even after deletion
- **Collaboration**: Large repos are difficult to clone and work with
- **Storage Limits**: GitHub has strict file size limits (100MB warning, 100GB repo limit)

## File Types That Must Never Be Committed

The following file extensions are explicitly blocked in `.gitignore`:

### Data Files
- `*.npz` - NumPy compressed arrays
- `*.npy` - NumPy arrays  
- `*.h5`, `*.hdf5` - HDF5 data files
- `*.mat` - MATLAB data files
- `*.pkl`, `*.pickle`, `*.dill` - Python pickled objects

### Model Files
- `*.h5.zip` - Compressed model weights
- `*.ckpt` - TensorFlow checkpoints
- `*.pb` - Protocol buffer files

### Directories
- `datasets/` - All dataset storage
- `data/` - General data directory
- `outputs/` - Training outputs
- `results/` - Experimental results
- `simulations/` - Simulation outputs
- `tike_outputs/` - Tike reconstruction outputs
- Any directory matching `*_generalization_study*/`

## Best Practices for Data Management

### 1. Before Starting Work

```bash
# Check that data files are properly ignored
git status --ignored

# Verify no large files are staged
git diff --cached --name-only | xargs -I {} ls -lh {}
```

### 2. Sharing Datasets

**Option A: Cloud Storage**
```python
# Document in README or scripts
DATASET_URL = "https://drive.google.com/file/d/YOUR_FILE_ID"
# Provide download instructions
```

**Option B: Symbolic Links**
```bash
# Link to data stored outside repo
ln -s /path/to/shared/data/fly001.npz datasets/fly001.npz
# Symbolic links can be committed (they're just pointers)
```

**Option C: Data Manifest**
```yaml
# datasets/manifest.yaml
fly001:
  url: "https://example.com/data/fly001.npz"
  md5: "abc123..."
  size: "220MB"
  description: "Fly wing ptychography dataset"
```

### 3. Working with Large Outputs

```bash
# Use output directories outside the repo
ptycho_train --output_dir ~/ptycho_outputs/experiment_001

# Or use ignored directories within repo
ptycho_train --output_dir outputs/experiment_001
```

### 4. Checking File Sizes Before Commit

```bash
# Find large files in staging area
git diff --cached --name-only | xargs -I {} sh -c 'ls -lh {} | awk "\$5 ~ /M|G/ {print}"'

# Set up pre-commit hook (see below)
```

## Pre-Commit Hook to Prevent Accidents

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Prevent commits of files larger than 5MB

MAX_SIZE=5242880  # 5MB in bytes
BLOCKED_EXTENSIONS="npz npy h5 hdf5 mat pkl pickle dill"

# Check staged files
for file in $(git diff --cached --name-only); do
    # Check file size
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        if [ "$size" -gt "$MAX_SIZE" ]; then
            echo "ERROR: File $file is larger than 5MB (size: $size bytes)"
            echo "Large data files must not be committed to git!"
            exit 1
        fi
    fi
    
    # Check file extension
    ext="${file##*.}"
    for blocked in $BLOCKED_EXTENSIONS; do
        if [ "$ext" = "$blocked" ]; then
            echo "ERROR: File $file has blocked extension .$ext"
            echo "Data files (.$ext) must not be committed to git!"
            exit 1
        fi
    done
done

echo "Pre-commit check passed: No large or data files detected"
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## If You Accidentally Committed Data Files

### Recent Commit (Not Pushed)

```bash
# Undo the last commit but keep changes
git reset --soft HEAD~1

# Remove the data file from staging
git reset HEAD path/to/data.npz

# Re-commit without the data file
git commit -m "your message"
```

### Already Pushed or Old Commits

Use BFG Repo-Cleaner to purge from history:

```bash
# Download BFG if not available
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar -O bfg.jar

# Remove all NPZ files from history
java -jar bfg.jar --delete-files "*.npz" --no-blob-protection

# Clean up the repository
git reflog expire --expire=now --all && git gc --prune=now --aggressive

# Force push (coordinate with team!)
git push --force
```

## Verification Commands

```bash
# Check what's ignored
git check-ignore -v *.npz

# Find large files in repo
find . -type f -size +5M ! -path "./.git/*" 

# Check repo size
du -sh .git

# List tracked files by size
git ls-files | xargs -I {} ls -la {} | sort -k5 -rn | head -20
```

## Team Guidelines

1. **Code Review**: Reviewers must check for accidental data file commits
2. **CI/CD**: Set up automated checks in GitHub Actions
3. **Documentation**: Always document where to obtain datasets
4. **Communication**: Inform team before force-pushing after cleanup

## GitHub Actions Check

Add to `.github/workflows/check-file-size.yml`:

```yaml
name: Check File Sizes

on: [push, pull_request]

jobs:
  check-large-files:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Check for large files
      run: |
        # Find files larger than 5MB
        large_files=$(find . -type f -size +5M ! -path "./.git/*" ! -path "./bfg.jar")
        if [ ! -z "$large_files" ]; then
          echo "ERROR: Large files detected:"
          echo "$large_files"
          exit 1
        fi
        
    - name: Check for data files
      run: |
        # Check for data file extensions
        data_files=$(find . -type f \( -name "*.npz" -o -name "*.npy" -o -name "*.h5" -o -name "*.hdf5" \) ! -path "./.git/*")
        if [ ! -z "$data_files" ]; then
          echo "ERROR: Data files detected:"
          echo "$data_files"
          exit 1
        fi
```

## Summary

- ✅ Use `.gitignore` to block data files
- ✅ Set up pre-commit hooks for local protection  
- ✅ Use cloud storage or external repositories for data
- ✅ Document data sources in README files
- ✅ Implement CI checks as a final safety net
- ❌ Never use `git add .` without checking
- ❌ Never force-add ignored files with `git add -f`