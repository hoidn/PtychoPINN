# EMERGENCY: Fix Large File Push Issue

## The Problem
Your repository contains large NPZ files (>100MB) in its git history that GitHub is rejecting:
- `test_sim_final.npz` (219MB) 
- `test_gs2.npz` (110MB)
- `test_perf.npz` (55MB)
- `test_gs1.npz` (3.2MB)

These files are blocking all pushes to GitHub, even though they've been removed from tracking.

## Immediate Solution Options

### Option 1: Force Clean with BFG (Recommended)

1. **Clone a fresh copy** (to preserve your current work):
```bash
cd ~/Documents
git clone --mirror https://github.com/hoidn/PtychoPINN.git PtychoPINN-mirror
cd PtychoPINN-mirror
```

2. **Run BFG on the mirror**:
```bash
# Download BFG if needed
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar -O bfg.jar

# Remove all NPZ files
java -jar bfg.jar --delete-files "*.npz" --no-blob-protection

# Also remove any file over 50MB
java -jar bfg.jar --strip-blobs-bigger-than 50M --no-blob-protection
```

3. **Clean up**:
```bash
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

4. **Force push the cleaned mirror**:
```bash
git push --force
```

5. **Update your local repo**:
```bash
cd ~/Documents/PtychoPINN2
git fetch origin
git reset --hard origin/feature/high-performance-patch-extraction
```

### Option 2: Create New Clean Branch

If BFG continues to fail:

1. **Export your patches** (preserves your work):
```bash
cd ~/Documents/PtychoPINN2
git format-patch origin/main..HEAD --stdout > my-changes.patch
```

2. **Clone fresh from GitHub**:
```bash
cd ~/Documents
git clone https://github.com/hoidn/PtychoPINN.git PtychoPINN-clean
cd PtychoPINN-clean
git checkout -b feature/high-performance-patch-extraction-clean
```

3. **Apply your patches**:
```bash
git apply ../PtychoPINN2/my-changes.patch
```

4. **Push the clean branch**:
```bash
git push origin feature/high-performance-patch-extraction-clean
```

### Option 3: Interactive Rebase (Advanced)

If you want to preserve the exact commit history minus the NPZ files:

```bash
# Find the commit before NPZ files were added
git log --oneline --all -- "*.npz" | tail -1

# Start interactive rebase from that commit
git rebase -i <commit-hash>^

# In the editor, mark commits with NPZ files as 'edit'
# For each commit, remove NPZ files:
git rm --cached *.npz
git commit --amend --no-edit
git rebase --continue

# Force push when done
git push --force origin feature/high-performance-patch-extraction
```

## Prevention Measures (Already Implemented)

✅ **`.gitignore` updated** - All data files blocked
✅ **Pre-commit hook installed** - Prevents large file commits
✅ **Documentation added** - `docs/DATA_MANAGEMENT_GUIDE.md`
✅ **README warning added** - Prominent notice about data files

## Team Communication Template

Send this to your team after fixing:

```
Subject: Important: Git History Cleanup Required

Team,

I've cleaned our repository to remove large data files (NPZ) that were accidentally committed. This was necessary because GitHub was blocking all pushes due to files exceeding 100MB.

Action Required:
1. Back up any local changes
2. Re-clone the repository OR run:
   git fetch origin
   git reset --hard origin/<your-branch>

Prevention:
- Run ./scripts/setup_git_hooks.sh to install protective hooks
- Never commit data files (*.npz, *.h5, etc.)
- See docs/DATA_MANAGEMENT_GUIDE.md for details

The repository is now protected against future data file commits.
```

## If All Else Fails

Contact GitHub support about temporarily increasing the file size limit while you clean the repository, or consider using Git LFS for legitimate large files that must be versioned.

## Verification After Cleanup

```bash
# Check no NPZ files remain in history
git rev-list --objects --all | grep -E "\.npz$"

# Check repository size
du -sh .git

# Verify push works
git push origin --dry-run --all
```