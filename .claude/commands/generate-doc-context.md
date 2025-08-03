# Command: /generate-doc-context <maskset-file-or-patterns>

**Goal:** Create an isolated, "documentation-only" view of the codebase in a temporary `git worktree` for AI context priming or architectural review.

**Usage:**
- `/generate-doc-context doc_context.maskset` (using a maskset file)
- `/generate-doc-context 'ptycho/**/*.py' 'scripts/tools/*.py'` (using direct glob patterns)

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**YOUR ROLE IS AN AUTONOMOUS ORCHESTRATOR.**
1.  You MUST parse the maskset (either from a file or from the command arguments).
2.  You MUST create a new, clean `git worktree` in a temporary directory.
3.  You MUST run the `strip_code.py` script inside that worktree with the specified masks.
4.  You MUST report the path to the new worktree and provide instructions for cleanup.

**DO NOT:**
-   ❌ Modify the user's current working directory. All modifications must happen inside the worktree.
-   ❌ Skip the `git worktree` step. Isolation is critical.

---

## 🤖 **YOUR EXECUTION WORKFLOW**

### Step 1: Parse Maskset and Define Paths

```bash
# The user's input is in $ARGUMENTS
MASK_INPUT="$ARGUMENTS"
WORKTREE_DIR="doc_context_worktree_$(date +%s)"
MASK_PATTERNS=()

# Check if the input is a file or direct patterns
if [ -f "$MASK_INPUT" ]; then
    echo "Reading glob patterns from maskset file: $MASK_INPUT"
    # Read non-empty, non-comment lines from the file into the array
    while IFS= read -r line; do
        [[ -n "$line" && ! "$line" =~ ^\s*# ]] && MASK_PATTERNS+=("$line")
    done < "$MASK_INPUT"
else
    echo "Using glob patterns provided directly as arguments."
    MASK_PATTERNS=($ARGUMENTS)
fi

if [ ${#MASK_PATTERNS[@]} -eq 0 ]; then
    echo "❌ ERROR: No valid patterns found in the maskset or arguments."
    exit 1
fi

echo "Patterns to be processed:"
printf " - %s\n" "${MASK_PATTERNS[@]}"
```

### Step 2: Create the Isolated Git Worktree

```bash
# Check if the worktree directory already exists
if [ -d "$WORKTREE_DIR" ]; then
    echo "⚠️ Warning: Worktree directory '$WORKTREE_DIR' already exists. Removing it first."
    git worktree remove --force "$WORKTREE_DIR"
fi

echo "Creating a new, clean worktree at: ./$WORKTREE_DIR"
# Create a worktree based on the current HEAD
git worktree add "$WORKTREE_DIR" HEAD

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to create git worktree. Please ensure you are in a git repository."
    exit 1
fi
```

### Step 3: Run the Code Stripping Script Inside the Worktree

```bash
echo "Running the code stripping script inside the worktree..."

# The script to run is at its original location, but we execute it from
# within the worktree directory to ensure paths are resolved correctly.
(cd "$WORKTREE_DIR" && python ../scripts/tools/strip_code.py "${MASK_PATTERNS[@]}")

if [ $? -ne 0 ]; then
    echo "❌ ERROR: The strip_code.py script failed. The worktree may be in a partial state."
    echo "You can inspect it at: ./$WORKTREE_DIR"
    exit 1
fi
```

### Step 4: Report Success and Provide Cleanup Instructions

```bash
echo ""
echo "✅ Success! A documentation-only view of the codebase has been created."
echo ""
echo "You can now inspect the result at:"
echo "  $WORKTREE_DIR"
echo ""
echo "This directory contains a full copy of the repository, but with the code stripped from the files matching your maskset, leaving only the module-level docstrings."
echo ""
echo "---"
echo "🧹 **Cleanup Instructions**"
echo "When you are finished, you can remove the worktree with the following command:"
echo "  git worktree remove --force $WORKTREE_DIR"
echo ""
echo "You can also remove the worktree's administrative files from the .git directory with:"
echo "  git worktree prune"
```

