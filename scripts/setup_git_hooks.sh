#!/bin/bash
# Setup script for git hooks to prevent committing large data files

echo "Setting up git hooks to prevent data file commits..."

# Create hooks directory if it doesn't exist
mkdir -p .git/hooks

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook to prevent committing large files and data files

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MAX_SIZE=5242880  # 5MB in bytes
BLOCKED_EXTENSIONS="npz npy h5 hdf5 mat pkl pickle dill zip tar gz"

echo "Running pre-commit checks..."

# Track if we found any issues
FOUND_ISSUES=0

# Check staged files
for file in $(git diff --cached --name-only); do
    # Skip if file doesn't exist (was deleted)
    if [ ! -f "$file" ]; then
        continue
    fi
    
    # Get file size (cross-platform)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        size=$(stat -f%z "$file" 2>/dev/null)
    else
        # Linux
        size=$(stat -c%s "$file" 2>/dev/null)
    fi
    
    # Check file size
    if [ -n "$size" ] && [ "$size" -gt "$MAX_SIZE" ]; then
        size_mb=$((size / 1048576))
        echo -e "${RED}ERROR: File '$file' is ${size_mb}MB (limit is 5MB)${NC}"
        echo "  Large data files must not be committed to git!"
        echo "  Store them in cloud storage or external repositories instead."
        FOUND_ISSUES=1
    fi
    
    # Check file extension
    filename=$(basename "$file")
    ext="${filename##*.}"
    
    # Check against blocked extensions
    for blocked in $BLOCKED_EXTENSIONS; do
        if [ "$ext" = "$blocked" ]; then
            echo -e "${RED}ERROR: File '$file' has blocked extension .$ext${NC}"
            echo "  Data files (.$ext) must not be committed to git!"
            echo "  See docs/DATA_MANAGEMENT_GUIDE.md for alternatives."
            FOUND_ISSUES=1
            break
        fi
    done
    
    # Special check for test data patterns
    if [[ "$filename" =~ ^test.*\.npz$ ]] || [[ "$filename" =~ .*_test\.npz$ ]] || [[ "$filename" =~ .*_train\.npz$ ]]; then
        echo -e "${RED}ERROR: Test/training data file detected: '$file'${NC}"
        echo "  These files should never be committed to git!"
        FOUND_ISSUES=1
    fi
done

# Check for common mistake patterns
if git diff --cached --name-only | grep -E "(datasets|data|outputs|results|simulations|tike_outputs)/" > /dev/null; then
    echo -e "${YELLOW}WARNING: You're committing files in a data directory.${NC}"
    echo "  Please verify these are not data files:"
    git diff --cached --name-only | grep -E "(datasets|data|outputs|results|simulations|tike_outputs)/"
    echo ""
fi

# If we found issues, block the commit
if [ "$FOUND_ISSUES" -eq 1 ]; then
    echo ""
    echo -e "${RED}Commit blocked due to data file policy violations.${NC}"
    echo ""
    echo "To fix this:"
    echo "1. Remove data files from staging: git reset HEAD <file>"
    echo "2. Add them to .gitignore"
    echo "3. See docs/DATA_MANAGEMENT_GUIDE.md for data management best practices"
    echo ""
    echo "If you absolutely must commit these files (NOT RECOMMENDED):"
    echo "  git commit --no-verify"
    exit 1
fi

echo -e "${GREEN}Pre-commit check passed: No data files detected${NC}"
EOF

# Make pre-commit hook executable
chmod +x .git/hooks/pre-commit

echo "Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will now:"
echo "  ✓ Block files larger than 5MB"
echo "  ✓ Block common data file extensions (npz, npy, h5, etc.)"
echo "  ✓ Warn about files in data directories"
echo ""
echo "To test the hook, try:"
echo "  touch test.npz && git add test.npz && git commit -m 'test'"
echo ""
echo "To bypass the hook in emergencies (NOT RECOMMENDED):"
echo "  git commit --no-verify"