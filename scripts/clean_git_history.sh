#!/bin/bash
# Script to clean git history of large data files using BFG

echo "============================================"
echo "Git History Cleaner for Data Files"
echo "============================================"
echo ""
echo "WARNING: This will rewrite git history!"
echo "Make sure all team members are aware before proceeding."
echo ""
read -p "Do you want to continue? (yes/no): " response

if [ "$response" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# Check if BFG is available
if [ ! -f "bfg.jar" ]; then
    echo "BFG not found. Downloading..."
    wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar -O bfg.jar
fi

echo ""
echo "Step 1: Creating backup of current state..."
git branch backup-before-cleanup-$(date +%Y%m%d-%H%M%S)

echo ""
echo "Step 2: Removing data files from history..."
echo "This may take several minutes for large repositories..."

# Remove various data file types
java -jar bfg.jar --delete-files "*.npz" --no-blob-protection
java -jar bfg.jar --delete-files "*.npy" --no-blob-protection  
java -jar bfg.jar --delete-files "*.h5" --no-blob-protection
java -jar bfg.jar --delete-files "*.hdf5" --no-blob-protection
java -jar bfg.jar --delete-files "*.mat" --no-blob-protection

# Remove files larger than 10MB
java -jar bfg.jar --strip-blobs-bigger-than 10M --no-blob-protection

echo ""
echo "Step 3: Cleaning up repository..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Step 4: Checking repository size..."
echo "Repository size before: (check .git size manually)"
echo "Repository size after:"
du -sh .git

echo ""
echo "============================================"
echo "Cleanup Complete!"
echo "============================================"
echo ""
echo "IMPORTANT NEXT STEPS:"
echo "1. Review the changes with: git log --oneline"
echo "2. Test that the repository still works correctly"
echo "3. Force push ONLY after coordinating with your team:"
echo "   git push origin --force --all"
echo "   git push origin --force --tags"
echo ""
echo "4. All team members must re-clone or reset their local repos:"
echo "   git fetch origin"
echo "   git reset --hard origin/$(git branch --show-current)"
echo ""
echo "5. Delete the bfg-report directory when satisfied:"
echo "   rm -rf ..bfg-report"
echo ""
echo "A backup branch was created: backup-before-cleanup-*"