#!/bin/bash

# Git LFS Setup Script for SearchRankMaster
echo "Setting up Git LFS for SearchRankMaster repository..."

# Navigate to your repository
cd "C:/Users/ankit/Downloads/SearchRankMaster - Copy"

# Install Git LFS if not already done
git lfs install

# Track large files with LFS
git lfs track "*.csv"
git lfs track "*.zip"
git lfs track "*.h5"
git lfs track "*.pkl"
git lfs track "*.joblib"
git lfs track "*.parquet"

# Add .gitattributes
git add .gitattributes

# Remove virtual environment if tracked
git rm -r --cached venv/ 2>/dev/null || echo "venv not tracked"

# Add .gitignore
git add .gitignore

# Commit changes
git add .
git commit -m "Configure Git LFS and cleanup repository"

# Check LFS status
echo "Git LFS tracked files:"
git lfs ls-files

echo "Repository setup complete!"
echo "You can now push to GitHub with: git push origin main"
