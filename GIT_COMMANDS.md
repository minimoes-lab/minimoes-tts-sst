# Git Commands for GitHub Upload

## Quick Upload (If repo already exists)

```bash
# Add all files (respecting .gitignore)
git add .

# Commit
git commit -m "feat: Complete Qwen3-TTS integration with streaming blendshapes"

# Push
git push origin main
```

## First Time Setup (New Repository)

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: Streaming Avatar API with Qwen3-TTS"

# Add remote (replace with your GitHub repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub
git push -u origin main
```

## Verify Before Pushing

```bash
# Check what will be committed
git status

# See what's ignored
git status --ignored

# Verify specific files are ignored
git check-ignore -v test_*.py
git check-ignore -v sample_outputs/*.wav
git check-ignore -v utils/model/model.pth

# See diff of changes
git diff --cached
```

## What Gets Uploaded

✅ **Included**:
- All source code (api.py, streaming/, utils/)
- Docker configuration
- Documentation (README.md, CONTRIBUTING.md)
- Requirements and configs
- Folder structure (.gitkeep files)

❌ **Excluded** (via .gitignore):
- Test files (test_*.py, demo_*.py, etc.)
- Generated outputs (*.wav, *.json, *.csv)
- Model weights (model.pth - too large)
- Python cache (__pycache__/)
- Virtual environments (venv/)
- IDE settings (.vscode/, .idea/)
- Temporary docs (*_SUMMARY.md, etc.)

## Repository Size

**~5-10 MB** (without model.pth)

The 600MB model file will be downloaded by users via `download_model.py`

## After Upload

Users will clone and run:

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Setup
echo "GROQ_API_KEY=your_key" > .env
python download_model.py

# Run
./docker-run.sh  # or docker-run.bat on Windows
```

## Troubleshooting

### If model.pth was accidentally committed:

```bash
# Remove from git but keep locally
git rm --cached utils/model/model.pth

# Commit the removal
git commit -m "Remove large model file from git"

# Push
git push origin main
```

### If test files were committed:

```bash
# Remove all test files
git rm --cached test_*.py

# Commit
git commit -m "Remove test files"

# Push
git push origin main
```

### Check repository size:

```bash
# See size of files to be committed
git ls-files | xargs du -ch | tail -1

# See largest files
git ls-files | xargs du -h | sort -rh | head -20
```

## Branch Strategy (Optional)

```bash
# Create development branch
git checkout -b develop

# Make changes
git add .
git commit -m "feat: new feature"

# Push to develop
git push origin develop

# Merge to main when ready
git checkout main
git merge develop
git push origin main
```

---

**Ready to upload!** Just run the commands above. 🚀
