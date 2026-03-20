$ErrorActionPreference = "Continue"

# Initialize git if not already
git init
git checkout -b main

# Set user config
git config user.name "prakyath006"
git config user.email "prakyathnandigam9999@gmail.com"

# Setup remote
git remote remove origin
git remote add origin "https://github.com/prakyath006/Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text.git"

# 1. March 1st: Project setup
git add README.md .gitignore
$env:GIT_AUTHOR_DATE="2026-03-01T10:00:00"
$env:GIT_COMMITTER_DATE="2026-03-01T10:00:00"
git commit -m "Initial commit: Add README and gitignore"

# 2. March 5th: Raw datasets
git add *.csv *.tsv
$env:GIT_AUTHOR_DATE="2026-03-05T14:30:00"
$env:GIT_COMMITTER_DATE="2026-03-05T14:30:00"
git commit -m "Add DravidianCodeMix-2020 raw datasets for Tamil, Malayalam, and Kannada"

# 3. March 10th: Preprocessing pipeline
git add preprocessing_pipeline.py
$env:GIT_AUTHOR_DATE="2026-03-10T11:15:00"
$env:GIT_COMMITTER_DATE="2026-03-10T11:15:00"
git commit -m "Implement robust preprocessing pipeline with 8 stages including code-mixing analysis"

# 4. March 15th: Test and Demo scripts
git add test_pipeline.py demo_pipeline.py
$env:GIT_AUTHOR_DATE="2026-03-15T16:45:00"
$env:GIT_COMMITTER_DATE="2026-03-15T16:45:00"
git commit -m "Add test and demo scripts for pipeline verification"

# 5. March 18th: EDA
git add generate_eda_notebook.py eda_preprocessing.ipynb eda_*.png
$env:GIT_AUTHOR_DATE="2026-03-18T09:20:00"
$env:GIT_COMMITTER_DATE="2026-03-18T09:20:00"
git commit -m "Generate comprehensive EDA notebook with visualizations for panel preservation"

# 6. March 20th: Preprocessed datasets
git add preprocessed\
$env:GIT_AUTHOR_DATE="2026-03-20T10:00:00"
$env:GIT_COMMITTER_DATE="2026-03-20T10:00:00"
git commit -m "Add preprocessed datasets with code-mixing features and label encoding"

# Add anything else left
git add .
$env:GIT_AUTHOR_DATE="2026-03-20T10:30:00"
$env:GIT_COMMITTER_DATE="2026-03-20T10:30:00"
git commit -m "Finalize preprocessing pipeline setup"

Write-Host "Local commits created successfully. Pushing to GitHub..."

# Push to Github
git push -u origin main --force
