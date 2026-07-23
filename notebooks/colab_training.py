# ============================================================================
# Opinion Evolution Tracker — Google Colab Training Notebook
# ============================================================================
# Run this notebook on Google Colab with GPU enabled:
#   Runtime → Change runtime type → T4 GPU
#
# This notebook will:
#   1. Clone your repo from GitHub
#   2. Install dependencies
#   3. Train the full model + all baselines
#   4. Run cross-domain evaluation
#   5. Save results for download
# ============================================================================

# ── Cell 1: Setup ──────────────────────────────────────────────────────────
# !nvidia-smi  # Verify GPU is available

# ── Cell 2: Clone repo ────────────────────────────────────────────────────
# !git clone https://github.com/prakyath006/Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text.git
# %cd Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text

# ── Cell 3: Install dependencies ──────────────────────────────────────────
# !pip install torch transformers scikit-learn pandas numpy tqdm

# ── Cell 4: Upload your data files ────────────────────────────────────────
# Since data/ is in .gitignore, you need to upload it.
# Option A: Upload manually via Colab file browser
# Option B: Upload to Google Drive and mount
#
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r /content/drive/MyDrive/project_data/* data/

# ── Cell 5: Verify data is present ───────────────────────────────────────
# !ls data/raw/
# !ls data/preprocessed/

# ── Cell 6: Run demo to verify everything works ──────────────────────────
# !python scripts/demo_full_project.py

# ── Cell 7: Train full model on Amazon data ──────────────────────────────
# !python scripts/train.py --domain amazon --epochs 20 --batch_size 16

# ── Cell 8: Run cross-domain evaluation ──────────────────────────────────
# !python scripts/cross_domain_eval.py

# ── Cell 9: Download results ─────────────────────────────────────────────
# from google.colab import files
# !zip -r results.zip outputs/
# files.download('results.zip')
