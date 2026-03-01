# Tracking Opinion Evolution in Multilingual Sequential Text

## DravidianCodeMix-2020 — Sentiment & Offensive Language Detection

A comprehensive preprocessing and EDA pipeline for the **DravidianCodeMix-2020** shared task dataset, targeting sentiment analysis and offensive language identification in code-mixed Dravidian languages (Tamil, Malayalam, and Kannada).

---

## 📋 Project Overview

Social media text in Dravidian languages is often **code-mixed** — users freely switch between their native script and English within the same comment. This project builds a robust preprocessing pipeline to clean, normalize, and analyze this challenging multilingual data.

### Dataset Summary

| Language | Task | Train | Dev | Test |
|----------|------|-------|-----|------|
| Tamil | Sentiment | 35,220 | — | — |
| Tamil | Offensive | 35,139 | — | — |
| Malayalam | Sentiment | 15,694 | — | — |
| Malayalam | Offensive | 16,011 | — | — |
| Kannada | Sentiment | 6,137 | — | — |
| Kannada | Offensive | 6,218 | — | — |

### Labels

- **Sentiment:** Positive, Negative, Mixed_feelings, unknown_state
- **Offensive:** Not_offensive, Offensive_Targeted_Insult_Individual, Offensive_Targeted_Insult_Group, Offensive_Targeted_Insult_Other, Offensive_Untargeted

---

## 🔧 Preprocessing Pipeline

An 8-stage pipeline implemented in `preprocessing_pipeline.py`:

1. **Data Loading & Unification** — Auto-detects delimiters (tab/semicolon) and column order
2. **Language Filtering** — Removes non-target-language samples (not-Tamil, not-Malayalam, not-Kannada)
3. **Noise Removal** — URLs, @mentions, emojis, special characters (Unicode-aware, preserves Dravidian scripts)
4. **Text Normalization** — Selective lowercasing (Latin only), Unicode NFC, punctuation normalization
5. **Code-Mix Handling** — Script detection, Code-Mixing Index (CMI) computation
6. **Sentence Segmentation** — Handles informal social media sentence boundaries
7. **Label Encoding** — String labels → integer encoding
8. **Statistics & Reporting** — Distribution reports and CMI analysis

### Code-Mixing Index (CMI)

```
CMI = (N - max(w_i)) / N × 100
```

| Language | Avg CMI | Monolingual % |
|----------|---------|---------------|
| Tamil | 3.7% | ~85% |
| Malayalam | 6.5% | ~78% |
| Kannada | 5.1% | ~80% |

---

## 📊 EDA (Exploratory Data Analysis)

The `eda_preprocessing.ipynb` notebook provides 14 visualizations:

- Dataset size comparisons
- Label distribution analysis
- Text length distributions (character & word counts)
- Before vs After preprocessing comparison
- Code-Mixing Index distribution by language
- Dominant script distribution
- Label vs CMI correlation (box plots)
- Word clouds per language
- Feature correlation heatmap
- Class imbalance analysis

---

## 🚀 Usage

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn wordcloud
```

### Run Preprocessing

```bash
# Process all datasets (train split)
python preprocessing_pipeline.py --split train

# Process a single dataset
python preprocessing_pipeline.py --dataset tamil_sentiment --split train

# Different filter modes
python preprocessing_pipeline.py --filter-mode tag    # keep non-target with tag
python preprocessing_pipeline.py --filter-mode keep   # keep everything
```

### Run EDA

Open `eda_preprocessing.ipynb` in Jupyter Notebook or VS Code and run all cells.

### Python API

```python
from preprocessing_pipeline import *

# Load data
df = DataLoader.load_dataset("tamil_sentiment", split="train")

# Run pipeline
pipeline = PreprocessingPipeline(filter_mode="remove", include_script_features=True)
processed = pipeline.process_dataframe(df, task="sentiment")

# Use processed data
X = processed["text"].values
y = processed["label_encoded"].values
```

---

## 📁 Project Structure

```
DravidianCodeMix-2020/
├── README.md
├── preprocessing_pipeline.py      # Main 8-stage pipeline
├── eda_preprocessing.ipynb        # EDA notebook (14 visualizations)
├── generate_eda_notebook.py       # EDA notebook generator
├── demo_pipeline.py               # Full demonstration script
├── test_pipeline.py               # Quick verification test
├── preprocessed/                  # Output directory
│   ├── tamil_sentiment_train_preprocessed.csv
│   ├── tamil_offensive_train_preprocessed.csv
│   ├── mal_sentiment_train_preprocessed.csv
│   ├── mal_offensive_train_preprocessed.csv
│   ├── kannada_sentiment_train_preprocessed.csv
│   └── kannada_offensive_train_preprocessed.csv
├── tamil_sentiment_full_train.csv # Raw dataset files
├── tamil_offensive_full_train.csv
├── mal_full_sentiment_train.csv
├── mal_full_offensive_train.csv
├── kannada_sentiment_train.csv
├── kannada_offensive_train.csv
└── ... (other raw dataset files)
```

---

## 📚 References

- B. R. Chakravarthi et al., "A Sentiment Analysis Dataset for Code-Mixed Malayalam-English," 2020
- B. R. Chakravarthi et al., "Corpus Creation for Sentiment Analysis in Code-Mixed Tamil-English Text," 2020
- DravidianCodeMix Shared Task @ FIRE 2020

---

## 👤 Author

**Prakyath Nandigam**
