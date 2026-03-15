"""
Quick demonstration of the DravidianCodeMix-2020 Preprocessing Pipeline.
"""
import sys
sys.path.insert(0, ".")
from preprocessing_pipeline import *

print("=" * 70)
print("  DravidianCodeMix-2020 Preprocessing Pipeline — DEMO")
print("=" * 70)

# ── 1. Load Tamil Sentiment ──
print("\n[1] Loading Tamil Sentiment Train...")
df = DataLoader.load_dataset("tamil_sentiment", split="train", task="sentiment")
print("    Rows:", len(df))
print("    Label dist:")
for label, count in df["label"].value_counts().items():
    pct = count / len(df) * 100
    print(f"      {label:30s} {count:6d} ({pct:.1f}%)")

# ── 2. Before / After cleaning examples ──
print("\n[2] Before / After Preprocessing (first 5 samples):")
pipeline = PreprocessingPipeline(
    filter_mode="remove",
    include_script_features=False,
    verbose=False,
)
for i in range(5):
    original = df.iloc[i]["text"]
    cleaned = pipeline.preprocess_text(original)
    print(f"\n    Sample {i+1}:")
    print(f"      BEFORE: {original[:100]}")
    print(f"      AFTER : {cleaned[:100]}")

# ── 3. Process full dataset ──
print("\n\n[3] Full Pipeline on Tamil Sentiment...")
pipeline_full = PreprocessingPipeline(
    filter_mode="remove",
    include_script_features=True,
    verbose=False,
)
processed = pipeline_full.process_dataframe(
    df, task="sentiment", dataset_name="tamil_sentiment_train"
)
print("    Rows after preprocessing:", len(processed))
print("    Columns:", list(processed.columns))
print("\n    Label encoded distribution:")
for val, count in processed["label_encoded"].value_counts().sort_index().items():
    label_names = {0: "Positive", 1: "Negative", 2: "Mixed_feelings", 3: "unknown_state"}
    name = label_names.get(int(val), str(val))
    print(f"      {int(val)} ({name:20s}) : {count:6d}")

print("\n    Code-mixing statistics:")
cmi = processed["code_mix_index"]
print(f"      Avg CMI        : {cmi.mean():.1f}%")
print(f"      Median CMI     : {cmi.median():.1f}%")
print(f"      Monolingual %  : {(cmi == 0).mean() * 100:.1f}%")
print(f"      High CM (>50%) : {(cmi > 50).mean() * 100:.1f}%")

print("\n    Dominant script distribution:")
for script, count in processed["dominant_script"].value_counts().items():
    print(f"      {script:15s} : {count:6d}")

# ── 4. Show a preprocessed sample ──
print("\n\n[4] Sample preprocessed rows:")
cols = ["text", "label", "label_encoded", "word_count", "dominant_script", "code_mix_index"]
print(processed[cols].head(10).to_string(max_colwidth=60))

# ── 5. Quick run on all datasets ──
print("\n\n[5] Processing ALL datasets (train split)...")
all_results = pipeline_full.process_all(split="train", save=True)

print("\n" + "=" * 70)
print("  DEMO COMPLETE — Preprocessed files saved to 'preprocessed/' folder")
print("=" * 70)
