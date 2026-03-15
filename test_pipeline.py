"""Minimal test to verify pipeline works correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing_pipeline import *

# 1. Load one dataset
df = DataLoader.load_dataset("tamil_sentiment", split="train", task="sentiment")
print("Loaded Tamil Sentiment Train:", len(df), "rows")
vc = df["label"].value_counts()
for lab in vc.index:
    print("  ", lab, ":", vc[lab])

# 2. Before/After cleaning
pipeline = PreprocessingPipeline(filter_mode="remove", include_script_features=False, verbose=False)
for i in range(3):
    orig = df.iloc[i]["text"]
    clean = pipeline.preprocess_text(orig)
    print("\nSample", i+1)
    print("  BEFORE:", repr(orig[:80]))
    print("  AFTER :", repr(clean[:80]))

# 3. Full pipeline
pipeline2 = PreprocessingPipeline(filter_mode="remove", include_script_features=True, verbose=False)
processed = pipeline2.process_dataframe(df, task="sentiment", dataset_name="tamil_train")
print("\nAfter preprocessing:", len(processed), "rows")
print("Columns:", ", ".join(processed.columns))
ec = processed["label_encoded"].value_counts().sort_index()
for idx in ec.index:
    print("  Encoded", int(idx), ":", ec[idx])

print("\nAvg CMI:", round(processed["code_mix_index"].mean(), 1))
ds = processed["dominant_script"].value_counts()
for s in ds.index:
    print("  Script", s, ":", ds[s])

# 4. Check preprocessed output directory
print("\nChecking preprocessed files...")
if os.path.exists("preprocessed"):
    for f in os.listdir("preprocessed"):
        sz = os.path.getsize(os.path.join("preprocessed", f))
        print("  ", f, "->", sz, "bytes")
else:
    print("  No preprocessed directory yet")

print("\nDone!")
