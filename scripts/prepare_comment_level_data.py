"""
Module 8a — Comment-level data for multilingual and cross-domain sentiment evaluation.

No sliding windows and no artificial histories: every row is one comment or review.

  tamil / malayalam / kannada : the OFFICIAL DravidianCodeMix train / dev / test files, preprocessed with the
                                project pipeline (src/preprocessing.py) and mapped through the ontology.
  amazon                      : the Amazon review texts of the existing user split (train/val/test users), flattened
                                to reviews; val is written as "dev".

Leakage controls (applied to dev and test only; training rows are never altered)
  * rows with empty text are dropped;
  * a dev/test text whose normalised form occurs in the training split is removed;
  * a test text that also occurs in the dev split is removed;
  * duplicate texts inside dev or test are kept once.
Every count is written to outputs/comment_level/data_report.json. Labels are ontology SentimentState values
(0 POSITIVE, 1 NEGATIVE, 2 MIXED, 3 UNKNOWN). Output: data/comment_level/<corpus>_<split>.csv.

    python scripts/prepare_comment_level_data.py
"""
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True
import logging                                                       # noqa: E402
logging.disable(logging.WARNING)
import eval_utils as eu                                              # noqa: E402
from ontology import SentimentState, map_labels_to_ontology          # noqa: E402

CORPORA = {"tamil": ("tamil_sentiment", "dravidian_tamil"),
           "malayalam": ("mal_sentiment", "dravidian_malayalam"),
           "kannada": ("kannada_sentiment", "dravidian_kannada")}


def norm(t):
    return re.sub(r"\s+", " ", str(t).strip().lower())


def clean_split(df, forbidden, name, report):
    """Drop empty, forbidden (seen in an earlier split) and duplicate texts; record every count."""
    r = {"rows_in": int(len(df))}
    df = df.assign(_n=df["text"].map(norm))
    df = df[df["_n"] != ""]
    r["empty_removed"] = r["rows_in"] - int(len(df))
    keep = ~df["_n"].isin(forbidden)
    r["removed_text_seen_in_earlier_split"] = int((~keep).sum())
    df = df[keep]
    before = len(df)
    df = df.drop_duplicates("_n", keep="first")
    r["within_split_duplicates_removed"] = before - int(len(df))
    r["rows_out"] = int(len(df))
    report[name] = r
    return df.drop(columns="_n")


def write(df, corpus, split, report_entry):
    os.makedirs(eu.COMMENT_LEVEL_DIR, exist_ok=True)
    df = df[["text", "label"]].copy()
    df["label_name"] = df["label"].map(lambda v: eu.SENTIMENT_NAMES[int(v)])
    df.to_csv(os.path.join(eu.COMMENT_LEVEL_DIR, f"{corpus}_{split}.csv"), index=False, encoding="utf-8")
    report_entry["label_counts"] = {n: int((df["label"] == i).sum()) for i, n in enumerate(eu.SENTIMENT_NAMES)}


def build_corpus(corpus, splits):
    """splits: {"train": df, "dev": df, "test": df} with columns text,label."""
    rep = {}
    train = splits["train"]
    train_norm = set(train["text"].map(norm))
    rep["train"] = {"rows": int(len(train)), "distinct_texts": len(train_norm),
                    "texts_with_conflicting_labels": int((train.assign(_n=train["text"].map(norm)).groupby("_n")["label"].nunique() > 1).sum())}
    write(train, corpus, "train", rep["train"])
    dev = clean_split(splits["dev"], train_norm, "dev", rep)
    write(dev, corpus, "dev", rep["dev"])
    test = clean_split(splits["test"], train_norm | set(dev["text"].map(norm)), "test", rep)
    write(test, corpus, "test", rep["test"])
    return rep


def dravidian_splits(key, domain):
    from preprocessing import DataLoader, PreprocessingPipeline
    pipe = PreprocessingPipeline(filter_mode="remove", include_script_features=False, verbose=False)
    out = {}
    for split in ("train", "dev", "test"):
        raw = DataLoader.load_dataset(key, split, task="sentiment")
        proc = pipe.process_dataframe(raw, task="sentiment", dataset_name=f"{key}_{split}")
        states = map_labels_to_ontology(proc["label"].astype(str).tolist(), domain)
        out[split] = pd.DataFrame({"text": proc["text"].tolist(), "label": [s.value for s in states]})
    return out


def amazon_splits():
    from dataset import AmazonSequenceDataset
    out = {}
    for split, name in (("train", "train"), ("val", "dev"), ("test", "test")):
        ds = AmazonSequenceDataset(split=split)
        texts = [t for s in ds.sequences for t in s["texts"]]
        labels = [l for s in ds.sequences for l in s["sentiments"]]
        out[name] = pd.DataFrame({"text": texts, "label": labels})
    return out


def main():
    report = {"description": "Comment-level data; official DravidianCodeMix splits and Amazon reviews (user-level split).",
              "label_space": dict(enumerate(eu.SENTIMENT_NAMES))}
    for corpus, (key, domain) in CORPORA.items():
        print("preparing", corpus, flush=True)
        report[corpus] = build_corpus(corpus, dravidian_splits(key, domain))
    print("preparing amazon", flush=True)
    report["amazon"] = build_corpus("amazon", amazon_splits())
    eu.save_json(report, os.path.join(ROOT, "outputs", "comment_level", "data_report.json"))
    for c in ("tamil", "malayalam", "kannada", "amazon"):
        r = report[c]
        print(f"{c:10s} train {r['train']['rows']:6d} | dev {r['dev']['rows_out']:5d} (removed {r['dev']['removed_text_seen_in_earlier_split']}) "
              f"| test {r['test']['rows_out']:5d} (removed {r['test']['removed_text_seen_in_earlier_split']} seen in train/dev, "
              f"{r['test']['within_split_duplicates_removed']} duplicates)")


if __name__ == "__main__":
    main()
