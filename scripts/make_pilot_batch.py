"""
Pilot annotation batch 01: 50 real Amazon All Beauty reviews from 10 product timelines, annotations left blank.

A PRODUCT_TIMELINE is the reviews of ONE product (parent_asin) from different reviewers, ordered by real timestamp.
Procedure (details and rationale in docs/pilot_batch_sampling.md):
  1. keep reviews with 10 to 300 words; sort by timestamp (then file line);
  2. per product keep the earliest review of each reviewer, drop later repeats of the same text, drop repeated timestamps;
  3. a product is eligible if it keeps at least 20 reviews spanning at least 365 days and is not already in the 16-record
     same-reviewer pilot (data/annotations/aspect_annotation_template.csv);
  4. choose 10 eligible products with a seeded random draw over the sorted product ids;
  5. in each product take 5 reviews at evenly spaced ranks of its chronological order (first, quartiles, last).
Nothing about the text, rating or any model output is used to choose products or reviews, and nothing is labelled.

    python scripts/make_pilot_batch.py            # writes the batch and its manifest (refuses to overwrite an existing batch)
    python scripts/make_pilot_batch.py --verify   # independent re-check against the raw file
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
RAW = os.path.join(ROOT, "data", "raw", "All_Beauty.jsonl")
ANN = os.path.join(ROOT, "data", "annotations")
TEMPLATE = os.path.join(ANN, "aspect_annotation_template.csv")
BATCH = os.path.join(ANN, "pilot_batch_01_product_timelines.csv")
MANIFEST = os.path.join(ANN, "pilot_batch_01_manifest.json")
SEED, N_PRODUCTS, K = 20261114, 10, 5
MIN_WORDS, MAX_WORDS, MIN_ELIGIBLE, MIN_SPAN_DAYS = 10, 300, 20, 365
BLANK_COLUMNS = ["aspect_status", "aspect_span_text", "span_start", "span_end", "span_type", "aspect_category", "sentiment",
                 "annotator_id", "notes"]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def template_columns():
    return pd.read_csv(TEMPLATE, encoding="utf-8-sig", nrows=0).columns.tolist()


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_raw():
    raw = pd.read_json(RAW, lines=True, convert_dates=False)
    raw["line"] = raw.index + 1                                    # 1-based line number in the jsonl file
    raw["words"] = raw.text.fillna("").str.split().str.len()
    return raw


def build():
    if os.path.exists(BATCH):
        sys.exit(f"{BATCH} already exists and may contain annotations; refusing to overwrite.")
    cols = template_columns()
    template_sha = sha256(TEMPLATE)
    existing_products = set(pd.read_csv(TEMPLATE, encoding="utf-8-sig", dtype=str).entity_id)
    raw = load_raw()
    r = raw[(raw.words >= MIN_WORDS) & (raw.words <= MAX_WORDS)].sort_values(["timestamp", "line"])
    r = r.drop_duplicates(["parent_asin", "user_id"]).drop_duplicates(["parent_asin", "text"]).drop_duplicates(["parent_asin", "timestamp"])
    g = r.groupby("parent_asin").timestamp.agg(["size", "min", "max"])
    g["span_days"] = (g["max"] - g["min"]) / 86400000
    eligible = g[(g["size"] >= MIN_ELIGIBLE) & (g.span_days >= MIN_SPAN_DAYS)]
    eligible = eligible[~eligible.index.isin(existing_products)].sort_index()
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(eligible.index.to_numpy(), size=N_PRODUCTS, replace=False).tolist())
    rows, picked = [], []
    for p in chosen:
        t = r[r.parent_asin == p].sort_values(["timestamp", "line"]).reset_index(drop=True)
        idx = np.unique(np.round(np.linspace(0, len(t) - 1, K)).astype(int))
        assert len(idx) == K
        sid = "AMZ-PT-" + hashlib.sha1(p.encode()).hexdigest()[:10]
        for pos, i in enumerate(idx, 1):
            x = t.iloc[i]
            rows.append({"sequence_id": sid, "sequence_type": "PRODUCT_TIMELINE", "source": "amazon_all_beauty_2023", "entity_id": p,
                         "message_id": f"AMZ-AB-L{int(x.line):06d}", "position": pos, "timestamp_ms": int(x.timestamp),
                         "timestamp_utc": utc(int(x.timestamp)), "language": "en", "title": x.title, "text": x.text,
                         "parent_message_id": "", "annotation_status": "NOT_STARTED"})
        picked.append({"sequence_id": sid, "product": p, "eligible_reviews": int(len(t)), "span_days": round(float((t.timestamp.iloc[-1] - t.timestamp.iloc[0]) / 86400000), 1),
                       "selected_ranks_in_product_stream": [int(i) + 1 for i in idx]})
    df = pd.DataFrame(rows).reindex(columns=cols).fillna("")
    df.to_csv(BATCH, index=False, encoding="utf-8-sig")
    manifest = {"batch": "pilot_batch_01", "file": "data/annotations/pilot_batch_01_product_timelines.csv",
                "parameters": {"seed": SEED, "products": N_PRODUCTS, "reviews_per_product": K, "min_words": MIN_WORDS, "max_words": MAX_WORDS,
                               "min_eligible_reviews_per_product": MIN_ELIGIBLE, "min_span_days": MIN_SPAN_DAYS},
                "eligible_products": int(len(eligible)), "products_excluded_as_in_existing_pilot": sorted(existing_products),
                "sequences": picked, "messages": int(len(df)), "annotated_rows": 0,
                "existing_template_sha256_at_generation": template_sha, "batch_sha256_at_generation": sha256(BATCH)}
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in ("eligible_products", "messages")}), "->", BATCH)


def verify():
    """Independent of build(): re-reads the raw file and the written batch and checks every selected review."""
    problems = []
    cols = template_columns()
    b = pd.read_csv(BATCH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    t = pd.read_csv(TEMPLATE, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    man = json.load(open(MANIFEST, encoding="utf-8"))
    if b.columns.tolist() != cols:
        problems.append("columns differ from the template")
    if len(b) != N_PRODUCTS * K or b.sequence_id.nunique() != N_PRODUCTS:
        problems.append(f"expected {N_PRODUCTS * K} rows in {N_PRODUCTS} sequences, found {len(b)} in {b.sequence_id.nunique()}")
    if not b.message_id.is_unique:
        problems.append("message ids are not unique")
    for c in BLANK_COLUMNS:
        if (b[c] != "").any():
            problems.append(f"annotation column {c} is not blank")
    if set(b.annotation_status) != {"NOT_STARTED"} or set(b.sequence_type) != {"PRODUCT_TIMELINE"}:
        problems.append("status or sequence_type not as prepared")
    if set(b.message_id) & set(t.message_id) or set(b.entity_id) & set(t.entity_id):
        problems.append("overlaps the existing 16-record pilot")
    if sha256(TEMPLATE) != man["existing_template_sha256_at_generation"]:
        problems.append("the existing template changed since the batch was generated")
    raw = load_raw().set_index("line")
    revs = []
    for sid, g in b.groupby("sequence_id", sort=False):
        if g.entity_id.nunique() != 1:
            problems.append(f"{sid}: more than one product")
        ts = g.timestamp_ms.astype("int64").tolist()
        if ts != sorted(ts) or len(set(ts)) != len(ts):
            problems.append(f"{sid}: timestamps not strictly increasing")
        if g.position.astype(int).tolist() != list(range(1, len(g) + 1)):
            problems.append(f"{sid}: positions not 1..n")
        users = []
        for _, x in g.iterrows():
            line = int(x.message_id.split("-L")[1])
            src = raw.loc[line]
            users.append(src.user_id)
            if src.parent_asin != x.entity_id:
                problems.append(f"{x.message_id}: product differs from the raw file")
            if int(src.timestamp) != int(x.timestamp_ms) or utc(int(src.timestamp)) != x.timestamp_utc:
                problems.append(f"{x.message_id}: timestamp differs from the raw file")
            if str(src.text) != x.text or str(src.title) != x.title:
                problems.append(f"{x.message_id}: text or title differs from the raw file")
            if not (MIN_WORDS <= len(x.text.split()) <= MAX_WORDS):
                problems.append(f"{x.message_id}: length outside the rule")
        if len(set(users)) != len(users):
            problems.append(f"{sid}: a reviewer appears twice")
        revs.append((sid, g.entity_id.iloc[0], len(g), len(set(users)), round((ts[-1] - ts[0]) / 86400000, 1)))
    for r in revs:
        print("%s product %s: %d reviews, %d distinct reviewers, spans %.1f days" % r)
    print("VERIFIED: all checks passed" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        sys.exit(1 if verify() else 0)
    build()
    sys.exit(1 if verify() else 0)
