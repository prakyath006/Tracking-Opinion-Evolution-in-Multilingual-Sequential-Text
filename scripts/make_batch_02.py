"""
Annotation batch 02: 400 real Amazon All Beauty reviews = 40 product timelines x 10 reviews, annotations left blank.

Why a second batch: the 50-review pilot is far below the training pipeline's floors (docs/annotation_requirements_and_batch_02.md).
Why not the pilot's rule again: batch 01 drew uniformly from products with >= 20 reviews over >= 365 days, which over-represents popular,
long-lived products. Batch 02 keeps the sequence definition (docs/aspect_annotation_guidelines.md section 2: reviews of ONE product from different
reviewers, strictly increasing timestamps, distinct text, 10 to 300 words) and changes only WHICH products are drawn:

  1. same review filter and de-duplication as batch 01 (per product: earliest review per reviewer, distinct text, distinct timestamp);
  2. four popularity strata by the number of kept reviews n (and span in days d):
       SMALL  12 <= n < 20, d >= 180   (RELAXED: batch 01 required n >= 20 and d >= 365; flagged in the manifest, pending team approval)
       MEDIUM 20 <= n < 50,  d >= 365
       LARGE  50 <= n < 200, d >= 365
       XLARGE n >= 200,      d >= 365
     quotas 10 / 12 / 12 / 6 products, so popular products are a minority (batch 01 was dominated by them);
  3. aspect-category breadth: each product gets a lexical profile, the share of its reviews that contain an alias of each category of
     ontology/product_aspect_taxonomy.json (a sampling aid only: no label is created and no text is judged). Its distinctive category is the
     category with the highest lift (product share / corpus share) among categories with >= 3 hit reviews. Inside every stratum the draw goes
     round-robin over distinctive categories (seeded shuffle each round), so rare categories (packaging, delivery, skin reaction, size, price ...)
     are not drowned out by effectiveness and fragrance;
  4. products of batch 01 and of the 16-record same-reviewer pilot are excluded;
  5. 10 reviews per product at evenly spaced ranks of its chronological order (first and last included), as in batch 01;
  6. products are assigned alternately to wave 1 and wave 2 inside each stratum, so each wave is a balanced half (annotate wave 1 first).

    python scripts/make_batch_02.py            # writes the batch and its manifest (refuses to overwrite)
    python scripts/make_batch_02.py --verify   # independent re-check against the raw file
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.dont_write_bytecode = True
import make_pilot_batch as mp  # noqa: E402

ANN = mp.ANN
BATCH = os.path.join(ANN, "pilot_batch_02_product_timelines.csv")
MANIFEST = os.path.join(ANN, "pilot_batch_02_manifest.json")
TAXONOMY = os.path.join(ROOT, "ontology", "product_aspect_taxonomy.json")
SEED, K = 20261120, 10
MIN_WORDS, MAX_WORDS = mp.MIN_WORDS, mp.MAX_WORDS
STRATA = {"SMALL": dict(lo=12, hi=20, days=180, quota=10, relaxed=True), "MEDIUM": dict(lo=20, hi=50, days=365, quota=12, relaxed=False),
          "LARGE": dict(lo=50, hi=200, days=365, quota=12, relaxed=False), "XLARGE": dict(lo=200, hi=10 ** 9, days=365, quota=6, relaxed=False)}
MIN_HIT_REVIEWS = 3


def lexicon():
    cats = json.load(open(TAXONOMY, encoding="utf-8"))["categories"]
    out = {}
    for c in cats:
        alts = [(re.escape(a[:-1]) + r"\w*") if a.endswith("*") else re.escape(a) for a in c["aliases"]]
        out[c["id"]] = re.compile(r"\b(?:" + "|".join(alts) + r")\b" if alts else r"(?!x)x", re.I)
    return out


def hits(texts, lex):
    s = pd.Series(list(texts), dtype=str)
    return pd.DataFrame({c: s.str.contains(rx, na=False).to_numpy() for c, rx in lex.items()})


def filtered_reviews(raw):
    r = raw[(raw.words >= MIN_WORDS) & (raw.words <= MAX_WORDS)].sort_values(["timestamp", "line"])
    return r.drop_duplicates(["parent_asin", "user_id"]).drop_duplicates(["parent_asin", "text"]).drop_duplicates(["parent_asin", "timestamp"])


def build():
    if os.path.exists(BATCH):
        sys.exit(f"{BATCH} already exists and may contain annotations; refusing to overwrite.")
    cols = mp.template_columns()
    b1 = pd.read_csv(mp.BATCH, encoding="utf-8-sig", dtype=str)
    tmpl = pd.read_csv(mp.TEMPLATE, encoding="utf-8-sig", dtype=str)
    excluded = set(b1.entity_id) | set(tmpl.entity_id)
    lex = lexicon()
    r = filtered_reviews(mp.load_raw())
    g = r.groupby("parent_asin").timestamp.agg(["size", "min", "max"])
    g["span_days"] = (g["max"] - g["min"]) / 86400000
    g["stratum"] = ""
    for name, s in STRATA.items():
        g.loc[(g["size"] >= s["lo"]) & (g["size"] < s["hi"]) & (g.span_days >= s["days"]), "stratum"] = name
    g = g[(g.stratum != "") & ~g.index.isin(excluded)].sort_index()
    pool = r[r.parent_asin.isin(g.index)]
    H = hits(pool.text, lex)
    H.index = pool.index
    H["parent_asin"] = pool.parent_asin.to_numpy()
    corpus_share = H.drop(columns="parent_asin").mean()
    cnt = H.groupby("parent_asin").sum()
    share = cnt.div(H.groupby("parent_asin").size(), axis=0)
    lift = share.div(corpus_share.replace(0, np.nan), axis=1)
    lift = lift.where(cnt >= MIN_HIT_REVIEWS)
    dist = lift.fillna(-1).idxmax(axis=1).where(lift.notna().any(axis=1), "(none)")
    g["distinctive"] = dist.reindex(g.index).fillna("(none)")
    rng = np.random.default_rng(SEED)
    chosen = []
    for name, s in STRATA.items():
        cand = {c: sorted(g[(g.stratum == name) & (g.distinctive == c)].index) for c in lex}
        picked = []
        while len(picked) < s["quota"] and any(cand.values()):
            order = [c for c in rng.permutation(sorted(cand)) if cand[c]]
            for c in order:
                if len(picked) == s["quota"]:
                    break
                p = cand[c].pop(int(rng.integers(len(cand[c]))))
                picked.append(p)
        assert len(picked) == s["quota"], f"stratum {name}: only {len(picked)} products"
        for w, p in enumerate(picked):
            chosen.append((name, p, 1 + w % 2))
    rows, picks = [], []
    for name, p, wave in chosen:
        t = r[r.parent_asin == p].sort_values(["timestamp", "line"]).reset_index(drop=True)
        idx = np.unique(np.round(np.linspace(0, len(t) - 1, K)).astype(int))
        assert len(idx) == K
        sid = "AMZ-PT-" + hashlib.sha1(p.encode()).hexdigest()[:10]
        for pos, i in enumerate(idx, 1):
            x = t.iloc[i]
            rows.append({"sequence_id": sid, "sequence_type": "PRODUCT_TIMELINE", "source": "amazon_all_beauty_2023", "entity_id": p,
                         "message_id": f"AMZ-AB-L{int(x.line):06d}", "position": pos, "timestamp_ms": int(x.timestamp), "timestamp_utc": mp.utc(int(x.timestamp)),
                         "language": "en", "title": x.title, "text": x.text, "parent_message_id": "", "annotation_status": "NOT_STARTED"})
        picks.append({"sequence_id": sid, "product": p, "stratum": name, "wave": wave, "eligible_reviews": int(len(t)),
                      "span_days": round(float((t.timestamp.iloc[-1] - t.timestamp.iloc[0]) / 86400000), 1), "distinctive_category": g.loc[p, "distinctive"],
                      "selected_ranks_in_product_stream": [int(i) + 1 for i in idx]})
    df = pd.DataFrame(rows).reindex(columns=cols).fillna("")
    df.to_csv(BATCH, index=False, encoding="utf-8-sig")
    sel_hits = hits(df.text, lex).mean().round(3).to_dict()
    b1_hits = hits(b1.text, lex).mean().round(3).to_dict()
    manifest = {"batch": "pilot_batch_02", "file": "data/annotations/pilot_batch_02_product_timelines.csv",
                "parameters": {"seed": SEED, "reviews_per_product": K, "min_words": MIN_WORDS, "max_words": MAX_WORDS,
                               "strata": {k: {kk: vv for kk, vv in v.items()} for k, v in STRATA.items()}},
                "deviation_from_sequence_definition": "SMALL stratum uses n >= 12 and span >= 180 days instead of n >= 20 and span >= 365 days; "
                                                      "pending team approval (guidelines section 12 rule 3 is a pilot decision to be revisited)",
                "eligible_products_by_stratum": {k: int((g.stratum == k).sum()) for k in STRATA}, "excluded_products": sorted(excluded),
                "sequences": picks, "messages": int(len(df)),
                "lexical_category_coverage": {"note": "share of reviews containing an alias of the category (sampling aid, not a label)",
                                              "batch_02": sel_hits, "batch_01": b1_hits, "corpus_eligible_products": corpus_share.round(3).to_dict()},
                "distinctive_category_counts": pd.Series([p["distinctive_category"] for p in picks]).value_counts().to_dict(),
                "annotated_rows": 0, "batch_sha256_at_generation": mp.sha256(BATCH)}
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in ("eligible_products_by_stratum", "messages", "distinctive_category_counts")}), "->", BATCH)


def verify():
    problems = []
    cols = mp.template_columns()
    b = pd.read_csv(BATCH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    man = json.load(open(MANIFEST, encoding="utf-8"))
    b1 = pd.read_csv(mp.BATCH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    t = pd.read_csv(mp.TEMPLATE, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    n_seq = sum(s["quota"] for s in STRATA.values())
    if b.columns.tolist() != cols:
        problems.append("columns differ from the template")
    if len(b) != n_seq * K or b.sequence_id.nunique() != n_seq:
        problems.append(f"expected {n_seq * K} rows in {n_seq} sequences, found {len(b)} in {b.sequence_id.nunique()}")
    if not b.message_id.is_unique:
        problems.append("message ids are not unique")
    for c in mp.BLANK_COLUMNS:
        if (b[c] != "").any():
            problems.append(f"annotation column {c} is not blank")
    if set(b.annotation_status) != {"NOT_STARTED"} or set(b.sequence_type) != {"PRODUCT_TIMELINE"}:
        problems.append("status or sequence_type not as prepared")
    if set(b.message_id) & (set(b1.message_id) | set(t.message_id)) or set(b.entity_id) & (set(b1.entity_id) | set(t.entity_id)):
        problems.append("overlaps batch 01 or the 16-record pilot")
    if mp.sha256(BATCH) != man["batch_sha256_at_generation"]:
        problems.append("the batch file differs from the manifest hash (it may contain annotations now)")
    raw = mp.load_raw().set_index("line")
    strata = {s["sequence_id"]: s for s in man["sequences"]}
    for sid, g in b.groupby("sequence_id", sort=False):
        ts = g.timestamp_ms.astype("int64").tolist()
        if g.entity_id.nunique() != 1 or ts != sorted(ts) or len(set(ts)) != len(ts) or g.position.astype(int).tolist() != list(range(1, len(g) + 1)):
            problems.append(f"{sid}: product, timestamp order or positions wrong")
        users = []
        for _, x in g.iterrows():
            src = raw.loc[int(x.message_id.split("-L")[1])]
            users.append(src.user_id)
            if (src.parent_asin != x.entity_id or int(src.timestamp) != int(x.timestamp_ms) or mp.utc(int(src.timestamp)) != x.timestamp_utc
                    or str(src.text) != x.text or str(src.title) != x.title or not (MIN_WORDS <= len(x.text.split()) <= MAX_WORDS)):
                problems.append(f"{x.message_id}: differs from the raw file or breaks the length rule")
        if len(set(users)) != len(users):
            problems.append(f"{sid}: a reviewer appears twice")
        s = STRATA[strata[sid]["stratum"]]
        if not (s["lo"] <= strata[sid]["eligible_reviews"] < s["hi"]) or strata[sid]["span_days"] < s["days"]:
            problems.append(f"{sid}: stratum size or span rule broken")
    cnt = pd.Series([s["stratum"] for s in man["sequences"]]).value_counts().to_dict()
    if cnt != {k: v["quota"] for k, v in STRATA.items()}:
        problems.append(f"stratum counts {cnt}")
    print("strata:", cnt, "| waves:", pd.Series([s["wave"] for s in man["sequences"]]).value_counts().to_dict())
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
