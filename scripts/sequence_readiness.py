"""
Phase 1: which sequences in our data are genuinely chronological, and which can support aspect-level evaluation?

Read-only on every dataset. Counts only what the files contain (timestamps, user and product identifiers, text length);
nothing is simulated and no annotation is invented. Writes:

    outputs/diagnostics/sequence_readiness/readiness_stats.json     all numbers quoted in docs/sequence_readiness.md
    data/annotations/aspect_annotation_template.csv                 header plus real, UNANNOTATED messages of the strictest
                                                                    Amazon sequences (same user, same product, >= 3 reviews, spanning >= 1 day)

    python scripts/sequence_readiness.py            # stats and template (about 2 minutes)

Keyword matching against ontology/product_aspect_taxonomy.json is a PRE-SCREEN of where annotation is worth the effort,
not an aspect detector and not a measurement of aspect accuracy.
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
RAW = os.path.join(ROOT, "data", "raw", "All_Beauty.jsonl")
SEQ = os.path.join(ROOT, "data", "preprocessed", "amazon_beauty_sequences.csv")
OUT_JSON = os.path.join(ROOT, "outputs", "diagnostics", "sequence_readiness", "readiness_stats.json")
TEMPLATE = os.path.join(ROOT, "data", "annotations", "aspect_annotation_template.csv")
MIN_WORDS = 10                                        # shorter reviews rarely contain an aspect with a sentiment
TEMPLATE_COLUMNS = ["sequence_id", "sequence_type", "source", "entity_id", "message_id", "position", "timestamp_ms",
                    "timestamp_utc", "language", "title", "text", "parent_message_id",
                    "aspect_status", "aspect_span_text", "span_start", "span_end", "span_type", "aspect_category", "sentiment",
                    "annotation_status", "annotator_id", "notes"]


def alias_regexes():
    tax = json.load(open(os.path.join(ROOT, "ontology", "product_aspect_taxonomy.json"), encoding="utf-8"))
    out = {}
    for c in tax["categories"]:
        parts = [re.escape(a[:-1]) + r"\w*" if a.endswith("*") else re.escape(a) for a in c["aliases"]]
        out[c["id"]] = re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I)
    return out


def match_matrix(texts, regexes):
    s = pd.Series(texts).fillna("").astype(str)
    return pd.DataFrame({k: s.str.contains(r, regex=True) for k, r in regexes.items()})


def main():
    raw = pd.read_json(RAW, lines=True, convert_dates=False)           # row index = line order in the file
    raw["line"] = raw.index + 1
    raw["words"] = raw.text.fillna("").str.split().str.len()
    seq = pd.read_csv(SEQ)
    rx = alias_regexes()
    stats = {"amazon_full_local_file": {
        "file": "data/raw/All_Beauty.jsonl", "reviews": int(len(raw)), "reviewers": int(raw.user_id.nunique()),
        "products_parent_asin": int(raw.parent_asin.nunique()),
        "first_review_utc": datetime.fromtimestamp(raw.timestamp.min() / 1000, timezone.utc).isoformat(),
        "last_review_utc": datetime.fromtimestamp(raw.timestamp.max() / 1000, timezone.utc).isoformat(),
        "reviewers_with_ge3_reviews": int((raw.groupby("user_id").size() >= 3).sum()),
        "build_script_parse_cap_records": 250000}}

    # ── A. user timelines: the project's existing OET sequences ──────────────────────────────────────────────
    s = seq.sort_values(["user_id", "sequence_position"])
    L = s.groupby("user_id").size()
    gaps = s.groupby("user_id").timestamp.diff().dropna() / 86400000
    n_asin = s.groupby("user_id").asin.nunique()
    stats["A_user_timelines_project_subset"] = {
        "reviewers": int(len(L)), "reviews": int(len(s)), "length_counts": {str(k): int(v) for k, v in L.value_counts().sort_index().items() if k <= 10},
        "length_ge5": int((L >= 5).sum()), "length_ge10": int((L >= 10).sum()), "length_ge20": int((L >= 20).sum()),
        "order_equals_timestamp_order_share": float(s.groupby("user_id").timestamp.apply(lambda t: t.is_monotonic_increasing).mean()),
        "consecutive_pairs": int(len(gaps)), "pairs_with_identical_timestamp": int((gaps == 0).sum()),
        "pairs_within_one_day_share": float((gaps < 1).mean()), "median_gap_days": float(gaps.median()),
        "reviewers_whose_reviews_are_all_different_products_share": float((n_asin == L).mean()),
        "reviewers_with_a_repeated_product": int((n_asin < L).sum())}
    texts = s.text_original.fillna("").astype(str)
    M = match_matrix(texts.values, rx)
    M["user"] = s.user_id.values
    M["words"] = texts.str.split().str.len().values
    cat_share = {k: float(M[k].mean()) for k in rx}
    per_user = M.groupby("user")[list(rx)].sum()
    gen = [c for c in rx if c != "GENERAL"]
    stats["A_user_timelines_project_subset"]["prescreen"] = {
        "share_of_reviews_with_alias_match_by_category": cat_share,
        "share_of_reviews_with_any_specific_alias": float(M[gen].any(axis=1).mean()),
        "reviewers_with_a_category_matched_in_ge2_reviews": int((per_user[gen] >= 2).any(axis=1).sum()),
        "reviewers_with_a_category_matched_in_ge3_reviews": int((per_user[gen] >= 3).any(axis=1).sum()),
        "note": "keyword pre-screen over text_original; an upper-bound hint, not annotation"}

    # ── B. same reviewer, same product, strictly valid ───────────────────────────────────────────────────────
    pairs_raw = raw.groupby(["user_id", "parent_asin"]).size()
    ok = raw[raw.words >= MIN_WORDS].drop_duplicates(["user_id", "parent_asin", "text"]).drop_duplicates(["user_id", "parent_asin", "timestamp"])
    n = ok.groupby(["user_id", "parent_asin"]).size()
    keep = n[n >= 2].index
    sub = ok.set_index(["user_id", "parent_asin"]).loc[keep].reset_index().sort_values(["user_id", "parent_asin", "timestamp"])
    span = sub.groupby(["user_id", "parent_asin"]).timestamp.apply(lambda x: (x.max() - x.min()) / 86400000)
    rating_changes = sub.groupby(["user_id", "parent_asin"]).rating.apply(lambda r: r.nunique() > 1)
    SM = match_matrix(sub.text.values, rx)
    SM[["user_id", "parent_asin"]] = sub[["user_id", "parent_asin"]].values
    sp_cat = SM.groupby(["user_id", "parent_asin"])[gen].sum()
    stats["B_same_reviewer_same_product"] = {
        "raw_pairs_with_ge2_reviews_before_filters": int((pairs_raw >= 2).sum()),
        "filters": f"review has >= {MIN_WORDS} words; identical text per pair dropped; identical timestamp per pair dropped",
        "pairs_ge2": int((n >= 2).sum()), "pairs_ge3": int((n >= 3).sum()), "pairs_ge4": int((n >= 4).sum()),
        "reviews_in_pairs_ge2": int(n[n >= 2].sum()), "median_span_days": float(span.median()),
        "pairs_ge2_spanning_at_least_1_day": int((span >= 1).sum()), "pairs_ge2_spanning_at_least_30_days": int((span >= 30).sum()),
        "pairs_ge3_spanning_at_least_1_day": int(((n.reindex(span.index) >= 3) & (span >= 1)).sum()),
        "pairs_ge3_spanning_at_least_30_days": int(((n.reindex(span.index) >= 3) & (span >= 30)).sum()),
        "pairs_ge3_posted_within_one_day": int(((n.reindex(span.index) >= 3) & (span < 1)).sum()),
        "share_of_pairs_with_a_rating_change": float(rating_changes.mean()),
        "pairs_with_a_category_matched_in_ge2_reviews": int((sp_cat >= 2).any(axis=1).sum())}

    # ── C. product timelines (different reviewers, one product) ──────────────────────────────────────────────
    pr = raw[raw.words >= MIN_WORDS].drop_duplicates(["user_id", "parent_asin"]).copy()
    cnt = pr.groupby("parent_asin").size()
    span_y = pr.groupby("parent_asin").timestamp.apply(lambda x: (x.max() - x.min()) / 86400000 / 365.25)
    stats["C_product_timelines"] = {
        "rule": f"reviews with >= {MIN_WORDS} words, one review per reviewer per product, ordered by timestamp",
        "unique_timestamp_share_all_reviews_per_product_mean": float(raw.groupby("parent_asin").timestamp.apply(lambda x: x.nunique() / len(x)).mean()),
        **{f"products_ge{k}": int((cnt >= k).sum()) for k in (10, 20, 50, 100)},
        **{f"reviews_in_products_ge{k}": int(cnt[cnt >= k].sum()) for k in (10, 20, 50, 100)},
        **{f"median_span_years_products_ge{k}": float(span_y[cnt >= k].median()) for k in (10, 20, 50, 100)},
        "in_project_subset_products_ge10_reviews": int((seq.groupby("asin").size() >= 10).sum())}
    big = pr[pr.parent_asin.isin(cnt[cnt >= 10].index)]
    PM = match_matrix(big.text.values, rx)
    PM["p"] = big.parent_asin.values
    pc = PM.groupby("p")[gen].sum()
    stats["C_product_timelines"]["prescreen"] = {
        "products_with_a_category_matched_in_ge10_reviews": int((pc >= 10).any(axis=1).sum()),
        "products_with_a_category_matched_in_ge20_reviews": int((pc >= 20).any(axis=1).sum()),
        "product_category_pairs_ge10": int((pc >= 10).sum().sum()),
        "note": "keyword pre-screen; an upper-bound hint, not annotation"}

    # ── D. DravidianCodeMix: what metadata exists ────────────────────────────────────────────────────────────
    rep = json.load(open(os.path.join(ROOT, "outputs", "comment_level", "data_report.json"), encoding="utf-8"))
    lines = at = 0
    for f in ("tamil_sentiment_full_train.csv", "mal_full_sentiment_train.csv", "kannada_sentiment_train.csv"):
        for line in open(os.path.join(ROOT, "data", "raw", f), encoding="utf-8", errors="replace"):
            lines += 1
            at += bool(re.search(r"@\w", line))
    stats["D_dravidiancodemix"] = {
        "comments_after_cleaning": {l: {sp: rep[l][sp].get("rows") or rep[l][sp].get("rows_out") for sp in ("train", "dev", "test")} for l in ("tamil", "malayalam", "kannada")},
        "fields_in_raw_files": "text and label only (sentiment: text;label; offensive: text<TAB>label<TAB>)",
        "timestamp_field": False, "author_field": False, "comment_or_video_id_field": False, "parent_comment_field": False,
        "raw_sentiment_train_lines": lines, "raw_lines_containing_an_at_mention": at,
        "valid_chronological_threads": 0}

    # ── template: real, unannotated messages of the strictest sequences ───────────────────────────────────────
    # Template sequences: >= 3 reviews AND spanning at least one day (reviews posted in one burst are not an evolution over time).
    ids = span[(n.reindex(span.index) >= 3) & (span >= 1)].index
    rows = []
    for (u, p), g in ok.set_index(["user_id", "parent_asin"]).loc[ids].reset_index().sort_values(["user_id", "parent_asin", "timestamp"]).groupby(["user_id", "parent_asin"], sort=False):
        sid = "AMZ-SUP-" + hashlib.sha1(f"{u}|{p}".encode()).hexdigest()[:10]
        for pos, (_, r) in enumerate(g.iterrows(), 1):
            rows.append({"sequence_id": sid, "sequence_type": "SAME_REVIEWER_SAME_PRODUCT", "source": "amazon_all_beauty_2023",
                         "entity_id": p, "message_id": f"AMZ-AB-L{int(r.line):06d}", "position": pos, "timestamp_ms": int(r.timestamp),
                         "timestamp_utc": datetime.fromtimestamp(r.timestamp / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "language": "en", "title": r.title, "text": r.text, "parent_message_id": "",
                         "annotation_status": "NOT_STARTED"})
    os.makedirs(os.path.dirname(TEMPLATE), exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=TEMPLATE_COLUMNS).fillna("")
    df.to_csv(TEMPLATE, index=False, encoding="utf-8-sig")
    stats["template"] = {"file": "data/annotations/aspect_annotation_template.csv", "sequences": int(df.sequence_id.nunique()),
                         "messages": int(len(df)), "annotated_rows": 0, "columns": TEMPLATE_COLUMNS}

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
