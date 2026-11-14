"""
Local, model-free readiness check of the reliable-annotation training files (nothing is trained, no tokenizer or encoder is loaded):

  1. the files are valid and contain exactly the 63 confirmed Batch 01 rows and the 398 agreed Wave 1 examples;
  2. product-level splits share no product, no message and no review text across train/val/test (exact text check);
  3. aspect-extraction safety: every message in the extraction files has identical, agreed spans in BOTH original annotator files and is finished by
     both, so no disputed, unmatched, unfinished or NEEDS_REVIEW message can contribute background (O) tokens;
  4. the existing readiness rules of each stage (src/aspect_training.py): size warnings of extraction and sentiment, cv_blockers for the
     out-of-fold run, series_blockers for the sequence model on the exact product splits the scripts would make.

    python scripts/aspect_readiness.py [--split_seeds 0 1 2 3 4 5]
Exit code 0 always; the table says PASS / PASS WITH WARNINGS / BLOCKED per stage.
"""
import argparse
import os
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src")):
    sys.path.insert(0, p)
sys.dont_write_bytecode = True
import logging  # noqa: E402
logging.disable(logging.WARNING)
import aspect_data as ad  # noqa: E402
import aspect_training as at  # noqa: E402
import validate_aspect_annotations as va  # noqa: E402

D = os.path.join(ROOT, "data", "annotations", "training")
ANN = os.path.join(ROOT, "data", "annotations")
SENT = [f"{D}/aspect_training_sentiment_batch01.csv", f"{D}/aspect_training_sentiment_wave1.csv"]
EXT = [f"{D}/aspect_training_extraction_batch01.csv", f"{D}/aspect_training_extraction_wave1.csv"]
ORIG = {"batch01": (f"{ANN}/pilot_batch_01_annotator_A.csv", f"{ANN}/pilot_batch_01_annotator_B.csv"),
        "wave1": (f"{ANN}/batch_02_wave1_annotator_1_DONE.csv", f"{ANN}/batch_02_wave1_annotator_2_DONE.csv")}
RANKED = ("POSITIVE", "NEGATIVE", "MIXED")


def rd(p):
    return pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def norm(t):
    return " ".join(t.lower().split())


def split_checks(splits):
    prod = {n: {m.entity_id for m in ms} for n, ms in splits.items()}
    msgs = {n: {m.message_id for m in ms} for n, ms in splits.items()}
    txt = {n: {norm(m.text) for m in ms} for n, ms in splits.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {"shared_products": sum(len(prod[a] & prod[b]) for a, b in pairs), "shared_messages": sum(len(msgs[a] & msgs[b]) for a, b in pairs),
            "shared_review_texts": sum(len(txt[a] & txt[b]) for a, b in pairs)}


def extraction_safety():
    """Every extraction message: finished in both originals, the two annotators' spans are identical sets, and equal the file's spans."""
    bad, n = [], 0
    for name, (pa, pb) in ORIG.items():
        A, B = rd(pa), rd(pb)
        E = rd(f"{D}/aspect_training_extraction_{name}.csv")
        spans = lambda df, m: {(r.span_start, r.span_end, r.aspect_category, r.sentiment) for r in df[(df.message_id == m) & (df.aspect_status == "ASPECT_PRESENT")].itertuples()}   # noqa: E731
        for m in E.message_id.unique():
            n += 1
            a, b, e = A[A.message_id == m], B[B.message_id == m], E[E.message_id == m]
            ok = (spans(A, m) == spans(B, m) == spans(E, m) and (set(a.annotation_status) | set(b.annotation_status)) == {"ANNOTATED"}
                  and set(a.aspect_status) == set(b.aspect_status) == set(e.aspect_status))
            if not ok:
                bad.append(m)
    return n, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    a = ap.parse_args(argv)
    out = {}

    # 1. the data
    s = at.load_corpus(SENT)
    ex = ad.build_sentiment_examples(s.messages, level="message_category")
    rows = sum(len(rd(p)) for p in SENT)
    b01 = rd(SENT[0])
    print(f"[data] sentiment files: {len(s.messages)} messages, {len({m.entity_id for m in s.messages})} products, {rows} rows; "
          f"Batch 01 rows {len(b01)} (expected 63), Wave 1 (message, category) examples {len([e for e in ex if e.message_id in set(rd(SENT[1]).message_id)])} (expected 398); "
          f"unfinished messages {len(s.unfinished_message_ids)}; label counts {dict(Counter(e.label for e in ex))}")
    out["data"] = len(b01) == 63 and len([e for e in ex if e.message_id in set(rd(SENT[1]).message_id)]) == 398 and not s.unfinished_message_ids
    e_corp = at.load_corpus(EXT)
    n_ext, bad = extraction_safety()
    print(f"[extraction safety] {n_ext} messages, {len(bad)} unsafe (must be 0): {bad[:5]}; mentions {sum(len(m.mentions) for m in e_corp.messages)}; "
          f"products {len({m.entity_id for m in e_corp.messages})}")
    out["safe"] = not bad

    # 2/4. stage by stage, on the exact splits the scripts make (default ratios and split_seed 0)
    sent_splits = at.make_splits(s.messages, (0.7, 0.15, 0.15), 0)
    ext_splits = at.make_splits(e_corp.messages, (0.7, 0.15, 0.15), 0)
    for label, sp in (("sentiment", sent_splits), ("extraction", ext_splits)):
        chk = split_checks(sp)
        print(f"[leakage, {label} splits, seed 0] {chk}; products per split {at.split_summary(sp) and {n: v['n_products'] for n, v in at.split_summary(sp).items()}}")
        out[f"leak_{label}"] = not any(chk.values())
    summ = at.split_summary(sent_splits)
    exs = {n: ad.build_sentiment_examples(m, level="message_category") for n, m in sent_splits.items()}
    stats = {n: {"products": summ[n]["n_products"], "items": len(exs[n]), "class_counts": Counter(e.label for e in exs[n])} for n in sent_splits}
    stats = {n: {**v, "class_counts": {c: v["class_counts"].get(c, 0) for c in ("POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED")}} for n, v in stats.items()}
    w_sent = at.size_warnings("sentiment", stats, annotators=s.annotators, duplicate_texts=len(ad.cross_split_duplicate_texts(sent_splits)))
    esum = at.split_summary(ext_splits)
    mentions = {n: sum(len(m.mentions) for m in ms if m.status == "ASPECT_PRESENT") for n, ms in ext_splits.items()}
    w_ext = at.size_warnings("extraction", {n: {"products": esum[n]["n_products"], "items": mentions[n]} for n in ext_splits}, annotators=e_corp.annotators,
                             duplicate_texts=len(ad.cross_split_duplicate_texts(ext_splits)))
    products = sorted({m.entity_id for m in s.messages})
    cv = at.cv_blockers(len(products), len(ex), 5)
    ext_block = [] if mentions["val"] > 0 and mentions["train"] > 0 else [f"split mentions {mentions}: a checkpoint cannot be chosen on span F1"]
    seq = {}
    for seed in a.split_seeds:
        outer = ad.split_entities(products, (0.85, 0.15), seed, ("pool", "test"))
        inner = ad.split_entities(outer["pool"], (0.8, 0.2), seed, ("train", "val"))
        counts = {}
        for n, ps in {"train": inner["train"], "val": inner["val"], "test": outer["test"]}.items():
            es = ad.build_sentiment_examples([m for m in s.messages if m.entity_id in set(ps)], level="message_category")
            ser = Counter((e.entity_id, e.category) for e in es if e.label in RANKED)
            counts[n] = sum(v >= 2 for v in ser.values())
        seq[seed] = (counts, at.series_blockers(counts))
    print("\nSTAGE READINESS (existing rules, thresholds unchanged)")
    print(f"  aspect extraction : {'BLOCKED' if ext_block else 'PASS WITH WARNINGS'}  mentions per split {mentions}; warnings: {len(w_ext)}")
    for w in w_ext:
        print("      -", w)
    per_split = ", ".join(n + " " + str(stats[n]["items"]) for n in stats)
    print(f"  aspect sentiment  : PASS WITH WARNINGS  examples {{{per_split}}}; warnings: {len(w_sent)}")
    for w in w_sent:
        print("      -", w)
    print(f"  OOF generation    : {'BLOCKED: ' + '; '.join(cv) if cv else 'PASS'}  ({len(products)} products, {len(ex)} examples, 5 folds)")
    ok_seeds = [k for k, (c, b) in seq.items() if not b]
    print(f"  aspect sequence   : {'PASS on split_seed ' + str(ok_seeds) if ok_seeds else 'BLOCKED on every tested split_seed'}")
    for k, (c, b) in seq.items():
        print(f"      split_seed {k}: usable series {c}  {b or 'ok'}")
    return out


if __name__ == "__main__":
    main()
