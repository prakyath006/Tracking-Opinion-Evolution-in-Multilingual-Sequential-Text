"""
Training files from reliable annotations ONLY (no adjudication, no model predictions, no gold labels created):

  Batch 01: the 63 CONFIRMED_AGREED rows of the provisional consolidation (both annotators, identical span, category, sentiment, type).
  Batch 02 Wave 1: spans both annotators marked (overlap >= 0.5) with the same category and sentiment, in messages both finished.

Everything disputed, unresolved, low-confidence or provisional is left out. Two files are written to data/annotations/training/ (never overwritten):

  aspect_training_sentiment_<batch>.csv   every agreed mention (annotator 1's span for the 27 pairs whose boundaries differ; the label is the agreed one).
                                  Used by aspect sentiment and the aspect series; a disputed mention is simply absent, which cannot change an agreed label.
  aspect_training_extraction_<batch>.csv  ONLY messages whose A and B annotations are identical (same spans, category, sentiment), plus messages both marked NO_ASPECT.
                                  A message with any disputed, unmatched, unfinished or boundary-different span is excluded entirely, so its unmarked
                                  or contested text can never become a background (O) label.

    python scripts/build_training_annotations.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.dont_write_bytecode = True
import validate_aspect_annotations as va  # noqa: E402

ANN = os.path.join(ROOT, "data", "annotations")
P1 = os.path.join(ROOT, "outputs", "annotation_consolidation", "pilot_batch_01")
OUT = os.path.join(ANN, "training")
SOURCES = {"batch01": dict(a=f"{ANN}/pilot_batch_01_annotator_A.csv", b=f"{ANN}/pilot_batch_01_annotator_B.csv", ref=f"{ANN}/pilot_batch_01_product_timelines.csv",
                           who="annotator_A|annotator_B"),
           "wave1": dict(a=f"{ANN}/batch_02_wave1_annotator_1_DONE.csv", b=f"{ANN}/batch_02_wave1_annotator_2_DONE.csv", ref=f"{ANN}/batch_02_wave1_reference.csv",
                         who="annotator_1|annotator_2")}


def rd(p):
    return pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def confirmed_batch01():
    c, p = rd(f"{P1}/pilot_batch_01_PROVISIONAL_consolidated.csv"), rd(f"{P1}/pilot_batch_01_PROVISIONAL_provenance.csv")
    assert len(c) == len(p) and (c.message_id.to_numpy() == p.message_id.to_numpy()).all()
    keep = (p.training_eligible == "yes").to_numpy()
    assert (p.tier[keep] == "CONFIRMED_AGREED").all() and int(keep.sum()) == 63
    return {(r.message_id, int(r.span_start), int(r.span_end)) for r in c[keep].itertuples()}


def messages(df):
    return {m: g for m, g in df.groupby("message_id", sort=False)}


def build(name, cfg, tax, confirmed=None):
    A, B, REF = rd(cfg["a"]), rd(cfg["b"]), rd(cfg["ref"])
    ra, rb = va.validate(A, tax, REF), va.validate(B, tax, REF)
    assert not [i for i in ra["issues"] + rb["issues"] if i["severity"] == va.ERROR], "validation errors"
    ok = ra["complete_messages"] & rb["complete_messages"]                       # finished and valid in both; NEEDS_REVIEW messages are never complete
    prepared = {r["message_id"]: {c: r[c] for c in va.PREPARED} for _, r in REF.drop_duplicates("message_id").iterrows()}
    ga, gb = messages(A), messages(B)
    sent_rows, ext_rows, stats = [], [], dict(messages=len(prepared), finished_in_both=len(ok), agreed_mentions=0, extraction_messages=0, extraction_no_aspect=0)

    def row(mid, s, e, text, typ, cat, sent):
        r = dict(prepared[mid])
        r.update(aspect_status="ASPECT_PRESENT", aspect_span_text=text, span_start=str(s), span_end=str(e), span_type=typ, aspect_category=cat, sentiment=sent,
                 annotation_status="ANNOTATED", annotator_id=cfg["who"], notes="")
        return r

    for mid in prepared:
        if mid not in ok:
            continue
        a, b = ga[mid], gb[mid]
        sa = [r for r in a.itertuples() if r.aspect_status == "ASPECT_PRESENT"]
        sb = [r for r in b.itertuples() if r.aspect_status == "ASPECT_PRESENT"]
        if not sa and not sb:                                                    # both say there is no aspect: a valid all-background message
            r = dict(prepared[mid]); r.update(aspect_status="NO_ASPECT", annotation_status="ANNOTATED", annotator_id=cfg["who"])
            if set(a.aspect_status) == set(b.aspect_status) == {"NO_ASPECT"}:
                ext_rows.append(r); stats["extraction_no_aspect"] += 1
            continue
        key = lambda r: (int(r.span_start), int(r.span_end), r.aspect_category, r.sentiment)          # noqa: E731
        pairs = va.match_spans([key(r) for r in sa], [key(r) for r in sb])
        agreed = [(sa[i], sb[j]) for i, j, _ in pairs if sa[i].aspect_category == sb[j].aspect_category and sa[i].sentiment == sb[j].sentiment and sa[i].sentiment]
        if confirmed is not None:                                                # batch 01: only the 63 rows confirmed by span, category, sentiment and type
            agreed = [(x, y) for x, y in agreed if (mid, int(x.span_start), int(x.span_end)) in confirmed and (x.span_start, x.span_end) == (y.span_start, y.span_end)]
        if confirmed is None:                                                    # batch 01 sentiment rows come straight from the 63 confirmed rows (below)
            for x, y in agreed:
                sent_rows.append(row(mid, int(x.span_start), int(x.span_end), x.aspect_span_text, x.span_type, x.aspect_category, x.sentiment))
            stats["agreed_mentions"] += len(agreed)
        identical = (len(agreed) == len(sa) == len(sb) and all((x.span_start, x.span_end) == (y.span_start, y.span_end) for x, y in agreed))
        if identical:
            for x, y in agreed:
                ext_rows.append(row(mid, int(x.span_start), int(x.span_end), x.aspect_span_text, x.span_type, x.aspect_category, x.sentiment))
            stats["extraction_messages"] += 1
    if confirmed is not None:
        c, p = rd(f"{P1}/pilot_batch_01_PROVISIONAL_consolidated.csv"), rd(f"{P1}/pilot_batch_01_PROVISIONAL_provenance.csv")
        c = c[(p.training_eligible == "yes").to_numpy()].copy()
        c["notes"] = ""
        sent_rows += c.to_dict("records")
        stats["agreed_mentions"] = len(c)
    return sent_rows, ext_rows, stats


def main():
    os.makedirs(OUT, exist_ok=True)
    # one file per source batch, because each is checked against its own unannotated reference (a message id may be in only one file)
    paths = {(task, name): os.path.join(OUT, f"aspect_training_{task}_{name}.csv") for task in ("sentiment", "extraction") for name in SOURCES}
    for p in paths.values():
        if os.path.exists(p):
            sys.exit(f"{p} already exists; refusing to overwrite")
    tax = va.load_taxonomies()
    report = {}
    for name, cfg in SOURCES.items():
        s, e, st = build(name, cfg, tax, confirmed_batch01() if name == "batch01" else None)
        report[name] = st
        for task, rows in (("sentiment", s), ("extraction", e)):
            df = pd.DataFrame(rows, columns=va.COLUMNS)
            assert not df.duplicated(["message_id", "span_start", "span_end"]).any()
            df.to_csv(paths[(task, name)], index=False, encoding="utf-8-sig")
            res = va.validate(df, tax, rd(cfg["ref"]))
            assert not [i for i in res["issues"] if i["severity"] == va.ERROR] and not res["unfinished_messages"], (task, name)
    print(report)
    for (task, name), p in paths.items():
        d = rd(p)
        print(task, name, os.path.relpath(p, ROOT), "| rows", len(d), "| messages", d.message_id.nunique(), "| products", d.entity_id.nunique(),
              "| (message, category) examples", len(d[d.aspect_status == "ASPECT_PRESENT"].drop_duplicates(["message_id", "aspect_category"])))


if __name__ == "__main__":
    main()
