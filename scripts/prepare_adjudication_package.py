"""
Prepare an adjudication package from two annotators' files for the same batch.

READ-ONLY on both annotation files. It does not choose a winner, infer a label, or merge anything: every adjudicated_* and decision column is written
BLANK. It lists, side by side with the review text and timestamp, every place where the two annotators differ:
    UNMATCHED_SPAN_A_ONLY / UNMATCHED_SPAN_B_ONLY   a span the other annotator has no matching span for (overlap/union < 0.5)
    CATEGORY_DISAGREEMENT                           matched spans with different categories
    SENTIMENT_DISAGREEMENT                          matched spans with different sentiments
    NEEDS_REVIEW                                    a span either annotator marked NEEDS_REVIEW
    MESSAGE_STATUS_DISAGREEMENT                     one annotator found no/unclear aspect where the other found aspects
Spans are matched exactly as scripts/validate_aspect_annotations.py --compare does, so the numbers agree with it. Messages that are not finished in both files
(for example NEEDS_REVIEW) are outside the agreement statistics but are listed too, flagged in_agreement_statistics = no.

Output (a NEW directory, refused if it exists, so filled-in files are never overwritten):
    adjudication_items.csv   the working file (UTF-8 with BOM, opens in Excel)
    adjudication_items.xlsx  the same table, with wrapped text and the blank decision columns highlighted
    README.md                column dictionary and rules          summary.json   counts

    python scripts/prepare_adjudication_package.py --a data/annotations/<A>.csv --b data/annotations/<B>.csv --reference data/annotations/<batch>.csv --out_dir <new dir>
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.dont_write_bytecode = True
import validate_aspect_annotations as va                                  # noqa: E402

SIDE = ["span_text", "start", "end", "span_type", "category", "sentiment", "annotation_status", "notes"]
ADJ = ["adjudicated_span_text", "adjudicated_span_start", "adjudicated_span_end", "adjudicated_category", "adjudicated_sentiment", "decision_reason",
       "adjudicator_id"]
COLUMNS = (["item_id", "item_type", "conflict_group", "in_agreement_statistics", "guideline_topics", "message_id", "sequence_id", "product_id", "position",
            "timestamp_utc", "timestamp_ms", "review_title", "review_text", "match_kind", "overlap_iou"]
           + [f"A_{c}" for c in SIDE] + [f"B_{c}" for c in SIDE] + ["other_side_shows"] + ADJ)
REFERRING = re.compile(r"^(it|this|that|these|those|they|them)(\s+\S+){0,2}$|^(the |this )?(product|item|stuff)$", re.I)


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def spans_of(df, mid):
    g = df[(df.message_id == mid) & (df.aspect_status == "ASPECT_PRESENT")]
    return [{"span_text": r.aspect_span_text, "start": int(r.span_start), "end": int(r.span_end), "span_type": r.span_type, "category": r.aspect_category,
             "sentiment": r.sentiment, "annotation_status": r.annotation_status, "notes": r.notes} for r in g.itertuples() if r.span_start != ""]


def iou(a, b):
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union else 0.0


def is_referring(span):
    return bool(REFERRING.match(span["span_text"].strip().strip(".,!?")))


def topics(a, b, kind, item_type, iou_v=None):
    """Deterministic tags from observable fields only (they point to a guideline question; they do not say who is right)."""
    t = []
    sides = [s for s in (a, b) if s]
    if item_type in ("UNMATCHED_SPAN_A_ONLY", "UNMATCHED_SPAN_B_ONLY") and kind.startswith("partial"):
        t.append("span_boundary")
    if item_type in ("UNMATCHED_SPAN_A_ONLY", "UNMATCHED_SPAN_B_ONLY") and kind.startswith("no overlapping"):
        t.append("aspect_present_in_one_annotation_only")
    if a and b and kind.startswith("matched") and iou_v is not None and iou_v < 1.0:
        t.append("span_boundary")
    if any(is_referring(s) for s in sides):
        t.append("referring_expression_or_pronoun")
    if any(s["category"] == "OTHER_ASPECT" for s in sides):
        t.append("OTHER_ASPECT")
    if any(s["category"] == "GENERAL" for s in sides):
        t.append("GENERAL")
    if any(s["span_type"] == "IMPLICIT" for s in sides) or (a and b and a["span_type"] != b["span_type"]):
        t.append("implicit_aspect")
    if item_type == "SENTIMENT_DISAGREEMENT" and a and b and "NEUTRAL" in (a["sentiment"], b["sentiment"]):
        t.append("NEUTRAL_vs_polarity")
    if any("UNSURE" in s["notes"] for s in sides):
        t.append("annotator_marked_UNSURE")
    if item_type == "NEEDS_REVIEW":
        t.append("NEEDS_REVIEW")
    if item_type == "MESSAGE_STATUS_DISAGREEMENT":
        t.append("message_level_aspect_status")
    return ";".join(dict.fromkeys(t))


def build_items(A, B, ref, tax):
    rA, rB = va.validate(A, tax, ref), va.validate(B, tax, ref)
    both_ok = rA["complete_messages"] & rB["complete_messages"]
    order = list(dict.fromkeys((list(ref.message_id) if ref is not None else []) + list(A.message_id) + list(B.message_id)))
    prep = {}
    for mid in order:
        ra = A[A.message_id == mid].head(1)
        rb = B[B.message_id == mid].head(1)
        src = ra if len(ra) else rb
        if len(ra) and len(rb):
            for c in va.PREPARED:
                if ra.iloc[0][c] != rb.iloc[0][c]:
                    raise ValueError(f"message {mid}: column {c} differs between the two files; they are not annotations of the same batch")
        if len(src):
            prep[mid] = src.iloc[0]
    items, counter = [], Counter()

    def add(item_type, mid, a, b, iou_v, kind, other_shows, group=""):
        counter[item_type] += 1
        p = prep[mid]
        row = {c: "" for c in COLUMNS}
        row.update({"item_type": item_type, "conflict_group": group, "in_agreement_statistics": "yes" if mid in both_ok else "no", "message_id": mid,
                    "sequence_id": p.sequence_id, "product_id": p.entity_id, "position": p.position, "timestamp_utc": p.timestamp_utc,
                    "timestamp_ms": p.timestamp_ms, "review_title": p.title, "review_text": p.text, "match_kind": kind,
                    "overlap_iou": "" if iou_v is None else f"{iou_v:.2f}", "other_side_shows": other_shows,
                    "guideline_topics": topics(a, b, kind, item_type, iou_v)})
        for side, s in (("A", a), ("B", b)):
            if s:
                for c in SIDE:
                    row[f"{side}_{c}"] = s[c]
        starts = [int(s["start"]) for s in (a, b) if s and s["start"] != ""]
        items.append((order.index(mid), min(starts or [0]), item_type, row))

    gcount = 0
    for mid in order:
        if mid not in prep:
            continue
        sa, sb = spans_of(A, mid), spans_of(B, mid)
        st_a = "ASPECT_PRESENT" if sa else (A[A.message_id == mid].aspect_status.iloc[0] if (A.message_id == mid).any() else "MISSING")
        st_b = "ASPECT_PRESENT" if sb else (B[B.message_id == mid].aspect_status.iloc[0] if (B.message_id == mid).any() else "MISSING")
        if (st_a == "ASPECT_PRESENT") != (st_b == "ASPECT_PRESENT"):
            add("MESSAGE_STATUS_DISAGREEMENT", mid, {**{c: "" for c in SIDE}, "span_text": f"(message status: {st_a})"} if not sa else sa[0],
                {**{c: "" for c in SIDE}, "span_text": f"(message status: {st_b})"} if not sb else sb[0], None,
                f"A: {st_a}; B: {st_b}", "message-level status; the first span of the annotator who found aspects is shown")
            # no `continue`: the spans of the annotator who found aspects are also unmatched spans and are listed below, as --compare counts them
        m = va.match_spans([(s["start"], s["end"]) for s in sa], [(s["start"], s["end"]) for s in sb])
        ua, ub = {x[0] for x in m}, {x[1] for x in m}
        for ia, ib, v in m:
            a, b = sa[ia], sb[ib]
            kind = f"matched (IoU {v:.2f})"
            if "NEEDS_REVIEW" in (a["annotation_status"], b["annotation_status"]):
                add("NEEDS_REVIEW", mid, a, b, v, kind, "the other annotator's matched span")
                continue
            if a["category"] != b["category"]:                          # a pair that differs in both gets one item of each type, as --compare counts them
                add("CATEGORY_DISAGREEMENT", mid, a, b, v, kind, "the other annotator's matched span")
            if a["sentiment"] != b["sentiment"]:
                add("SENTIMENT_DISAGREEMENT", mid, a, b, v, kind, "the other annotator's matched span")
        # unmatched spans, with mirrored partial overlaps linked into one conflict group
        un = [("A", i, sa[i]) for i in range(len(sa)) if i not in ua] + [("B", j, sb[j]) for j in range(len(sb)) if j not in ub]
        parent = list(range(len(un)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for x in range(len(un)):
            for y in range(x + 1, len(un)):
                if un[x][0] != un[y][0] and iou(un[x][2], un[y][2]) > 0:
                    parent[find(x)] = find(y)
        gid = {}
        for x in range(len(un)):
            r = find(x)
            if r not in gid:
                gcount += 1
                gid[r] = f"G{gcount:03d}"
        for x, (who, _, s) in enumerate(un):
            others = [o for w2, _, o in un if w2 != who and iou(s, o) > 0]
            near = max(others, key=lambda o: iou(s, o), default=None)
            if near is None:                              # also allow the nearest span even if it was matched to another span of this annotator
                pool = sb if who == "A" else sa
                near = max(pool, key=lambda o: iou(s, o), default=None)
                near = near if near and iou(s, near) > 0 else None
            v = iou(s, near) if near else 0.0
            kind = (f"partial overlap with the other annotator's span (IoU {v:.2f}, below 0.5)" if near else "no overlapping span from the other annotator")
            a, b = (s, near) if who == "A" else (near, s)
            add(f"UNMATCHED_SPAN_{who}_ONLY", mid, a, b, v if near else None, kind,
                "the nearest overlapping span (IoU below 0.5)" if near else "nothing: this annotator has no span here", gid[find(x)])
    items.sort(key=lambda t: (t[0], t[1], t[2]))
    rows = []
    for n, (_, _, _, row) in enumerate(items, 1):
        row["item_id"] = f"ADJ-{n:03d}"
        rows.append(row)
    return rows, counter, both_ok, order


README = """# Adjudication package (to be filled in by the team)

Nothing here was decided by a script. No annotator was chosen, no label was inferred, no annotation was merged. The `adjudicated_*`, `decision_reason` and
`adjudicator_id` columns are **blank on purpose**; the originals are untouched.

## How to use `adjudication_items.csv` / `.xlsx`
* One row per item to settle. Rows with the same `conflict_group` are the two sides of one boundary conflict (an A span and a B span that partly overlap): settle the group once.
* `in_agreement_statistics = no` marks messages that were not finished in both files (they are outside the agreement figures, listed so the whole message can be settled).
* `guideline_topics` tags the kind of question (span boundary, referring expression, OTHER_ASPECT, GENERAL, implicit aspect, NEUTRAL vs polarity, UNSURE note...). The tags come from
  observable fields only; they do not say who is right.
* For an unmatched span the other annotator's columns show the nearest overlapping span (see `other_side_shows`) or are empty.
* Fill `adjudicated_span_text` with `adjudicated_span_start`/`_end` (0-based character offsets in `review_text`, end exclusive) when the span itself is decided, `adjudicated_category` (a product-taxonomy id
  or OTHER_ASPECT), `adjudicated_sentiment` (POSITIVE, NEGATIVE, NEUTRAL or MIXED) and a short `decision_reason`. Leave the span columns empty for an item you decide to drop and say so in the reason.
* The completed file is **not yet** an annotation file: a later, separate step turns the decisions into the 22-column format and validates it.

## Columns
`item_type`, `conflict_group`, `guideline_topics`: see above. `message_id`, `sequence_id`, `product_id`, `position`, `timestamp_utc`/`timestamp_ms`, `review_title`, `review_text`: copied from the original batch
(the title is context only; spans point into `review_text`). `match_kind`, `overlap_iou`: how the two spans relate. `A_*` / `B_*`: each annotator's span text, offsets, span type, category, sentiment, annotation status and notes.
"""


def write_package(rows, counter, both_ok, order, out_dir, paths):
    os.makedirs(out_dir)                                                  # raises FileExistsError: filled-in files are never overwritten
    csv_path = os.path.join(out_dir, "adjudication_items.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "items"
    ws.append(COLUMNS)
    for r in rows:
        ws.append([r[c] for c in COLUMNS])
    yellow, grey = PatternFill("solid", fgColor="FFF2CC"), PatternFill("solid", fgColor="E7E6E6")
    for i, c in enumerate(COLUMNS, 1):
        cell = ws.cell(row=1, column=i)
        cell.font = Font(bold=True)
        cell.fill = yellow if c in ADJ else grey
        ws.column_dimensions[cell.column_letter].width = 70 if c == "review_text" else 34 if c in ("A_notes", "B_notes", "guideline_topics", "review_title") else 22 if "span_text" in c else 14
        for r in range(2, len(rows) + 2):
            ws.cell(row=r, column=i).alignment = Alignment(wrap_text=True, vertical="top")
            if c in ADJ:
                ws.cell(row=r, column=i).fill = yellow
    ws.freeze_panes = "F2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(os.path.join(out_dir, "adjudication_items.xlsx"))
    open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8", newline="\n").write(README)
    tops = Counter(t for r in rows for t in r["guideline_topics"].split(";") if t)
    summary = {"inputs": paths, "items": len(rows), "by_item_type": dict(counter), "by_guideline_topic": dict(tops.most_common()),
               "conflict_groups": len({r["conflict_group"] for r in rows if r["conflict_group"]}), "messages": len(order),
               "messages_in_agreement_statistics": len(both_ok), "items_outside_agreement_statistics": sum(r["in_agreement_statistics"] == "no" for r in rows),
               "adjudication_columns_blank": all(r[c] == "" for r in rows for c in ADJ)}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), indent=1)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--reference", default=None, help="the unannotated batch (message order, texts, timestamps)")
    ap.add_argument("--out_dir", required=True, help="a NEW directory")
    a = ap.parse_args(argv)
    A, B = read(a.a), read(a.b)
    for name, df in (("A", A), ("B", B)):
        miss = [c for c in va.COLUMNS if c not in df.columns]
        if miss:
            raise ValueError(f"annotator {name}'s file lacks columns {miss}")
    ref = read(a.reference) if a.reference else None
    rows, counter, both_ok, order = build_items(A, B, ref, va.load_taxonomies())
    s = write_package(rows, counter, both_ok, order, a.out_dir, {"A": a.a, "B": b_ if (b_ := a.b) else a.b, "reference": a.reference})
    print(json.dumps(s, indent=1))
    return a.out_dir


if __name__ == "__main__":
    main()
