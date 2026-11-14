"""
Validate manually completed aspect-annotation CSV files and measure agreement between two annotators.

Read-only: it never edits, completes or infers an annotation, and it never writes next to an input file.
Schema and rules: docs/aspect_annotation_guidelines.md. Usage and issue codes: docs/aspect_annotation_validation.md.

    python scripts/validate_aspect_annotations.py FILE.csv                     # validate and show statistics
    python scripts/validate_aspect_annotations.py FILE.csv --strict            # unfinished messages also fail
    python scripts/validate_aspect_annotations.py A.csv --compare B.csv        # validate both, then agreement
    python scripts/validate_aspect_annotations.py FILE.csv --json report.json  # also write a JSON report

Exit code 0 = no errors (and, with --strict, nothing unfinished); 1 = errors or unfinished messages under --strict; 2 = bad usage.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True

COLUMNS = ["sequence_id", "sequence_type", "source", "entity_id", "message_id", "position", "timestamp_ms", "timestamp_utc",
           "language", "title", "text", "parent_message_id", "aspect_status", "aspect_span_text", "span_start", "span_end",
           "span_type", "aspect_category", "sentiment", "annotation_status", "annotator_id", "notes"]
PREPARED = COLUMNS[:12]                                    # filled by the preparation scripts; annotators never edit them
SPAN_FIELDS = ["aspect_span_text", "span_start", "span_end", "span_type"]
CONTENT = ["aspect_status"] + SPAN_FIELDS + ["aspect_category", "sentiment"]

SEQUENCE_TYPES = {"PRODUCT_TIMELINE", "SAME_REVIEWER_SAME_PRODUCT", "USER_TIMELINE", "COMMENT_LEVEL", "DISCUSSION_THREAD"}
ASPECT_STATUSES = {"ASPECT_PRESENT", "NO_ASPECT", "UNCLEAR_ASPECT"}
SENTIMENTS = {"POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"}
SPAN_TYPES = {"EXPLICIT", "IMPLICIT"}
ANNOTATION_STATUSES = {"NOT_STARTED", "IN_PROGRESS", "ANNOTATED", "NEEDS_REVIEW", "ADJUDICATED"}
COMPLETE = {"ANNOTATED", "ADJUDICATED"}
ENUMS = {"sequence_type": SEQUENCE_TYPES, "aspect_status": ASPECT_STATUSES, "sentiment": SENTIMENTS, "span_type": SPAN_TYPES,
         "annotation_status": ANNOTATION_STATUSES}
REFERENCE_FILES = ["pilot_batch_01_product_timelines.csv", "aspect_annotation_template.csv", "batch_02_wave1_reference.csv"]
IOU_MATCH = 0.5                                            # guidelines section 10: spans match at overlap/union >= 0.5

ERROR, WARNING = "ERROR", "WARNING"


# ───────────────────────────── taxonomy ─────────────────────────────
def load_taxonomies(root=ROOT):
    prod = json.load(open(os.path.join(root, "ontology", "product_aspect_taxonomy.json"), encoding="utf-8"))
    film = json.load(open(os.path.join(root, "ontology", "aspect_taxonomy.json"), encoding="utf-8"))
    extra = {"GENERAL", "OTHER_ASPECT"}
    return {"product": {c["id"] for c in prod["categories"]} | extra, "social": {c["id"] for c in film["categories"]} | extra}


def domain_of(source):
    s = (source or "").lower()
    return "product" if s.startswith("amazon") else "social" if s.startswith("dravidiancodemix") else None


# ───────────────────────────── reading ─────────────────────────────
def read_annotations(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def header_issues(df):
    out = []
    missing, extra = [c for c in COLUMNS if c not in df.columns], [c for c in df.columns if c not in COLUMNS]
    if missing:
        out.append(_issue(ERROR, "BAD_HEADER", None, None, None, f"missing columns: {missing}"))
    if extra:
        out.append(_issue(ERROR, "BAD_HEADER", None, None, None, f"unexpected columns: {extra}"))
    return out


def _issue(sev, code, row, message_id, field, detail):
    return {"severity": sev, "code": code, "row": row, "message_id": message_id, "field": field, "detail": detail}


def _is_int(v):
    return v.isdigit() and (v == "0" or not v.startswith("0"))


# ───────────────────────────── validation ─────────────────────────────
def validate(df, taxonomies, reference=None):
    """
    Returns a dict: issues (list), error_rows (set of 0-based row indices with an ERROR), unfinished_messages (set),
    complete_messages (set of message ids whose rows are all finished and error-free).
    Row numbers in issues are 1-based data rows (header excluded).
    """
    issues = header_issues(df)
    if any(i["code"] == "BAD_HEADER" and "missing" in i["detail"] for i in issues):
        return {"issues": issues, "error_rows": set(), "unfinished_messages": set(), "complete_messages": set(), "missing_messages": [], "rows": 0}
    err_rows, unfinished_rows = set(), set()

    def add(sev, code, i, field, detail):
        issues.append(_issue(sev, code, i + 1, df.at[i, "message_id"], field, detail))
        if sev == ERROR:
            err_rows.add(i)

    for i in df.index:
        r = df.loc[i]
        # labels
        for col, allowed in ENUMS.items():
            v = r[col]
            if v == "":
                continue
            if v != v.strip():
                add(ERROR, "WHITESPACE", i, col, f"{v!r} has leading or trailing whitespace")
            elif v not in allowed:
                hint = f" (did you mean {v.upper()}?)" if v.upper() in allowed else ""
                add(ERROR, "BAD_LABEL", i, col, f"{v!r} is not one of {sorted(allowed)}{hint}")
        for col in ("aspect_category", "span_start", "span_end", "annotator_id"):
            if r[col] != r[col].strip():
                add(ERROR, "WHITESPACE", i, col, f"{r[col]!r} has leading or trailing whitespace")
        if r["annotation_status"] == "":
            add(ERROR, "MISSING_STATUS", i, "annotation_status", "annotation_status is empty")
        if r["message_id"] == "" or r["sequence_id"] == "":
            add(ERROR, "MISSING_ID", i, "message_id", "sequence_id or message_id is empty")
        dom = domain_of(r["source"])
        if dom is None:
            add(ERROR, "UNKNOWN_SOURCE", i, "source", f"source {r['source']!r} is not a known dataset")

        status, astat = r["annotation_status"], r["aspect_status"]
        has_content = any(r[c] != "" for c in CONTENT)
        finished = status in COMPLETE
        if status == "NOT_STARTED" and has_content:
            add(ERROR, "STATUS_CONTENT_MISMATCH", i, "annotation_status", "annotation content present but annotation_status is NOT_STARTED")
        if status in ("NOT_STARTED", "IN_PROGRESS"):
            unfinished_rows.add(i)
        if status in COMPLETE | {"NEEDS_REVIEW"} and r["annotator_id"] == "":
            add(ERROR, "MISSING_ANNOTATOR", i, "annotator_id", f"annotator_id is required when annotation_status is {status}")
        if finished and astat == "":
            add(ERROR, "MISSING_ASPECT_STATUS", i, "aspect_status",
                f"{status} row has no aspect_status (a message with no aspect must be NO_ASPECT, never left blank)")
        if status == "NEEDS_REVIEW":
            unfinished_rows.add(i)
            if r["notes"].strip() == "":
                add(WARNING, "NEEDS_REVIEW_NO_NOTES", i, "notes", "NEEDS_REVIEW should explain why in notes")

        if astat == "ASPECT_PRESENT":
            _check_present(r, i, status, dom, taxonomies, add, unfinished_rows)
        elif astat in ("NO_ASPECT", "UNCLEAR_ASPECT"):
            if r["sentiment"] != "":
                add(ERROR, "SENTIMENT_ON_NON_ASPECT", i, "sentiment", f"{astat} must have an empty sentiment, found {r['sentiment']!r}")
            filled = [c for c in SPAN_FIELDS + ["aspect_category"] if r[c] != ""]
            if filled:
                add(ERROR, "FIELDS_ON_NON_ASPECT", i, filled[0], f"{astat} must have empty {filled}")
        elif astat == "" and has_content:
            add(ERROR, "CONTENT_WITHOUT_ASPECT_STATUS", i, "aspect_status", "span, category or sentiment filled but aspect_status is empty")

    missing_messages = _check_messages(df, issues, err_rows, unfinished_rows, reference)
    msgs = defaultdict(list)
    for i in df.index:
        msgs[df.at[i, "message_id"]].append(i)
    bad_msgs = {df.at[i, "message_id"] for i in err_rows}
    unfinished_msgs = {m for m, rows in msgs.items() if any(i in unfinished_rows for i in rows)}
    complete = {m for m, rows in msgs.items() if m not in bad_msgs and m not in unfinished_msgs
                and all(df.at[i, "annotation_status"] in COMPLETE for i in rows)}
    todo = sorted(unfinished_msgs - bad_msgs)
    if todo:
        issues.append(_issue(WARNING, "UNFINISHED_MESSAGES", None, None, "annotation_status",
                             f"{len(todo)} messages are not finished (NOT_STARTED, IN_PROGRESS, NEEDS_REVIEW or incomplete); first: {todo[:5]}"))
    return {"issues": issues, "error_rows": err_rows, "unfinished_messages": unfinished_msgs, "complete_messages": complete,
            "missing_messages": missing_messages, "rows": len(df)}


def _check_present(r, i, status, dom, tax, add, unfinished_rows):
    needs_all = status in COMPLETE
    needs_span = status in COMPLETE or status == "NEEDS_REVIEW"
    missing = [c for c in SPAN_FIELDS if r[c] == ""]
    if missing:
        if needs_span:
            add(ERROR, "MISSING_SPAN", i, missing[0], f"ASPECT_PRESENT row is missing {missing}")
        else:
            unfinished_rows.add(i)
    else:
        text, span = r["text"], r["aspect_span_text"]
        if not (_is_int(r["span_start"]) and _is_int(r["span_end"])):
            add(ERROR, "SPAN_NOT_INTEGER", i, "span_start", f"span_start={r['span_start']!r}, span_end={r['span_end']!r} must be whole numbers")
        else:
            s, e = int(r["span_start"]), int(r["span_end"])
            if not (0 <= s < e <= len(text)):
                add(ERROR, "SPAN_RANGE", i, "span_start", f"offsets {s}:{e} are not inside the text (length {len(text)}) or start >= end")
            elif text[s:e] != span:
                add(ERROR, "SPAN_MISMATCH", i, "aspect_span_text", f"text[{s}:{e}] is {text[s:e]!r} but aspect_span_text is {span!r}")
    if r["aspect_category"] == "":
        if needs_span:
            add(ERROR, "MISSING_CATEGORY", i, "aspect_category", "ASPECT_PRESENT row has no aspect_category")
        else:
            unfinished_rows.add(i)
    elif dom and r["aspect_category"] not in tax[dom]:
        add(ERROR, "BAD_CATEGORY", i, "aspect_category", f"{r['aspect_category']!r} is not a {dom} category (see docs/aspect_taxonomy.md)")
    if r["sentiment"] == "":
        if needs_all:
            add(ERROR, "MISSING_SENTIMENT", i, "sentiment", "ASPECT_PRESENT row has no sentiment")
        elif status != "NEEDS_REVIEW":
            unfinished_rows.add(i)


def _iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def _check_messages(df, issues, err_rows, unfinished_rows, reference):
    def add(sev, code, i, field, detail):
        issues.append(_issue(sev, code, i + 1, df.at[i, "message_id"], field, detail))
        if sev == ERROR:
            err_rows.add(i)

    groups = defaultdict(list)
    for i in df.index:
        groups[df.at[i, "message_id"]].append(i)
    for m, rows in groups.items():
        first = rows[0]
        for i in rows[1:]:
            diff = [c for c in PREPARED if df.at[i, c] != df.at[first, c]]
            if diff:
                add(ERROR, "MESSAGE_ROWS_DISAGREE", i, diff[0], f"rows of one message differ in prepared columns {diff}")
        aspect_rows = [i for i in rows if df.at[i, "aspect_status"] == "ASPECT_PRESENT"]
        none_rows = [i for i in rows if df.at[i, "aspect_status"] in ("NO_ASPECT", "UNCLEAR_ASPECT")]
        if len(none_rows) > 1:
            for i in none_rows[1:]:
                add(ERROR, "MULTIPLE_NO_ASPECT", i, "aspect_status", "a message with no or an unclear aspect has exactly one row")
        if none_rows and aspect_rows:
            add(ERROR, "MIXED_NO_ASPECT_AND_ASPECT", none_rows[0], "aspect_status",
                f"message has both {df.at[none_rows[0], 'aspect_status']} and ASPECT_PRESENT rows")
        seen = {}
        spans = []
        for i in aspect_rows:
            s, e = df.at[i, "span_start"], df.at[i, "span_end"]
            if _is_int(s) and _is_int(e):
                key = (int(s), int(e))
                if key in seen:
                    j = seen[key]
                    same = all(df.at[i, c] == df.at[j, c] for c in ("aspect_category", "sentiment"))
                    add(ERROR, "DUPLICATE_ROW" if same else "CONFLICTING_DUPLICATE", i, "span_start",
                        f"same span {key[0]}:{key[1]} as row {j + 1}" + ("" if same else
                        f" with different labels ({df.at[j, 'aspect_category']}/{df.at[j, 'sentiment']} vs {df.at[i, 'aspect_category']}/{df.at[i, 'sentiment']})"))
                else:
                    seen[key] = i
                    for k2, j in spans:
                        if _iou(key, k2) >= IOU_MATCH and df.at[i, "aspect_category"] == df.at[j, "aspect_category"] != "":
                            add(WARNING, "OVERLAPPING_SAME_CATEGORY", i, "span_start",
                                f"span overlaps row {j + 1} (IoU {_iou(key, k2):.2f}) with the same category; possible duplicate")
                    spans.append((key, i))
        statuses = {df.at[i, "annotation_status"] for i in rows}
        if len(statuses) > 1 and statuses & COMPLETE:
            add(WARNING, "MIXED_STATUSES", first, "annotation_status", f"rows of one message have statuses {sorted(statuses)}")
    # chronology inside sequences (annotators must not reorder)
    for sid, g in df.groupby("sequence_id", sort=False):
        try:
            g2 = g.drop_duplicates("message_id")
            pos = g2.position.astype(int).tolist()
            ts = g2.timestamp_ms.astype("int64").tolist()
        except ValueError:
            add(ERROR, "BAD_PREPARED_VALUE", g.index[0], "position", f"{sid}: position or timestamp_ms is not a number")
            continue
        order = sorted(range(len(pos)), key=lambda k: pos[k])
        if [ts[k] for k in order] != sorted(ts):
            add(ERROR, "CHRONOLOGY", g.index[0], "timestamp_ms", f"{sid}: timestamps are not in the order of position")
    # reference: the prepared batch the annotator was given
    if reference is None:
        issues.append(_issue(WARNING, "REFERENCE_NOT_FOUND", None, None, None, "no reference batch matched; prepared columns and completeness were not checked against the original"))
        return []
    ref = reference.drop_duplicates("message_id").set_index("message_id")
    for m, rows in groups.items():
        if m not in ref.index:
            add(ERROR, "UNKNOWN_MESSAGE", rows[0], "message_id", "message_id is not in the reference batch")
            continue
        for c in PREPARED:
            if c == "message_id":
                continue                      # it is the key of the lookup
            if any(df.at[i, c] != ref.at[m, c] for i in rows):
                i = next(i for i in rows if df.at[i, c] != ref.at[m, c])
                add(ERROR, "PREPARED_COLUMN_CHANGED", i, c, f"{c} differs from the reference batch (expected {ref.at[m, c][:60]!r}, found {df.at[i, c][:60]!r})")
    missing = [m for m in ref.index if m not in groups]
    for m in missing:
        issues.append(_issue(WARNING, "MESSAGE_NOT_IN_FILE", None, m, None, "reference message has no row in the file"))
    return missing


def find_reference(df, root=ROOT, explicit=None):
    if explicit:
        return read_annotations(explicit)
    ids = set(df["message_id"]) if "message_id" in df.columns else set()
    for name in REFERENCE_FILES:
        p = os.path.join(root, "data", "annotations", name)
        if os.path.exists(p):
            ref = read_annotations(p)
            if ids & set(ref["message_id"]):
                return ref
    return None


# ───────────────────────────── statistics ─────────────────────────────
def statistics(df, result):
    """Counts over finished, error-free messages only; other messages are summarised by status."""
    usable = df[df.message_id.isin(result["complete_messages"])]
    aspect = usable[usable.aspect_status == "ASPECT_PRESENT"]
    out = {"rows": int(len(df)), "messages": int(df.message_id.nunique()),
           "messages_finished_and_valid": len(result["complete_messages"]),
           "messages_unfinished": len(result["unfinished_messages"]),
           "annotation_status_rows": dict(Counter(df.annotation_status)),
           "annotators": dict(Counter(df.annotator_id[df.annotator_id != ""])),
           "message_level_aspect_status": dict(Counter(
               "ASPECT_PRESENT" if (g.aspect_status == "ASPECT_PRESENT").any() else g.aspect_status.iloc[0]
               for _, g in usable.groupby("message_id"))),
           "aspect_rows": int(len(aspect)),
           "by_aspect_category": dict(Counter(aspect.aspect_category)),
           "by_sentiment": dict(Counter(aspect.sentiment)),
           "aspect_by_sentiment": {c: dict(Counter(g.sentiment)) for c, g in aspect.groupby("aspect_category")},
           "by_sequence_type": {}, "by_sequence": {}}
    for t, g in usable.groupby("sequence_type"):
        a = g[g.aspect_status == "ASPECT_PRESENT"]
        out["by_sequence_type"][t] = {"messages": int(g.message_id.nunique()), "aspect_rows": int(len(a)),
                                      "by_sentiment": dict(Counter(a.sentiment)), "by_aspect_category": dict(Counter(a.aspect_category))}
    for sid, g_all in df.groupby("sequence_id", sort=False):
        g = usable[usable.sequence_id == sid]
        a = g[g.aspect_status == "ASPECT_PRESENT"]
        out["by_sequence"][sid] = {
            "sequence_type": g_all.sequence_type.iloc[0], "entity_id": g_all.entity_id.iloc[0],
            "messages": int(g_all.message_id.nunique()), "messages_finished_and_valid": int(g.message_id.nunique()),
            "messages_with_aspect": int(a.message_id.nunique()),
            "messages_without_aspect": int(g[g.aspect_status == "NO_ASPECT"].message_id.nunique()),
            "messages_unclear": int(g[g.aspect_status == "UNCLEAR_ASPECT"].message_id.nunique()),
            "aspect_rows": int(len(a)), "by_sentiment": dict(Counter(a.sentiment)), "by_aspect_category": dict(Counter(a.aspect_category))}
    return out


# ───────────────────────────── agreement ─────────────────────────────
def cohens_kappa(a, b):
    """Cohen's kappa for two equally long label lists; None when it is undefined (no items, or chance agreement is 1)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in set(a) | set(b))
    return None if pe >= 1 else (po - pe) / (1 - pe)


def match_spans(spans_a, spans_b, threshold=IOU_MATCH):
    """One-to-one matching of (start, end, ...) spans by overlap/union, best pairs first. Returns [(index_a, index_b, iou)]."""
    pairs = sorted(((_iou(sa[:2], sb[:2]), ia, ib) for ia, sa in enumerate(spans_a) for ib, sb in enumerate(spans_b)
                    if _iou(sa[:2], sb[:2]) >= threshold), key=lambda p: (-p[0], p[1], p[2]))
    used_a, used_b, out = set(), set(), []
    for v, ia, ib in pairs:
        if ia not in used_a and ib not in used_b:
            used_a.add(ia); used_b.add(ib); out.append((ia, ib, v))
    return out


def _message_spans(g):
    return [(int(r.span_start), int(r.span_end), r.aspect_category, r.sentiment) for r in g.itertuples()
            if r.aspect_status == "ASPECT_PRESENT"]


def _message_status(g):
    return "ASPECT_PRESENT" if (g.aspect_status == "ASPECT_PRESENT").any() else g.aspect_status.iloc[0]


def _agreement_block(labels_a, labels_b):
    n = len(labels_a)
    return {"n": n, "raw_agreement": (sum(x == y for x, y in zip(labels_a, labels_b)) / n) if n else None,
            "cohens_kappa": cohens_kappa(labels_a, labels_b),
            "kappa_note": None if cohens_kappa(labels_a, labels_b) is not None else "undefined (no items, or both annotators used one label throughout)"}


def compare(df_a, res_a, df_b, res_b):
    """Agreement per (sequence_type, language) group on messages finished and error-free in BOTH files. Groups are never pooled."""
    ids = sorted(res_a["complete_messages"] & res_b["complete_messages"])
    fa, fb = df_a.drop_duplicates("message_id").set_index("message_id"), df_b.drop_duplicates("message_id").set_index("message_id")
    keyed = [c for c in PREPARED if c != "message_id"]
    changed = [m for m in ids if any(fa.at[m, c] != fb.at[m, c] for c in keyed)]
    ids = [m for m in ids if m not in changed]
    a = df_a[df_a.message_id.isin(ids)]
    b = df_b[df_b.message_id.isin(ids)]
    out = {"messages_in_a": int(df_a.message_id.nunique()), "messages_in_b": int(df_b.message_id.nunique()),
           "messages_compared": len(ids),
           "excluded_not_finished_or_invalid_in_a_or_b": int(len(set(df_a.message_id) | set(df_b.message_id)) - len(ids)),
           "prepared_columns_differ_between_files": changed,
           "only_in_a": sorted(set(df_a.message_id) - set(df_b.message_id)), "only_in_b": sorted(set(df_b.message_id) - set(df_a.message_id)),
           "annotators_a": sorted(set(df_a.annotator_id) - {""}), "annotators_b": sorted(set(df_b.annotator_id) - {""}),
           "groups": {}}
    out["warnings"] = []
    if set(out["annotators_a"]) & set(out["annotators_b"]):
        out["warnings"].append("the two files share an annotator_id; they may not be independent annotations")
    if changed:
        out["warnings"].append(f"{len(changed)} messages differ in prepared columns (text, timestamp, ...) between the files and were excluded")
    if not out["annotators_a"] or not out["annotators_b"]:
        out["warnings"].append("annotator_id is missing in one file")
    meta = a.drop_duplicates("message_id").set_index("message_id")[["sequence_type", "language"]]
    groups = defaultdict(list)
    for m in ids:
        groups[(meta.at[m, "sequence_type"], meta.at[m, "language"])].append(m)
    for (stype, lang), ms in sorted(groups.items()):
        st_a, st_b, cat_a, cat_b, sen_a, sen_b = [], [], [], [], [], []
        n_a = n_b = n_match = n_sent_missing = 0
        for m in ms:
            ga, gb = a[a.message_id == m], b[b.message_id == m]
            st_a.append(_message_status(ga)); st_b.append(_message_status(gb))
            sa, sb = _message_spans(ga), _message_spans(gb)
            n_a += len(sa); n_b += len(sb)
            for ia, ib, _ in match_spans(sa, sb):
                n_match += 1
                cat_a.append(sa[ia][2]); cat_b.append(sb[ib][2])
                if sa[ia][3] and sb[ib][3]:
                    sen_a.append(sa[ia][3]); sen_b.append(sb[ib][3])
                else:
                    n_sent_missing += 1
        out["groups"][f"{stype} / {lang}"] = {
            "messages": len(ms),
            "aspect_status_per_message": _agreement_block(st_a, st_b),
            "spans": {"in_a": n_a, "in_b": n_b, "matched_iou_ge_0.5": n_match,
                      "span_f1": (2 * n_match / (n_a + n_b)) if (n_a + n_b) else None,
                      "only_in_a": n_a - n_match, "only_in_b": n_b - n_match},
            "category_on_matched_spans": _agreement_block(cat_a, cat_b),
            "sentiment_on_matched_spans": {**_agreement_block(sen_a, sen_b), "matched_spans_without_sentiment_in_a_or_b": n_sent_missing}}
    return out


# ───────────────────────────── reporting ─────────────────────────────
def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def print_validation(name, df, result, stats, max_issues=60):
    errs = [i for i in result["issues"] if i["severity"] == ERROR]
    warns = [i for i in result["issues"] if i["severity"] == WARNING]
    print(f"\n== {name}: {result['rows']} rows, {stats['messages']} messages ==")
    print(f"errors: {len(errs)}   warnings: {len(warns)}   finished and valid messages: {stats['messages_finished_and_valid']}   unfinished: {stats['messages_unfinished']}")
    by_code = Counter(i["code"] for i in result["issues"])
    if by_code:
        print("issue codes: " + ", ".join(f"{c} x{n}" for c, n in sorted(by_code.items())))
    for i in (errs + warns)[:max_issues]:
        where = f"row {i['row']}" if i["row"] else "file"
        print(f"  {i['severity']:7s} {i['code']:28s} {where:8s} {i['message_id'] or '':22s} {i['field'] or '':18s} {i['detail']}")
    if len(errs) + len(warns) > max_issues:
        print(f"  ... {len(errs) + len(warns) - max_issues} more (use --json for the full list)")
    print(f"annotation_status rows: {stats['annotation_status_rows']}")
    if stats["messages_finished_and_valid"]:
        print(f"message-level aspect status: {stats['message_level_aspect_status']}")
        print(f"aspect rows: {stats['aspect_rows']}   by category: {stats['by_aspect_category']}")
        print(f"by sentiment: {stats['by_sentiment']}")
        for t, v in stats["by_sequence_type"].items():
            print(f"  {t}: {v['messages']} messages, {v['aspect_rows']} aspect rows, sentiment {v['by_sentiment']}")
        print("  per sequence (finished/valid messages of all):")
        for sid, v in stats["by_sequence"].items():
            print(f"    {sid} entity {v['entity_id']}: {v['messages_finished_and_valid']}/{v['messages']} messages, aspect rows {v['aspect_rows']}, "
                  f"without aspect {v['messages_without_aspect']}, unclear {v['messages_unclear']}, sentiment {v['by_sentiment']}")
    else:
        print("no finished, valid messages yet: no annotation statistics")


def print_agreement(ag):
    print(f"\n== agreement: {ag['messages_compared']} messages compared (a has {ag['messages_in_a']}, b has {ag['messages_in_b']}; "
          f"{ag['excluded_not_finished_or_invalid_in_a_or_b']} left out as unfinished, invalid or in only one file) ==")
    for w in ag["warnings"]:
        print("  WARNING:", w)
    if ag["only_in_a"] or ag["only_in_b"]:
        print(f"  messages only in a: {len(ag['only_in_a'])}, only in b: {len(ag['only_in_b'])}")
    if not ag["groups"]:
        print("  nothing to compare")
    for g, v in ag["groups"].items():
        s, sp, c, se = v["aspect_status_per_message"], v["spans"], v["category_on_matched_spans"], v["sentiment_on_matched_spans"]
        print(f"  [{g}] (not pooled with other groups) {v['messages']} messages")
        print(f"    aspect status per message : n={s['n']}  agreement {_fmt(s['raw_agreement'])}  kappa {_fmt(s['cohens_kappa'])}")
        print(f"    spans (IoU>=0.5)          : a={sp['in_a']} b={sp['in_b']} matched={sp['matched_iou_ge_0.5']}  span F1 {_fmt(sp['span_f1'])}")
        print(f"    category on matched spans : n={c['n']}  agreement {_fmt(c['raw_agreement'])}  kappa {_fmt(c['cohens_kappa'])}")
        print(f"    sentiment on matched spans: n={se['n']}  agreement {_fmt(se['raw_agreement'])}  kappa {_fmt(se['cohens_kappa'])}")


def run(path, taxonomies, reference_path=None, root=ROOT):
    """(dataframe, validation result, statistics); statistics is None when required columns are missing."""
    df = read_annotations(path)
    result = validate(df, taxonomies, find_reference(df, root, reference_path))
    broken = any(i["code"] == "BAD_HEADER" and "missing" in i["detail"] for i in result["issues"])
    return df, result, (None if broken else statistics(df, result))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="completed annotation CSV")
    ap.add_argument("--compare", metavar="OTHER.csv", help="second annotator's file for the same messages")
    ap.add_argument("--reference", help="the unannotated batch the file was made from (default: detected in data/annotations)")
    ap.add_argument("--strict", action="store_true", help="unfinished messages also make the exit code 1")
    ap.add_argument("--json", metavar="OUT.json", help="write the full report as JSON")
    a = ap.parse_args(argv)
    inputs = {os.path.abspath(p) for p in (a.file, a.compare, a.reference) if p}
    if a.json and os.path.abspath(a.json) in inputs:
        print("--json must not be one of the input files", file=sys.stderr)
        return 2
    tax = load_taxonomies()
    report, failed = {}, False
    runs = [("file", a.file)] + ([("compare", a.compare)] if a.compare else [])
    dfs = {}
    for key, path in runs:
        df, res, stats = run(path, tax, a.reference)
        if stats is None:
            print(f"{path}: invalid header: " + "; ".join(i["detail"] for i in res["issues"] if i["code"] == "BAD_HEADER"))
            return 1
        print_validation(path, df, res, stats)
        n_err = sum(i["severity"] == ERROR for i in res["issues"])
        failed |= n_err > 0 or (a.strict and bool(res["unfinished_messages"] or res["missing_messages"]))
        dfs[key] = (df, res)
        report[key] = {"path": path, "issues": res["issues"], "statistics": stats}
    if a.compare:
        ag = compare(dfs["file"][0], dfs["file"][1], dfs["compare"][0], dfs["compare"][1])
        print_agreement(ag)
        report["agreement"] = ag
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        print("\nwrote", a.json)
    print("\nRESULT:", "FAILED" if failed else "no errors")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
