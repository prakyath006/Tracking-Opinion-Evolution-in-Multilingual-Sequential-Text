"""
Tests for scripts/prepare_adjudication_package.py.

SYNTHETIC FIXTURES ONLY: two invented annotators label a handful of invented messages in a temporary folder. No project annotation file is read or changed.
"""
import hashlib
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import prepare_adjudication_package as pap
import validate_aspect_annotations as va
from test_aspect_data import COLUMNS, T1, T2, T3, make_rows

T6, T7 = "Great price but not worth it.", "Love it."
TS = {f"M{i}": 1_600_000_000_000 + i * 86_400_000 for i in range(1, 9)}


def rows(who):
    """Annotator A or B on the same eight messages; the differences are the point."""
    a = who == "A"
    r = []

    def add(mid, pos, text, mentions=(), **kw):
        r.extend(make_rows("SEQ-1", "P1", mid, pos, TS[mid], text, mentions, annotator=f"annotator_{who}", **kw))
    add("M1", 1, T1, [("scent", "FRAGRANCE", "POSITIVE"), ("cap", "PACKAGING" if a else "OTHER_ASPECT", "NEGATIVE")])             # category disagreement
    add("M2", 2, T2, [("Smells", "FRAGRANCE", "MIXED" if a else "NEGATIVE")])                                                         # sentiment disagreement
    add("M3", 3, "It comes in a bottle.", [("bottle" if a else "a bottle", "PACKAGING", "NEUTRAL")])                                  # matched, IoU 0.75
    if a:
        add("M4", 4, T3, aspect_status="NO_ASPECT")                                                                                   # message status disagreement
    else:
        add("M4", 4, T3, [("wife", "OTHER_ASPECT", "NEUTRAL")])
    add("M5", 5, "Very strong.", [("Very strong" if a else "strong", "FRAGRANCE", "NEGATIVE")])                                       # matched, IoU 0.55
    if a:
        add("M6", 6, T6, [("Great price", "PRICE_VALUE", "POSITIVE"), ("not worth it", "PRICE_VALUE", "NEGATIVE", 0, "IMPLICIT")])  # partial overlap + A-only
    else:
        add("M6", 6, T6, [("price", "PRICE_VALUE", "POSITIVE")])
    if a:
        add("M7", 7, T7, [("it", "GENERAL", "POSITIVE")])                                                                             # pronoun, partial overlap
    else:
        add("M7", 7, T7, [("Love it", "GENERAL", "POSITIVE", 0, "IMPLICIT")])
    r.extend(make_rows("SEQ-1", "P1", "M8", 8, TS["M8"], T1, [("scent", "FRAGRANCE", "POSITIVE")], annotator=f"annotator_{who}"))
    cap = "NEGATIVE" if not a else ""
    r.extend(make_rows("SEQ-1", "P1", "M8", 8, TS["M8"], T1, [("cap", "PACKAGING", cap)], annotator=f"annotator_{who}",
                       status="ANNOTATED" if not a else "NEEDS_REVIEW"))                                                           # A is unsure here
    return r


def write(rs, path):
    pd.DataFrame(rs, columns=COLUMNS, dtype=str).fillna("").to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    d = tmp_path_factory.mktemp("adj")
    a, b = write(rows("A"), d / "A.csv"), write(rows("B"), d / "B.csv")
    before = (sha(a), sha(b))
    out = str(d / "package")
    pap.main(["--a", a, "--b", b, "--out_dir", out])
    return {"a": a, "b": b, "before": before, "out": out, "items": pd.read_csv(os.path.join(out, "adjudication_items.csv"), encoding="utf-8-sig", dtype=str, keep_default_na=False),
            "dir": d}


def test_every_kind_of_difference_becomes_one_item(pkg):
    it = pkg["items"]
    c = it.item_type.value_counts().to_dict()
    assert c == {"UNMATCHED_SPAN_A_ONLY": 3, "UNMATCHED_SPAN_B_ONLY": 3, "CATEGORY_DISAGREEMENT": 1, "SENTIMENT_DISAGREEMENT": 1, "NEEDS_REVIEW": 1,
                 "MESSAGE_STATUS_DISAGREEMENT": 1}
    cat = it[it.item_type == "CATEGORY_DISAGREEMENT"].iloc[0]
    assert (cat.message_id, cat.A_category, cat.B_category, cat.A_span_text, cat.B_span_text) == ("M1", "PACKAGING", "OTHER_ASPECT", "cap", "cap")
    sen = it[it.item_type == "SENTIMENT_DISAGREEMENT"].iloc[0]
    assert (sen.message_id, sen.A_sentiment, sen.B_sentiment) == ("M2", "MIXED", "NEGATIVE")
    nr = it[it.item_type == "NEEDS_REVIEW"].iloc[0]
    assert (nr.message_id, nr.A_annotation_status, nr.A_sentiment, nr.B_sentiment, nr.in_agreement_statistics) == ("M8", "NEEDS_REVIEW", "", "NEGATIVE", "no")
    ms = it[it.item_type == "MESSAGE_STATUS_DISAGREEMENT"].iloc[0]
    assert ms.message_id == "M4" and "NO_ASPECT" in ms.match_kind and ms.B_span_text == "wife"
    wife = it[(it.message_id == "M4") & (it.item_type == "UNMATCHED_SPAN_B_ONLY")]
    assert len(wife) == 1 and wife.B_span_text.iloc[0] == "wife"                                                                         # also listed as an unmatched span
    assert set(it[it.in_agreement_statistics == "no"].message_id) == {"M8"}                                                           # the unfinished message is flagged


def test_matched_spans_with_the_same_labels_create_no_item(pkg):
    it = pkg["items"]
    assert not set(it.message_id) & {"M3", "M5"}                                                                                      # matched, only the boundary differs slightly


def test_the_counts_agree_with_the_validators_compare(pkg):
    A, B = pkg["items"], None
    a, b = pap.read(pkg["a"]), pap.read(pkg["b"])
    tax = va.load_taxonomies()
    ra, rb = va.validate(a, tax), va.validate(b, tax)
    g = va.compare(a, ra, b, rb)["groups"]["PRODUCT_TIMELINE / en"]["spans"]
    it = pkg["items"]
    inside = it[it.in_agreement_statistics == "yes"]
    assert g["only_in_a"] == (inside.item_type == "UNMATCHED_SPAN_A_ONLY").sum() and g["only_in_b"] == (inside.item_type == "UNMATCHED_SPAN_B_ONLY").sum()


def test_mirrored_partial_overlaps_share_a_conflict_group_and_unmatched_spans_show_the_nearest_other_span(pkg):
    it = pkg["items"]
    price = it[(it.message_id == "M6") & it.item_type.str.startswith("UNMATCHED")]
    a_great = price[price.A_span_text == "Great price"].iloc[0]
    b_price = price[price.B_span_text == "price"].iloc[0]
    assert a_great.conflict_group == b_price.conflict_group != ""
    assert a_great.B_span_text == "price" and a_great.match_kind.startswith("partial overlap") and "nearest overlapping" in a_great.other_side_shows
    lone = price[price.A_span_text == "not worth it"].iloc[0]
    assert lone.B_span_text == "" and lone.match_kind.startswith("no overlapping") and lone.conflict_group not in ("", a_great.conflict_group)
    assert lone.A_span_type == "IMPLICIT"


def test_guideline_topic_tags_come_from_observable_fields(pkg):
    it = pkg["items"].set_index("item_id")
    t = lambda mid, typ: pkg["items"][(pkg["items"].message_id == mid) & (pkg["items"].item_type.str.startswith(typ))].guideline_topics.tolist()    # noqa: E731
    assert all("OTHER_ASPECT" in x for x in t("M1", "CATEGORY"))
    part = pkg["items"][(pkg["items"].message_id == "M6") & pkg["items"].match_kind.str.startswith("partial")]
    assert len(part) == 2 and all("span_boundary" in x for x in part.guideline_topics)
    lone = pkg["items"][(pkg["items"].message_id == "M6") & pkg["items"].match_kind.str.startswith("no overlapping")]
    assert len(lone) == 1 and "aspect_present_in_one_annotation_only" in lone.guideline_topics.iloc[0] and "span_boundary" not in lone.guideline_topics.iloc[0]
    pron = pkg["items"][(pkg["items"].message_id == "M7")].guideline_topics.tolist()
    assert pron and all("referring_expression_or_pronoun" in x and "GENERAL" in x and "implicit_aspect" in x for x in pron)
    assert "message_level_aspect_status" in t("M4", "MESSAGE")[0] and "NEEDS_REVIEW" in t("M8", "NEEDS")[0]


def test_text_timestamps_and_offsets_are_copied_exactly_and_nothing_is_filled_in(pkg):
    it = pkg["items"]
    for r in it.itertuples():
        assert r.timestamp_ms == str(TS[r.message_id]) and r.review_text and r.product_id == "P1"
        for s in "AB":
            t, st, en = getattr(r, f"{s}_span_text"), getattr(r, f"{s}_start"), getattr(r, f"{s}_end")
            if st != "" and r.item_type != "MESSAGE_STATUS_DISAGREEMENT":
                assert r.review_text[int(st):int(en)] == t
    blank = [c for c in it.columns if c.startswith("adjudicated_") or c in ("decision_reason", "adjudicator_id")]
    assert len(blank) == 7 and (it[blank] == "").all().all()
    assert it.item_id.is_unique and list(it.columns) == pap.COLUMNS


def test_both_input_files_are_untouched_and_the_package_has_all_parts(pkg):
    assert (sha(pkg["a"]), sha(pkg["b"])) == pkg["before"]
    for f in ("adjudication_items.csv", "adjudication_items.xlsx", "README.md", "summary.json"):
        assert os.path.exists(os.path.join(pkg["out"], f))
    summ = json.load(open(os.path.join(pkg["out"], "summary.json")))
    assert summ["items"] == len(pkg["items"]) == 10 and summ["adjudication_columns_blank"] is True and summ["messages_in_agreement_statistics"] == 7
    x = pd.read_excel(os.path.join(pkg["out"], "adjudication_items.xlsx"), dtype=str, keep_default_na=False)
    assert x.shape == pkg["items"].shape and x.item_id.tolist() == pkg["items"].item_id.tolist()


def test_an_existing_package_is_never_overwritten_and_mismatched_batches_are_refused(pkg):
    with pytest.raises(FileExistsError):
        pap.main(["--a", pkg["a"], "--b", pkg["b"], "--out_dir", pkg["out"]])
    other = rows("B")
    other[0]["text"] = other[0]["text"] + " edited"
    b2 = write(other, pkg["dir"] / "B2.csv")
    with pytest.raises(ValueError, match="not annotations of the same batch"):
        pap.main(["--a", pkg["a"], "--b", b2, "--out_dir", str(pkg["dir"] / "package2")])
    assert not os.path.exists(pkg["dir"] / "package2")
