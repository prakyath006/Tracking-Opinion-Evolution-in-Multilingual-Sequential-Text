"""
Tests for scripts/validate_aspect_annotations.py.

The CSV rows here are TEST FIXTURES built in memory or in a temporary directory so the validator can be exercised. They are not project
data, are not saved in the repository, and no project annotation file is read or written (except a read-only smoke test of the
unannotated pilot batch, which is skipped when it is absent).
"""
import hashlib
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import validate_aspect_annotations as va

TAX = va.load_taxonomies()
TEXT = "The scent is lovely but the cap leaks."          # scent = 4:9, cap = 28:31
BASE = {"sequence_id": "S1", "sequence_type": "PRODUCT_TIMELINE", "source": "amazon_all_beauty_2023", "entity_id": "P1",
        "message_id": "M1", "position": "1", "timestamp_ms": "1000", "timestamp_utc": "2020-01-01T00:00:00Z", "language": "en",
        "title": "t", "text": TEXT, "parent_message_id": "", "aspect_status": "", "aspect_span_text": "", "span_start": "",
        "span_end": "", "span_type": "", "aspect_category": "", "sentiment": "", "annotation_status": "NOT_STARTED",
        "annotator_id": "", "notes": ""}


def row(**kw):
    r = dict(BASE)
    r.update(kw)
    return r


def present(message="M1", start=4, end=9, cat="FRAGRANCE", sent="POSITIVE", status="ANNOTATED", annotator="a1", **kw):
    r = row(message_id=message, aspect_status="ASPECT_PRESENT", aspect_span_text=TEXT[start:end], span_start=str(start),
            span_end=str(end), span_type="EXPLICIT", aspect_category=cat, sentiment=sent, annotation_status=status,
            annotator_id=annotator)
    r.update(kw)
    return r


def none_row(message="M1", kind="NO_ASPECT", annotator="a1", **kw):
    r = row(message_id=message, aspect_status=kind, annotation_status="ANNOTATED", annotator_id=annotator)
    r.update(kw)
    return r


def frame(*rows):
    return pd.DataFrame(list(rows), columns=va.COLUMNS, dtype=str).fillna("")


def codes(df, reference=None, severity=None):
    res = va.validate(df, TAX, reference)
    return sorted({i["code"] for i in res["issues"] if severity is None or i["severity"] == severity})


def errors(df, reference=None):
    return codes(df, reference, va.ERROR)


# ───────────────────────── 1. labels ─────────────────────────
def test_valid_annotations_pass_cleanly():
    df = frame(present("M1"), present("M1", start=28, end=31, cat="PACKAGING", sent="NEGATIVE"),
               none_row("M2", position="2", timestamp_ms="2000"), none_row("M3", "UNCLEAR_ASPECT", position="3", timestamp_ms="3000"))
    res = va.validate(df, TAX)
    assert [i for i in res["issues"] if i["severity"] == va.ERROR] == []
    assert res["complete_messages"] == {"M1", "M2", "M3"} and res["unfinished_messages"] == set()


@pytest.mark.parametrize("field, value", [("sentiment", "GOOD"), ("sentiment", "UNKNOWN"), ("aspect_status", "PRESENT"),
                                          ("span_type", "BOTH"), ("annotation_status", "DONE"), ("sequence_type", "REVIEW")])
def test_labels_outside_the_schema_are_errors(field, value):
    r = present()
    r[field] = value
    assert "BAD_LABEL" in errors(frame(r))


def test_lowercase_and_padded_labels_are_reported_not_fixed():
    df = frame(present(sent="positive"))
    res = va.validate(df, TAX)
    assert any(i["code"] == "BAD_LABEL" and "POSITIVE" in i["detail"] for i in res["issues"])        # hint, no silent fix
    assert df.at[0, "sentiment"] == "positive"
    assert "WHITESPACE" in errors(frame(present(sent="POSITIVE ")))


def test_category_must_belong_to_the_rows_domain():
    assert errors(frame(present(cat="FRAGRANCE"))) == []
    assert "BAD_CATEGORY" in errors(frame(present(cat="music_bgm")))                                   # film category on a product review
    assert "BAD_CATEGORY" in errors(frame(present(cat="SCENT")))
    social = present(cat="music_bgm")
    social.update(source="dravidiancodemix_youtube", language="ta")
    assert errors(frame(social)) == []
    social["aspect_category"] = "FRAGRANCE"
    assert "BAD_CATEGORY" in errors(frame(social))
    assert errors(frame(present(cat="GENERAL"))) == [] and errors(frame(present(cat="OTHER_ASPECT"))) == []


def test_bad_header_and_unknown_source():
    df = frame(present())
    assert "BAD_HEADER" in codes(df.drop(columns=["sentiment"]))
    bad = present(); bad["source"] = "somewhere"
    assert "UNKNOWN_SOURCE" in errors(frame(bad))


# ───────────────────────── 2. spans ─────────────────────────
def test_span_must_match_the_text_at_the_offsets():
    assert errors(frame(present())) == []
    wrong_text = present(); wrong_text["aspect_span_text"] = "smell"
    assert "SPAN_MISMATCH" in errors(frame(wrong_text))
    shifted = present(); shifted["span_start"], shifted["span_end"] = "5", "10"
    assert "SPAN_MISMATCH" in errors(frame(shifted))
    res = va.validate(frame(shifted), TAX)
    assert any("'cent '" in i["detail"] for i in res["issues"])           # the message shows what the offsets actually select


@pytest.mark.parametrize("start, end", [("-1", "5"), ("9", "4"), ("4", "4"), ("4", "999"), ("4.0", "9"), ("abc", "9"), (" 4", "9")])
def test_span_offsets_must_be_valid_whole_numbers_inside_the_text(start, end):
    r = present(); r["span_start"], r["span_end"] = start, end
    assert {"SPAN_RANGE", "SPAN_NOT_INTEGER", "WHITESPACE"} & set(errors(frame(r)))


def test_aspect_present_needs_its_span_fields():
    r = present(); r["span_type"] = ""
    assert "MISSING_SPAN" in errors(frame(r))
    r = present(); r["aspect_span_text"] = ""
    assert "MISSING_SPAN" in errors(frame(r))


# ───────────────────────── 3. NO_ASPECT / UNCLEAR_ASPECT ─────────────────────────
@pytest.mark.parametrize("kind", ["NO_ASPECT", "UNCLEAR_ASPECT"])
def test_non_aspect_rows_may_not_carry_sentiment_span_or_category(kind):
    assert errors(frame(none_row(kind=kind))) == []
    assert "SENTIMENT_ON_NON_ASPECT" in errors(frame(none_row(kind=kind, sentiment="NEUTRAL")))
    assert "FIELDS_ON_NON_ASPECT" in errors(frame(none_row(kind=kind, aspect_category="FRAGRANCE")))
    assert "FIELDS_ON_NON_ASPECT" in errors(frame(none_row(kind=kind, aspect_span_text="scent", span_start="4", span_end="9", span_type="EXPLICIT")))


# ───────────────────────── 4. ASPECT_PRESENT completeness ─────────────────────────
def test_aspect_present_needs_category_and_sentiment_when_finished():
    r = present(); r["aspect_category"] = ""
    assert "MISSING_CATEGORY" in errors(frame(r))
    r = present(); r["sentiment"] = ""
    assert "MISSING_SENTIMENT" in errors(frame(r))
    for status in ("ANNOTATED", "ADJUDICATED"):
        r = present(status=status); r["sentiment"] = ""
        assert "MISSING_SENTIMENT" in errors(frame(r))


def test_needs_review_may_leave_the_sentiment_empty_but_should_explain():
    r = present(status="NEEDS_REVIEW", sent=""); r["notes"] = "sarcasm, cannot decide"
    res = va.validate(frame(r), TAX)
    assert [i for i in res["issues"] if i["severity"] == va.ERROR] == []
    assert res["unfinished_messages"] == {"M1"}                                  # not finished, not counted as complete
    r["notes"] = ""
    assert "NEEDS_REVIEW_NO_NOTES" in codes(frame(r), severity=va.WARNING)
    r2 = present(status="NEEDS_REVIEW", sent=""); r2["aspect_category"] = ""
    assert "MISSING_CATEGORY" in errors(frame(r2))                              # the aspect itself must still be identified


# ───────────────────────── 5. duplicates ─────────────────────────
def test_duplicate_and_conflicting_rows():
    assert "DUPLICATE_ROW" in errors(frame(present(), present()))
    assert "CONFLICTING_DUPLICATE" in errors(frame(present(), present(sent="NEGATIVE")))
    assert "CONFLICTING_DUPLICATE" in errors(frame(present(), present(cat="TEXTURE")))
    assert errors(frame(present(), present(start=28, end=31, cat="PACKAGING"))) == []             # different spans are fine
    assert "OVERLAPPING_SAME_CATEGORY" in codes(frame(present(start=4, end=9), present(start=4, end=8)), severity=va.WARNING)


def test_no_aspect_rows_are_exclusive_and_unique():
    assert "MULTIPLE_NO_ASPECT" in errors(frame(none_row(), none_row()))
    assert "MIXED_NO_ASPECT_AND_ASPECT" in errors(frame(none_row(), present()))
    assert "MIXED_NO_ASPECT_AND_ASPECT" in errors(frame(none_row(kind="UNCLEAR_ASPECT"), present()))


def test_rows_of_one_message_must_agree_on_prepared_columns():
    edited = present(start=28, end=31, cat="PACKAGING"); edited["text"] = TEXT + " "
    assert "MESSAGE_ROWS_DISAGREE" in errors(frame(present(), edited))


# ───────────────────────── 6. incomplete and annotation status ─────────────────────────
def test_unstarted_and_in_progress_rows_are_unfinished_not_errors():
    res = va.validate(frame(row(), row(message_id="M2", annotation_status="IN_PROGRESS", position="2", timestamp_ms="2000")), TAX)
    assert [i for i in res["issues"] if i["severity"] == va.ERROR] == []
    assert res["unfinished_messages"] == {"M1", "M2"} and res["complete_messages"] == set()


def test_status_must_match_content():
    assert "STATUS_CONTENT_MISMATCH" in errors(frame(present(status="NOT_STARTED")))
    blank_done = row(annotation_status="ANNOTATED", annotator_id="a1")
    assert "MISSING_ASPECT_STATUS" in errors(frame(blank_done))                  # annotated but nothing said: must be NO_ASPECT
    assert "MISSING_ANNOTATOR" in errors(frame(present(annotator="")))
    assert "MISSING_STATUS" in errors(frame(row(annotation_status="")))
    only_sentiment = row(sentiment="POSITIVE", annotation_status="IN_PROGRESS")
    assert "CONTENT_WITHOUT_ASPECT_STATUS" in errors(frame(only_sentiment))


def test_partly_filled_in_progress_row_is_unfinished_but_still_label_checked():
    r = present(status="IN_PROGRESS"); r["sentiment"] = ""
    res = va.validate(frame(r), TAX)
    assert [i for i in res["issues"] if i["severity"] == va.ERROR] == [] and res["unfinished_messages"] == {"M1"}
    r["aspect_category"] = "NOPE"
    assert "BAD_CATEGORY" in errors(frame(r))


def test_unfinished_message_with_an_error_is_not_counted_as_complete():
    df = frame(present("M1"), present("M2", sent="GOOD", position="2", timestamp_ms="2000"))
    res = va.validate(df, TAX)
    assert res["complete_messages"] == {"M1"}


# ───────────────────────── reference batch and chronology ─────────────────────────
def test_prepared_columns_must_match_the_reference_batch():
    ref = frame(row(), row(message_id="M2", position="2", timestamp_ms="2000"))
    ok = frame(present("M1"), none_row("M2", position="2", timestamp_ms="2000"))
    assert errors(ok, ref) == []
    edited = frame(present("M1"), none_row("M2", position="2", timestamp_ms="2000", text="changed text"))
    assert "PREPARED_COLUMN_CHANGED" in errors(edited, ref)
    retimed = frame(present("M1", timestamp_ms="999"), none_row("M2", position="2", timestamp_ms="2000"))
    assert "PREPARED_COLUMN_CHANGED" in errors(retimed, ref)
    assert "UNKNOWN_MESSAGE" in errors(frame(present("M9")), ref)
    res = va.validate(frame(present("M1")), TAX, ref)                       # M2 was never returned
    assert res["missing_messages"] == ["M2"] and "MESSAGE_NOT_IN_FILE" in codes(frame(present("M1")), ref, va.WARNING)


def test_chronology_must_follow_position():
    swapped = frame(present("M1", timestamp_ms="5000"), none_row("M2", position="2", timestamp_ms="2000"))
    assert "CHRONOLOGY" in errors(swapped)


# ───────────────────────── 7. statistics ─────────────────────────
def test_statistics_by_aspect_sentiment_and_sequence():
    r3 = none_row("M3", position="3", timestamp_ms="3000")
    other = present("M4", sequence_id="S2", entity_id="P2", cat="PRICE_VALUE", sent="NEUTRAL")
    other.update(position="1")
    unfinished = row(message_id="M5", sequence_id="S2", entity_id="P2", position="2", timestamp_ms="2000")
    df = frame(present("M1"), present("M1", start=28, end=31, cat="PACKAGING", sent="NEGATIVE"),
               present("M2", cat="FRAGRANCE", sent="MIXED", position="2", timestamp_ms="2000"), r3, other, unfinished)
    res = va.validate(df, TAX)
    st = va.statistics(df, res)
    assert st["by_aspect_category"] == {"FRAGRANCE": 2, "PACKAGING": 1, "PRICE_VALUE": 1}
    assert st["by_sentiment"] == {"POSITIVE": 1, "NEGATIVE": 1, "MIXED": 1, "NEUTRAL": 1}
    assert st["aspect_by_sentiment"]["FRAGRANCE"] == {"POSITIVE": 1, "MIXED": 1}
    assert st["message_level_aspect_status"] == {"ASPECT_PRESENT": 3, "NO_ASPECT": 1}
    assert st["messages_finished_and_valid"] == 4 and st["messages_unfinished"] == 1
    s1, s2 = st["by_sequence"]["S1"], st["by_sequence"]["S2"]
    assert (s1["messages"], s1["messages_finished_and_valid"], s1["messages_with_aspect"], s1["messages_without_aspect"], s1["aspect_rows"]) == (3, 3, 2, 1, 3)
    assert (s2["messages"], s2["messages_finished_and_valid"], s2["aspect_rows"], s2["entity_id"]) == (2, 1, 1, "P2")
    assert st["by_sequence_type"]["PRODUCT_TIMELINE"]["aspect_rows"] == 4
    assert st["annotation_status_rows"] == {"ANNOTATED": 5, "NOT_STARTED": 1}


def test_statistics_exclude_invalid_messages():
    df = frame(present("M1"), present("M2", sent="GOOD", position="2", timestamp_ms="2000"))
    st = va.statistics(df, va.validate(df, TAX))
    assert st["by_sentiment"] == {"POSITIVE": 1} and st["messages_finished_and_valid"] == 1


# ───────────────────────── 8. agreement ─────────────────────────
def test_cohens_kappa_known_values():
    assert va.cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert va.cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)          # po 0.75, pe 0.5
    assert va.cohens_kappa(["a", "b"], ["b", "a"]) == pytest.approx(-1.0)
    assert va.cohens_kappa(["a", "a", "a"], ["a", "a", "a"]) is None                  # chance agreement is 1: undefined
    assert va.cohens_kappa([], []) is None


def test_span_matching_by_overlap_over_union():
    assert va.match_spans([(0, 10)], [(0, 5)]) == [(0, 0, 0.5)]                       # exactly the threshold
    assert va.match_spans([(0, 10)], [(0, 4)]) == []
    assert va.match_spans([(0, 10)], [(10, 20)]) == []
    m = va.match_spans([(0, 10), (20, 30)], [(1, 10), (20, 30)])
    assert sorted((a, b) for a, b, _ in m) == [(0, 0), (1, 1)]
    one_to_one = va.match_spans([(0, 10)], [(0, 10), (0, 9)])
    assert len(one_to_one) == 1 and one_to_one[0][1] == 0                             # best partner, never two


def _pair(rows_a, rows_b):
    a, b = frame(*rows_a), frame(*rows_b)
    return a, va.validate(a, TAX), b, va.validate(b, TAX)


def test_agreement_is_computed_on_hand_checked_numbers():
    def msg(i, **kw):
        return {"message_id": f"M{i}", "position": str(i), "timestamp_ms": str(i * 1000), **kw}
    a_rows = [present(**msg(1)),                                                      # M1: scent POS
              present(**msg(2), sent="NEGATIVE"),                                     # M2: scent NEG
              none_row(**msg(3)),                                                     # M3: no aspect
              present(**msg(4), cat="TEXTURE")]                                       # M4: texture POS
    b_rows = [present(**msg(1), annotator="b1"),                                      # agrees
              present(**msg(2), sent="POSITIVE", annotator="b1"),                     # sentiment differs
              none_row(**msg(3), annotator="b1"),                                     # agrees
              present(**msg(4), cat="FRAGRANCE", annotator="b1")]                     # category differs
    ag = va.compare(*_pair(a_rows, b_rows))
    g = ag["groups"]["PRODUCT_TIMELINE / en"]
    assert ag["messages_compared"] == 4 and list(ag["groups"]) == ["PRODUCT_TIMELINE / en"]
    assert g["aspect_status_per_message"]["raw_agreement"] == pytest.approx(1.0)
    assert g["aspect_status_per_message"]["cohens_kappa"] == pytest.approx(1.0)
    assert (g["spans"]["in_a"], g["spans"]["in_b"], g["spans"]["matched_iou_ge_0.5"], g["spans"]["span_f1"]) == (3, 3, 3, 1.0)
    assert g["category_on_matched_spans"]["n"] == 3 and g["category_on_matched_spans"]["raw_agreement"] == pytest.approx(2 / 3)
    assert g["sentiment_on_matched_spans"]["n"] == 3 and g["sentiment_on_matched_spans"]["raw_agreement"] == pytest.approx(2 / 3)
    # sentiment labels: a = POS,NEG,POS ; b = POS,POS,POS -> po 2/3, pe = (2/3*1 + 1/3*0) = 2/3 -> kappa 0
    assert g["sentiment_on_matched_spans"]["cohens_kappa"] == pytest.approx(0.0)


def test_span_disagreement_lowers_span_f1():
    a_rows = [present("M1"), present("M1", start=28, end=31, cat="PACKAGING")]
    b_rows = [present("M1", annotator="b1")]
    g = va.compare(*_pair(a_rows, b_rows))["groups"]["PRODUCT_TIMELINE / en"]
    assert (g["spans"]["in_a"], g["spans"]["in_b"], g["spans"]["matched_iou_ge_0.5"], g["spans"]["only_in_a"]) == (2, 1, 1, 1)
    assert g["spans"]["span_f1"] == pytest.approx(2 / 3)


def test_groups_are_never_pooled():
    def other_group(msg):
        r = present(msg, annotator="a1"); r.update(sequence_id="S2", sequence_type="SAME_REVIEWER_SAME_PRODUCT", position="1")
        return r
    a_rows = [present("M1"), other_group("M2")]
    b_rows = [present("M1", annotator="b1"), {**other_group("M2"), "annotator_id": "b1"}]
    ag = va.compare(*_pair(a_rows, b_rows))
    assert sorted(ag["groups"]) == ["PRODUCT_TIMELINE / en", "SAME_REVIEWER_SAME_PRODUCT / en"]
    assert "overall" not in ag and "pooled" not in json.dumps(ag).lower().replace("never pooled", "")


def test_agreement_leaves_out_unfinished_invalid_and_one_sided_messages_and_warns_about_one_annotator():
    a_rows = [present("M1"), present("M2", sent="GOOD", position="2", timestamp_ms="2000"), present("M3", position="3", timestamp_ms="3000")]
    b_rows = [present("M1"), present("M2", position="2", timestamp_ms="2000"), row(message_id="M4", position="4", timestamp_ms="4000")]
    ag = va.compare(*_pair(a_rows, b_rows))
    assert ag["messages_compared"] == 1                                               # only M1 is finished and valid in both
    assert ag["only_in_a"] == ["M3"] and ag["only_in_b"] == ["M4"]
    assert any("share an annotator_id" in w for w in ag["warnings"])
    differing = [present("M1", annotator="b1")]
    differing[0]["title"] = "other title"
    ag = va.compare(*_pair([present("M1")], differing))
    assert ag["messages_compared"] == 0 and ag["prepared_columns_differ_between_files"] == ["M1"]


def test_sentiment_kappa_skips_matched_spans_without_a_sentiment():
    a_rows = [present("M1", status="NEEDS_REVIEW", sent="", notes="unsure")]
    b_rows = [present("M1", annotator="b1")]
    ag = va.compare(*_pair(a_rows, b_rows))
    assert ag["messages_compared"] == 0                                               # a NEEDS_REVIEW message is not finished: excluded


# ───────────────────────── command line ─────────────────────────
def _write(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_cli_exit_codes_and_input_files_are_never_modified(tmp_path, capsys):
    good = _write(frame(present("M1"), none_row("M2", position="2", timestamp_ms="2000")), tmp_path / "good.csv")
    bad = _write(frame(present(sent="GOOD")), tmp_path / "bad.csv")
    unfinished = _write(frame(row()), tmp_path / "todo.csv")
    before = {p: _sha(p) for p in (good, bad, unfinished)}
    assert va.main([good]) == 0
    assert va.main([bad]) == 1
    assert va.main([unfinished]) == 0 and va.main([unfinished, "--strict"]) == 1
    out = capsys.readouterr().out
    assert "BAD_LABEL" in out and "RESULT: no errors" in out and "FAILED" in out
    assert {p: _sha(p) for p in before} == before


def test_cli_compare_json_and_refusal_to_overwrite_an_input(tmp_path, capsys):
    a = _write(frame(present("M1")), tmp_path / "a.csv")
    b = _write(frame(present("M1", annotator="b1")), tmp_path / "b.csv")
    out = tmp_path / "report.json"
    assert va.main([a, "--compare", b, "--json", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["agreement"]["messages_compared"] == 1 and rep["file"]["statistics"]["aspect_rows"] == 1
    before = _sha(a)
    assert va.main([a, "--json", a]) == 2 and _sha(a) == before                          # refuses to overwrite an input
    capsys.readouterr()


def test_cli_missing_columns_is_reported(tmp_path, capsys):
    p = _write(frame(present()).drop(columns=["span_start"]), tmp_path / "x.csv")
    assert va.main([p]) == 1
    assert "invalid header" in capsys.readouterr().out


def test_unannotated_pilot_batch_validates_with_no_errors_and_nothing_finished(capsys):
    path = os.path.join(ROOT, "data", "annotations", "pilot_batch_01_product_timelines.csv")
    if not os.path.exists(path):
        pytest.skip("pilot batch not present")
    before = _sha(path)
    df, res, st = va.run(path, TAX)
    assert [i for i in res["issues"] if i["severity"] == va.ERROR] == []
    assert st["messages"] == 50 and st["messages_finished_and_valid"] == 0 and st["aspect_rows"] == 0
    assert va.main([path, "--strict"]) == 1
    assert _sha(path) == before
    capsys.readouterr()
