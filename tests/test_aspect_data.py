"""
Tests for src/aspect_data.py.

Every annotated row and sequence here is a SYNTHETIC FIXTURE written to a temporary folder or built in memory. None is project data and
none is a label for the real 50-review pilot. The only real file touched is the unannotated pilot batch, read-only, in one smoke test.
"""
import hashlib
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import aspect_data as ad
from ontology import SentimentState

COLUMNS = ["sequence_id", "sequence_type", "source", "entity_id", "message_id", "position", "timestamp_ms", "timestamp_utc", "language",
           "title", "text", "parent_message_id", "aspect_status", "aspect_span_text", "span_start", "span_end", "span_type",
           "aspect_category", "sentiment", "annotation_status", "annotator_id", "notes"]


# ───────────────────────── fixtures ─────────────────────────
def fake_tokenizer():
    """A small offline WordPiece fast tokenizer (lowercasing, [CLS] ... [SEP]) so alignment logic is tested without downloads."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    words = ("[PAD] [UNK] [CLS] [SEP] . , the scent is lovely but cap leaks smells great at first then turns sour price worth every penny "
             "not bad it comes in a bottle bought for my wife un ##scent ##ed very strong love this product and sized").split()
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                                       special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")


@pytest.fixture(scope="module")
def tok():
    return fake_tokenizer()


def make_rows(seq, entity, mid, pos, ts, text, mentions=(), status="ANNOTATED", aspect_status=None, stype="PRODUCT_TIMELINE",
              annotator="a1", source="amazon_all_beauty_2023", language="en"):
    """One CSV row per mention; mentions = (span text, category, sentiment[, occurrence index, span_type])."""
    base = {"sequence_id": seq, "sequence_type": stype, "source": source, "entity_id": entity, "message_id": mid, "position": str(pos),
            "timestamp_ms": str(ts), "timestamp_utc": "2020-01-01T00:00:00Z", "language": language, "title": "t", "text": text,
            "parent_message_id": "", "aspect_status": "", "aspect_span_text": "", "span_start": "", "span_end": "", "span_type": "",
            "aspect_category": "", "sentiment": "", "annotation_status": status, "annotator_id": annotator if status != "NOT_STARTED" else "",
            "notes": ""}
    if not mentions:
        return [{**base, "aspect_status": aspect_status or ("" if status == "NOT_STARTED" else "NO_ASPECT")}]
    rows = []
    for m in mentions:
        span, cat, sent = m[:3]
        occ = m[3] if len(m) > 3 else 0
        kind = m[4] if len(m) > 4 else "EXPLICIT"
        start = -1
        for _ in range(occ + 1):
            start = text.index(span, start + 1)
        rows.append({**base, "aspect_status": "ASPECT_PRESENT", "aspect_span_text": span, "span_start": str(start),
                     "span_end": str(start + len(span)), "span_type": kind, "aspect_category": cat, "sentiment": sent})
    return rows


T1 = "The scent is lovely but the cap leaks."
T2 = "Smells great at first, then turns sour."
T3 = "Bought for my wife."
T4 = "It comes in a bottle."
T5 = "Very strong."
T6 = "Love this product."
TB1 = "The price is great but not worth it."
TB2 = "The bottle is great and it comes in a bottle."


def fixture_rows():
    r = []
    r += make_rows("SEQ-A", "P1", "A1", 1, 1000, T1, [("scent", "FRAGRANCE", "POSITIVE"), ("cap", "PACKAGING", "NEGATIVE")])
    r += make_rows("SEQ-A", "P1", "A2", 2, 2000, T2, [("Smells", "FRAGRANCE", "MIXED")])
    r += make_rows("SEQ-A", "P1", "A3", 3, 3000, T3)                                              # NO_ASPECT
    r += make_rows("SEQ-A", "P1", "A4", 4, 4000, T4, [("bottle", "PACKAGING", "NEUTRAL")])
    r += make_rows("SEQ-A", "P1", "A5", 5, 5000, T5, aspect_status="UNCLEAR_ASPECT")
    r += make_rows("SEQ-A", "P1", "A6", 6, 6000, T6, status="NOT_STARTED")                       # unfinished
    r += make_rows("SEQ-B", "P2", "B1", 1, 1500, TB1, [("price", "PRICE_VALUE", "POSITIVE"),
                                                         ("not worth it", "PRICE_VALUE", "NEGATIVE", 0, "IMPLICIT")])
    r += make_rows("SEQ-B", "P2", "B2", 2, 2500, TB2, [("bottle", "PACKAGING", "POSITIVE", 0), ("bottle", "PACKAGING", "NEUTRAL", 1)])
    return r


def write_csv(rows, path):
    pd.DataFrame(rows, columns=COLUMNS, dtype=str).fillna("").to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


@pytest.fixture()
def loaded(tmp_path):
    path = write_csv(fixture_rows(), tmp_path / "ann.csv")
    return path, ad.load_annotations(path)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ───────────────────────── reading and validation ─────────────────────────
def test_loads_only_finished_valid_messages_and_never_treats_unfinished_as_no_aspect(loaded):
    path, ann = loaded
    ids = [m.message_id for m in ann.messages]
    assert sorted(ids) == ["A1", "A2", "A3", "A4", "A5", "B1", "B2"]
    assert ann.unfinished_message_ids == ["A6"] and ann.skipped_invalid_message_ids == []
    status = {m.message_id: m.status for m in ann.messages}
    assert status["A3"] == "NO_ASPECT" and status["A5"] == "UNCLEAR_ASPECT" and status["A1"] == "ASPECT_PRESENT"
    assert ann.domains == ["product"]


def test_mentions_carry_offsets_category_sentiment_and_span_type(loaded):
    _, ann = loaded
    a1 = next(m for m in ann.messages if m.message_id == "A1")
    assert [(x.start, x.end, x.text, x.category, x.sentiment) for x in a1.mentions] == \
        [(4, 9, "scent", "FRAGRANCE", "POSITIVE"), (28, 31, "cap", "PACKAGING", "NEGATIVE")]
    assert all(T1[x.start:x.end] == x.text for x in a1.mentions)
    b1 = next(m for m in ann.messages if m.message_id == "B1")
    assert [x.span_type for x in b1.mentions] == ["EXPLICIT", "IMPLICIT"]


def test_original_file_is_never_modified(tmp_path):
    path = write_csv(fixture_rows(), tmp_path / "ann.csv")
    before, mtime = sha(path), os.path.getmtime(path)
    ad.load_annotations(path)
    assert sha(path) == before and os.path.getmtime(path) == mtime


def test_invalid_annotations_raise_or_are_skipped_per_message(tmp_path):
    rows = fixture_rows()
    rows[0]["sentiment"] = "GOOD"                                                   # A1 now has an invalid label
    path = write_csv(rows, tmp_path / "bad.csv")
    with pytest.raises(ad.AnnotationError) as e:
        ad.load_annotations(path)
    assert any(i["code"] == "BAD_LABEL" for i in e.value.issues)
    ann = ad.load_annotations(path, on_invalid="skip")
    assert ann.skipped_invalid_message_ids == ["A1"]
    assert "A1" not in [m.message_id for m in ann.messages] and len(ann.messages) == 6
    with pytest.raises(ValueError):
        ad.load_annotations(path, on_invalid="ignore")


@pytest.mark.parametrize("edit", [
    lambda r: r[0].update(span_start="5", span_end="10"),                           # offsets no longer match the span text
    lambda r: r[0].update(aspect_category="music_bgm"),                             # film category on a product review
    lambda r: r[3].update(sentiment="NEUTRAL"),                                     # sentiment on A3, a NO_ASPECT row
])
def test_offset_category_and_status_rules_come_from_the_validator(tmp_path, edit):
    rows = fixture_rows()
    assert rows[3]["aspect_status"] == "NO_ASPECT"
    edit(rows)
    with pytest.raises(ad.AnnotationError):
        ad.load_annotations(write_csv(rows, tmp_path / "x.csv"))


def test_missing_columns_raise(tmp_path):
    df = pd.DataFrame(fixture_rows(), columns=COLUMNS, dtype=str).drop(columns=["span_start"])
    p = tmp_path / "h.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ad.AnnotationError, match="invalid header"):
        ad.load_annotations(str(p))


def test_unannotated_pilot_batch_yields_no_data_and_is_not_modified():
    path = os.path.join(ROOT, "data", "annotations", "pilot_batch_01_product_timelines.csv")
    if not os.path.exists(path):
        pytest.skip("pilot batch not present")
    before = sha(path)
    ann = ad.load_annotations(path)
    assert ann.messages == [] and len(ann.unfinished_message_ids) == 50          # nothing annotated, nothing invented
    assert sha(path) == before


def test_category_lists_follow_the_two_taxonomies_and_social_rows_load(tmp_path):
    prod, soc = ad.category_ids("product"), ad.category_ids("social")
    assert prod[0] == "EFFECTIVENESS" and prod[-2:] == ["GENERAL", "OTHER_ASPECT"] and len(prod) == len(set(prod)) == 14
    assert soc[:2] == ["fan_stardom", "trailer_teaser"] and soc[-2:] == ["GENERAL", "OTHER_ASPECT"] and len(soc) == 9
    with pytest.raises(ValueError):
        ad.category_ids("music")
    text = "Trailer super bgm mass"
    rows = make_rows("SEQ-S", "", "S1", 1, 1000, text, [("Trailer", "trailer_teaser", "POSITIVE"), ("bgm", "music_bgm", "POSITIVE")],
                     stype="COMMENT_LEVEL", source="dravidiancodemix_youtube", language="ta")
    ann = ad.load_annotations(write_csv(rows, tmp_path / "social.csv"))
    assert ann.domains == ["social"] and [x.category for x in ann.messages[0].mentions] == ["trailer_teaser", "music_bgm"]
    assert [e.category for e in ad.build_sentiment_examples(ann.messages)] == ["trailer_teaser", "music_bgm"]
    with pytest.raises(ValueError, match="entity_id"):                      # comments have no product: no leakage-safe split exists
        ad.split_by_entity(ann.messages)


# ───────────────────────── NEUTRAL stays out of the ontology ─────────────────────────
def test_neutral_is_separate_from_the_ontology():
    assert ad.AspectSentiment.label_names() == ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]
    assert "NEUTRAL" not in SentimentState.label_names() and len(SentimentState) == 4
    assert ad.AspectSentiment.NEUTRAL.to_ontology_name() is None
    for name in ("POSITIVE", "NEGATIVE", "MIXED"):
        assert ad.AspectSentiment[name].to_ontology_name() == SentimentState[name].name
    assert ad.AspectSentiment.MIXED.value != SentimentState.MIXED.value             # own ids: convert by name, never by id
    with pytest.raises(ValueError):
        ad.AspectSentiment.from_label("UNKNOWN")


# ───────────────────────── BIO labels ─────────────────────────
def tags_of(enc):
    return enc.tags


def test_bio_for_multiple_aspects_in_one_review(tok):
    m = [ad.Mention(4, 9, "scent", "FRAGRANCE", "POSITIVE", "EXPLICIT"), ad.Mention(28, 31, "cap", "PACKAGING", "NEGATIVE", "EXPLICIT")]
    enc = ad.bio_encode(T1, m, tok)
    # [CLS] the scent is lovely but the cap leaks . [SEP]
    assert enc.tags == [None, "O", "B-ASP", "O", "O", "O", "O", "B-ASP", "O", "O", None]
    assert enc.labels == [-100, 0, 1, 0, 0, 0, 0, 1, 0, 0, -100]
    assert enc.tagset == ["O", "B-ASP", "I-ASP"] and enc.truncated_mentions == [] and enc.partial_boundary_mentions == []


def test_multiword_span_is_b_then_i(tok):
    enc = ad.bio_encode(TB1, [ad.Mention(23, 35, "not worth it", "PRICE_VALUE", "NEGATIVE", "IMPLICIT")], tok)
    assert [t for t in enc.tags if t and t != "O"] == ["B-ASP", "I-ASP", "I-ASP"]


def test_subwords_of_one_word_are_b_then_i(tok):
    text = "It is unscented."
    enc = ad.bio_encode(text, [ad.Mention(6, 15, "unscented", "FRAGRANCE", "NEUTRAL", "EXPLICIT")], tok)
    toks = tok.convert_ids_to_tokens(enc.input_ids)
    assert toks == ["[CLS]", "it", "is", "un", "##scent", "##ed", ".", "[SEP]"]
    assert enc.tags == [None, "O", "O", "B-ASP", "I-ASP", "I-ASP", "O", None] and enc.partial_boundary_mentions == []


def test_span_ending_inside_a_token_is_reported_not_hidden(tok):
    enc = ad.bio_encode("It is unscented.", [ad.Mention(7, 12, "nscen", "FRAGRANCE", "NEUTRAL", "EXPLICIT")], tok)
    assert [t for t in enc.tags if t not in (None, "O")] == ["B-ASP", "I-ASP"]       # tokens it overlaps are labelled
    assert len(enc.partial_boundary_mentions) == 1


def test_category_specific_tags(tok):
    cats = ad.category_ids("product")
    m = [ad.Mention(4, 9, "scent", "FRAGRANCE", "POSITIVE", "EXPLICIT"), ad.Mention(28, 31, "cap", "PACKAGING", "NEGATIVE", "EXPLICIT")]
    enc = ad.bio_encode(T1, m, tok, categories=cats)
    assert enc.tags[2] == "B-FRAGRANCE" and enc.tags[7] == "B-PACKAGING"
    assert enc.tagset[0] == "O" and enc.tagset[enc.labels[2]] == "B-FRAGRANCE" and len(enc.tagset) == 1 + 2 * len(cats)
    with pytest.raises(ValueError, match="not in the category list"):
        ad.bio_encode(T1, m, tok, categories=["FRAGRANCE"])


def test_adjacent_spans_get_two_b_tags_and_overlaps_follow_the_policy(tok):
    adj = [ad.Mention(0, 3, "The", "OTHER_ASPECT", "NEUTRAL", "EXPLICIT"), ad.Mention(4, 9, "scent", "FRAGRANCE", "POSITIVE", "EXPLICIT")]
    assert ad.bio_encode(T1, adj, tok).tags[1:3] == ["B-ASP", "B-ASP"]
    over = [ad.Mention(0, 9, "The scent", "FRAGRANCE", "POSITIVE", "EXPLICIT"), ad.Mention(4, 9, "scent", "FRAGRANCE", "POSITIVE", "EXPLICIT")]
    with pytest.raises(ad.SpanOverlapError):
        ad.bio_encode(T1, over, tok)
    assert ad.bio_encode(T1, over, tok, overlap="merge").tags[1:3] == ["B-ASP", "I-ASP"]
    with pytest.raises(ValueError):
        ad.bio_encode(T1, over, tok, categories=["FRAGRANCE"], overlap="merge")


def test_truncation_drops_cut_off_spans_and_says_so(tok):
    m = [ad.Mention(4, 9, "scent", "FRAGRANCE", "POSITIVE", "EXPLICIT"), ad.Mention(28, 31, "cap", "PACKAGING", "NEGATIVE", "EXPLICIT")]
    enc = ad.bio_encode(T1, m, tok, max_length=5)                                   # [CLS] the scent is [SEP]
    assert len(enc.input_ids) == 5 and enc.tags == [None, "O", "B-ASP", "O", None]
    assert [x.text for x in enc.truncated_mentions] == ["cap"]


def test_extraction_examples_handle_missing_unclear_and_unfinished_messages(tok, loaded):
    _, ann = loaded
    ex, rep = ad.build_extraction_examples(ann.messages, tok)
    ids = [e.message_id for e in ex]
    assert "A5" not in ids and "A6" not in ids and len(ex) == 6                    # unclear excluded; unfinished never loaded
    a3 = next(e for e in ex if e.message_id == "A3")
    assert set(a3.encoding.labels) == {0, -100} and a3.status == "NO_ASPECT"       # NO_ASPECT is a real all-O example
    a1 = next(e for e in ex if e.message_id == "A1")
    assert a1.encoding.tags.count("B-ASP") == 2                                     # two aspects in one review
    assert rep["unclear_excluded"] == 1 and rep["no_aspect_all_O"] == 1 and rep["mentions"] == 8 and rep["examples"] == 6
    ex2, rep2 = ad.build_extraction_examples(ann.messages, tok, unclear="mask")
    a5 = next(e for e in ex2 if e.message_id == "A5")
    assert set(a5.encoding.labels) == {-100} and rep2["unclear_masked"] == 1     # never all-O: an aspect may be present
    with pytest.raises(ValueError):
        ad.build_extraction_examples(ann.messages, tok, unclear="zero")


def test_a_slow_tokenizer_is_refused():
    with pytest.raises(TypeError, match="fast tokenizer"):
        ad.bio_encode("text", [], SimpleNamespace(is_fast=False))


# ───────────────────────── aspect-conditioned sentiment ─────────────────────────
@pytest.mark.parametrize("labels, expected", [
    (["POSITIVE"], "POSITIVE"), (["NEUTRAL"], "NEUTRAL"), (["NEUTRAL", "NEUTRAL"], "NEUTRAL"),
    (["POSITIVE", "NEUTRAL"], "POSITIVE"), (["NEGATIVE", "NEUTRAL"], "NEGATIVE"), (["POSITIVE", "POSITIVE"], "POSITIVE"),
    (["POSITIVE", "NEGATIVE"], "MIXED"), (["MIXED"], "MIXED"), (["MIXED", "POSITIVE"], "MIXED"),
    (["POSITIVE", "NEGATIVE", "NEUTRAL"], "MIXED"),
])
def test_aggregate_polarity_follows_the_guideline_rules(labels, expected):
    assert ad.aggregate_polarity(labels) == expected


def test_aggregate_polarity_rejects_bad_input():
    with pytest.raises(ValueError):
        ad.aggregate_polarity([])
    with pytest.raises(ValueError):
        ad.aggregate_polarity(["UNKNOWN"])


def test_sentiment_examples_per_message_and_category(loaded):
    _, ann = loaded
    ex = ad.build_sentiment_examples(ann.messages)
    got = {e.example_id: (e.label, e.label_id) for e in ex}
    assert got == {"A1|FRAGRANCE": ("POSITIVE", 0), "A1|PACKAGING": ("NEGATIVE", 1),         # two aspects, two separate labels
                   "A2|FRAGRANCE": ("MIXED", 3),                                             # annotated MIXED stays MIXED
                   "A4|PACKAGING": ("NEUTRAL", 2),                                           # NEUTRAL kept as its own label
                   "B1|PRICE_VALUE": ("MIXED", 3),                                           # POSITIVE + NEGATIVE, same category
                   "B2|PACKAGING": ("POSITIVE", 0)}                                          # POSITIVE + NEUTRAL -> POSITIVE
    assert not any(e.message_id in ("A3", "A5", "A6") for e in ex)                           # no aspect, unclear, unfinished: no example
    b1 = next(e for e in ex if e.example_id == "B1|PRICE_VALUE")
    assert b1.mention_labels == ("POSITIVE", "NEGATIVE") and b1.spans == ((4, 9), (23, 35)) and b1.span_texts == ("price", "not worth it")
    a1 = [e for e in ex if e.message_id == "A1"]
    assert [e.category for e in a1] == ["FRAGRANCE", "PACKAGING"] and a1[0].category_order < a1[1].category_order


def test_mention_level_examples_keep_each_mentions_own_label(loaded):
    _, ann = loaded
    ex = ad.build_sentiment_examples(ann.messages, level="mention")
    b1 = sorted((e.span_texts[0], e.label) for e in ex if e.message_id == "B1")
    assert b1 == [("not worth it", "NEGATIVE"), ("price", "POSITIVE")]
    assert len(ex) == 8 and len({e.example_id for e in ex}) == 8                            # the two B2 'bottle' mentions get distinct ids
    with pytest.raises(ValueError):
        ad.build_sentiment_examples(ann.messages, level="sentence")


def test_conditioning_text_and_pair_encoding(tok, loaded):
    _, ann = loaded
    ex = {e.example_id: e for e in ad.build_sentiment_examples(ann.messages)}
    b1 = ex["B1|PRICE_VALUE"]
    assert ad.conditioning_text(b1, "category") == "price value"
    assert ad.conditioning_text(b1, "span") == "price | not worth it"
    assert ad.conditioning_text(b1, "category_and_span") == "price value: price | not worth it"
    with pytest.raises(ValueError):
        ad.conditioning_text(b1, "none")
    enc = ad.encode_sentiment_examples([ex["A1|FRAGRANCE"], ex["A1|PACKAGING"]], tok, conditioning="span", max_length=32)
    assert [e["label"] for e in enc] == [0, 1] and [e["example_id"] for e in enc] == ["A1|FRAGRANCE", "A1|PACKAGING"]
    toks = tok.convert_ids_to_tokens(enc[0]["input_ids"])
    assert toks[:4] == ["[CLS]", "scent", "[SEP]", "the"]                                   # aspect first, review second
    short = ad.encode_sentiment_examples([ex["A1|FRAGRANCE"]], tok, conditioning="span", max_length=6)[0]
    assert len(short["input_ids"]) == 6 and tok.convert_ids_to_tokens(short["input_ids"])[1] == "scent"   # only the review is cut


# ───────────────────────── timelines ─────────────────────────
def test_grouping_preserves_chronological_order_and_keeps_sequences_apart(loaded):
    _, ann = loaded
    shuffled = list(reversed(ann.messages))
    groups = ad.group_by_sequence(shuffled)
    assert list(groups) == ["SEQ-A", "SEQ-B"]                                              # ordered by first timestamp (1000 < 1500)
    assert [m.message_id for m in groups["SEQ-A"]] == ["A1", "A2", "A3", "A4", "A5"]
    assert [m.message_id for m in groups["SEQ-B"]] == ["B1", "B2"]
    assert all(a.timestamp_ms <= b.timestamp_ms for g in groups.values() for a, b in zip(g, g[1:]))
    assert list(ad.group_by_sequence(shuffled, sequence_type="SAME_REVIEWER_SAME_PRODUCT")) == []


def test_grouping_refuses_a_sequence_whose_order_was_changed():
    a = SimpleNamespace(sequence_id="S", sequence_type="PRODUCT_TIMELINE", position=1, timestamp_ms=2000)
    b = SimpleNamespace(sequence_id="S", sequence_type="PRODUCT_TIMELINE", position=2, timestamp_ms=1000)
    with pytest.raises(ad.AnnotationError, match="order of position"):
        ad.group_by_sequence([a, b])


def test_examples_group_by_timeline_and_give_per_category_series(loaded):
    _, ann = loaded
    ex = ad.build_sentiment_examples(ann.messages)
    groups = ad.group_by_sequence(reversed(ex))
    seq_a = groups["SEQ-A"]
    assert [(e.position, e.category) for e in seq_a] == [(1, "FRAGRANCE"), (1, "PACKAGING"), (2, "FRAGRANCE"), (4, "PACKAGING")]
    series = ad.category_series(seq_a)
    assert [(s["position"], s["label"]) for s in series["FRAGRANCE"]] == [(1, "POSITIVE"), (2, "MIXED")]
    assert [(s["position"], s["label"]) for s in series["PACKAGING"]] == [(1, "NEGATIVE"), (4, "NEUTRAL")]


# ───────────────────────── leakage-safe splits ─────────────────────────
def items(n_entities=10, per=3):
    return [SimpleNamespace(entity_id=f"P{e:02d}", text=f"review {e} {k}", sequence_id=f"S{e:02d}")
            for e in range(n_entities) for k in range(per)]


def test_entities_stay_whole_and_splits_cover_everything():
    data = items()
    sp = ad.split_by_entity(data, seed=1)
    ad.assert_no_entity_leakage(sp)
    assert sum(len(v) for v in sp.values()) == len(data)
    where = {}
    for name, its in sp.items():
        for it in its:
            where.setdefault(it.entity_id, set()).add(name)
    assert all(len(v) == 1 for v in where.values())                                       # a product is never split
    assert all(len(v) > 0 for v in sp.values())
    assert [len(set(i.entity_id for i in sp[n])) for n in ("train", "val", "test")] == [7, 2, 1] or sum(
        len(set(i.entity_id for i in v)) for v in sp.values()) == 10


def test_split_is_deterministic_independent_of_item_order_and_changes_with_the_seed():
    data = items()
    a = ad.split_entities([i.entity_id for i in data], seed=3)
    b = ad.split_entities([i.entity_id for i in reversed(data)], seed=3)
    c = ad.split_entities([i.entity_id for i in data], seed=4)
    assert a == b and a != c
    assert [i.text for i in ad.split_by_entity(data, seed=3)["train"]] == [i.text for i in ad.split_by_entity(data, seed=3)["train"]]


def test_every_positive_ratio_split_gets_an_entity_and_sizes_add_up():
    for n in (3, 4, 5, 10, 11, 23):
        sp = ad.split_entities([f"P{i}" for i in range(n)], seed=0)
        assert sum(len(v) for v in sp.values()) == n and all(len(v) >= 1 for v in sp.values())
    sp = ad.split_entities([f"P{i}" for i in range(4)], ratios=(0.5, 0.5, 0.0))
    assert sp["test"] == [] and len(sp["train"]) == 2
    assert ad.split_entities([f"P{i}" for i in range(10)], seed=0)["train"].__len__() == 7


def test_split_errors():
    with pytest.raises(ValueError, match="cannot fill"):
        ad.split_entities(["P1", "P2"], seed=0)
    with pytest.raises(ValueError, match="sum to 1"):
        ad.split_entities(["P1", "P2", "P3"], ratios=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError, match="entity_id"):
        ad.split_entities(["P1", "", "P3"])
    with pytest.raises(ValueError):
        ad.split_entities(["P1", "P2", "P3"], ratios=(0.5, 0.5), names=("a", "b", "c"))


def test_leakage_check_and_duplicate_text_report():
    a, b = SimpleNamespace(entity_id="P1", text="Great  scent"), SimpleNamespace(entity_id="P1", text="ok")
    with pytest.raises(ValueError, match="appears in both"):
        ad.assert_no_entity_leakage({"train": [a], "test": [b]})
    same = SimpleNamespace(entity_id="P2", text="great scent")
    dup = ad.cross_split_duplicate_texts({"train": [a], "test": [same], "val": [SimpleNamespace(entity_id="P3", text="different")]})
    assert dup == {"great scent": ["test", "train"]}                                       # entity-disjoint, yet identical text


def test_messages_and_examples_split_with_the_same_function(loaded):
    _, ann = loaded
    sp = ad.split_by_entity(ann.messages, ratios=(0.5, 0.5, 0.0), seed=0)
    assert sorted(len({m.entity_id for m in v}) for v in sp.values()) == [0, 1, 1]
    ex = ad.build_sentiment_examples(ann.messages)
    sp2 = ad.split_by_entity(ex, ratios=(0.5, 0.5, 0.0), seed=0)
    ad.assert_no_entity_leakage(sp2)
    assert sum(len(v) for v in sp2.values()) == len(ex)


# ───────────────────────── real mBERT and XLM-R tokenizers (skipped when unavailable) ─────────────────────────
def real_tokenizer(name):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(name)
        except Exception:
            pytest.skip(f"tokenizer {name} not available offline")


@pytest.mark.parametrize("name", ["bert-base-multilingual-cased", "xlm-roberta-base"])
def test_offset_alignment_with_real_fast_tokenizers(name):
    tk = real_tokenizer(name)
    assert tk.is_fast
    spans = [ad.Mention(4, 9, "scent", "FRAGRANCE", "POSITIVE", "EXPLICIT"), ad.Mention(28, 31, "cap", "PACKAGING", "NEGATIVE", "EXPLICIT")]
    enc = ad.bio_encode(T1, spans, tk)
    assert enc.partial_boundary_mentions == [] and enc.truncated_mentions == [] and enc.unaligned_mentions == []
    assert enc.labels[0] == -100 and enc.labels[-1] == -100                                 # <s>/[CLS] and </s>/[SEP] ignored
    covered = [[i for i, t in enumerate(enc.tags) if t and t != "O" and enc.offsets[i][0] < m.end and enc.offsets[i][1] > m.start] for m in spans]
    for m, idx in zip(spans, covered):
        assert idx and enc.tags[idx[0]] == "B-ASP" and all(enc.tags[i] == "I-ASP" for i in idx[1:])
        assert "".join(T1[enc.offsets[i][0]:enc.offsets[i][1]] for i in idx).replace(" ", "") == m.text   # exactly the span's characters
    labelled = {i for idx in covered for i in idx}
    assert {i for i, t in enumerate(enc.tags) if t and t != "O"} == labelled              # nothing else is labelled


@pytest.mark.parametrize("name", ["bert-base-multilingual-cased", "xlm-roberta-base"])
def test_real_tokenizers_in_the_whole_pipeline(name, loaded):
    tk = real_tokenizer(name)
    _, ann = loaded
    ex, rep = ad.build_extraction_examples(ann.messages, tk, categories=ad.category_ids("product"))
    assert rep["examples"] == 6 and rep["truncated_mentions"] == 0 and rep["unaligned_mentions"] == 0
    a1 = next(e for e in ex if e.message_id == "A1")
    assert "B-FRAGRANCE" in a1.encoding.tags and "B-PACKAGING" in a1.encoding.tags
    sent = ad.build_sentiment_examples(ann.messages)
    pairs = ad.encode_sentiment_examples(sent, tk, conditioning="category_and_span", max_length=24)
    assert len(pairs) == len(sent) and all(len(p["input_ids"]) <= 24 for p in pairs)
    assert all(p["input_ids"][0] == tk.cls_token_id for p in pairs)
    assert ("token_type_ids" in pairs[0]) == (name == "bert-base-multilingual-cased")      # mBERT has segment ids, XLM-R does not
    first_sep = pairs[0]["input_ids"].index(tk.sep_token_id)
    assert tk.decode(pairs[0]["input_ids"][1:first_sep]).startswith(ad.category_phrase(sent[0].category))   # the aspect comes first
