"""
Tests for src/aspect_encoders.py, aspect_extractor.py, aspect_sentiment_model.py and aspect_sequence_model.py.

Everything here is a synthetic fixture: tiny randomly initialised encoders (no download, nothing pretrained), a toy tokenizer, invented
sentences, and made-up probability vectors standing in for a sentiment model. Nothing is trained, no performance is measured or implied, and no
annotation of the real pilot is created or read.
"""
import inspect
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import aspect_data as ad
import aspect_encoders as ae
import aspect_extractor as ax
import aspect_sentiment_model as sm
import aspect_sequence_model as sq
from ontology import TrajectoryType, TransitionType
from test_aspect_data import fake_tokenizer, fixture_rows, write_csv

torch.manual_seed(0)
CATS = ad.category_ids("product")
DAY = 86_400_000


# ───────────────────────── tiny encoders and data ─────────────────────────
def tiny_encoder(kind="bert"):
    from transformers import BertConfig, BertModel, XLMRobertaConfig, XLMRobertaModel
    kw = dict(vocab_size=64, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
    return BertModel(BertConfig(**kw), add_pooling_layer=False) if kind == "bert" else XLMRobertaModel(XLMRobertaConfig(**kw), add_pooling_layer=False)


@pytest.fixture(scope="module")
def tok():
    return fake_tokenizer()


@pytest.fixture(scope="module")
def ann(tmp_path_factory):
    return ad.load_annotations(write_csv(fixture_rows(), tmp_path_factory.mktemp("a") / "ann.csv"))


def ids_batch(b=3, t=7, vocab=40, pad_tail=0):
    ids = torch.randint(4, vocab, (b, t))
    mask = torch.ones(b, t, dtype=torch.long)
    if pad_tail:
        mask[:, -pad_tail:] = 0
        ids[:, -pad_tail:] = 0
    return ids, mask


# ───────────────────────── shared helpers ─────────────────────────
def test_pooling_cls_and_masked_mean():
    h = torch.arange(24, dtype=torch.float).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    assert torch.equal(ae.pool(h, mask, "cls"), h[:, 0])
    mean = ae.pool(h, mask, "mean")
    assert torch.allclose(mean[0], h[0].mean(0)) and torch.allclose(mean[1], h[1, :2].mean(0))     # padding is not averaged in
    with pytest.raises(ValueError):
        ae.pool(h, mask, "max")


def test_encoder_modes_set_what_trains_and_counts_add_up():
    for mode, layers, expect_trainable in (("frozen", 2, False), ("partial", 1, True), ("full", 2, True)):
        m = ax.AspectExtractor(tiny_encoder(), finetune=mode, finetune_layers=layers)
        c = m.parameter_counts()
        assert c["total"] == c["encoder_total"] + c["new_layers_total"] and c["trainable"] == c["encoder_trainable"] + c["new_layers_trainable"]
        assert (c["encoder_trainable"] > 0) == expect_trainable and c["new_layers_trainable"] == c["new_layers_total"]
    frozen = ax.AspectExtractor(tiny_encoder(), finetune="frozen").parameter_counts()
    part = ax.AspectExtractor(tiny_encoder(), finetune="partial", finetune_layers=1).parameter_counts()
    full = ax.AspectExtractor(tiny_encoder(), finetune="full").parameter_counts()
    assert frozen["encoder_trainable"] == 0 < part["encoder_trainable"] < full["encoder_trainable"] < full["encoder_total"] + 1
    with pytest.raises(ValueError):
        ax.AspectExtractor(tiny_encoder(), finetune="lora")


# ───────────────────────── extractor ─────────────────────────
@pytest.mark.parametrize("kind", ["bert", "xlmr"])
def test_extractor_logit_shapes_for_both_encoder_families_and_both_tag_sets(kind):
    ids, mask = ids_batch()
    generic = ax.AspectExtractor(tiny_encoder(kind))
    assert generic.tagset == ["O", "B-ASP", "I-ASP"] and generic(ids, mask)["logits"].shape == (3, 7, 3)
    cat = ax.AspectExtractor(tiny_encoder(kind), tagset=ad.bio_tagset(CATS))
    assert cat(ids, mask)["logits"].shape == (3, 7, 1 + 2 * len(CATS))
    with pytest.raises(ValueError, match="'O'"):
        ax.AspectExtractor(tiny_encoder(kind), tagset=["B-ASP", "O"])


def test_extractor_loss_ignores_masked_labels_and_never_gives_nan():
    m = ax.AspectExtractor(tiny_encoder())
    ids, mask = ids_batch(2, 6)
    labels = torch.tensor([[-100, 0, 1, 2, 0, -100], [-100, 1, 0, 0, 0, -100]])
    out = m(ids, mask, labels=labels)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"]) and out["loss"] > 0
    out["loss"].backward()
    assert m.classifier.weight.grad is not None
    m.zero_grad()
    none = m(ids, mask, labels=torch.full((2, 6), -100))                                   # every message masked
    assert float(none["loss"].detach()) == 0.0
    none["loss"].backward()                                                                  # still differentiable
    assert all(not torch.isnan(p.grad).any() for p in m.parameters() if p.grad is not None)


def test_extractor_gradients_reach_the_encoder_only_when_it_trains():
    ids, mask = ids_batch(2, 6)
    labels = torch.randint(0, 3, (2, 6))
    frozen = ax.AspectExtractor(tiny_encoder(), finetune="frozen")
    frozen(ids, mask, labels=labels)["loss"].backward()
    assert all(p.grad is None for p in frozen.encoder.parameters()) and frozen.classifier.weight.grad is not None
    full = ax.AspectExtractor(tiny_encoder(), finetune="full")
    full(ids, mask, labels=labels)["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in full.encoder.parameters())


def test_extractor_padding_does_not_change_real_token_logits():
    m = ax.AspectExtractor(tiny_encoder()).eval()
    ids, mask = ids_batch(2, 5)
    padded_ids, padded_mask = torch.cat([ids, torch.zeros(2, 3, dtype=torch.long)], 1), torch.cat([mask, torch.zeros(2, 3, dtype=torch.long)], 1)
    a, b = m(ids, mask)["logits"], m(padded_ids, padded_mask)["logits"][:, :5]
    assert torch.allclose(a, b, atol=1e-5)


def test_bio_to_spans_cases():
    ts = ["O", "B-ASP", "I-ASP"]
    text = "xx scent yy cap zz"
    offs = [(0, 0), (3, 8), (9, 11), (12, 15), (16, 18), (0, 0)]
    special = [1, 0, 0, 0, 0, 1]
    span = lambda tags, **k: ax.bio_to_spans(tags, offs, special, k.get("tagset", ts), text)                    # noqa: E731
    assert span([0, 1, 2, 1, 0, 0]) == [{"start": 3, "end": 11, "text": "scent yy", "category": None, "n_tokens": 2},
                                        {"start": 12, "end": 15, "text": "cap", "category": None, "n_tokens": 1}]       # two aspects
    assert span([0, 0, 0, 0, 0, 0]) == []                                                                             # no aspect
    assert [s["text"] for s in span([0, 2, 2, 0, 0, 0])] == ["scent yy"]                                              # I without B starts a span
    assert [s["text"] for s in span([0, 1, 1, 0, 0, 0])] == ["scent", "yy"]                                           # B B: two spans
    assert [s["text"] for s in span([1, 1, 0, 0, 0, 1])] == ["scent"]                                                 # special tokens never labelled
    cat = ad.bio_tagset(["FRAGRANCE", "PACKAGING"])
    f, p = cat.index("B-FRAGRANCE"), cat.index("B-PACKAGING")
    res = ax.bio_to_spans([0, f, cat.index("I-FRAGRANCE"), p, cat.index("I-PACKAGING"), 0], offs, special, cat, text)
    assert [(s["text"], s["category"]) for s in res] == [("scent yy", "FRAGRANCE"), ("cap zz", "PACKAGING")]
    mixed = ax.bio_to_spans([0, f, cat.index("I-PACKAGING"), 0, 0, 0], offs, special, cat, text)                      # category change splits
    assert [(s["text"], s["category"]) for s in mixed] == [("scent", "FRAGRANCE"), ("yy", "PACKAGING")]


def test_gold_labels_round_trip_through_bio_to_spans(tok, ann):
    cats = CATS
    exs, _ = ad.build_extraction_examples(ann.messages, tok, categories=cats)
    tagset = ad.bio_tagset(cats)
    by_id = {m.message_id: m for m in ann.messages}
    for ex in exs:
        e = ex.encoding
        ids = [0 if l == ad.IGNORE_INDEX else l for l in e.labels]
        got = [(s["start"], s["end"], s["category"]) for s in ax.bio_to_spans(ids, e.offsets, e.special_tokens_mask, tagset, ex.text)]
        assert got == [(m.start, m.end, m.category) for m in by_id[ex.message_id].mentions]           # multiple aspects and none at all


def test_extraction_batch_pads_labels_with_ignore_index(tok, ann):
    exs, _ = ad.build_extraction_examples(ann.messages, tok)
    b = ax.extraction_batch(exs, tok.pad_token_id)
    n = max(len(e.encoding.input_ids) for e in exs)
    assert b["input_ids"].shape == b["attention_mask"].shape == b["labels"].shape == (len(exs), n)
    short = min(range(len(exs)), key=lambda i: len(exs[i].encoding.input_ids))
    k = len(exs[short].encoding.input_ids)
    assert (b["labels"][short, k:] == -100).all() and (b["attention_mask"][short, k:] == 0).all() and (b["input_ids"][short, k:] == tok.pad_token_id).all()
    out = ax.AspectExtractor(tiny_encoder())(**b)                                                    # the batch runs through the model
    assert out["logits"].shape == (len(exs), n, 3) and torch.isfinite(out["loss"])


def test_predict_spans_returns_valid_spans_per_text_including_none(tok):
    m = ax.AspectExtractor(tiny_encoder())
    texts = ["The scent is lovely but the cap leaks.", "Bought for my wife.", ""]
    res = m.predict_spans(texts, tok, max_length=16, batch_size=2)
    assert len(res) == 3 and res[2] == []
    for text, spans in zip(texts, res):
        for s in spans:
            assert 0 <= s["start"] < s["end"] <= len(text) and text[s["start"]:s["end"]] == s["text"]
    assert m.training is True                                                                        # restores the mode it found
    with pytest.raises(TypeError):
        m.predict_spans(["x"], type("T", (), {"is_fast": False})())


# ───────────────────────── sentiment model ─────────────────────────
def test_sentiment_classifier_outputs_four_classes_with_aspect_sentiment_labels():
    assert sm.LABELS == ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]
    ids, mask = ids_batch(4, 9)
    m = sm.AspectSentimentClassifier(tiny_encoder())
    out = m(ids, mask, labels=torch.tensor([0, 1, 2, 3]), return_embedding=True)                       # NEUTRAL=2, MIXED=3 are valid targets
    assert out["logits"].shape == (4, 4) and out["embedding"].shape == (4, 16) and torch.isfinite(out["loss"])
    with pytest.raises(ValueError):
        sm.AspectSentimentClassifier(tiny_encoder(), pooling="max")


@pytest.mark.parametrize("kind", ["bert", "xlmr"])
def test_each_aspect_of_one_review_is_scored_separately(tok, ann, kind):
    m = sm.AspectSentimentClassifier(tiny_encoder(kind)).eval()
    exs = [e for e in ad.build_sentiment_examples(ann.messages) if e.message_id == "A1"]
    assert [e.category for e in exs] == ["FRAGRANCE", "PACKAGING"]                                      # two aspects, one review
    batch = sm.sentiment_batch(ad.encode_sentiment_examples(exs, tok, conditioning="span", max_length=32), tok.pad_token_id)
    assert batch["labels"].tolist() == [0, 1]
    kw = {k: v for k, v in batch.items() if k != "labels"}
    if kind == "xlmr":
        kw.pop("token_type_ids", None)
    logits = m(**kw)["logits"]
    assert logits.shape == (2, 4)                                                                       # one row per aspect, not one per review
    assert not torch.allclose(logits[0], logits[1])                                                      # the aspect text reaches the model
    assert batch["input_ids"][0].tolist() != batch["input_ids"][1].tolist()


def test_predict_for_aspects_gives_one_distribution_per_aspect_and_nothing_for_none(tok):
    m = sm.AspectSentimentClassifier(tiny_encoder())
    review = "The scent is lovely but the cap leaks."
    p = m.predict_for_aspects(review, ["scent", "cap", "price"], tok, max_length=24)
    assert p.shape == (3, 4) and torch.allclose(p.sum(-1), torch.ones(3), atol=1e-5) and m.training is True
    assert not torch.allclose(p[0], p[1])
    assert m.predict_for_aspects(review, [], tok).shape == (0, 4)                                       # missing aspects: no output at all
    solo = m.predict_for_aspects(review, ["cap"], tok, max_length=24)
    assert torch.allclose(solo[0], m.eval().predict_for_aspects(review, ["cap"], tok, max_length=24)[0], atol=1e-5)
    with pytest.raises(TypeError):
        m.predict_for_aspects(review, ["cap"], type("T", (), {"is_fast": False})())


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_sentiment_padding_is_masked_for_both_pooling_modes(pooling):
    m = sm.AspectSentimentClassifier(tiny_encoder(), pooling=pooling).eval()
    ids, mask = ids_batch(2, 6)
    pad = lambda x, v: torch.cat([x, torch.full((2, 4), v, dtype=x.dtype)], 1)                          # noqa: E731
    a = m(ids, mask)["logits"]
    b = m(pad(ids, 0), pad(mask, 0))["logits"]
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(sm.AspectSentimentClassifier(tiny_encoder(), pooling="cls").eval()(ids, mask)["logits"],
                              sm.AspectSentimentClassifier(tiny_encoder(), pooling="mean").eval()(ids, mask)["logits"])


def test_sentiment_batch_handles_token_types_and_gradients_follow_the_mode():
    enc = [{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "token_type_ids": [0, 0, 1], "label": 2, "example_id": "a"},
           {"input_ids": [5, 6], "attention_mask": [1, 1], "token_type_ids": [0, 0], "label": 3, "example_id": "b"}]
    b = sm.sentiment_batch(enc, pad_token_id=9)
    assert b["input_ids"].tolist() == [[5, 6, 7], [5, 6, 9]] and b["token_type_ids"].tolist() == [[0, 0, 1], [0, 0, 0]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]] and b["labels"].tolist() == [2, 3]
    assert "token_type_ids" not in sm.sentiment_batch([{k: v for k, v in e.items() if k != "token_type_ids"} for e in enc], 9)
    m = sm.AspectSentimentClassifier(tiny_encoder(), finetune="frozen")
    m(**b)["loss"].backward()
    assert all(p.grad is None for p in m.encoder.parameters()) and m.head.classifier[0].weight.grad is not None
    c = m.parameter_counts()
    assert c["encoder_trainable"] == 0 and c["new_layers_trainable"] > 0


# ───────────────────────── sequence model: events, series, targets ─────────────────────────
def probs_for(label_index, sharp=0.6):
    p = np.full(4, (1 - sharp) / 3)
    p[label_index] = sharp                                   # a made-up "prediction": not a one-hot, not a gold label
    return list(p)


def ev(entity, cat, mid, pos, day, gold="POSITIVE", p=None, stype="PRODUCT_TIMELINE", **kw):
    return sq.AspectEvent(stype, entity, cat, mid, pos, int(day * DAY), p if p is not None else probs_for(ad.AspectSentiment[gold].value), gold, **kw)


def toy_events():
    return [ev("P1", "FRAGRANCE", "m3", 3, 30, "NEGATIVE"), ev("P1", "FRAGRANCE", "m1", 1, 0, "POSITIVE"), ev("P1", "FRAGRANCE", "m2", 2, 10, "MIXED"),
            ev("P1", "PACKAGING", "m2", 2, 10, "NEGATIVE"), ev("P1", "PACKAGING", "m4", 4, 40, "POSITIVE"),
            ev("P2", "FRAGRANCE", "n1", 1, 5, "POSITIVE"), ev("P2", "FRAGRANCE", "n2", 2, 8, "POSITIVE"), ev("P2", "FRAGRANCE", "n3", 3, 9, "POSITIVE"),
            ev("P2", "FRAGRANCE", "n4", 4, 70, "NEUTRAL")]                      # P2 has no PACKAGING events at all


def test_series_never_mix_aspects_products_or_types_and_are_chronological():
    series = sq.build_series(toy_events())
    assert [s.key for s in series] == [("PRODUCT_TIMELINE", "P1", "FRAGRANCE"), ("PRODUCT_TIMELINE", "P1", "PACKAGING"),
                                       ("PRODUCT_TIMELINE", "P2", "FRAGRANCE")]            # no series for P2's missing aspect
    for s in series:
        assert {(e.sequence_type, e.entity_id, e.category) for e in s.events} == {s.key}
        ts = [e.timestamp_ms for e in s.events]
        assert ts == sorted(ts)
    assert [e.message_id for e in series[0].events] == ["m1", "m2", "m3"]                    # shuffled input, ordered output
    other_type = sq.build_series(toy_events() + [ev("P1", "FRAGRANCE", "x1", 1, 1, stype="SAME_REVIEWER_SAME_PRODUCT")])
    assert len(other_type) == 4                                                              # sequence types stay apart


def test_collate_refuses_a_series_that_mixes_aspects_or_is_out_of_order():
    mixed = sq.AspectSeries(("PRODUCT_TIMELINE", "P1", "FRAGRANCE"), [ev("P1", "FRAGRANCE", "a", 1, 0), ev("P1", "PACKAGING", "b", 2, 1)])
    with pytest.raises(ValueError, match="mixes"):
        sq.collate_series([mixed], CATS)
    unordered = sq.AspectSeries(("PRODUCT_TIMELINE", "P1", "FRAGRANCE"), [ev("P1", "FRAGRANCE", "a", 1, 5), ev("P1", "FRAGRANCE", "b", 2, 1)])
    with pytest.raises(ValueError, match="chronological"):
        sq.collate_series([unordered], CATS)
    with pytest.raises(ValueError, match="unknown aspect"):
        sq.collate_series([sq.AspectSeries(("PRODUCT_TIMELINE", "P1", "BATTERY"), [ev("P1", "BATTERY", "a", 1, 0)])], CATS)
    with pytest.raises(ValueError):
        sq.collate_series([], CATS)


def test_time_gap_features():
    f = sq.time_features([0, 1 * DAY, 3 * DAY])
    assert f.shape == (3, 2)
    assert np.allclose(f[:, 0], np.log1p([0, 1, 2])) and np.allclose(f[:, 1], np.log1p([0, 1, 3]))      # gap to previous, time since first
    assert sq.time_features([5 * DAY]).tolist() == [[0.0, 0.0]]
    with pytest.raises(ValueError):
        sq.time_features([DAY, 0])


@pytest.mark.parametrize("golds, transitions, trajectory", [
    (["POSITIVE", "MIXED", "NEGATIVE"], [-100, "DOWNGRADE", "DOWNGRADE"], "DECLINING"),
    (["POSITIVE", "POSITIVE", "POSITIVE"], [-100, "STABLE", "STABLE"], "STABLE"),
    (["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"], [-100, "DOWNGRADE", "UPGRADE", "DOWNGRADE"], "VOLATILE"),
    (["POSITIVE", "NEUTRAL", "NEGATIVE"], [-100, -100, "DOWNGRADE"], "DECLINING"),                  # NEUTRAL skipped: P -> N across it
    (["NEUTRAL", "POSITIVE"], [-100, -100], -100),                                                    # one ranked event: no transition, no trajectory
    (["NEUTRAL", "NEUTRAL"], [-100, -100], -100),
    (["POSITIVE"], [-100], -100),
])
def test_targets_from_gold_follow_the_ontology_and_skip_neutral(golds, transitions, trajectory):
    s = sq.AspectSeries(("PRODUCT_TIMELINE", "P", "FRAGRANCE"), [ev("P", "FRAGRANCE", f"m{i}", i, i, g) for i, g in enumerate(golds)])
    t, j = sq.series_targets(s)
    assert t == [x if x == -100 else TransitionType[x].value for x in transitions]
    assert j == (-100 if trajectory == -100 else TrajectoryType[trajectory].value)


def test_targets_for_an_upgrade_come_straight_from_the_ontology():
    from ontology import SentimentState
    s = sq.AspectSeries(("PRODUCT_TIMELINE", "P", "FRAGRANCE"), [ev("P", "FRAGRANCE", "a", 1, 0, "NEGATIVE"), ev("P", "FRAGRANCE", "b", 2, 1, "POSITIVE")])
    t, j = sq.series_targets(s)
    assert t == [-100, TransitionType.UPGRADE.value]
    assert j == TrajectoryType.compute([SentimentState.NEGATIVE, SentimentState.POSITIVE]).value


def test_targets_are_ignored_when_there_are_no_gold_labels():
    s = sq.AspectSeries(("PRODUCT_TIMELINE", "P", "FRAGRANCE"), [ev("P", "FRAGRANCE", "a", 1, 0, gold=None, p=probs_for(0)),
                                                                  ev("P", "FRAGRANCE", "b", 2, 1, gold=None, p=probs_for(1))])
    assert sq.series_targets(s) == ([-100, -100], -100)


# ───────────────────────── sequence model: forward, masking, independence ─────────────────────────
def model(**kw):
    torch.manual_seed(1)
    return sq.AspectSequenceModel(len(CATS), hidden_dim=12, attention_dim=8, head_hidden=8, category_dim=4, **kw).eval()


def run(m, batch):
    return m(batch["sentiment_probs"], batch["time_features"], batch["seq_lens"], batch["category_ids"], batch.get("text_features"))


def test_forward_shapes_attention_and_multiple_aspects():
    batch = sq.collate_series(sq.build_series(toy_events()), CATS)
    out = run(model(), batch)
    B, T = 3, 4
    assert batch["sentiment_probs"].shape == (B, T, 4) and batch["time_features"].shape == (B, T, 2) and batch["seq_lens"].tolist() == [3, 2, 4]
    assert out["transition_logits"].shape == (B, T, 3) and out["trajectory_logits"].shape == (B, 4) and out["attention_weights"].shape == (B, T)
    w = out["attention_weights"]
    assert torch.allclose(w.sum(1), torch.ones(B), atol=1e-5)
    assert w[1, 2:].abs().sum() == 0 and w[0, 3:].abs().sum() == 0                                   # nothing attends to padding
    assert batch["category_ids"].tolist() == [CATS.index("FRAGRANCE"), CATS.index("PACKAGING"), CATS.index("FRAGRANCE")]


def test_padding_and_other_series_do_not_change_a_series_output():
    series = sq.build_series(toy_events())
    m = model()
    alone = run(m, sq.collate_series([series[0]], CATS))
    together_batch = sq.collate_series(series, CATS)                                                   # padded to length 4 next to other series
    together = run(m, together_batch)
    assert torch.allclose(alone["trajectory_logits"][0], together["trajectory_logits"][0], atol=1e-5)
    assert torch.allclose(alone["transition_logits"][0, :3], together["transition_logits"][0, :3], atol=1e-5)
    assert torch.allclose(alone["attention_weights"][0], together["attention_weights"][0, :3], atol=1e-5)
    changed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in together_batch.items()}
    changed["sentiment_probs"][2] = torch.rand_like(changed["sentiment_probs"][2])                      # rewrite the OTHER series entirely
    again = run(m, changed)
    for b in (0, 1):                                                                                   # series of other aspects/products unchanged
        assert torch.allclose(together["trajectory_logits"][b], again["trajectory_logits"][b], atol=1e-5)
    assert not torch.allclose(together["trajectory_logits"][2], again["trajectory_logits"][2])
    garbage = together_batch["sentiment_probs"].clone()
    garbage[0, 3:] = 99.0                                                                              # values in padding positions are irrelevant
    out = m(garbage, together_batch["time_features"], together_batch["seq_lens"], together_batch["category_ids"])
    assert torch.allclose(out["trajectory_logits"][0], together["trajectory_logits"][0], atol=1e-5)


def test_time_gaps_and_event_order_reach_the_model():
    base = [ev("P", "FRAGRANCE", "a", 1, 0, p=probs_for(0)), ev("P", "FRAGRANCE", "b", 2, 1, p=probs_for(1)), ev("P", "FRAGRANCE", "c", 3, 2, p=probs_for(0))]
    spaced = [ev("P", "FRAGRANCE", "a", 1, 0, p=probs_for(0)), ev("P", "FRAGRANCE", "b", 2, 100, p=probs_for(1)), ev("P", "FRAGRANCE", "c", 3, 400, p=probs_for(0))]
    m = model()
    a = run(m, sq.collate_series(sq.build_series(base), CATS))["trajectory_logits"]
    b = run(m, sq.collate_series(sq.build_series(spaced), CATS))["trajectory_logits"]
    assert not torch.allclose(a, b)                                                                    # same predictions, different time gaps
    order = [ev("P", "FRAGRANCE", "a", 1, 0, p=probs_for(0)), ev("P", "FRAGRANCE", "b", 2, 1, p=probs_for(1)), ev("P", "FRAGRANCE", "c", 3, 2, p=probs_for(2))]
    reordered = [ev("P", "FRAGRANCE", "a", 1, 2, p=probs_for(0)), ev("P", "FRAGRANCE", "b", 2, 1, p=probs_for(1)), ev("P", "FRAGRANCE", "c", 3, 0, p=probs_for(2))]
    c1 = sq.collate_series(sq.build_series(order), CATS)
    c2 = sq.collate_series(sq.build_series(reordered), CATS)
    assert [mid for mid in c2["message_ids"][0]] == ["c", "b", "a"]                                    # builder sorts by time, not by input order
    assert not torch.allclose(run(m, c1)["trajectory_logits"], run(m, c2)["trajectory_logits"])


def test_a_single_event_series_and_decoding():
    s = sq.build_series([ev("P", "FRAGRANCE", "a", 1, 0, "POSITIVE"), ev("Q", "FRAGRANCE", "b", 1, 0, "POSITIVE"), ev("Q", "FRAGRANCE", "c", 2, 3, "NEGATIVE")])
    batch = sq.collate_series(s, CATS)
    assert batch["trajectory_targets"].tolist() == [-100, TrajectoryType.DECLINING.value] and batch["transition_targets"][0].tolist() == [-100, -100]
    out = run(model(), batch)
    dec = sq.decode_predictions(out, batch["seq_lens"].tolist())
    assert len(dec[0]["transitions"]) == 1 and dec[0]["transitions"][0] is None and len(dec[1]["transitions"]) == 2
    assert dec[1]["transitions"][0] is None and dec[1]["transitions"][1] in sq.TRANSITIONS and dec[0]["trajectory"] in sq.TRAJECTORIES
    assert len(dec[1]["attention"]) == 2


# ───────────────────────── no gold labels as input ─────────────────────────
def test_forward_has_no_parameter_that_could_carry_gold_labels_and_outputs_ignore_them():
    params = list(inspect.signature(sq.AspectSequenceModel.forward).parameters)
    assert params == ["self", "sentiment_probs", "time_features", "seq_lens", "category_ids", "text_features"]
    a = toy_events()
    b = [sq.AspectEvent(e.sequence_type, e.entity_id, e.category, e.message_id, e.position, e.timestamp_ms, e.sentiment_probs,
                        "NEGATIVE" if e.gold_label != "NEGATIVE" else "POSITIVE") for e in a]              # different gold labels, same predictions
    ba, bb = sq.collate_series(sq.build_series(a), CATS), sq.collate_series(sq.build_series(b), CATS)
    for k in ("sentiment_probs", "time_features", "seq_lens", "category_ids"):
        assert torch.equal(ba[k], bb[k])                                                               # inputs do not depend on gold
    assert not torch.equal(ba["transition_targets"], bb["transition_targets"])                          # only the targets do
    m = model()
    assert torch.allclose(run(m, ba)["trajectory_logits"], run(m, bb)["trajectory_logits"])


def make_examples(ann):
    return [e for e in ad.build_sentiment_examples(ann.messages)]


def test_events_require_predicted_probabilities_and_reject_gold(ann):
    exs = make_examples(ann)
    probs = {e.example_id: list(np.random.default_rng(i).dirichlet(np.ones(4) * 2)) for i, e in enumerate(exs)}
    events = sq.events_from_examples(exs, probs, source="aspect_sentiment_model (fixture)")
    assert len(events) == len(exs) and all(e.gold_label for e in events) and len(events[0].sentiment_probs) == 4
    gold_onehot = {e.example_id: [1.0 if i == e.label_id else 0.0 for i in range(4)] for e in exs}
    with pytest.raises(sq.LeakageError, match="one-hot"):
        sq.events_from_examples(exs, gold_onehot, source="whatever")
    for bad_source in ("gold", "Labels", "", "  "):
        with pytest.raises(sq.LeakageError):
            sq.events_from_examples(exs, probs, source=bad_source)
    missing = dict(probs); missing.pop(exs[0].example_id)
    with pytest.raises(KeyError):
        sq.events_from_examples(exs, missing, source="m")
    with pytest.raises(ValueError, match="summing to 1"):
        sq.events_from_examples(exs, {**probs, exs[0].example_id: [0.5, 0.5, 0.5, 0.5]}, source="m")
    with pytest.raises(ValueError, match="message_category"):
        sq.events_from_examples(ad.build_sentiment_examples(ann.messages, level="mention"), probs, source="m")


def test_fixture_annotations_to_series_end_to_end(ann):
    exs = make_examples(ann)
    probs = {e.example_id: list(np.random.default_rng(i).dirichlet(np.ones(4) * 2)) for i, e in enumerate(exs)}
    series = sq.build_series(sq.events_from_examples(exs, probs, source="aspect_sentiment_model (fixture)"))
    keys = {s.key: [e.message_id for e in s.events] for s in series}
    assert keys == {("PRODUCT_TIMELINE", "P1", "FRAGRANCE"): ["A1", "A2"], ("PRODUCT_TIMELINE", "P1", "PACKAGING"): ["A1", "A4"],
                    ("PRODUCT_TIMELINE", "P2", "PRICE_VALUE"): ["B1"], ("PRODUCT_TIMELINE", "P2", "PACKAGING"): ["B2"]}
    batch = sq.collate_series(series, CATS)
    got = {k: (batch["trajectory_targets"][i].item(), batch["transition_targets"][i].tolist()) for i, k in enumerate(batch["keys"])}
    assert got[("PRODUCT_TIMELINE", "P1", "FRAGRANCE")] == (TrajectoryType.DECLINING.value, [-100, TransitionType.DOWNGRADE.value])     # POSITIVE -> MIXED
    assert got[("PRODUCT_TIMELINE", "P1", "PACKAGING")] == (-100, [-100, -100])                  # NEGATIVE then NEUTRAL: one ranked event only
    assert got[("PRODUCT_TIMELINE", "P2", "PRICE_VALUE")][0] == -100
    out = run(model(), batch)
    assert out["trajectory_logits"].shape == (4, 4)


# ───────────────────────── loss, optional text features, counts ─────────────────────────
def test_sequence_loss_ignores_undefined_targets_and_never_gives_nan():
    series = sq.build_series(toy_events())
    batch = sq.collate_series(series, CATS)
    m = model().train()
    out = run(m, batch)
    loss = sq.AspectSequenceLoss()(out, batch)
    assert torch.isfinite(loss["total_loss"]) and loss["total_loss"] > 0
    loss["total_loss"].backward()
    assert m.transition_head.classifier[0].weight.grad is not None and m.bilstm.lstm.weight_ih_l0.grad is not None
    m.zero_grad()
    empty = dict(batch)
    empty["transition_targets"] = torch.full_like(batch["transition_targets"], -100)
    empty["trajectory_targets"] = torch.full_like(batch["trajectory_targets"], -100)
    z = sq.AspectSequenceLoss()(run(m, batch), empty)
    assert float(z["total_loss"].detach()) == 0.0
    z["total_loss"].backward()
    assert all(not torch.isnan(p.grad).any() for p in m.parameters() if p.grad is not None)
    only_traj = sq.AspectSequenceLoss(transition_weight=0.0)(run(m, batch), batch)
    assert torch.allclose(only_traj["total_loss"], only_traj["trajectory_loss"])


def test_optional_text_features_are_required_when_declared_and_counted():
    events = [ev("P", "FRAGRANCE", "a", 1, 0, text_features=[0.1] * 5), ev("P", "FRAGRANCE", "b", 2, 4, "NEGATIVE", text_features=[0.2] * 5)]
    batch = sq.collate_series(sq.build_series(events), CATS)
    assert batch["text_features"].shape == (1, 2, 5)
    m = sq.AspectSequenceModel(len(CATS), text_dim=5, hidden_dim=8, attention_dim=4, head_hidden=4).eval()
    assert run(m, batch)["trajectory_logits"].shape == (1, 4)
    with pytest.raises(ValueError, match="text_features"):
        m(batch["sentiment_probs"], batch["time_features"], batch["seq_lens"], batch["category_ids"])
    with pytest.raises(ValueError):
        sq.collate_series(sq.build_series(events + [ev("P", "FRAGRANCE", "c", 3, 5)]), CATS)        # text vectors on some events only
    plain = sq.AspectSequenceModel(len(CATS), hidden_dim=8, attention_dim=4, head_hidden=4)
    c = plain.parameter_counts()
    assert c["trainable"] == c["total"] > 0 and c["encoder_total"] == 0                                   # no pretrained encoder in this model
