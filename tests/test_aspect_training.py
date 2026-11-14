"""
Tests for src/aspect_training.py, scripts/train_aspect_extractor.py and scripts/train_aspect_sentiment.py.

SYNTHETIC FIXTURES ONLY: invented sentences and labels written to temporary folders, tiny randomly initialised encoders and a toy tokenizer.
No project annotation is read (none exists) and none is created; no pretrained model is downloaded; nothing here measures model quality.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import aspect_data as ad
import aspect_training as at
import train_aspect_extractor as tae
import train_aspect_sentiment as tas
from test_aspect_data import COLUMNS, T1, T2, T3, T4, fake_tokenizer, make_rows
from test_aspect_models import tiny_encoder

SENTS = ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]
FLIP = {"POSITIVE": "NEGATIVE", "NEGATIVE": "POSITIVE", "NEUTRAL": "NEUTRAL", "MIXED": "MIXED"}


@pytest.fixture(scope="module")
def tok():
    return fake_tokenizer()


def corpus_rows(n_products=12, annotator="a1", flip_for=(), only_scent_for=(), prefix="", stype="PRODUCT_TIMELINE"):
    """Invented annotated data: n_products products x 4 messages (two aspects, one aspect, no aspect, one aspect)."""
    rows = []
    for p in range(n_products):
        e = f"{prefix}P{p:02d}"
        s = lambda k: (lambda lab: FLIP[lab] if e in flip_for else lab)(SENTS[(p + k) % 4])          # noqa: E731
        m0 = [("scent", "FRAGRANCE", s(0))] + ([] if e in only_scent_for else [("cap", "PACKAGING", s(1))])
        rows += make_rows(f"SEQ-{e}", e, f"{e}-m0", 1, 1000 + p, T1, m0, annotator=annotator, stype=stype)
        rows += make_rows(f"SEQ-{e}", e, f"{e}-m1", 2, 2000 + p, T2, [("Smells", "FRAGRANCE", s(2))], annotator=annotator, stype=stype)
        rows += make_rows(f"SEQ-{e}", e, f"{e}-m2", 3, 3000 + p, T3, annotator=annotator, stype=stype)
        rows += make_rows(f"SEQ-{e}", e, f"{e}-m3", 4, 4000 + p, T4, [("bottle", "PACKAGING", s(3))], annotator=annotator, stype=stype)
    return rows


def write(rows, path):
    pd.DataFrame(rows, columns=COLUMNS, dtype=str).fillna("").to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture()
def ann_path(tmp_path):
    return write(corpus_rows(), tmp_path / "ann.csv")


def common(tmp_path, ann, *extra):
    return ["--annotations", ann, "--output_root", str(tmp_path / "out"), "--epochs", "2", "--batch_size", "4", "--patience", "0",
            "--device", "cpu", "--max_token_length", "32", *extra]


def seeded_encoder(kind):
    torch.manual_seed(123)                        # the tiny encoder is built outside the script, so fix its random weights for comparisons
    return tiny_encoder(kind)


def run_extractor(tmp_path, ann, name, *extra, kind="bert"):
    return tae.main(common(tmp_path, ann, "--name", name, *extra), encoder=seeded_encoder(kind), tokenizer=fake_tokenizer())


def run_sentiment(tmp_path, ann, name, *extra, kind="bert"):
    return tas.main(common(tmp_path, ann, "--name", name, *extra), encoder=seeded_encoder(kind), tokenizer=fake_tokenizer())


def jload(d, f):
    return json.load(open(os.path.join(d, f), encoding="utf-8"))


# ───────────────────────── metrics ─────────────────────────
def test_span_scores_exact_typed_untyped_and_relaxed():
    gold = {"m1": [(0, 5, "A"), (10, 15, "B")], "m2": [], "m3": [(0, 4, "A")]}
    pred = {"m1": [(0, 5, "A"), (10, 15, "A")], "m2": [(2, 6, "A")], "m3": [(0, 3, "A")]}
    s = at.span_scores(gold, pred, typed=True)
    t = s["exact_typed"]                                            # 1 of 4 predictions, 1 of 3 gold spans
    assert (t["tp"], t["n_pred"], t["n_gold"]) == (1, 4, 3)
    assert t["precision"] == pytest.approx(0.25) and t["recall"] == pytest.approx(1 / 3) and t["f1"] == pytest.approx(2 / 7)
    u = s["exact_untyped"]                                          # boundaries only: the (10,15) span now counts
    assert (u["tp"], u["precision"], u["recall"]) == (2, 0.5, pytest.approx(2 / 3)) and u["f1"] == pytest.approx(4 / 7)
    r = s["relaxed_untyped"]                                        # (0,4) vs (0,3): overlap/union 0.75
    assert (r["tp"], r["precision"], r["recall"]) == (3, 0.75, 1.0) and r["f1"] == pytest.approx(6 / 7)
    assert s["by_category"]["A"]["f1"] == pytest.approx(1 / 3) and s["by_category"]["B"]["f1"] == 0.0
    assert s["macro_f1_by_category"] == pytest.approx(1 / 6)
    assert at.span_scores(gold, pred, typed=False)["exact_typed"] is None


def test_span_scores_count_duplicates_and_handle_empty_sets():
    s = at.span_scores({"m": [(0, 5, None)]}, {"m": [(0, 5, None), (0, 5, None)]}, typed=False)["exact_untyped"]
    assert (s["tp"], s["n_pred"], s["n_gold"]) == (1, 2, 1)                                      # a repeated prediction is not a second hit
    z = at.span_scores({}, {}, typed=False)["exact_untyped"]
    assert (z["precision"], z["recall"], z["f1"], z["n_gold"]) == (0.0, 0.0, 0.0, 0)
    none_pred = at.span_scores({"m": [(0, 5, None)]}, {"m": []}, typed=False)["exact_untyped"]
    assert none_pred["recall"] == 0.0 and none_pred["precision"] == 0.0


def test_token_macro_f1_ignores_tags_that_never_occur():
    assert at.token_macro_f1([0, 0, 1, 2], [0, 0, 1, 2]) == pytest.approx(1.0)
    assert at.token_macro_f1([0, 0, 1], [0, 0, 0]) == pytest.approx((0.8 + 0.0) / 2)
    assert at.token_macro_f1([], []) == 0.0


# ───────────────────────── small-data warnings ─────────────────────────
def test_size_warnings_flag_tiny_data_and_stay_quiet_for_adequate_data():
    tiny = {"train": {"products": 7, "items": 30}, "val": {"products": 2, "items": 10}, "test": {"products": 1, "items": 5}}
    w = " ".join(at.size_warnings("extraction", tiny, annotators={"a1"}, duplicate_texts=2))
    for fragment in ("only 10 products in total", "val split has only 2 product", "test split has only 5 aspect mentions",
                     "only 30 training", "single annotator", "2 identical review text", "pipeline check"):
        assert fragment in w
    ok = {"train": {"products": 100, "items": 2000}, "val": {"products": 20, "items": 400}, "test": {"products": 20, "items": 400}}
    assert at.size_warnings("extraction", ok, annotators={"a1", "a2"}) == []


def test_sentiment_warnings_name_absent_and_rare_classes():
    cc = {"POSITIVE": 80, "NEGATIVE": 15, "NEUTRAL": 0, "MIXED": 5}
    stats = {"train": {"products": 100, "items": 2000}, "val": {"products": 20, "items": 100, "class_counts": cc},
             "test": {"products": 20, "items": 150, "class_counts": {k: 40 for k in cc}}}
    w = " ".join(at.size_warnings("sentiment", stats, annotators={"a", "b"}))
    assert "no example of ['NEUTRAL']" in w and "NEGATIVE (15)" in w and "MIXED (5)" in w and "test split has no example" not in w
    no_test = {"train": {"products": 100, "items": 2000}, "val": {"products": 20, "items": 400}}
    assert "no test split" in " ".join(at.size_warnings("sentiment", no_test, annotators={"a", "b"}))


# ───────────────────────── plumbing ─────────────────────────
class WithEncoder(nn.Module):
    def __init__(self, train_encoder):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.encoder.requires_grad_(train_encoder)
        self.head = nn.Linear(2, 2)

    def forward(self, x, labels=None):
        return {"loss": ((self.head(self.encoder(x)) - labels) ** 2).mean()}


def test_checkpoint_leaves_out_a_frozen_encoder_and_param_groups_split_learning_rates():
    frozen, trained = WithEncoder(False), WithEncoder(True)
    state, inc = at.checkpoint_state(frozen)
    assert inc is False and set(state) == {"head.weight", "head.bias"}
    other = WithEncoder(False)
    at.load_checkpoint_state(other, state, inc)
    assert torch.equal(other.head.weight, frozen.head.weight)
    state, inc = at.checkpoint_state(trained)
    assert inc is True and "encoder.weight" in state
    with pytest.raises(RuntimeError):
        at.load_checkpoint_state(WithEncoder(True), {k: v for k, v in state.items() if k.startswith("head")}, True)
    with pytest.raises(RuntimeError):
        at.load_checkpoint_state(WithEncoder(False), {"head.weight": state["head.weight"], "head.bias": state["head.bias"], "zzz": torch.zeros(1)}, False)
    g = at.aspect_param_groups(trained, 2e-5, 1e-3, 0.01)
    assert {x["name"]: x["lr"] for x in g} == {"head": 1e-3, "encoder": 2e-5}
    assert [x["name"] for x in at.aspect_param_groups(frozen, 2e-5, 1e-3, 0.01)] == ["head"]


def test_batches_cover_everything_once_and_are_seeded():
    items = list(range(10))
    got = list(at.iter_batches(items, 4))
    assert got == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    a = [b for b in at.iter_batches(items, 3, np.random.default_rng(1))]
    b = [b for b in at.iter_batches(items, 3, np.random.default_rng(1))]
    assert a == b and sorted(x for c in a for x in c) == items and a != list(at.iter_batches(items, 3))


# ───────────────────────── corpus and splits ─────────────────────────
def test_load_corpus_reads_finished_messages_only_and_never_modifies_files(tmp_path):
    rows = corpus_rows(4) + make_rows("SEQ-X", "PX", "PX-m0", 1, 9000, T1, status="NOT_STARTED")
    path = write(rows, tmp_path / "a.csv")
    before = sha(path)
    c = at.load_corpus([path])
    assert len(c.messages) == 16 and c.unfinished_message_ids == ["PX-m0"] and c.annotators == {"a1"} and c.domains == ["product"]
    assert sha(path) == before
    only = write(corpus_rows(3, stype="SAME_REVIEWER_SAME_PRODUCT", prefix="Q"), tmp_path / "b.csv")
    both = at.load_corpus([path, only])
    assert {m.sequence_type for m in both.messages} == {"PRODUCT_TIMELINE", "SAME_REVIEWER_SAME_PRODUCT"}
    assert {m.sequence_type for m in at.load_corpus([path, only], sequence_types=["SAME_REVIEWER_SAME_PRODUCT"]).messages} == {"SAME_REVIEWER_SAME_PRODUCT"}


def test_load_corpus_errors(tmp_path):
    a = write(corpus_rows(3), tmp_path / "a.csv")
    b = write(corpus_rows(3, annotator="a2"), tmp_path / "b.csv")                                  # the same messages from a second annotator
    with pytest.raises(ValueError, match="adjudicate"):
        at.load_corpus([a, b])
    nothing = write(make_rows("S", "P", "m", 1, 1, T1, status="NOT_STARTED"), tmp_path / "n.csv")      # an unannotated file
    with pytest.raises(ValueError, match="no finished, valid annotations"):
        at.load_corpus([nothing])
    with pytest.raises(ValueError, match="none of the finished ones"):
        at.load_corpus([a], sequence_types=["DISCUSSION_THREAD"])
    bad = corpus_rows(3)
    bad[0]["sentiment"] = "GOOD"
    with pytest.raises(ad.AnnotationError):
        at.load_corpus([write(bad, tmp_path / "bad.csv")])
    assert len(at.load_corpus([write(bad, tmp_path / "bad2.csv")], on_invalid="skip").messages) == 11


def test_splits_keep_products_whole_and_are_reproducible(tmp_path):
    c = at.load_corpus([write(corpus_rows(), tmp_path / "a.csv")])
    sp = at.make_splits(c.messages, seed=0)
    assert [len({m.entity_id for m in sp[n]}) for n in ("train", "val", "test")] == [8, 2, 2]
    ent = {n: {m.entity_id for m in ms} for n, ms in sp.items()}
    assert not (ent["train"] & ent["val"]) and not (ent["train"] & ent["test"]) and not (ent["val"] & ent["test"])
    assert at.make_splits(c.messages, seed=0).keys() == sp.keys() and {m.message_id for m in at.make_splits(c.messages, seed=0)["test"]} == {m.message_id for m in sp["test"]}
    s = at.split_summary(sp)
    assert s["train"]["n_products"] == 8 and s["val"]["n_mentions"] == 2 * 4 and set(s["val"]["products"]) == ent["val"]


# ───────────────────────── fit: validation-only selection ─────────────────────────
def test_fit_selects_on_validation_only_and_stops_early(tmp_path):
    exp = str(tmp_path / "run"); os.makedirs(exp)
    model = WithEncoder(False)
    scripted = iter([0.2, 0.5, 0.4, 0.3, 0.9])
    seen = []

    def evaluate(m, items):
        seen.append(tuple(items))
        v = next(scripted)
        return {"score": v}, {"summary": {"score": v}}, []

    train, val = [torch.ones(1, 2)] * 4, ["VAL-A", "VAL-B"]
    collate = lambda chunk: {"x": torch.cat(chunk), "labels": torch.zeros(len(chunk), 2)}                      # noqa: E731
    res = at.fit(model, train, val, collate, evaluate, device=torch.device("cpu"), epochs=5, batch_size=2, accumulation=1, lr_head=1e-2,
                 lr_encoder=None, weight_decay=0.0, schedule="constant", warmup_ratio=0.0, grad_clip=1.0, selection_metric="score", patience=2,
                 seed=0, exp_dir=exp, config={"k": 1}, log=lambda s: None)
    assert res["best_epoch"] == 2 and len(res["history"]) == 4                                                # epochs 3 and 4 did not improve: stop
    assert all(s == tuple(val) for s in seen)                                                                 # only the validation items were evaluated
    ck = torch.load(res["checkpoint"], weights_only=False)
    assert ck["epoch"] == 2 and ck["selection_value"] == 0.5 and ck["config"] == {"k": 1}
    assert jload(exp, "best_validation.json") == {"summary": {"score": 0.5}}
    lines = open(os.path.join(exp, "train_log.jsonl")).read().strip().split("\n")
    assert len(lines) == 4 and [json.loads(l)["is_best"] for l in lines] == [True, True, False, False]
    with pytest.raises(ValueError, match="validation split is empty"):
        at.fit(model, train, [], collate, evaluate, device=torch.device("cpu"), epochs=1, batch_size=2, accumulation=1, lr_head=1e-2, lr_encoder=None,
               weight_decay=0.0, schedule="constant", warmup_ratio=0.0, grad_clip=0.0, selection_metric="score", patience=0, seed=0, exp_dir=exp,
               config={}, log=print)
    other = str(tmp_path / "other"); os.makedirs(other)
    with pytest.raises(KeyError, match="selection metric"):
        at.fit(model, train, val, collate, lambda m, i: ({"other": 1.0}, {}, []), device=torch.device("cpu"), epochs=1, batch_size=2, accumulation=1,
               lr_head=1e-2, lr_encoder=None, weight_decay=0.0, schedule="constant", warmup_ratio=0.0, grad_clip=0.0, selection_metric="score",
               patience=0, seed=0, exp_dir=other, config={}, log=print)


# ───────────────────────── extractor script ─────────────────────────
def test_extractor_run_writes_every_artifact_and_reports_span_metrics(tmp_path, ann_path):
    d = run_extractor(tmp_path, ann_path, "e1")
    for f in ("config.json", "splits.json", "train_log.jsonl", "history.json", "best_checkpoint.pt", "best_validation.json", "test_metrics.json",
              "summary.json", os.path.join("predictions", "validation_predictions.csv"), os.path.join("predictions", "test_predictions.csv")):
        assert os.path.exists(os.path.join(d, f)), f
    cfg, summ = jload(d, "config.json"), jload(d, "summary.json")
    assert cfg["task"] == "extraction" and cfg["selection_uses_test"] is False and cfg["tagset"] == ["O", "B-ASP", "I-ASP"]
    assert cfg["parameters"]["trainable"] > 0 and cfg["annotators"] == ["a1"]
    test = jload(d, "test_metrics.json")
    for k in ("exact_untyped", "relaxed_untyped"):
        assert set(test["span_scores"][k]) >= {"precision", "recall", "f1", "tp", "n_pred", "n_gold"}
    assert all(0.0 <= summ["test"][k] <= 1.0 for k in ("span_precision", "span_recall", "span_f1", "token_macro_f1"))
    assert summ["test"]["span_f1"] == summ["test"]["macro_span_f1"]                                  # generic tags: macro equals micro
    sp = jload(d, "splits.json")
    prods = {n: set(v["products"]) for n, v in sp["splits"].items()}
    assert not (prods["train"] & prods["val"]) and not (prods["train"] & prods["test"]) and not (prods["val"] & prods["test"])
    pred = pd.read_csv(os.path.join(d, "predictions", "test_predictions.csv"))
    assert len(pred) == sp["splits"]["test"]["n_messages"] and {"message_id", "gold_spans", "pred_spans", "n_gold", "n_pred"} <= set(pred.columns)
    assert set(pred.message_id) == {m for m in pd.read_csv(ann_path, encoding="utf-8-sig").message_id if m.rsplit("-m", 1)[0] in prods["test"]}


def test_extractor_selects_the_epoch_with_the_best_validation_score(tmp_path, ann_path):
    d = run_extractor(tmp_path, ann_path, "e2", "--epochs", "4", "--selection_metric", "token_macro_f1", "--lr_head", "5e-3")
    hist = jload(d, "history.json")
    best, expect = None, 0
    for r in hist:
        if best is None or r["selection_value"] > best + 1e-4:
            best, expect = r["selection_value"], r["epoch"]
    assert jload(d, "summary.json")["best_epoch"] == expect
    assert torch.load(os.path.join(d, "best_checkpoint.pt"), weights_only=False)["epoch"] == expect
    assert all(r["selection_metric"] == "token_macro_f1" for r in hist)


def test_extractor_category_tags_report_per_category_scores_and_accept_xlmr_style_encoder(tmp_path, ann_path):
    d = run_extractor(tmp_path, ann_path, "e3", "--category_tags", "--finetune", "partial", "--finetune_layers", "1", kind="xlmr")
    cfg, test = jload(d, "config.json"), jload(d, "test_metrics.json")
    assert cfg["category_tags"] is True and len(cfg["tagset"]) == 1 + 2 * 14 and cfg["parameters"]["encoder_trainable"] > 0
    assert set(test["span_scores"]["by_category"]) <= set(ad.category_ids("product")) and test["span_scores"]["exact_typed"]["n_gold"] > 0
    assert test["span_scores"]["macro_f1_by_category"] is not None


def test_frozen_extractor_checkpoint_omits_the_encoder(tmp_path, ann_path):
    d = run_extractor(tmp_path, ann_path, "e4", "--finetune", "frozen", "--class_weights", "sqrt")
    ck = torch.load(os.path.join(d, "best_checkpoint.pt"), weights_only=False)
    assert ck["includes_encoder"] is False and not any(k.startswith("encoder.") for k in ck["model_state_dict"])
    assert jload(d, "config.json")["parameters"]["encoder_trainable"] == 0


def test_test_labels_cannot_influence_extractor_training_or_selection(tmp_path):
    base = corpus_rows()
    c = at.load_corpus([write(base, tmp_path / "a.csv")])
    test_products = {m.entity_id for m in at.make_splits(c.messages, seed=0)["test"]}
    changed = corpus_rows(only_scent_for=test_products)                                              # different gold spans in the test products only
    da = run_extractor(tmp_path, write(base, tmp_path / "A.csv"), "ta")
    db = run_extractor(tmp_path, write(changed, tmp_path / "B.csv"), "tb")
    strip = lambda d: [(r["epoch"], r["train_loss"], r["validation"], r["selection_value"]) for r in jload(d, "history.json")]    # noqa: E731
    assert strip(da) == strip(db)                                                                    # identical training and validation
    assert jload(da, "best_validation.json") == jload(db, "best_validation.json")
    ta, tb = jload(da, "test_metrics.json"), jload(db, "test_metrics.json")
    assert ta["span_scores"]["exact_untyped"]["n_gold"] != tb["span_scores"]["exact_untyped"]["n_gold"]       # only the test result moves


# ───────────────────────── sentiment script ─────────────────────────
def test_sentiment_run_reports_four_class_metrics_and_per_example_predictions(tmp_path, ann_path):
    d = run_sentiment(tmp_path, ann_path, "s1")
    cfg, summ, test, sp = jload(d, "config.json"), jload(d, "summary.json"), jload(d, "test_metrics.json"), jload(d, "splits.json")
    assert cfg["labels"] == ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"] and cfg["selection_metric"] == "macro_f1" and cfg["selection_uses_test"] is False
    assert list(test["per_class"]) == cfg["labels"] and np.array(test["confusion_matrix"]).shape == (4, 4)
    assert all({"precision", "recall", "f1", "support"} <= set(v) for v in test["per_class"].values())
    n = sum(sum(r) for r in test["confusion_matrix"])
    assert n == test["n"] > 0
    assert test["accuracy"] == pytest.approx(np.trace(np.array(test["confusion_matrix"])) / n)
    assert test["macro_f1"] == pytest.approx(np.mean([v["f1"] for v in test["per_class"].values()]))
    pred = pd.read_csv(os.path.join(d, "predictions", "test_predictions.csv"))
    assert len(pred) == n and {"example_id", "category", "gold", "pred", "correct", "prob_POSITIVE", "prob_NEUTRAL", "prob_MIXED"} <= set(pred.columns)
    assert np.allclose(pred[[f"prob_{c}" for c in cfg["labels"]]].sum(axis=1), 1.0, atol=1e-4) and set(pred.gold) <= set(cfg["labels"])
    assert (pred.correct == (pred.gold == pred.pred).astype(int)).all()
    assert os.path.exists(os.path.join(d, "predictions", "validation_predictions.csv")) and summ["test"]["accuracy"] == pytest.approx(test["accuracy"])


def test_sentiment_selection_uses_validation_macro_f1_and_all_four_classes_exist(tmp_path, ann_path):
    d = run_sentiment(tmp_path, ann_path, "s2", "--epochs", "4", "--lr_head", "5e-3", "--level", "mention", "--conditioning", "span", "--pooling", "mean")
    hist = jload(d, "history.json")
    best, expect = None, 0
    for r in hist:
        if best is None or r["selection_value"] > best + 1e-4:
            best, expect = r["selection_value"], r["epoch"]
    assert jload(d, "summary.json")["best_epoch"] == expect and all(r["selection_metric"] == "macro_f1" for r in hist)
    assert set(jload(d, "config.json")["class_counts"]["train"]) == {"POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"}
    w = run_sentiment(tmp_path, ann_path, "s3", "--class_weights", "full", "--finetune", "frozen", kind="xlmr")
    cw = jload(w, "class_weights.json")
    assert cw["mode"] == "full" and len(cw["weights"]) == 4 and sum(cw["training_class_counts"].values()) > 0


def test_test_labels_cannot_influence_sentiment_training_or_selection(tmp_path):
    c = at.load_corpus([write(corpus_rows(), tmp_path / "a.csv")])
    test_products = {m.entity_id for m in at.make_splits(c.messages, seed=0)["test"]}
    da = run_sentiment(tmp_path, write(corpus_rows(), tmp_path / "A.csv"), "ua")
    db = run_sentiment(tmp_path, write(corpus_rows(flip_for=test_products), tmp_path / "B.csv"), "ub")      # POSITIVE <-> NEGATIVE in test products only
    strip = lambda d: [(r["epoch"], r["train_loss"], r["validation"], r["selection_value"]) for r in jload(d, "history.json")]    # noqa: E731
    assert strip(da) == strip(db) and jload(da, "best_validation.json") == jload(db, "best_validation.json")
    pa, pb = pd.read_csv(os.path.join(da, "predictions", "test_predictions.csv")), pd.read_csv(os.path.join(db, "predictions", "test_predictions.csv"))
    assert pa.pred.tolist() == pb.pred.tolist()                                                      # same model, same predictions on the test inputs
    assert pa.gold.tolist() != pb.gold.tolist()                                                      # only the gold the test is scored against moved


# ───────────────────────── warnings, refusals and argument checks ─────────────────────────
def test_small_data_is_flagged_in_the_run_output(tmp_path, ann_path, capsys):
    d = run_sentiment(tmp_path, ann_path, "w1")
    out = capsys.readouterr().out
    assert "THE DATASET IS TOO SMALL FOR A MEANINGFUL EVALUATION" in out and "single annotator" in out
    summ = jload(d, "summary.json")
    assert summ["dataset_too_small_for_meaningful_evaluation"] is True and any("products in total" in w for w in summ["warnings"])
    assert jload(d, "config.json")["warnings"] == summ["warnings"]


def test_runs_refuse_to_overwrite_and_unusable_data_stops_cleanly(tmp_path, ann_path):
    run_extractor(tmp_path, ann_path, "dup")
    with pytest.raises(FileExistsError):
        run_extractor(tmp_path, ann_path, "dup")
    with pytest.raises(FileExistsError):
        run_sentiment(tmp_path, ann_path, "dup")                                                     # same directory: refused by the other script too
    unannotated = write(make_rows("S", "P", "m", 1, 1, T1, status="NOT_STARTED"), tmp_path / "todo.csv")
    with pytest.raises(ValueError, match="no finished, valid annotations"):
        run_extractor(tmp_path, unannotated, "x1")
    assert not os.path.exists(os.path.join(str(tmp_path / "out"), "x1"))                              # nothing is created for unusable data


def test_a_validation_split_without_aspects_is_an_error_not_a_zero_score(tmp_path):
    rows = []
    for p in range(12):                                                                              # every message lacks an aspect
        rows += make_rows(f"S{p}", f"P{p:02d}", f"P{p:02d}-m0", 1, 1000 + p, T3)
    path = write(rows, tmp_path / "none.csv")
    with pytest.raises(ValueError, match="no aspect mention"):
        run_extractor(tmp_path, path, "z1")
    with pytest.raises(ValueError, match="no aspect example"):
        run_sentiment(tmp_path, path, "z2")


def test_no_test_flag_skips_the_test_split_and_arguments_are_validated(tmp_path, ann_path):
    d = run_sentiment(tmp_path, ann_path, "n1", "--no_test")
    assert not os.path.exists(os.path.join(d, "test_metrics.json")) and jload(d, "summary.json")["test"] is None
    for i, bad in enumerate((["--finetune", "frozen", "--lr_encoder", "1e-5"], ["--split_ratios", "0.5", "0.2", "0.2"], ["--split_ratios", "1.0", "0.0", "0.0"],
                             ["--finetune", "partial", "--finetune_layers", "0"], ["--max_token_length", "4"], ["--epochs", "0"])):
        with pytest.raises(ValueError):
            run_sentiment(tmp_path, ann_path, f"bad{i}", *bad)
    with pytest.raises(SystemExit):
        tae.parse_args(["--annotations", ann_path, "--selection_metric", "test_f1"])
    with pytest.raises(SystemExit):
        tas.parse_args(["--annotations", ann_path, "--encoder", "mbert", "--pooling", "max"])
    with pytest.raises(TypeError, match="fast tokenizer"):
        tas.main(common(tmp_path, ann_path, "--name", "slow"), encoder=tiny_encoder(), tokenizer=type("T", (), {"is_fast": False})())
