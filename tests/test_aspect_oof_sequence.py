"""
Tests for scripts/generate_aspect_oof_predictions.py and scripts/train_aspect_sequence.py (and the helpers they add to src/aspect_training.py).

SYNTHETIC FIXTURES ONLY. The annotated rows are invented, written to temporary folders, and the encoders are tiny random networks, so every number
produced here is a pipeline check, never a research result. The unannotated pilot is not used and no annotation is created for it.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import aspect_data as ad
import aspect_sequence_model as sq
import aspect_training as at
import generate_aspect_oof_predictions as oof
import train_aspect_sequence as seq
from ontology import TrajectoryType, TransitionType
from test_aspect_data import COLUMNS, T1, T2, fake_tokenizer, make_rows
from test_aspect_models import tiny_encoder

DAY = 86_400_000
PATTERNS = [["NEGATIVE", "NEGATIVE", "MIXED", "POSITIVE", "POSITIVE"], ["POSITIVE", "POSITIVE", "MIXED", "NEGATIVE", "NEGATIVE"],
            ["POSITIVE"] * 5, ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"], ["POSITIVE", "NEUTRAL", "NEGATIVE", "NEUTRAL", "NEGATIVE"]]
OFFSETS = [0, 3, 10, 40, 100]


def corpus_rows(n_products=20, flip=(), annotator="a1"):
    """Invented annotated corpus: n_products products x 5 messages; FRAGRANCE in every message, PACKAGING in messages 1, 3 and 5."""
    rows = []
    for p in range(n_products):
        e = f"P{p:02d}"
        fr, pk = PATTERNS[p % 5], PATTERNS[(p + 1) % 5]
        f = lambda lab: {"POSITIVE": "NEGATIVE", "NEGATIVE": "POSITIVE"}.get(lab, lab) if e in flip else lab        # noqa: E731
        for k in range(5):
            ts = (p * 7 + OFFSETS[k]) * DAY + k
            if k % 2 == 0:
                mentions = [("scent", "FRAGRANCE", f(fr[k])), ("cap", "PACKAGING", f(pk[k]))]
                text = T1
            else:
                mentions, text = [("Smells", "FRAGRANCE", f(fr[k]))], T2
            rows += make_rows(f"SEQ-{e}", e, f"{e}-m{k}", k + 1, ts, text, mentions, annotator=annotator)
    return rows


def write(rows, path):
    pd.DataFrame(rows, columns=COLUMNS, dtype=str).fillna("").to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def factory():
    torch.manual_seed(123)                          # a fixed random tiny encoder for every fold, so runs can be compared exactly
    return tiny_encoder()


def run_oof(tmp, ann, name, *extra, allow=True):
    argv = ["--annotations", ann, "--output_root", str(tmp / "oof"), "--name", name, "--folds", "3", "--epochs", "2", "--batch_size", "8",
            "--patience", "0", "--device", "cpu", "--max_token_length", "32", *extra]
    if allow:
        argv.append("--allow_small_data")
    return oof.main(argv, encoder_factory=factory, tokenizer=fake_tokenizer())


def run_seq(tmp, ann, oof_dir, name, *extra, allow=True):
    argv = ["--annotations", ann, "--oof_dir", oof_dir, "--output_root", str(tmp / "seq"), "--name", name, "--epochs", "3", "--batch_size", "8",
            "--patience", "0", "--device", "cpu", "--hidden_dim", "16", "--attention_dim", "8", "--head_hidden", "8", *extra]
    if allow:
        argv.append("--allow_small_data")
    return seq.main(argv)


def jl(d, *parts):
    return json.load(open(os.path.join(d, *parts), encoding="utf-8"))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One annotation file and one out-of-fold run, shared by the tests that only read them."""
    tmp = tmp_path_factory.mktemp("w")
    ann = write(corpus_rows(), tmp / "ann.csv")
    return {"tmp": tmp, "ann": ann, "oof": run_oof(tmp, ann, "oof_a")}


# ───────────────────────── folds ─────────────────────────
def test_kfold_products_are_disjoint_cover_everything_and_are_deterministic():
    prods = [f"P{i:02d}" for i in range(17)]
    folds = oof.kfold_products(prods, 3, seed=0)
    assert sorted(len(f) for f in folds) == [5, 6, 6] and sorted(x for f in folds for x in f) == prods
    assert len({x for f in folds for x in f}) == 17
    assert folds == oof.kfold_products(list(reversed(prods)), 3, seed=0) != oof.kfold_products(prods, 3, seed=1)
    with pytest.raises(ValueError):
        oof.kfold_products(prods, 1, 0)
    with pytest.raises(ValueError):
        oof.kfold_products(prods[:2], 3, 0)


def test_size_floors():
    assert at.cv_blockers(10, 50, 5) and not at.cv_blockers(200, 3000, 5)
    assert any("per fold" in b for b in at.cv_blockers(35, 3000, 20))
    assert at.series_blockers({"train": 10, "val": 5, "test": 5}) and not at.series_blockers({"train": 80, "val": 30, "test": 30})
    assert not any("test" in b for b in at.series_blockers({"train": 80, "val": 30}, test_expected=False))


# ───────────────────────── out-of-fold generation ─────────────────────────
def test_oof_layout_and_every_example_is_predicted_by_a_model_that_never_saw_its_product(world):
    d = world["oof"]
    for f in ("config/config.json", "config/product_split.json", "logs/fold_0/train_log.jsonl", "logs/fold_2/history.json",
              "metrics/fold_1/best_validation.json", "metrics/oof_quality.json", "predictions/oof_predictions.csv",
              "predictions/heldout_predictions.csv", "summary.json"):
        assert os.path.exists(os.path.join(d, *f.split("/"))), f
    assert os.path.isdir(os.path.join(d, "checkpoints")) and not os.path.exists(os.path.join(d, "checkpoints", "fold_0"))      # not kept by default
    split = jl(d, "config", "product_split.json")
    pool, test = set(split["pool"]), set(split["test"])
    assert not pool & test and len(pool) == 17 and len(test) == 3
    held_union = set()
    for f in split["folds"]:
        held, train, val = set(f["heldout"]), set(f["train"]), set(f["val"])
        assert not held & train and not held & val and not train & val                      # the model of fold k never saw a product of fold k
        assert not (held | train | val) & test                                              # test products are in no fold at all
        assert held | train | val == pool
        held_union |= held
    assert held_union == pool
    oofp = pd.read_csv(os.path.join(d, "predictions", "oof_predictions.csv"))
    fold_of = {p: f["fold"] for f in split["folds"] for p in f["heldout"]}
    assert len(oofp) == 17 * 8 and oofp.example_id.is_unique and set(oofp.entity_id) == pool          # 8 aspect examples per product
    assert all(fold_of[r.entity_id] == r.fold for r in oofp.itertuples())                              # predicted by the fold that excluded the product
    assert set(oofp.role) == {"oof"} and set(oofp.n_models) == {1}
    held = pd.read_csv(os.path.join(d, "predictions", "heldout_predictions.csv"))
    assert set(held.entity_id) == test and set(held.role) == {"heldout_ensemble"} and set(held.n_models) == {3} and len(held) == 3 * 8


def test_prediction_files_hold_probabilities_only_and_never_a_gold_label(world):
    for f in ("oof_predictions.csv", "heldout_predictions.csv"):
        df = pd.read_csv(os.path.join(world["oof"], "predictions", f))
        assert not [c for c in df.columns if "gold" in c or c in ("label", "label_id", "sentiment")]
        probs = df[oof.PROB_COLUMNS].to_numpy()
        assert probs.shape[1] == 4 and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5) and (probs >= 0).all()
        assert (probs.max(axis=1) < 1.0 - 1e-9).all()                                               # soft predictions, not a one-hot of anything
    cfg = jl(world["oof"], "config", "config.json")
    assert cfg["gold_labels_are_model_input"] is False and cfg["selection_uses_heldout_fold"] is False and cfg["selection_uses_test"] is False


def test_prediction_calls_the_model_without_labels(world):
    seen = {}
    ann = at.load_corpus([world["ann"]])
    items = oof.build_items(ann.messages[:6], fake_tokenizer(), "category_and_span", 32)
    model = oof.sm.AspectSentimentClassifier(tiny_encoder())
    real = model.forward
    model.forward = lambda **kw: (seen.update(kw), real(**kw))[1]
    p = oof.predict_probs(model, items, 0, 4, torch.device("cpu"))
    assert "labels" not in seen and set(seen) <= {"input_ids", "attention_mask", "token_type_ids"} and p.shape == (len(items), 4)


def test_labels_of_a_product_never_influence_its_own_out_of_fold_probabilities(world):
    tmp = world["tmp"]
    split = jl(world["oof"], "config", "product_split.json")
    pool_product, test_product = split["folds"][0]["heldout"][0], split["test"][0]
    a = pd.read_csv(os.path.join(world["oof"], "predictions", "oof_predictions.csv")).set_index("example_id")
    # (1) flip every gold label of ONE pool product: its own held-out probabilities must not move, other products' (trained with its labels) do
    d1 = run_oof(tmp, write(corpus_rows(flip={pool_product}), tmp / "flip_pool.csv"), "oof_b")
    b = pd.read_csv(os.path.join(d1, "predictions", "oof_predictions.csv")).set_index("example_id")
    own = a.entity_id == pool_product
    assert own.sum() == 8 and np.allclose(a.loc[own, oof.PROB_COLUMNS], b.loc[own.index[own], oof.PROB_COLUMNS], atol=0)
    assert not np.allclose(a.loc[~own, oof.PROB_COLUMNS], b.loc[a.index[~own], oof.PROB_COLUMNS])
    # (2) flip every gold label of an outer TEST product: no model has seen it, so nothing at all changes
    d2 = run_oof(tmp, write(corpus_rows(flip={test_product}), tmp / "flip_test.csv"), "oof_c")
    for f in ("oof_predictions.csv", "heldout_predictions.csv"):
        x, y = pd.read_csv(os.path.join(world["oof"], "predictions", f)), pd.read_csv(os.path.join(d2, "predictions", f))
        assert x[oof.PROB_COLUMNS].equals(y[oof.PROB_COLUMNS])


def test_oof_refuses_small_data_without_the_override_and_marks_overridden_runs(world):
    tmp = world["tmp"]
    with pytest.raises(at.DataTooSmallError, match="too small for meaningful cross-validation"):
        run_oof(tmp, world["ann"], "refused", allow=False)
    assert not os.path.exists(os.path.join(str(tmp / "oof"), "refused"))                           # nothing was created
    cfg, summ = jl(world["oof"], "config", "config.json"), jl(world["oof"], "summary.json")
    assert cfg["research_result"] is False and summ["research_result"] is False and any("too small" in w for w in cfg["warnings"])
    assert jl(world["oof"], "metrics", "oof_quality.json")["research_result"] is False
    with pytest.raises(FileExistsError):
        run_oof(tmp, world["ann"], "oof_a")


def test_oof_can_keep_fold_checkpoints_and_rejects_unannotated_data(tmp_path):
    ann = write(corpus_rows(), tmp_path / "ann.csv")
    d = run_oof(tmp_path, ann, "keep", "--keep_fold_checkpoints", "--finetune", "frozen")
    assert all(os.path.exists(os.path.join(d, "checkpoints", f"fold_{k}", "best_checkpoint.pt")) for k in range(3))
    todo = write(make_rows("S", "P", "m", 1, 1, T1, status="NOT_STARTED"), tmp_path / "todo.csv")
    with pytest.raises(ValueError, match="no finished, valid annotations"):
        run_oof(tmp_path, todo, "x")


# ───────────────────────── series construction and baselines ─────────────────────────
def test_series_inputs_never_depend_on_gold_and_are_chronological(world):
    probs = {r.example_id: [getattr(r, c) for c in oof.PROB_COLUMNS] for r in pd.read_csv(os.path.join(world["oof"], "predictions", "oof_predictions.csv")).itertuples()}
    msgs = [m for m in at.load_corpus([world["ann"]]).messages if m.entity_id == pd.read_csv(os.path.join(world["oof"], "predictions", "oof_predictions.csv")).entity_id[0]]
    shuffled = list(reversed(msgs))
    s1, _ = seq.series_for_split(shuffled, probs, "oof (fixture)")
    assert {s.key[2] for s in s1} == {"FRAGRANCE", "PACKAGING"} and all(len({e.category for e in s.events}) == 1 for s in s1)
    for s in s1:
        ts = [e.timestamp_ms for e in s.events]
        assert ts == sorted(ts) and [e.position for e in s.events] == sorted(e.position for e in s.events)       # input order was reversed
    flipped = [ad.Message(**{**m.__dict__, "mentions": tuple(ad.Mention(x.start, x.end, x.text, x.category, "NEGATIVE" if x.sentiment == "POSITIVE" else "POSITIVE", x.span_type) for x in m.mentions)}) for m in msgs]
    s2, _ = seq.series_for_split(flipped, probs, "oof (fixture)")
    c1, c2 = sq.collate_series(s1, ad.category_ids("product")), sq.collate_series(s2, ad.category_ids("product"))
    for k in ("sentiment_probs", "time_features", "seq_lens", "category_ids"):
        assert torch.equal(c1[k], c2[k])                                                          # gold labels changed, inputs identical
    assert not torch.equal(c1["transition_targets"], c2["transition_targets"])                    # only the targets moved
    with pytest.raises(KeyError):
        seq.series_for_split(msgs, {k: v for k, v in list(probs.items())[1:]}, "oof (fixture)")


def _series(labels, probs_idx):
    events = [sq.AspectEvent("PRODUCT_TIMELINE", "P", "FRAGRANCE", f"m{i}", i, i * DAY,
                             list(np.eye(4)[j] * 0.6 + 0.1), g) for i, (g, j) in enumerate(zip(labels, probs_idx))]
    return sq.AspectSeries(("PRODUCT_TIMELINE", "P", "FRAGRANCE"), events)


def test_rule_baseline_applies_the_ontology_to_predicted_sentiment():
    P, N, NEU, MIX = (ad.AspectSentiment[x].value for x in ("POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"))
    traj, trans, fb_t, fb_e = seq.rule_predictions(_series(["POSITIVE"] * 3, [P, MIX, N]))             # predicted POSITIVE, MIXED, NEGATIVE
    assert traj == TrajectoryType.DECLINING.value and not fb_t
    assert trans == [TransitionType.STABLE.value, TransitionType.DOWNGRADE.value, TransitionType.DOWNGRADE.value] and fb_e == [True, False, False]
    traj, trans, fb_t, fb_e = seq.rule_predictions(_series(["POSITIVE"] * 3, [P, NEU, N]))             # a predicted NEUTRAL is skipped (guidelines)
    assert traj == TrajectoryType.DECLINING.value and trans == [2, 2, TransitionType.DOWNGRADE.value] and fb_e == [True, True, False]
    traj, trans, fb_t, fb_e = seq.rule_predictions(_series(["POSITIVE"] * 2, [NEU, P]))                # one ranked predicted event: no evidence
    assert traj == TrajectoryType.STABLE.value and fb_t is True and trans == [2, 2]
    from ontology import SentimentState
    traj, *_ = seq.rule_predictions(_series(["POSITIVE"] * 2, [N, P]))
    assert traj == TrajectoryType.compute([SentimentState.NEGATIVE, SentimentState.POSITIVE]).value


def test_majority_baseline_uses_training_targets_only():
    train = [_series(["POSITIVE", "POSITIVE", "POSITIVE"], [0, 0, 0]), _series(["POSITIVE", "POSITIVE", "NEGATIVE"], [0, 0, 1]),
             _series(["POSITIVE", "POSITIVE"], [0, 0])]
    traj, trans = seq.majority_labels(train)
    assert traj == TrajectoryType.STABLE.value and trans == TransitionType.STABLE.value


def test_grouped_paired_bootstrap():
    y = [0, 1, 0, 1, 0, 1]
    good, bad = y, [1 - v for v in y]
    groups = ["a", "a", "b", "b", "c", "c"]
    r = at.grouped_paired_bootstrap(y, good, bad, groups, [0, 1], n_resamples=200, seed=0)
    assert r["estimate"] == pytest.approx(1.0) and r["ci_low"] > 0 and r["n_groups"] == 3
    assert r == at.grouped_paired_bootstrap(y, good, bad, groups, [0, 1], n_resamples=200, seed=0)
    assert at.grouped_paired_bootstrap(y, good, bad, ["a"] * 6, [0, 1]) is None


def test_bootstrap_macro_f1_equals_sklearns_on_the_full_sample():
    rng = np.random.default_rng(3)
    y, a, b = rng.integers(0, 4, 60), rng.integers(0, 4, 60), rng.integers(0, 4, 60)
    import eval_utils
    f = eval_utils.macro_f1_fn([0, 1, 2, 3])
    r = at.grouped_paired_bootstrap(y, a, b, [str(i) for i in range(60)], [0, 1, 2, 3], n_resamples=20)
    assert r["estimate"] == pytest.approx(f(y, a) - f(y, b))


def test_oof_reader_refuses_a_file_with_gold_columns(world, tmp_path):
    import shutil
    d = tmp_path / "oofcopy"
    shutil.copytree(world["oof"], d, ignore=shutil.ignore_patterns("checkpoints"))
    f = d / "predictions" / "oof_predictions.csv"
    df = pd.read_csv(f)
    df["gold"] = "POSITIVE"
    df.to_csv(f, index=False)
    with pytest.raises(sq.LeakageError, match="gold-label column"):
        seq.read_oof(str(d))


# ───────────────────────── sequence training ─────────────────────────
def test_sequence_training_end_to_end_layout_baselines_and_split_integrity(world, monkeypatch):
    tmp, ann, d_oof = world["tmp"], world["ann"], world["oof"]
    calls = []
    real_fit = at.fit
    monkeypatch.setattr(at, "fit", lambda *a, **k: (calls.append(a[1:3]), real_fit(*a, **k))[1])
    d = run_seq(tmp, ann, d_oof, "seq_a")
    for f in ("config/config.json", "config/splits.json", "logs/train_log.jsonl", "logs/history.json", "checkpoints/best_checkpoint.pt",
              "metrics/best_validation.json", "metrics/validation_metrics.json", "metrics/test_metrics.json", "predictions/test_series_predictions.csv",
              "predictions/test_transition_predictions.csv", "predictions/validation_series_predictions.csv", "summary.json"):
        assert os.path.exists(os.path.join(d, *f.split("/"))), f
    sp, split = jl(d, "config", "splits.json")["products"], jl(d_oof, "config", "product_split.json")
    tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    assert not tr & va and not tr & te and not va & te and tr | va == set(split["pool"]) and te == set(split["test"])      # same test products as the OOF run
    (train_items, val_items), = calls                                                              # fit saw train and validation series only
    assert {s.key[1] for s in train_items} <= tr and {s.key[1] for s in val_items} <= va and not {s.key[1] for s in train_items + val_items} & te
    for s in train_items + val_items:
        ts = [e.timestamp_ms for e in s.events]
        assert ts == sorted(ts) and len({e.category for e in s.events}) == 1                      # chronological, one aspect per series
    cfg = jl(d, "config", "config.json")
    assert cfg["research_result"] is False and cfg["gold_labels_are_model_input"] is False and cfg["selection_uses_test"] is False
    assert "fold-ensemble" in cfg["prediction_sources"]["test"] and "out-of-fold" in cfg["prediction_sources"]["train"]
    rep = jl(d, "metrics", "test_metrics.json")
    assert rep["research_result"] is False
    for who in ("model", "rule_baseline", "majority_baseline"):
        tj, trn = rep[who]["trajectory"], rep[who]["transition"]
        assert list(tj["per_class"]) == sq.TRAJECTORIES and np.array(tj["confusion_matrix"]).shape == (4, 4)
        assert list(trn["per_class"]) == sq.TRANSITIONS and np.array(trn["confusion_matrix"]).shape == (3, 3)
        assert 0 <= tj["accuracy"] <= 1 and 0 <= tj["macro_f1"] <= 1 and all("f1" in v for v in tj["per_class"].values())
    assert "model_minus_rule_macro_f1_product_grouped_bootstrap" in rep and set(rep["rule_baseline_fallbacks"])
    summ = jl(d, "summary.json")
    assert summ["research_result"] is False and set(summ["test"]) == {"model", "rule_baseline", "majority_baseline"}


def test_the_reported_rule_baseline_matches_an_independent_recomputation(world):
    d = os.path.join(str(world["tmp"] / "seq"), "seq_a")
    rows = pd.read_csv(os.path.join(d, "predictions", "test_series_predictions.csv"))
    held = pd.read_csv(os.path.join(world["oof"], "predictions", "heldout_predictions.csv")).set_index("example_id")
    from ontology import SentimentState
    names = ad.AspectSentiment.label_names()
    assert len(rows) > 0
    for r in rows.itertuples():
        states = []
        for mid in json.loads(r.message_ids):
            p = held.loc[f"{mid}|{r.category}", oof.PROB_COLUMNS].to_numpy(dtype=float)
            lab = names[int(np.argmax(p))]
            if lab != "NEUTRAL":
                states.append(SentimentState[lab])
        expected = TrajectoryType.compute(states).name                                            # the ontology rule, written out again here
        assert r.rule_trajectory == expected
    tm = pd.read_csv(os.path.join(d, "predictions", "test_transition_predictions.csv"))
    assert set(tm.rule_transition) <= set(sq.TRANSITIONS) and (tm.gold_transition.isin(sq.TRANSITIONS)).all()


def test_sequence_training_refuses_small_data_and_unmatched_inputs(world, tmp_path):
    tmp, ann, d_oof = world["tmp"], world["ann"], world["oof"]
    with pytest.raises(at.DataTooSmallError) as e:
        run_seq(tmp, ann, d_oof, "refused_seq", allow=False)
    assert "usable training series" in str(e.value) and "pipeline check" in str(e.value)
    assert not os.path.exists(os.path.join(str(tmp / "seq"), "refused_seq"))
    other = write(corpus_rows(flip={"P00"}), tmp_path / "other.csv")
    with pytest.raises(ValueError, match="differ from those the OOF predictions were made from"):
        run_seq(tmp, other, d_oof, "mismatch")
    with pytest.raises(FileExistsError):
        run_seq(tmp, ann, d_oof, "seq_a")


def test_selection_uses_validation_and_no_test_skips_the_test_split(world):
    tmp, ann, d_oof = world["tmp"], world["ann"], world["oof"]
    d = run_seq(tmp, ann, d_oof, "seq_b", "--no_test", "--selection_metric", "mean_macro_f1", "--epochs", "4", "--class_weights", "sqrt")
    assert not os.path.exists(os.path.join(d, "metrics", "test_metrics.json")) and "test" not in jl(d, "summary.json")
    hist = jl(d, "logs", "history.json")
    best, expect = None, 0
    for r in hist:
        if best is None or r["selection_value"] > best + 1e-4:
            best, expect = r["selection_value"], r["epoch"]
    assert jl(d, "summary.json")["best_epoch"] == expect and all(r["selection_metric"] == "mean_macro_f1" for r in hist)
    assert torch.load(os.path.join(d, "checkpoints", "best_checkpoint.pt"), weights_only=False)["epoch"] == expect


def test_padding_in_a_real_batch_does_not_change_a_series_output(world):
    probs = {r.example_id: [getattr(r, c) for c in oof.PROB_COLUMNS] for r in pd.read_csv(os.path.join(world["oof"], "predictions", "oof_predictions.csv")).itertuples()}
    msgs = [m for m in at.load_corpus([world["ann"]]).messages if f"{m.message_id}|FRAGRANCE" in probs]       # products that have OOF rows
    series, _ = seq.series_for_split([m for m in msgs if m.entity_id in {"P00", "P01", "P02"} and f"{m.message_id}|FRAGRANCE" in probs], probs, "oof (fixture)")
    cats = ad.category_ids("product")
    model = sq.AspectSequenceModel(len(cats), hidden_dim=12, attention_dim=8, head_hidden=8).eval()
    batch = sq.collate_series(series, cats)
    both = seq.forward(model, batch)
    short = min(range(len(series)), key=lambda i: len(series[i].events))
    alone = seq.forward(model, sq.collate_series([series[short]], cats))
    assert len(series[short].events) < batch["sentiment_probs"].shape[1]                           # it really was padded in the batch
    assert torch.allclose(alone["trajectory_logits"][0], both["trajectory_logits"][short], atol=1e-5)


# ───────────────────────── review additions: provenance, ties, NEUTRAL and missing aspects, masked positions ─────────────────────────
def _copy_oof(world, tmp_path):
    import shutil
    d = tmp_path / "oofcopy"
    shutil.copytree(world["oof"], d, ignore=shutil.ignore_patterns("checkpoints"))
    return d


def test_the_sequence_stage_verifies_the_recorded_provenance_of_every_probability(world, tmp_path):
    seq.read_oof(world["oof"])                                                                      # the genuine run passes
    d = _copy_oof(world, tmp_path)
    f = d / "predictions" / "oof_predictions.csv"
    df = pd.read_csv(f)
    df.loc[0, "fold"] = (int(df.loc[0, "fold"]) + 1) % 3                                            # claims another fold produced this product's row
    df.to_csv(f, index=False)
    with pytest.raises(sq.LeakageError, match="not the one that held the product out"):
        seq.read_oof(str(d))
    d2 = tmp_path / "oofcopy2"
    import shutil
    shutil.copytree(world["oof"], d2, ignore=shutil.ignore_patterns("checkpoints"))
    ps = json.load(open(d2 / "config" / "product_split.json"))
    ps["folds"][0]["train"].append(ps["folds"][0]["heldout"][0])                                    # a fold that trained on its own held-out product
    json.dump(ps, open(d2 / "config" / "product_split.json", "w"))
    with pytest.raises(sq.LeakageError, match="held-out product is also in"):
        seq.read_oof(str(d2))
    d3 = tmp_path / "oofcopy3"
    shutil.copytree(world["oof"], d3, ignore=shutil.ignore_patterns("checkpoints"))
    ps = json.load(open(d3 / "config" / "product_split.json"))
    ps["folds"][1]["val"].append(ps["test"][0])                                                     # a test product inside a fold
    json.dump(ps, open(d3 / "config" / "product_split.json", "w"))
    with pytest.raises(sq.LeakageError, match="test product appears"):
        seq.read_oof(str(d3))
    d4 = tmp_path / "oofcopy4"
    shutil.copytree(world["oof"], d4, ignore=shutil.ignore_patterns("checkpoints"))
    h = pd.read_csv(d4 / "predictions" / "heldout_predictions.csv")
    h.loc[0, "entity_id"] = jl(world["oof"], "config", "product_split.json")["pool"][0]             # an ensemble row for a pool product
    h.to_csv(d4 / "predictions" / "heldout_predictions.csv", index=False)
    with pytest.raises(sq.LeakageError, match="not test products"):
        seq.read_oof(str(d4))


def test_identical_review_texts_across_product_groups_are_reported_not_hidden(world):
    cfg = jl(world["oof"], "config", "config.json")
    assert cfg["identical_texts_across_groups"]["test_vs_pool"] > 0                                  # the invented products reuse two sentences
    assert any("identical review texts" in w for w in cfg["warnings"])


def test_events_with_equal_timestamps_keep_position_order_and_zero_gap():
    e = [sq.AspectEvent("PRODUCT_TIMELINE", "P", "FRAGRANCE", f"m{i}", pos, 5 * DAY, [0.25] * 4, "POSITIVE") for i, pos in enumerate([3, 1, 2])]
    s, = sq.build_series(e)
    assert [x.position for x in s.events] == [1, 2, 3]
    assert sq.time_features([x.timestamp_ms for x in s.events]).tolist() == [[0.0, 0.0]] * 3
    two_types = sq.build_series(e + [sq.AspectEvent("SAME_REVIEWER_SAME_PRODUCT", "P", "FRAGRANCE", "x", 1, 6 * DAY, [0.25] * 4, "POSITIVE")])
    assert len(two_types) == 2                                                                       # same product and aspect, different sequence type


def test_messages_without_a_usable_aspect_create_no_events_and_neutral_only_series_have_no_targets(world):
    probs = {r.example_id: [getattr(r, c) for c in oof.PROB_COLUMNS] for r in pd.read_csv(os.path.join(world["oof"], "predictions", "oof_predictions.csv")).itertuples()}
    base = [m for m in at.load_corpus([world["ann"]]).messages if m.entity_id == "P03"]
    none = ad.Message("SEQ-P03", "PRODUCT_TIMELINE", "P03", "P03-none", 9, 10**13, "en", "product", "Bought it.", "t", "NO_ASPECT", ())
    unclear = ad.Message("SEQ-P03", "PRODUCT_TIMELINE", "P03", "P03-unc", 10, 10**13 + 1, "en", "product", "Very strong.", "t", "UNCLEAR_ASPECT", ())
    with_missing, _ = seq.series_for_split(base + [none, unclear], probs, "oof (fixture)")             # no probabilities exist, none are needed
    plain, _ = seq.series_for_split(base, probs, "oof (fixture)")
    assert [[e.message_id for e in s.events] for s in with_missing] == [[e.message_id for e in s.events] for s in plain]
    neutral_only = sq.AspectSeries(("PRODUCT_TIMELINE", "P", "FRAGRANCE"), [sq.AspectEvent("PRODUCT_TIMELINE", "P", "FRAGRANCE", f"m{i}", i, i * DAY, [0.1, 0.1, 0.7, 0.1], "NEUTRAL") for i in range(4)])
    one_ranked = sq.AspectSeries(("PRODUCT_TIMELINE", "P", "PACKAGING"), [sq.AspectEvent("PRODUCT_TIMELINE", "P", "PACKAGING", f"n{i}", i, i * DAY, [0.7, 0.1, 0.1, 0.1], g) for i, g in enumerate(["NEUTRAL", "POSITIVE", "NEUTRAL"])])
    keep, excluded = seq.usable_series([neutral_only, one_ranked, _series(["POSITIVE", "NEGATIVE"], [0, 1])])
    assert len(keep) == 1 and excluded == 2                                                          # only the series with two ranked gold events is evaluated


def test_oof_run_with_missing_and_unclear_aspect_messages(tmp_path):
    rows = corpus_rows()
    for p in range(20):
        e = f"P{p:02d}"
        rows += make_rows(f"SEQ-{e}", e, f"{e}-none", 6, (p * 7 + 120) * DAY, "Bought it for my wife.")
        rows += make_rows(f"SEQ-{e}", e, f"{e}-unc", 7, (p * 7 + 130) * DAY, "Very strong.", aspect_status="UNCLEAR_ASPECT")
    d = run_oof(tmp_path, write(rows, tmp_path / "ann.csv"), "with_missing")
    oofp = pd.read_csv(os.path.join(d, "predictions", "oof_predictions.csv"))
    held = pd.read_csv(os.path.join(d, "predictions", "heldout_predictions.csv"))
    ids = set(oofp.message_id) | set(held.message_id)
    assert not any(i.endswith("-none") or i.endswith("-unc") for i in ids) and len(oofp) == 17 * 8 and len(held) == 3 * 8


def test_masked_and_undefined_positions_never_enter_the_metrics():
    s1 = _series(["POSITIVE", "NEUTRAL", "NEGATIVE", "NEGATIVE"], [0, 2, 1, 1])       # gold transitions: -, -(neutral), DOWNGRADE, STABLE
    s2 = _series(["NEUTRAL", "POSITIVE"], [2, 0])                                       # one ranked event: no trajectory, so not evaluated at all
    preds = [{"trajectory": 1, "transitions": [2, 2, 1, 2]}, {"trajectory": 0, "transitions": [2, 2]}]
    keep, excluded = seq.usable_series([s1, s2])
    assert keep == [s1] and excluded == 1
    t, e, _, _ = seq.arrays(keep, preds[:1], (0, 2))
    assert t["y"] == [TrajectoryType.DECLINING.value] and len(e["y"]) == 2                          # two defined transitions, not four positions
    assert e["y"] == [TransitionType.DOWNGRADE.value, TransitionType.STABLE.value] and e["model"] == [1, 2]
    blk = seq.block(e["y"], e["model"], seq.TRANS_LABELS, sq.TRANSITIONS)
    assert blk["n"] == 2 and blk["accuracy"] == 1.0 and np.array(blk["confusion_matrix"]).sum() == 2
    rep, _, _ = seq.report(keep, preds[:1], (0, 2), research=False)
    assert rep["n_series"] == 1 and rep["n_transition_events"] == 2
    assert np.array(rep["model"]["trajectory"]["confusion_matrix"]).sum() == 1


def test_predictions_do_not_depend_on_how_series_are_batched():
    cats = ad.category_ids("product")
    series = [_series(["POSITIVE"] * n, [0, 1, 3, 0, 1][:n]) for n in (2, 5, 3, 4)]
    for i, s in enumerate(series):
        s.events = [sq.AspectEvent(e.sequence_type, f"P{i}", e.category, e.message_id, e.position, e.timestamp_ms, e.sentiment_probs, e.gold_label) for e in s.events]
        s.key = ("PRODUCT_TIMELINE", f"P{i}", "FRAGRANCE")
    torch.manual_seed(0)
    model = sq.AspectSequenceModel(len(cats), hidden_dim=12, attention_dim=8, head_hidden=8).eval()
    loss = sq.AspectSequenceLoss()
    one, _ = seq.predict_series(model, series, cats, torch.device("cpu"), 1, loss)
    many, _ = seq.predict_series(model, series, cats, torch.device("cpu"), 32, loss)
    assert one == many                                                                               # padding in a batch does not change any prediction
    assert [len(p["transitions"]) for p in many] == [2, 5, 3, 4]
