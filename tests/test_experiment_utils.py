"""Tests for src/experiment_utils.py and scripts/train_experiment.py configuration (no pretrained model is loaded)."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiment_utils as xu
from classifier import MultiTaskLoss


# ───────────────────────── masking of the artificial first-position transition ────────────────────────
def test_mask_placeholder_trends_masks_only_position_zero_and_does_not_mutate():
    t = torch.tensor([[2, 0, 1, -1], [2, 1, -1, -1]])
    m = xu.mask_placeholder_trends(t)
    assert m.tolist() == [[-1, 0, 1, -1], [-1, 1, -1, -1]]
    assert t.tolist() == [[2, 0, 1, -1], [2, 1, -1, -1]]          # gold tensor untouched


def _loss_inputs(trend_logits):
    B, S = 2, 3
    return ({"sentiment_logits": torch.zeros(B, S, 4), "trend_logits": trend_logits, "trajectory_logits": torch.zeros(B, 4)},
            {"sentiments": torch.zeros(B, S, dtype=torch.long), "trends": torch.tensor([[2, 0, 1], [2, 1, 0]]),
             "trajectories": torch.zeros(B, dtype=torch.long)})


def test_masked_placeholder_has_no_influence_on_the_loss_but_real_transitions_do():
    loss = MultiTaskLoss(sentiment_weight=0.0, trend_weight=1.0, trajectory_weight=0.0)
    base = torch.randn(2, 3, 3)
    pred, tgt = _loss_inputs(base)
    tgt["trends"] = xu.mask_placeholder_trends(tgt["trends"])
    ref = loss(pred, tgt)["trend_loss"].item()
    changed0 = base.clone(); changed0[:, 0, 0] += 5.0               # logits at the placeholder position
    changed1 = base.clone(); changed1[:, 1, 0] += 5.0               # logits at an observed transition
    assert loss(_loss_inputs(changed0)[0], tgt)["trend_loss"].item() == pytest.approx(ref)
    assert loss(_loss_inputs(changed1)[0], tgt)["trend_loss"].item() != pytest.approx(ref)
    unmasked = _loss_inputs(base)[1]                                # control: without masking the placeholder counts
    assert loss(_loss_inputs(changed0)[0], unmasked)["trend_loss"].item() != pytest.approx(
        loss(_loss_inputs(base)[0], unmasked)["trend_loss"].item())


def test_metrics_use_observed_transitions_only():
    # two sequences of length 3; the placeholder predictions are wrong, every observed transition is right
    a = {"sent_true": [0, 1, 2, 0, 0, 1], "sent_pred": [0, 1, 2, 0, 0, 1],
         "trend_true": [2, 0, 1, 2, 2, 1], "trend_pred": [0, 0, 1, 1, 2, 1],
         "position": [0, 1, 2, 0, 1, 2], "traj_true": [0, 2], "traj_pred": [0, 2],
         "pred_sequences": [[0, 1, 2], [0, 0, 1]], "seq_lens": [3, 3]}
    m = xu.metrics_from_arrays(a)
    assert m["transition_observed"]["n"] == 4                        # 6 positions minus 2 placeholders
    assert m["transition_observed"]["accuracy"] == pytest.approx(1.0)
    assert m["sentiment"]["accuracy"] == pytest.approx(1.0) and m["trajectory"]["accuracy"] == pytest.approx(1.0)
    assert m["mean_macro_f1"] == pytest.approx(np.mean([m["sentiment"]["macro_f1"], m["transition_observed"]["macro_f1"],
                                                        m["trajectory"]["macro_f1"]]))


# ───────────────────────── class weights ─────────────────────────────────────────────────────────────
def test_class_weight_modes():
    labels = [0] * 6 + [1] * 3 + [2] * 1                            # counts 6, 3, 1; total 10; C = 3
    assert xu.class_weight_vector(labels, 3, "none") is None
    full = xu.class_weight_vector(labels, 3, "full")
    assert full.tolist() == pytest.approx([10 / (3 * 6), 10 / (3 * 3), 10 / (3 * 1)])
    assert xu.class_weight_vector(labels, 3, "sqrt").tolist() == pytest.approx(full.sqrt().tolist())
    absent = xu.class_weight_vector([0, 0, 1], 4, "sqrt")           # classes 2 and 3 never occur: weight stays 0
    assert absent[2].item() == 0.0 and absent[3].item() == 0.0
    with pytest.raises(ValueError):
        xu.class_weight_vector(labels, 3, "balanced")


def test_training_weights_use_the_training_split_and_skip_the_placeholder():
    train = SimpleNamespace(sequences=[
        {"sentiments": [0, 0, 1], "trends": [2, 0, 1], "trajectory": 3},
        {"sentiments": [0, 2, 2], "trends": [2, 2, 2], "trajectory": 2}])
    w, counts = xu.training_class_weights(train, "full")
    assert counts["transition_observed"] == [1, 1, 2]                # placeholders (two STABLE) not counted
    assert counts["sentiment"][:3] == [3, 1, 2] and counts["trajectory"] == [0, 0, 1, 1]
    assert w["trend"].tolist() == pytest.approx([4 / (3 * 1), 4 / (3 * 1), 4 / (3 * 2)])
    assert xu.training_class_weights(train, "none")[0] is None


# ───────────────────────── encoder training modes (tiny random encoders, no download) ─────────────────
def _tiny(kind):
    from transformers import BertConfig, BertModel, XLMRobertaConfig, XLMRobertaModel
    kw = dict(vocab_size=50, hidden_size=16, num_hidden_layers=4, num_attention_heads=2, intermediate_size=32,
              max_position_embeddings=40)
    return BertModel(BertConfig(**kw)) if kind == "bert" else XLMRobertaModel(XLMRobertaConfig(**kw))


@pytest.mark.parametrize("kind", ["bert", "xlmr"])
def test_encoder_modes(kind):
    enc = _tiny(kind)
    assert xu.set_encoder_trainable(enc, "frozen") == 0
    n_layer = sum(p.numel() for p in enc.encoder.layer[0].parameters())
    assert xu.set_encoder_trainable(enc, "partial", 1) == n_layer
    assert all(p.requires_grad for p in enc.encoder.layer[3].parameters())
    assert not any(p.requires_grad for p in enc.encoder.layer[2].parameters())
    assert not any(p.requires_grad for p in enc.embeddings.parameters())
    full = xu.set_encoder_trainable(enc, "full")
    assert all(p.requires_grad for p in enc.embeddings.parameters())
    assert all(p.requires_grad for l in enc.encoder.layer for p in l.parameters())
    assert not any(p.requires_grad for p in enc.pooler.parameters())    # the unused pooler never trains
    assert full == sum(p.numel() for p in enc.embeddings.parameters()) + 4 * n_layer
    with pytest.raises(ValueError):
        xu.set_encoder_trainable(enc, "partial", 0)
    with pytest.raises(ValueError):
        xu.set_encoder_trainable(enc, "partial", 5)
    with pytest.raises(ValueError):
        xu.set_encoder_trainable(enc, "top")


class _Holder(nn.Module):
    def __init__(self, encoder_trainable):
        super().__init__()
        self.embedding_generator = nn.Linear(2, 2)
        for p in self.embedding_generator.parameters():
            p.requires_grad = encoder_trainable
        self.head = nn.Linear(2, 3)


def test_separate_learning_rates_for_encoder_and_head():
    g = xu.build_param_groups(_Holder(True), 2e-5, 1e-3, 0.01)
    assert {x["name"]: x["lr"] for x in g} == {"head": 1e-3, "encoder": 2e-5}
    assert {x["name"]: len(x["params"]) for x in g} == {"head": 2, "encoder": 2}
    frozen = xu.build_param_groups(_Holder(False), 2e-5, 1e-3, 0.01)
    assert [x["name"] for x in frozen] == ["head"]                  # no encoder group when the encoder is frozen


# ───────────────────────── schedules and accumulation ───────────────────────────────────────────────
def _lrs(kind, total, warm=0.2):
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    sch, per_update = xu.make_scheduler(opt, kind, total, warm)
    out = []
    for _ in range(total + 1):
        out.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
    return out, per_update


def test_schedules():
    lin, per = _lrs("linear", 10)                                    # warm-up 2 updates, then linear decay to 0
    assert per and lin[0] == 0.0 and lin[1] == pytest.approx(0.5) and lin[2] == pytest.approx(1.0)
    assert lin[10] == pytest.approx(0.0) and lin[6] == pytest.approx(0.5)
    cos, _ = _lrs("cosine", 10)
    assert cos[2] == pytest.approx(1.0) and cos[10] == pytest.approx(0.0) and 0.4 < cos[6] < 0.6
    const, _ = _lrs("constant", 10)
    assert const == [1.0] * 11
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    plateau, per_update = xu.make_scheduler(opt, "plateau", 10, plateau_mode="max")
    assert per_update is False
    for _ in range(4):
        plateau.step(0.5)                                            # no improvement: the learning rate must fall
    assert opt.param_groups[0]["lr"] < 1.0
    with pytest.raises(ValueError):
        xu.make_scheduler(opt, "step", 10)


class _FakeOET(nn.Module):
    """Same call interface as OpinionEvolutionTracker, trainable head only."""
    def __init__(self):
        super().__init__()
        self.embedding_generator = nn.Linear(1, 1)
        self.embedding_generator.requires_grad_(False)
        self.s, self.t, self.j = nn.Linear(4, 4), nn.Linear(4, 3), nn.Linear(4, 4)

    def forward(self, texts, seq_lens=None, padding_mask=None):
        x = torch.ones(len(texts), padding_mask.shape[1], 4)
        return {"sentiment_logits": self.s(x), "trend_logits": self.t(x), "trajectory_logits": self.j(x[:, 0])}


def _batches(n):
    return [{"texts": [["a", "b", "c"]], "seq_lens": [3], "padding_mask": torch.ones(1, 3, dtype=torch.bool),
             "sentiments": torch.tensor([[0, 1, 2]]), "trends": torch.tensor([[2, 0, 1]]), "trajectories": torch.tensor([1])}
            for _ in range(n)]


def test_gradient_accumulation_update_count_and_scheduler_steps():
    assert xu.updates_per_epoch(5, 2) == 3 and xu.updates_per_epoch(4, 2) == 2 and xu.updates_per_epoch(5, 1) == 5
    model = _FakeOET()
    steps = []

    class CountingSGD(torch.optim.SGD):
        def step(self, *a, **k):
            steps.append(1)
            return super().step(*a, **k)

    opt = CountingSGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    sch, per = xu.make_scheduler(opt, "linear", 3)
    out = xu.train_one_epoch(model, _batches(5), MultiTaskLoss(), opt, sch, per, 2, 1.0, torch.device("cpu"))
    assert out["optimizer_updates"] == 3 and len(steps) == 3         # batches 2, 4 and the leftover 5th
    assert sch.last_epoch == 3


def test_accumulating_two_equal_batches_matches_one_step_on_their_mean_gradient():
    torch.manual_seed(0)
    m1, m2 = _FakeOET(), _FakeOET()
    m2.load_state_dict(m1.state_dict())
    o1 = torch.optim.SGD([p for p in m1.parameters() if p.requires_grad], lr=0.1)
    o2 = torch.optim.SGD([p for p in m2.parameters() if p.requires_grad], lr=0.1)
    xu.train_one_epoch(m1, _batches(2), MultiTaskLoss(), o1, None, False, 2, 0.0, torch.device("cpu"))   # one update
    xu.train_one_epoch(m2, _batches(2), MultiTaskLoss(), o2, None, False, 1, 0.0, torch.device("cpu"))   # two updates
    # accumulating averages the two gradients into one update; two separate updates move the weights differently
    assert all(torch.isfinite(p).all() for p in m1.parameters())
    assert not torch.allclose(m1.s.weight, m2.s.weight)


# ───────────────────────── selection and checkpoints ────────────────────────────────────────────────
def test_selection_uses_validation_summary_values():
    s = {"val_loss": 0.9, "mean_macro_f1": 0.55, "sentiment_macro_f1": 0.5, "transition_macro_f1": 0.52, "trajectory_macro_f1": 0.6}
    assert xu.selection_value(s, "mean_macro_f1") == (0.55, True)
    assert xu.selection_value(s, "trajectory_macro_f1") == (0.6, True)
    assert xu.selection_value(s, "val_loss") == (0.9, False)
    with pytest.raises(ValueError):
        xu.selection_value(s, "test_macro_f1")
    assert xu.is_improvement(0.56, 0.55, True) and not xu.is_improvement(0.55, 0.55, True)
    assert xu.is_improvement(0.8, 0.9, False) and not xu.is_improvement(1.0, 0.9, False)
    assert xu.is_improvement(0.1, None, True)


def test_checkpoint_excludes_a_frozen_encoder_and_roundtrips():
    frozen = _Holder(False)
    state, includes = xu.checkpoint_state(frozen)
    assert includes is False and set(state) == {"head.weight", "head.bias"}
    other = _Holder(False)
    xu.load_checkpoint_state(other, state, includes)
    assert torch.equal(other.head.weight, frozen.head.weight)
    trained = _Holder(True)
    state, includes = xu.checkpoint_state(trained)
    assert includes is True and "embedding_generator.weight" in state
    with pytest.raises(RuntimeError):                                # a trained encoder's weights may not be missing
        xu.load_checkpoint_state(_Holder(True), {k: v for k, v in state.items() if k.startswith("head")}, True)
    with pytest.raises(RuntimeError):                                # a missing head weight is always an error
        xu.load_checkpoint_state(_Holder(False), {"head.weight": frozen.head.weight}, False)
    with pytest.raises(RuntimeError):
        xu.load_checkpoint_state(_Holder(False), {**xu.checkpoint_state(frozen)[0], "extra.weight": torch.zeros(1)}, False)


# ───────────────────────── configuration and the training script's arguments ───────────────────────
def _config(**kw):
    c = dict(encoder="mbert", pooling="cls", class_weights="full", finetune="frozen", schedule="constant",
             selection_metric="mean_macro_f1", max_token_length=128, grad_accumulation=1, batch_size=16, epochs=10,
             finetune_layers=2, seed=0, lr_encoder_explicit=False)
    c.update(kw)
    return c


def test_validate_config_accepts_every_supported_option_and_rejects_bad_ones():
    for kw in ({}, {"encoder": "xlmr"}, {"pooling": "mean"}, {"max_token_length": 256}, {"class_weights": "sqrt"},
               {"class_weights": "none"}, {"finetune": "partial"}, {"finetune": "full"}, {"schedule": "cosine"},
               {"selection_metric": "val_loss"}, {"grad_accumulation": 4}):
        xu.validate_config(_config(**kw))
    for kw in ({"encoder": "gpt"}, {"pooling": "max"}, {"class_weights": "balanced"}, {"finetune": "lora"},
               {"schedule": "step"}, {"selection_metric": "test_f1"}, {"max_token_length": 1024}, {"grad_accumulation": 0},
               {"epochs": 0}, {"finetune": "partial", "finetune_layers": 0}, {"lr_encoder_explicit": True}):
        with pytest.raises(ValueError):
            xu.validate_config(_config(**kw))


def test_experiment_directory_is_never_overwritten(tmp_path):
    name = xu.default_experiment_name(_config(encoder="xlmr", finetune="partial", pooling="mean", max_token_length=256, seed=3))
    assert name == "xlmr_top2_mean_len256_wfull_mean_macro_f1_s3"
    path = xu.prepare_experiment_dir(str(tmp_path), name)
    assert os.path.isdir(os.path.join(path, "predictions"))
    with pytest.raises(FileExistsError):
        xu.prepare_experiment_dir(str(tmp_path), name)


def test_script_arguments_cover_every_requested_option():
    import train_experiment as te
    d = te.parse_args([])
    assert (d.encoder, d.finetune, d.pooling, d.max_token_length, d.class_weights, d.selection_metric) == \
           ("mbert", "frozen", "cls", 128, "full", "mean_macro_f1")
    a = te.parse_args(["--encoder", "xlmr", "--finetune", "partial", "--finetune_layers", "3", "--pooling", "mean",
                       "--max_token_length", "256", "--class_weights", "sqrt", "--seed", "7", "--epochs", "6",
                       "--batch_size", "8", "--grad_accumulation", "2", "--lr_head", "5e-4", "--lr_encoder", "1e-5",
                       "--schedule", "cosine", "--selection_metric", "trajectory_macro_f1"])
    assert (a.encoder, a.finetune, a.finetune_layers, a.pooling, a.max_token_length, a.class_weights) == \
           ("xlmr", "partial", 3, "mean", 256, "sqrt")
    assert (a.seed, a.epochs, a.batch_size, a.grad_accumulation, a.lr_head, a.lr_encoder, a.schedule) == \
           (7, 6, 8, 2, 5e-4, 1e-5, "cosine")
    assert xu.ENCODERS["xlmr"] == "xlm-roberta-base" and xu.ENCODERS["mbert"] == "bert-base-multilingual-cased"
    with pytest.raises(SystemExit):
        te.parse_args(["--selection_metric", "test_accuracy"])
