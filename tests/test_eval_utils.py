"""Unit tests for src/eval_utils.py (pure functions; no model, no data files)."""
import os
import sys

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import eval_utils as eu


def test_majority_baseline_uses_training_labels_only():
    train = [0, 0, 0, 1]                  # majority from TRAIN = class 0
    test = [1, 1, 1, 0]                   # test majority would be class 1: must NOT be used
    r = eu.majority_baseline(train, test, [0, 1, 2], ["a", "b", "c"])
    assert r["majority_class_from_train"] == 0
    assert abs(r["accuracy"] - 0.25) < 1e-9
    assert r["train_class_counts"] == {0: 3, 1: 1, 2: 0}


def test_observed_transition_metrics_exclude_position_zero():
    y_true = [2, 1, 0, 2, 0, 2]           # sequence A positions 0..2, sequence B positions 0..2
    y_pred = [0, 1, 0, 0, 0, 1]
    pos = [0, 1, 2, 0, 1, 2]
    r = eu.trend_metrics_observed_only(y_true, y_pred, pos, [0, 1, 2])
    assert r["n"] == 4                    # the two position-0 placeholders are dropped
    assert abs(r["accuracy"] - 0.75) < 1e-9


def test_gold_scs_matches_definition():
    r = eu.gold_scs([[0, 0, 1, 1], [0, 1, 0, 1], [2, 2, 2, 2], [1]])
    assert abs(r["scs_mean"] - np.mean([2 / 3, 0.0, 1.0, 1.0])) < 1e-9


def test_constant_sentiment_predictor_has_scs_one():
    assert eu.gold_scs([[3] * 5, [1] * 4])["scs_mean"] == 1.0


def test_bootstrap_is_deterministic_and_resamples_groups():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    f = eu.macro_f1_fn([0, 1])
    a = eu.bootstrap_ci(y, p, f, n_resamples=200, seed=3)
    b = eu.bootstrap_ci(y, p, f, n_resamples=200, seed=3)
    assert a == b and a["ci_low"] <= a["estimate"] <= a["ci_high"] and a["unit"] == "items"
    g = eu.bootstrap_ci(y, p, f, groups=[0, 0, 1, 1, 2, 2, 3, 3], n_resamples=200, seed=3)
    assert g["unit"] == "groups"


def test_comment_level_report_primary_view_ignores_unknown_gold():
    y = [0, 1, 2, 3, 3]
    p = [0, 1, 0, 3, 0]
    r = eu.comment_level_report(y, p)
    assert r["primary_3class"]["n"] == 3 and r["four_class"]["n"] == 5


def test_set_seed_makes_numpy_reproducible():
    eu.set_seed(7)
    a = np.random.rand(3)
    eu.set_seed(7)
    assert np.allclose(a, np.random.rand(3))
