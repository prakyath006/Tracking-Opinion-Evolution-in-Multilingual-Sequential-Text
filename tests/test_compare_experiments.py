"""Tests for scripts/compare_experiments.py. SYNTHETIC summary files only."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_experiments as ce


def make(root, name, mean, status="completed", smoke=False):
    d = root / name
    d.mkdir(parents=True)
    v = {"sentiment_macro_f1": mean, "transition_macro_f1": mean, "trajectory_macro_f1": mean, "mean_macro_f1": mean}
    t = {"sentiment_accuracy": 0.7, "sentiment_macro_f1": mean, "transition_macro_f1": mean, "trajectory_accuracy": 0.6, "trajectory_macro_f1": mean}
    json.dump({"status": status, "name": name, "best_epoch": 3, "selection_metric": "mean_macro_f1", "best_validation": v, "test": t, "smoke_test": smoke},
              open(d / "summary.json", "w"))
    json.dump({"trainable_params": 3_000_000}, open(d / "config.json", "w"))


def test_completed_experiments_are_compared_and_an_unfinished_optional_experiment_is_left_out(tmp_path):
    root = tmp_path / "exp"
    for n, m in (("E0_mbert", 0.50), ("E3c_xlmr", 0.55), ("E4_top2", 0.60)):
        make(root, n, m)
    make(root, "EX_extra", 0.9, status="running")
    out = str(tmp_path / "cmp.md")
    ce.main(["--root", str(root), "--ids", "E0", "E3c", "E4", "EX", "--out", out, "--panel1", str(tmp_path / "none.json")])
    text = open(out).read()
    assert "+0.100" in text and "E4_top2" in text and "EX (status is 'running')" in text
    assert "EX_extra" not in text
    with pytest.raises(FileExistsError):
        ce.main(["--root", str(root), "--out", out])


def test_a_missing_required_experiment_stops_and_nothing_is_invented(tmp_path):
    root = tmp_path / "exp"
    make(root, "E0_mbert", 0.5)
    with pytest.raises(SystemExit, match="E3c"):
        ce.main(["--root", str(root), "--out", str(tmp_path / "c.md")])
    assert not (tmp_path / "c.md").exists()
