"""
Tests for scripts/dravidian_evaluation.py.

SYNTHETIC FIXTURES ONLY: invented comments and random embeddings; no encoder is downloaded or run, no project data file is read or changed.
"""
import json
import os
import zlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import dravidian_evaluation as dve
import eval_comment_level as elc

TAMIL, LATIN = "இது நல்ல படம்", "super movie"


def fake_texts(lang, split, n=60):
    base = {"tamil": TAMIL, "malayalam": "ഇത് കൊള്ളാം", "kannada": "ಚೆನ್ನಾಗಿದೆ"}[lang]
    tag = {"train": 1, "dev": 2, "test": 3}[split] * 1000 + {"tamil": 100, "malayalam": 200, "kannada": 300}[lang]      # digits only: keeps the script groups clean
    return [f"{base} {tag + i}" if i % 3 else f"{LATIN} {tag + i}" for i in range(n)]


@pytest.fixture
def patched(monkeypatch):
    def load(corpus, split, max_rows=0):
        n = 60
        return fake_texts(corpus, split, n), [i % 4 for i in range(n)]

    def emb(corpus, split, encoder_key, device, max_rows):
        texts, labels = load(corpus, split)
        y = np.asarray(labels)
        X = np.random.default_rng(zlib.crc32(f"{corpus}-{split}".encode())).normal(size=(len(y), 16)).astype("float32")      # deterministic per file
        X[np.arange(len(y)), y] += 3.0                                    # separable on purpose, so the head learns something
        return X, y

    monkeypatch.setattr(dve.eu, "load_comment_csv", load)
    monkeypatch.setattr(dve.elc, "cached_embeddings", emb)
    monkeypatch.setattr(dve, "sha256", lambda p: "x")
    return monkeypatch


def test_script_groups_come_from_the_characters_of_the_text():
    assert dve.script_group(TAMIL, "tamil") == "native_script"
    assert dve.script_group(LATIN, "tamil") == "romanised"
    assert dve.script_group(TAMIL + " super", "tamil") == "mixed_script"
    assert dve.script_group("😀 123", "tamil") == "other"
    assert dve.script_group(TAMIL, "kannada") == "other"                  # another language's script is not this language's native script


def test_leakage_report_finds_overlap_between_splits_and_languages():
    data = {(lg, sp): pd.DataFrame({"text": fake_texts(lg, sp, 5)}) for lg in dve.LANGS for sp in ("train", "dev", "test")}
    rep = dve.leakage_report(data)
    assert all(v == 0 for lg in rep["within_language"].values() for v in lg.values()) and sum(rep["across_languages_train_vs_other_test"].values()) == 0
    data[("tamil", "test")] = pd.DataFrame({"text": ["  " + data[("tamil", "train")].text.iloc[0].upper()]})                 # same text, different case and spacing
    assert dve.leakage_report(data)["within_language"]["tamil"]["train_test"] == 1


def test_end_to_end_writes_configs_predictions_and_metrics_and_never_overwrites(patched, tmp_path):
    out = str(tmp_path / "out")
    res = dve.main(["--seeds", "0", "1", "--epochs", "3", "--bootstrap", "20", "--device", "cpu", "--out_dir", out])
    assert set(res) == {"tamil->tamil", "malayalam->malayalam", "kannada->kannada", "joint->tamil", "joint->malayalam", "joint->kannada"}
    for d in ("config", "metrics", "predictions"):
        assert os.listdir(os.path.join(out, d))
    cfg = json.load(open(os.path.join(out, "config", "config.json")))
    assert "no timestamps" in cfg["scope"] and cfg["encoder_frozen"] is True
    r = res["tamil->tamil"]
    cm = np.asarray(r["per_seed_reports"]["0"]["four_class"]["confusion_matrix"])
    assert cm.shape == (4, 4) and cm.sum() == 60 and r["four_class"]["accuracy_mean"] > 0.25 and set(r["four_class"]["per_class_f1_mean"]) == set(dve.eu.SENTIMENT_NAMES)
    pred = pd.read_csv(os.path.join(out, "predictions", "joint_to_kannada_test.csv"), encoding="utf-8")
    assert list(pred.columns[:5]) == ["row", "text", "script_group", "gold", "gold_name"] and {"pred_seed0", "pred_seed1"} <= set(pred.columns) and len(pred) == 60
    g = res["tamil->tamil"]["script_groups_seed_0"]
    assert {"native_script", "romanised"} <= set(g) and sum(v["n"] for v in g.values()) == 60
    assert "opinion evolution" in open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    with pytest.raises(FileExistsError):
        dve.main(["--seeds", "0", "--out_dir", out])


def test_every_model_saves_a_loadable_checkpoint_that_reproduces_its_predictions(patched, tmp_path):
    import torch
    out = str(tmp_path / "out")
    dve.main(["--seeds", "0", "1", "--epochs", "3", "--bootstrap", "10", "--device", "cpu", "--out_dir", out])
    files = sorted(os.listdir(os.path.join(out, "checkpoints")))
    assert files == sorted(f"{s}_seed{k}.pt" for s in ("tamil", "malayalam", "kannada", "joint") for k in (0, 1))      # separate language models and the joint model
    ck = torch.load(os.path.join(out, "checkpoints", "joint_seed1.pt"), weights_only=True)
    assert ck["train_languages"] == ["tamil", "malayalam", "kannada"] and ck["encoder_weights_in_checkpoint"] is False and ck["optimizer_state_saved"] is False
    assert ck["labels"] == ["POSITIVE", "NEGATIVE", "MIXED", "UNKNOWN"] and ck["training"]["epochs_max"] == 3 and ck["input_dim"] == 16 and "tamil_train" in ck["input_files_sha256"]
    for setup, tgt in (("tamil", "tamil"), ("joint", "kannada")):
        head, meta = dve.load_head(os.path.join(out, "checkpoints", f"{setup}_seed1.pt"))
        X, _ = dve.elc.cached_embeddings(tgt, "test", "xlmr", "cpu", 0)               # the patched, deterministic embeddings of that test split
        pred = dve.elc.predict(head, X, torch.device("cpu"))
        saved = pd.read_csv(os.path.join(out, "predictions", f"{setup}_to_{tgt}_test.csv"), encoding="utf-8")["pred_seed1"].to_numpy()
        assert (pred == saved).all() and meta["seed"] == 1
    other = torch.load(os.path.join(out, "checkpoints", "tamil_seed0.pt"), weights_only=True)
    other["head_state_dict"].pop(next(iter(other["head_state_dict"])))
    torch.save(other, str(tmp_path / "broken.pt"))
    with pytest.raises(RuntimeError):                                                  # a mismatching state dict is refused, never loaded loosely
        dve.load_head(str(tmp_path / "broken.pt"))
    with pytest.raises(FileExistsError):
        dve.save_head(head, os.path.join(out, "checkpoints", "joint_seed1.pt"), {})


def test_overlap_inside_a_language_stops_the_run(patched, tmp_path):
    orig = dve.eu.load_comment_csv

    def leaky(corpus, split, max_rows=0):
        t, y = orig(corpus, split, max_rows)
        if corpus == "tamil" and split == "test":
            t[0] = fake_texts("tamil", "train")[0]
        return t, y

    patched.setattr(dve.eu, "load_comment_csv", leaky)
    with pytest.raises(RuntimeError, match="overlap"):
        dve.main(["--seeds", "0", "--epochs", "1", "--device", "cpu", "--out_dir", str(tmp_path / "o")])
