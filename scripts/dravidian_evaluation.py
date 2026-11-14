"""
DravidianCodeMix comment-level sentiment evaluation with XLM-R (multilingual evaluation track).

COMMENT-LEVEL ONLY. Every row is one independent YouTube comment. The dataset has no timestamps, authors, threads or ordering, none are
invented, and nothing here measures opinion evolution; it is a separate experiment from the Amazon temporal (OET) experiments.

Reuses the existing components: the leakage-controlled official splits written by scripts/prepare_comment_level_data.py
(data/comment_level/<language>_{train,dev,test}.csv; ontology labels 0 POSITIVE, 1 NEGATIVE, 2 MIXED, 3 UNKNOWN), the frozen-encoder embedding
cache and sentence head of scripts/eval_comment_level.py, and src/eval_utils.py for metrics. The encoder is frozen XLM-R base ([CLS] vectors);
the head is trained on a language's TRAIN split with inverse-frequency class weights from TRAIN and early stopping on its DEV split; the TEST
split is only scored. Setups: one model per language (tamil, malayalam, kannada) and one joint model trained on the three TRAIN splits; every
model is scored on each language's TEST split separately. Script subsets (native script / romanised / mixed) are derived from the characters of
the text itself.

    python scripts/dravidian_evaluation.py --seeds 0 1 2 3 4 --device cpu          # writes outputs/dravidian_evaluation/
    python scripts/dravidian_evaluation.py --max_rows 64 --epochs 2 --seeds 0 --out_dir /tmp/smoke   # smoke test, caches nothing
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src")):
    sys.path.insert(0, p)
sys.dont_write_bytecode = True
import eval_comment_level as elc  # noqa: E402
import eval_utils as eu  # noqa: E402

LANGS = ["tamil", "malayalam", "kannada"]
BLOCKS = {"tamil": (0x0B80, 0x0BFF), "malayalam": (0x0D00, 0x0D7F), "kannada": (0x0C80, 0x0CFF)}
MIN_GROUP = 30
OUT = os.path.join(ROOT, "outputs", "dravidian_evaluation")


def script_group(text, lang):
    """native: only the language's own script (plus digits/punctuation); romanised: Latin letters only; mixed: both."""
    lo, hi = BLOCKS[lang]
    native = any(lo <= ord(c) <= hi for c in text)
    latin = any(("a" <= c.lower() <= "z") for c in text)
    return "mixed_script" if native and latin else "native_script" if native else "romanised" if latin else "other"


def norm(t):
    return " ".join(str(t).lower().split())


def leakage_report(data):
    """Exact (normalised) text overlap between splits, inside each language and across languages. The prepared files already remove dev/test texts that
    occur in an earlier split, so all counts must be 0; this recomputes them independently from the files actually used."""
    rep = {"within_language": {}, "across_languages_train_vs_other_test": {}}
    for lg in LANGS:
        s = {sp: {norm(t) for t in data[(lg, sp)]["text"]} for sp in ("train", "dev", "test")}
        rep["within_language"][lg] = {"train_dev": len(s["train"] & s["dev"]), "train_test": len(s["train"] & s["test"]), "dev_test": len(s["dev"] & s["test"])}
    for a in LANGS:
        ta = {norm(t) for t in data[(a, "train")]["text"]}
        for b in LANGS:
            if a != b:
                rep["across_languages_train_vs_other_test"][f"{a}_train->{b}_test"] = len(ta & {norm(t) for t in data[(b, "test")]["text"]})
    return rep


def load_data(max_rows):
    data = {}
    for lg in LANGS:
        for sp in ("train", "dev", "test"):
            texts, labels = eu.load_comment_csv(lg, sp, max_rows)
            data[(lg, sp)] = pd.DataFrame({"text": texts, "label": labels, "script_group": [script_group(t, lg) for t in texts]})
    return data


def distribution(data):
    out = {}
    for lg in LANGS:
        out[lg] = {}
        for sp in ("train", "dev", "test"):
            d = data[(lg, sp)]
            out[lg][sp] = {"rows": int(len(d)), "labels": {n: int((d.label == i).sum()) for i, n in enumerate(eu.SENTIMENT_NAMES)},
                           "script_groups": d.script_group.value_counts().to_dict()}
    return out


def group_metrics(y, pred, groups):
    out = {}
    for g in sorted(set(groups)):
        m = np.asarray(groups) == g
        r = eu.comment_level_report(y[m], pred[m])
        out[g] = {"n": int(m.sum()), "low_support": bool(m.sum() < MIN_GROUP), "accuracy_4class": r["four_class"]["accuracy"], "macro_f1_4class": r["four_class"]["macro_f1"],
                  "macro_f1_3class": r["primary_3class"].get("macro_f1"), "n_3class": r["primary_3class"]["n"]}
    return out


def save_head(head, path, meta):
    """Trained head weights plus everything needed to rebuild and use it. The frozen encoder is NOT stored: it is the public pretrained model named in
    meta["encoder"]. No optimizer state is stored (the head is trained to completion on cached vectors, there is no resume). Never overwrites."""
    import torch
    if os.path.exists(path):
        raise FileExistsError(path)
    torch.save({"head_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()}, **meta}, path)


def load_head(path, device="cpu"):
    """(head in eval mode, checkpoint dict). Strict: the state dict must match the rebuilt head exactly."""
    import torch
    ck = torch.load(path, map_location=device, weights_only=True)
    head = elc.make_head(ck["input_dim"], ck["num_classes"], ck["dropout"])
    head.load_state_dict(ck["head_state_dict"], strict=True)
    return head.to(device).eval(), ck


def predict_texts(path, texts, device="cpu", encoder=None):
    """Inference with a saved head: embeds the texts with the frozen encoder named in the checkpoint (or `encoder`, e.g. a local folder), then classifies."""
    import torch
    head, ck = load_head(path, device)
    dev = torch.device(device)
    X = elc.embed(list(texts), encoder or ck["encoder"], dev, max_len=ck["max_length"])
    return elc.predict(head, X, dev)


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=list(elc.ENCODERS), default="xlmr")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--max_rows", type=int, default=0, help="smoke test: sample this many rows per file; nothing is cached")
    ap.add_argument("--encoder_path", default=None, help="local folder with the encoder weights (instead of downloading the Hugging Face model id)")
    ap.add_argument("--out_dir", default=OUT)
    a = ap.parse_args(argv)
    encoder_id = elc.ENCODERS[a.encoder]                                  # the Hugging Face id, kept for the checkpoints
    if a.encoder_path:
        elc.ENCODERS[a.encoder] = a.encoder_path
    if os.path.exists(a.out_dir) and os.listdir(a.out_dir):
        raise FileExistsError(f"{a.out_dir} already has files; refusing to overwrite")
    import sklearn
    import torch
    import transformers
    device = torch.device("cuda" if (a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())) else "cpu")
    for d in ("config", "metrics", "predictions", "checkpoints"):
        os.makedirs(os.path.join(a.out_dir, d), exist_ok=True)

    data = load_data(a.max_rows)
    leak = leakage_report(data)
    if any(v for lg in leak["within_language"].values() for v in lg.values()):
        raise RuntimeError(f"train/dev/test text overlap inside a language: {leak['within_language']}; regenerate with scripts/prepare_comment_level_data.py")
    emb = {(lg, sp): elc.cached_embeddings(lg, sp, a.encoder, device, a.max_rows) for lg in LANGS for sp in ("train", "dev", "test")}
    for k, (X, y) in emb.items():
        assert len(X) == len(data[k]) and (y == data[k].label.to_numpy()).all(), f"embeddings misaligned with the split file for {k}"

    files = {f"{lg}_{sp}": sha256(os.path.join(eu.COMMENT_LEVEL_DIR, f"{lg}_{sp}.csv")) for lg in LANGS for sp in ("train", "dev", "test")}
    config = {"encoder": elc.ENCODERS[a.encoder], "encoder_frozen": True, "seeds": a.seeds, "epochs_max": a.epochs, "head": "Dropout-Linear(256)-LayerNorm-ReLU-Dropout-Linear(4), scripts/eval_comment_level.py::make_head",
              "class_weights": "inverse frequency from the training split", "early_stopping": "dev loss, patience 5", "bootstrap_resamples": a.bootstrap, "device": str(device),
              "smoke_test": bool(a.max_rows), "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
              "input_files_sha256": files, "labels": dict(enumerate(eu.SENTIMENT_NAMES)),
              "scope": "comment-level sentiment only; no timestamps, authors, threads or sequences exist or are used; not a temporal opinion-evolution experiment"}
    json.dump(config, open(f"{a.out_dir}/config/config.json", "w", encoding="utf-8"), indent=1)
    json.dump({"distribution": distribution(data), "leakage_checks_exact_normalised_text_overlap": leak,
               "split_policy": "official DravidianCodeMix train/dev/test; rows labelled not-<language> dropped by the project preprocessing; dev/test texts that also occur "
                               "in an earlier split removed (outputs/comment_level/data_report.json)"},
              open(f"{a.out_dir}/config/data_report.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    setups = {lg: ([lg]) for lg in LANGS}
    setups["joint"] = LANGS
    results = {}
    for name, srcs in setups.items():
        Xtr = np.concatenate([emb[(s, "train")][0] for s in srcs]); ytr = np.concatenate([emb[(s, "train")][1] for s in srcs])
        Xdv = np.concatenate([emb[(s, "dev")][0] for s in srcs]); ydv = np.concatenate([emb[(s, "dev")][1] for s in srcs])
        heads = {s: elc.train_head(Xtr, ytr, Xdv, ydv, s, device, epochs=a.epochs) for s in a.seeds}
        import inspect
        defaults = {k: v.default for k, v in inspect.signature(elc.train_head).parameters.items() if v.default is not inspect.Parameter.empty}
        counts = np.bincount(ytr, minlength=4).astype(float)
        for s, h in heads.items():                                       # the head chosen on DEV loss (early stopping), one file per setup and seed
            save_head(h, f"{a.out_dir}/checkpoints/{name}_seed{s}.pt", {
                "kind": "frozen_encoder_sentence_head", "setup": name, "train_languages": list(srcs), "seed": int(s), "labels": list(eu.SENTIMENT_NAMES),
                "input_dim": int(Xtr.shape[1]), "num_classes": 4, "dropout": float(inspect.signature(elc.make_head).parameters["dropout"].default),
                "encoder": encoder_id, "encoder_frozen": True, "encoder_weights_in_checkpoint": False, "pooling": "cls", "max_length": 128,
                "selection": "best dev loss (early stopping)", "train_rows": int(len(ytr)), "dev_rows": int(len(ydv)),
                "class_weights": [float(x) for x in np.where(counts > 0, counts.sum() / (4 * np.maximum(counts, 1)), 0.0)],
                "training": {"epochs_max": int(a.epochs), "weight_decay": 1e-4, **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in defaults.items() if k in ("lr", "patience", "batch_size")}},
                "optimizer": "AdamW", "optimizer_state_saved": False, "smoke_test": bool(a.max_rows),
                "input_files_sha256": {k: v for k, v in files.items() if k.split("_")[0] in srcs}})
        for tgt in LANGS:
            if name in LANGS and name != tgt:
                continue                                                  # a single-language model is scored only on its own test split
            Xte, yte = emb[(tgt, "test")]
            groups = data[(tgt, "test")].script_group.to_numpy()
            preds = {s: elc.predict(h, Xte, device) for s, h in heads.items()}
            reps = {s: eu.comment_level_report(yte, p, bootstrap=a.bootstrap if s == a.seeds[0] else 0) for s, p in preds.items()}
            maj3 = eu.majority_baseline(ytr, yte[np.isin(yte, eu.DETERMINATE)], eu.DETERMINATE, eu.SENTIMENT_NAMES[:3])
            maj4 = eu.majority_baseline(ytr, yte, eu.SENTIMENT_LABELS, eu.SENTIMENT_NAMES)
            f4 = [r["four_class"]["macro_f1"] for r in reps.values()]; f3 = [r["primary_3class"]["macro_f1"] for r in reps.values()]
            a4 = [r["four_class"]["accuracy"] for r in reps.values()]
            per_class = {n: float(np.mean([r["four_class"]["per_class"][n]["f1"] for r in reps.values()])) for n in eu.SENTIMENT_NAMES}
            key = f"{name}->{tgt}"
            results[key] = {"train_rows": int(len(ytr)), "test_rows": int(len(yte)), "seeds": a.seeds,
                            "four_class": {"accuracy_mean": float(np.mean(a4)), "accuracy_std": float(np.std(a4)), "macro_f1_mean": float(np.mean(f4)), "macro_f1_std": float(np.std(f4)),
                                           "per_class_f1_mean": per_class},
                            "primary_3class": {"macro_f1_mean": float(np.mean(f3)), "macro_f1_std": float(np.std(f3)), "n": reps[a.seeds[0]]["primary_3class"]["n"]},
                            "majority_baseline_4class": maj4, "majority_baseline_3class": maj3,
                            "per_seed_reports": {str(s): r for s, r in reps.items()},
                            "script_groups_seed_%d" % a.seeds[0]: group_metrics(yte, preds[a.seeds[0]], groups)}
            json.dump(results[key], open(f"{a.out_dir}/metrics/{name}_to_{tgt}.json", "w", encoding="utf-8"), indent=1)
            pdf = pd.DataFrame({"row": range(len(yte)), "text": data[(tgt, "test")].text, "script_group": groups, "gold": yte, "gold_name": [eu.SENTIMENT_NAMES[i] for i in yte]})
            for s, p in preds.items():
                pdf[f"pred_seed{s}"] = p
            pdf.to_csv(f"{a.out_dir}/predictions/{name}_to_{tgt}_test.csv", index=False, encoding="utf-8")
            print(f"{key:20s} 4-class acc {np.mean(a4):.4f} macro-F1 {np.mean(f4):.4f} | 3-class macro-F1 {np.mean(f3):.4f} +- {np.std(f3):.4f} | majority 4-class macro-F1 {maj4['macro_f1']:.4f}", flush=True)
    write_summary(a.out_dir, config, data, leak, results)
    print("wrote", a.out_dir)
    return results


def write_summary(out_dir, config, data, leak, results):
    L = [f"# DravidianCodeMix comment-level sentiment ({config['encoder']}, frozen encoder)", "",
         "**Scope.** Comment-level sentiment only. DravidianCodeMix has no timestamps, authors, threads or ordering and none were invented. These results say nothing about "
         "opinion evolution over time; the Amazon temporal experiments are separate.", "",
         f"Encoder `{config['encoder']}` (frozen), seeds {config['seeds']}, device {config['device']}{' (SMOKE TEST: not a result)' if config['smoke_test'] else ''}. "
         "Splits: official train/dev/test with dev/test texts already seen in an earlier split removed; the head is fitted on train, early-stopped on dev, and test is only scored.", "",
         "## Data", "| Language | train | dev | test | test POS / NEG / MIXED / UNKNOWN |", "|---|---|---|---|---|"]
    for lg in LANGS:
        t = {sp: len(data[(lg, sp)]) for sp in ("train", "dev", "test")}
        d = data[(lg, "test")]
        L.append(f"| {lg} | {t['train']} | {t['dev']} | {t['test']} | " + " / ".join(str(int((d.label == i).sum())) for i in range(4)) + " |")
    L += ["", f"Leakage check (exact normalised text overlap, must be 0): within-language {leak['within_language']}; cross-language train vs other test {sum(leak['across_languages_train_vs_other_test'].values())} texts in total.", "",
          "## Results (mean over seeds)", "| Setup -> test | acc (4-class) | macro-F1 (4-class) | macro-F1 (3-class POS/NEG/MIX) | majority macro-F1 (4-class) | F1 POS / NEG / MIXED / UNKNOWN |", "|---|---|---|---|---|---|"]
    for k, r in results.items():
        f = r["four_class"]
        L.append(f"| {k} | {f['accuracy_mean']:.3f} +- {f['accuracy_std']:.3f} | {f['macro_f1_mean']:.3f} +- {f['macro_f1_std']:.3f} | {r['primary_3class']['macro_f1_mean']:.3f} +- "
                 f"{r['primary_3class']['macro_f1_std']:.3f} | {r['majority_baseline_4class']['macro_f1']:.3f} | " + " / ".join(f"{f['per_class_f1_mean'][n]:.3f}" for n in eu.SENTIMENT_NAMES) + " |")
    L += ["", "## Confusion matrices (4-class, first seed; rows gold, columns predicted: POS, NEG, MIXED, UNKNOWN)"]
    for k, r in results.items():
        s0 = str(r["seeds"][0])
        L += [f"* `{k}`: {r['per_seed_reports'][s0]['four_class']['confusion_matrix']}"]
    L += ["", "## Script subsets of each test split (first seed; derived from the text characters)", "| Setup -> test | subset | n | acc (4-class) | macro-F1 (4-class) | macro-F1 (3-class) |", "|---|---|---|---|---|---|"]
    for k, r in results.items():
        gk = next(x for x in r if x.startswith("script_groups"))
        for g, m in r[gk].items():
            L.append(f"| {k} | {g}{' (n < 30: unreliable)' if m['low_support'] else ''} | {m['n']} | {m['accuracy_4class']:.3f} | {m['macro_f1_4class']:.3f} | "
                     f"{'n/a' if m['macro_f1_3class'] is None else format(m['macro_f1_3class'], '.3f')} |")
    open(f"{out_dir}/summary.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
