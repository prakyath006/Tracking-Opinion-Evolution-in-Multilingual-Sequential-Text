"""
Module 8b — Multilingual and cross-domain comment-level sentiment evaluation (supervised, frozen encoder).

For every source corpus S in {amazon, tamil, malayalam, kannada}: train the sentence-level head of the
`mbert_sentence` / `xlmr_sentence` baseline (same head architecture as src/baselines.py::SentenceLevelTransformer,
frozen encoder, inverse-frequency class weights from the TRAINING split, early stopping on the dev split) and score it on
the official TEST split of every target corpus T. S == T is the in-domain score; S != T is zero-shot transfer
(cross-lingual for Dravidian pairs, cross-domain and cross-lingual for pairs involving Amazon).

Because the encoder is frozen, [CLS] vectors are computed once per text and cached; the head is then trained on the
cached vectors. This is mathematically the same model as the frozen-encoder baseline, at a fraction of the cost.

Reported per (S, T): primary 3-class macro-F1 (gold POS/NEG/MIX only; comparable across all corpora), four-class
metrics, per-class P/R/F1, confusion matrix, a bootstrap 95% interval over comments, the mean/std over seeds, the
majority-class baseline chosen from S's TRAINING labels, and the relative degradation of the transfer score with
respect to the in-domain score of S.

Prerequisite: python scripts/prepare_comment_level_data.py
    python scripts/eval_comment_level.py --encoder mbert --seeds 0 1 2 3 4 --device cuda
    python scripts/eval_comment_level.py --encoder mbert --max_rows 64 --epochs 2 --seeds 0 --device cpu   # smoke test
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True
import eval_utils as eu                                         # noqa: E402

ENCODERS = {"mbert": "bert-base-multilingual-cased", "xlmr": "xlm-roberta-base"}
CORPORA = ["amazon", "tamil", "malayalam", "kannada"]


def embed(texts, encoder_name, device, batch_size=64, max_len=128):
    """[CLS] vectors of a frozen encoder."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not hasattr(embed, "_cache") or embed._cache[0] != (encoder_name, str(device)):
        tok = AutoTokenizer.from_pretrained(encoder_name)
        enc = AutoModel.from_pretrained(encoder_name).to(device).eval()
        embed._cache = ((encoder_name, str(device)), tok, enc)
    _, tok, enc = embed._cache
    out = []
    with torch.no_grad():
        order = np.argsort([len(t) for t in texts])               # length-sorted batches: less padding
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out.append(enc(**b).last_hidden_state[:, 0, :].float().cpu().numpy())
    vecs = np.concatenate(out)
    res = np.empty_like(vecs)
    res[order] = vecs
    return res


def cached_embeddings(corpus, split, encoder_key, device, max_rows):
    texts, labels = eu.load_comment_csv(corpus, split, max_rows)
    path = os.path.join(eu.COMMENT_LEVEL_DIR, "cache", f"{encoder_key}_{corpus}_{split}.npy")
    if not max_rows and os.path.exists(path):
        X = np.load(path)
        if len(X) == len(texts):
            return X, np.asarray(labels)
    X = embed(texts, ENCODERS[encoder_key], device)
    if not max_rows:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, X)
    return X, np.asarray(labels)


def make_head(dim=768, classes=4, dropout=0.3):
    import torch.nn as nn
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, classes))


def train_head(Xtr, ytr, Xdev, ydev, seed, device, epochs=30, lr=1e-3, patience=5, batch_size=64):
    import torch
    import torch.nn as nn
    eu.set_seed(seed)
    counts = np.bincount(ytr, minlength=4).astype(float)
    w = np.where(counts > 0, counts.sum() / (4 * np.maximum(counts, 1)), 0.0)         # same rule as classifier.compute_class_weights
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float, device=device))
    head = make_head(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    Xtr_t, ytr_t = torch.tensor(Xtr).to(device), torch.tensor(ytr).to(device)
    Xdev_t, ydev_t = torch.tensor(Xdev).to(device), torch.tensor(ydev).to(device)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(head(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            dev_loss = float(loss_fn(head(Xdev_t), ydev_t))
        if dev_loss < best:
            best, bad = dev_loss, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    return head.eval()


def predict(head, X, device):
    import torch
    with torch.no_grad():
        return head(torch.tensor(X).to(device)).argmax(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=list(ENCODERS), default="mbert")
    ap.add_argument("--sources", nargs="+", default=CORPORA, choices=CORPORA)
    ap.add_argument("--targets", nargs="+", default=CORPORA, choices=CORPORA)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--max_rows", type=int, default=0, help="smoke test: sample this many rows per file; nothing is cached")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    import torch
    device = torch.device("cuda" if (a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())) else "cpu")
    needed = {(c, s) for c in set(a.sources) | set(a.targets) for s in ("train", "dev", "test")}
    emb = {k: cached_embeddings(k[0], k[1], a.encoder, device, a.max_rows) for k in sorted(needed)}

    results = {"encoder": ENCODERS[a.encoder], "seeds": a.seeds, "device": str(device), "smoke_test": bool(a.max_rows),
               "protocol": "official dev/test files, duplicates removed (outputs/comment_level/data_report.json); frozen encoder; "
                           "head trained on source train, early stopping on source dev", "pairs": {}}
    for src in a.sources:
        Xtr, ytr = emb[(src, "train")]
        Xdv, ydv = emb[(src, "dev")]
        heads = [train_head(Xtr, ytr, Xdv, ydv, s, device, epochs=a.epochs) for s in a.seeds]
        for tgt in a.targets:
            Xte, yte = emb[(tgt, "test")]
            per_seed, report0 = [], None
            for s, h in zip(a.seeds, heads):
                rep = eu.comment_level_report(yte, predict(h, Xte, device), bootstrap=a.bootstrap if s == a.seeds[0] else 0)
                per_seed.append(rep["primary_3class"]["macro_f1"])
                report0 = report0 or rep
            maj = eu.majority_baseline(ytr, yte[np.isin(yte, eu.DETERMINATE)], eu.DETERMINATE, eu.SENTIMENT_NAMES[:3])
            results["pairs"][f"{src}->{tgt}"] = {
                "primary_3class_macro_f1_per_seed": per_seed, "mean": float(np.mean(per_seed)), "std": float(np.std(per_seed)),
                "report_seed_%d" % a.seeds[0]: report0, "majority_baseline_from_source_train_3class": maj}
            print(f"{src:10s}->{tgt:10s} macro-F1(3-class) {np.mean(per_seed):.4f} +- {np.std(per_seed):.4f}", flush=True)
    for src in a.sources:
        base = results["pairs"].get(f"{src}->{src}")
        if base:
            for tgt in a.targets:
                if tgt != src:
                    p = results["pairs"][f"{src}->{tgt}"]
                    p["relative_degradation_vs_in_domain_pct"] = float(100 * (base["mean"] - p["mean"]) / base["mean"])
    out = a.output or os.path.join(ROOT, "outputs", "comment_level", f"{a.encoder}_comment_level_results{'_smoke' if a.max_rows else ''}.json")
    eu.save_json(results, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
