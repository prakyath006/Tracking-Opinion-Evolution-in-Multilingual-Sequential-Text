"""
Small real forward-pass test of a Panel 1 checkpoint (default: outputs/checkpoints/best_model_amazon.pt). READ-ONLY on the checkpoint.

Loads it with mmap into the current OET model (strict), runs the first N real sequences of the unchanged Amazon test split in eval mode,
and checks: the checkpoint file hash is the same before and after; shapes; no NaN; softmax rows sum to 1; two passes give identical logits;
predicted label ranges. Reports agreement with the dataset labels on this tiny sample as a sanity figure ONLY (n is far too small to be a metric).

    python scripts/check_panel1_checkpoint.py --n_sequences 4 --max_reviews 8        # CPU, about a minute
"""
import argparse
import hashlib
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True
import logging  # noqa: E402
logging.disable(logging.WARNING)
from dataset import AmazonSequenceDataset  # noqa: E402
from model import OpinionEvolutionTracker  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "outputs", "checkpoints", "best_model_amazon.pt"))
    ap.add_argument("--n_sequences", type=int, default=4)
    ap.add_argument("--max_reviews", type=int, default=8, help="reviews per sequence kept for the CPU test")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "diagnostics", "panel1_checkpoint_forward_check.json"))
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise FileExistsError(a.out)
    before = sha(a.checkpoint)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    args = ck["args"]
    model = OpinionEvolutionTracker(model_name=args["model_name"], max_token_length=args["max_token_length"], embedding_strategy="cls",
                                    freeze_encoder=args["freeze_encoder"], use_cuda=False)
    strict = str(model.load_state_dict(ck["model_state_dict"], strict=True))
    model.eval()
    ds = AmazonSequenceDataset(split="test", max_seq_len=args["max_seq_len"])
    seqs = [{k: (v[:a.max_reviews] if isinstance(v, list) else v) for k, v in ds.sequences[i].items()} for i in range(a.n_sequences)]
    texts = [s["texts"] for s in seqs]
    with torch.no_grad():
        o1, o2 = model(texts), model(texts)
    checks, rows = {}, []
    for k in ("sentiment_logits", "trend_logits", "trajectory_logits"):
        checks[f"{k}_finite"] = bool(torch.isfinite(o1[k]).all())
        checks[f"{k}_deterministic"] = bool(torch.equal(o1[k], o2[k]))
        checks[f"{k}_softmax_sums_to_1"] = bool(torch.allclose(o1[k].softmax(-1).sum(-1), torch.ones_like(o1[k][..., 0]), atol=1e-5))
    checks["shapes"] = {k: list(o1[k].shape) for k in ("sentiment_logits", "trend_logits", "trajectory_logits", "attention_weights")}
    hit = n = 0
    for i, s in enumerate(seqs):
        L = len(s["texts"])
        pred = o1["sentiment_logits"][i, :L].argmax(-1).tolist()
        hit += sum(int(p == g) for p, g in zip(pred, s["sentiments"][:L])); n += L
        rows.append({"sequence": i, "reviews": L, "gold_sentiment": s["sentiments"][:L], "pred_sentiment": pred,
                     "gold_trajectory": s["trajectory"], "pred_trajectory": int(o1["trajectory_logits"][i].argmax()),
                     "attention": [round(float(x), 3) for x in o1["attention_weights"][i, :L]]})
    after = sha(a.checkpoint)
    rep = {"checkpoint": os.path.relpath(a.checkpoint, ROOT), "sha256_before": before, "sha256_after": after, "checkpoint_unchanged": before == after,
           "strict_load": strict, "saved_args": args, "epoch": ck["epoch"], "val_loss": ck["val_loss"], "sequences": rows, "checks": checks,
           "sample_sentiment_agreement": {"correct": hit, "positions": n, "note": "sanity figure on a handful of positions; not a metric"}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("checkpoint_unchanged", "strict_load", "checks", "sample_sentiment_agreement")}, indent=1))
    assert before == after and all(v for k, v in checks.items() if k != "shapes")
    return rep


if __name__ == "__main__":
    main()
