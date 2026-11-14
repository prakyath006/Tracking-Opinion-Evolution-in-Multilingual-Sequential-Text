"""
Train the aspect-conditioned sentiment classifier (POSITIVE / NEGATIVE / NEUTRAL / MIXED toward ONE aspect) on manually annotated aspect data.

New entry point; the existing OET scripts, model and checkpoints are not touched. Needs FINISHED annotations (docs/aspect_annotation_guidelines.md).
The labels are the annotators' per-aspect labels (one value per message and category, combined by the guideline rule; or per mention with
--level mention). Nothing is derived from review ratings or from OET.

Each run writes a new directory outputs/aspect_experiments/<name>/ and refuses to reuse an existing one:
    config.json  splits.json  class_weights.json  train_log.jsonl  history.json  best_checkpoint.pt  best_validation.json  test_metrics.json  summary.json
    predictions/validation_predictions.csv  predictions/test_predictions.csv     (one row per example, with the four class probabilities)

Protocol: products are split into train / val / test (a product is never split); only train is fitted, only val picks the checkpoint
(--selection_metric, default macro-F1 over the four classes); the test split is evaluated once at the end with that checkpoint.
Reported: accuracy, macro-F1 (fixed four classes, and over the classes that occur), per-class precision/recall/F1 and the confusion matrix.
A banner is printed, and stored in summary.json, when the data is too small for the numbers to mean anything.

    python scripts/train_aspect_sentiment.py --annotations data/annotations/<adjudicated>.csv --encoder xlmr --finetune full --seed 0
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True

import aspect_data as ad                                              # noqa: E402
import aspect_sentiment_model as sm                                   # noqa: E402
import aspect_training as at                                          # noqa: E402
import eval_utils as eu                                               # noqa: E402
import experiment_utils as xu                                         # noqa: E402

SELECTION_METRICS = ("macro_f1", "macro_f1_present_classes")
LABEL_IDS = list(range(len(sm.LABELS)))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    at.add_common_args(ap)
    ap.add_argument("--level", choices=["message_category", "mention"], default="message_category")
    ap.add_argument("--conditioning", choices=["category", "span", "category_and_span"], default="category_and_span")
    ap.add_argument("--pooling", choices=sm.POOLING, default="cls")
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--selection_metric", choices=SELECTION_METRICS, default="macro_f1")
    return ap.parse_args(argv)


@torch.no_grad()
def evaluate_sentiment(model, items, *, pad_id, batch_size, device, split="val"):
    """items = [(SentimentExample, encoded dict)]. Returns (summary, full metrics, per-example rows with class probabilities)."""
    model.eval()
    y_true, y_pred, rows = [], [], []
    loss_sum = 0.0
    for chunk in at.iter_batches(list(items), batch_size):
        batch = at.to_device(sm.sentiment_batch([e for _, e in chunk], pad_id), device)
        out = model(**batch)
        loss_sum += float(out["loss"]) * len(chunk)
        probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
        for (ex, _), p in zip(chunk, probs):
            pred = int(p.argmax())
            y_true.append(ex.label_id); y_pred.append(pred)
            rows.append({"split": split, "example_id": ex.example_id, "message_id": ex.message_id, "sequence_id": ex.sequence_id,
                         "entity_id": ex.entity_id, "category": ex.category, "gold": ex.label, "pred": sm.LABELS[pred],
                         "correct": int(pred == ex.label_id), **{f"prob_{n}": round(float(v), 6) for n, v in zip(sm.LABELS, p)}})
    full = eu.classification_metrics(y_true, y_pred, LABEL_IDS, sm.LABELS)
    present = [c["f1"] for c in full["per_class"].values() if c["support"] > 0]
    summary = {"accuracy": full["accuracy"], "macro_f1": full["macro_f1"],
               "macro_f1_present_classes": float(np.mean(present)) if present else 0.0, "loss": loss_sum / max(1, len(items))}
    full["macro_f1_present_classes"] = summary["macro_f1_present_classes"]
    full["loss"] = summary["loss"]
    return summary, full, rows


def main(argv=None, *, encoder=None, tokenizer=None):
    """`encoder` / `tokenizer` override the pretrained ones (used by the tests with tiny random models)."""
    args = parse_args(argv)
    cfg = at.common_config(args)
    log = at.make_log()
    corpus = at.load_corpus(args.annotations, sequence_types=args.sequence_types, on_invalid="skip" if args.skip_invalid else "raise")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(xu.ENCODERS.get(args.encoder, args.encoder))
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError("a fast tokenizer is required")
    splits = at.make_splits(corpus.messages, args.split_ratios, args.split_seed)
    items, counts = {}, {}
    for name, msgs in splits.items():
        exs = ad.build_sentiment_examples(msgs, level=args.level)
        enc = ad.encode_sentiment_examples(exs, tokenizer, conditioning=args.conditioning, max_length=args.max_token_length)
        items[name] = list(zip(exs, enc))
        counts[name] = {c: sum(e.label == c for e in exs) for c in sm.LABELS}
    if not items["val"]:
        raise ValueError("the validation split contains no aspect example: a checkpoint cannot be chosen")
    if not items["train"]:
        raise ValueError("the training split contains no aspect example")
    summ = at.split_summary(splits)
    stats = {n: {"products": summ[n]["n_products"], "items": len(items[n]), "class_counts": counts[n]} for n in splits}
    warnings = at.size_warnings("sentiment", stats, annotators=corpus.annotators, duplicate_texts=len(ad.cross_split_duplicate_texts(splits)))
    name = args.name or at.default_name("sentiment", cfg, f"{args.conditioning}_{args.pooling}")
    exp_dir = xu.prepare_experiment_dir(args.output_root, name)
    device = at.pick_device(args.device)
    eu.set_seed(args.seed)
    weights = None
    if args.class_weights != "none":
        weights = xu.class_weight_vector([e.label_id for e, _ in items["train"]], len(sm.LABELS), args.class_weights)
    model = sm.AspectSentimentClassifier(encoder if encoder is not None else args.encoder, pooling=args.pooling, hidden_dim=args.hidden_dim,
                                         dropout=args.dropout, finetune=args.finetune, finetune_layers=args.finetune_layers,
                                         class_weights=weights).to(device)
    pc = model.parameter_counts()
    cfg.update({"name": name, "task": "sentiment", "labels": sm.LABELS, "domains": corpus.domains, "parameters": pc, "device": str(device),
                "selection_uses_test": False, "environment": xu.environment_info(), "annotation_files": corpus.paths,
                "unfinished_messages_excluded": len(corpus.unfinished_message_ids), "annotators": sorted(corpus.annotators),
                "class_counts": counts, "warnings": warnings})
    xu.write_json(cfg, os.path.join(exp_dir, "config.json"))
    xu.write_json({"split_seed": args.split_seed, "ratios": list(args.split_ratios), "splits": summ, "class_counts": counts,
                   "identical_texts_across_splits": len(ad.cross_split_duplicate_texts(splits))}, os.path.join(exp_dir, "splits.json"))
    xu.write_json({"mode": args.class_weights, "training_class_counts": counts["train"],
                   "weights": weights.tolist() if weights is not None else None}, os.path.join(exp_dir, "class_weights.json"))
    log(f"run {name} on {device}; trainable parameters {pc['trainable']:,} of {pc['total']:,}")
    log("products per split: " + ", ".join(f"{n} {summ[n]['n_products']}" for n in splits) + "; aspect examples: " + ", ".join(f"{n} {len(items[n])}" for n in splits))
    at.announce_warnings(warnings, log)
    pad = tokenizer.pad_token_id
    ev = lambda m, its: evaluate_sentiment(m, its, pad_id=pad, batch_size=args.batch_size, device=device)               # noqa: E731
    res = at.fit(model, items["train"], items["val"], lambda c: sm.sentiment_batch([e for _, e in c], pad), ev, device=device, epochs=args.epochs,
                 batch_size=args.batch_size, accumulation=args.grad_accumulation, lr_head=args.lr_head, lr_encoder=cfg["lr_encoder"],
                 weight_decay=args.weight_decay, schedule=args.schedule, warmup_ratio=args.warmup_ratio, grad_clip=args.grad_clip,
                 selection_metric=args.selection_metric, patience=args.patience, seed=args.seed, exp_dir=exp_dir, config=cfg, log=log)
    at.finish(model, res, ev, items["val"], None if args.no_test else items["test"], exp_dir, cfg, warnings, log)
    log(f"wrote {exp_dir}")
    return exp_dir


if __name__ == "__main__":
    main()
