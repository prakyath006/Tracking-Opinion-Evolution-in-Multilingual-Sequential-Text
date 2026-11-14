"""
Train the aspect extractor (BIO span tagging) on manually annotated aspect data.

New entry point; the existing OET scripts, model and checkpoints are not touched. Needs FINISHED annotations (docs/aspect_annotation_guidelines.md).
Each run writes a new directory outputs/aspect_experiments/<name>/ and refuses to reuse an existing one:
    config.json  splits.json  train_log.jsonl  history.json  best_checkpoint.pt  best_validation.json  test_metrics.json  summary.json
    predictions/validation_predictions.csv  predictions/test_predictions.csv

Protocol: products are split into train / val / test (a product is never split); only train is fitted, only val picks the checkpoint
(--selection_metric, default macro span F1: the mean of the per-category span F1 with category tags, the span F1 with generic tags);
the test split is evaluated once at the end with that checkpoint. Reported: span precision, recall and F1 (exact boundaries, plus a relaxed
overlap >= 0.5 variant and, with category tags, per-category scores), token-level macro-F1.
A banner is printed, and stored in summary.json, when the data is too small for the numbers to mean anything.

    python scripts/train_aspect_extractor.py --annotations data/annotations/<adjudicated>.csv --encoder xlmr --finetune full --seed 0
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True

import aspect_data as ad                                              # noqa: E402
import aspect_extractor as ax                                         # noqa: E402
import aspect_training as at                                          # noqa: E402
import eval_utils as eu                                               # noqa: E402
import experiment_utils as xu                                         # noqa: E402

SELECTION_METRICS = ("macro_span_f1", "span_f1", "token_macro_f1")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    at.add_common_args(ap)
    ap.add_argument("--category_tags", action="store_true", help="tag B-<category>/I-<category> instead of the generic B-ASP/I-ASP")
    ap.add_argument("--unclear", choices=["exclude", "mask"], default="exclude", help="messages with an UNCLEAR_ASPECT: drop, or keep with every label ignored")
    ap.add_argument("--overlap", choices=["error", "merge"], default="error", help="two annotated spans sharing a token (merge: generic tags only)")
    ap.add_argument("--selection_metric", choices=SELECTION_METRICS, default="macro_span_f1")
    return ap.parse_args(argv)


def gold_and_predicted_spans(ex, pred_ids, tagset, typed):
    e = ex.encoding
    gold_ids = [0 if l == ad.IGNORE_INDEX else l for l in e.labels]
    conv = lambda spans: [(s["start"], s["end"], s["category"] if typed else None) for s in spans]            # noqa: E731
    gold = ax.bio_to_spans(gold_ids, e.offsets, e.special_tokens_mask, tagset, ex.text)
    pred = ax.bio_to_spans(pred_ids, e.offsets, e.special_tokens_mask, tagset, ex.text)
    return conv(gold), conv(pred)


@torch.no_grad()
def evaluate_extractor(model, examples, *, tagset, typed, pad_id, batch_size, device, split="val"):
    """(summary, full metrics, per-message prediction rows). Messages with an unclear aspect (masked) are not scored."""
    model.eval()
    gold, pred, rows, tok_true, tok_pred = {}, {}, [], [], []
    loss_sum, n = 0.0, 0
    for chunk in at.iter_batches(list(examples), batch_size):
        batch = at.to_device(ax.extraction_batch(chunk, pad_id), device)
        out = model(**batch)
        loss_sum += float(out["loss"]) * len(chunk); n += len(chunk)
        ids = out["logits"].argmax(-1).cpu()
        for i, ex in enumerate(chunk):
            if ex.status == "UNCLEAR_ASPECT":
                continue
            k = len(ex.encoding.input_ids)
            p_ids = ids[i, :k].tolist()
            g, p = gold_and_predicted_spans(ex, p_ids, tagset, typed)
            gold[ex.message_id], pred[ex.message_id] = g, p
            for lab, pid in zip(ex.encoding.labels, p_ids):
                if lab != ad.IGNORE_INDEX:
                    tok_true.append(lab); tok_pred.append(pid)
            rows.append({"split": split, "message_id": ex.message_id, "sequence_id": ex.sequence_id, "entity_id": ex.entity_id, "status": ex.status,
                         "n_gold": len(g), "n_pred": len(p), "n_exact_untyped": len(set((a, b) for a, b, _ in g) & set((a, b) for a, b, _ in p)),
                         "gold_spans": str([list(s) for s in g]), "pred_spans": str([list(s) for s in p])})
    sc = at.span_scores(gold, pred, typed=typed)
    main = sc["exact_typed"] if typed else sc["exact_untyped"]
    summary = {"span_precision": main["precision"], "span_recall": main["recall"], "span_f1": main["f1"],
               "macro_span_f1": sc["macro_f1_by_category"] if typed else main["f1"], "relaxed_span_f1": sc["relaxed_untyped"]["f1"],
               "token_macro_f1": at.token_macro_f1(tok_true, tok_pred), "loss": loss_sum / max(1, n)}
    full = {"summary": summary, "span_scores": sc, "typed": typed, "n_messages_scored": len(rows), "tagset": list(tagset)}
    return summary, full, rows


def main(argv=None, *, encoder=None, tokenizer=None):
    """`encoder` / `tokenizer` override the pretrained ones (used by the tests with tiny random models)."""
    args = parse_args(argv)
    cfg = at.common_config(args)
    log = at.make_log()
    corpus = at.load_corpus(args.annotations, sequence_types=args.sequence_types, on_invalid="skip" if args.skip_invalid else "raise")
    if args.category_tags and len(corpus.domains) != 1:
        raise ValueError(f"category tags need one domain, the files contain {corpus.domains}")
    categories = ad.category_ids(corpus.domains[0]) if args.category_tags else None
    tagset = ad.bio_tagset(categories)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(xu.ENCODERS.get(args.encoder, args.encoder))
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError("a fast tokenizer is required")
    splits = at.make_splits(corpus.messages, args.split_ratios, args.split_seed)
    examples, reports = {}, {}
    for name, msgs in splits.items():
        examples[name], reports[name] = ad.build_extraction_examples(msgs, tokenizer, max_length=args.max_token_length, categories=categories,
                                                                       unclear=args.unclear, overlap=args.overlap)
    summ = at.split_summary(splits)
    mentions = {n: sum(len(m.mentions) for m in splits[n] if m.status == "ASPECT_PRESENT") for n in splits}
    if mentions["val"] == 0:
        raise ValueError("the validation split contains no aspect mention: a checkpoint cannot be chosen on span F1")
    if not examples["train"]:
        raise ValueError("the training split has no usable message")
    stats = {n: {"products": summ[n]["n_products"], "items": mentions[n]} for n in splits}
    warnings = at.size_warnings("extraction", stats, annotators=corpus.annotators,
                                duplicate_texts=len(ad.cross_split_duplicate_texts(splits)))
    name = args.name or at.default_name("extractor", cfg, "categories" if args.category_tags else "generic")
    exp_dir = xu.prepare_experiment_dir(args.output_root, name)
    device = at.pick_device(args.device)
    eu.set_seed(args.seed)
    weights = None
    if args.class_weights != "none":
        labels = [l for ex in examples["train"] for l in ex.encoding.labels if l != ad.IGNORE_INDEX]
        weights = xu.class_weight_vector(labels, len(tagset), args.class_weights)
    model = ax.AspectExtractor(encoder if encoder is not None else args.encoder, tagset, finetune=args.finetune,
                               finetune_layers=args.finetune_layers, class_weights=weights).to(device)
    counts = model.parameter_counts()
    cfg.update({"name": name, "task": "extraction", "tagset": tagset, "category_tags": bool(args.category_tags), "domains": corpus.domains,
                "parameters": counts, "device": str(device), "selection_uses_test": False, "environment": xu.environment_info(),
                "annotation_files": corpus.paths, "unfinished_messages_excluded": len(corpus.unfinished_message_ids),
                "annotators": sorted(corpus.annotators), "extraction_reports": reports, "warnings": warnings})
    xu.write_json(cfg, os.path.join(exp_dir, "config.json"))
    xu.write_json({"split_seed": args.split_seed, "ratios": list(args.split_ratios), "splits": summ,
                   "identical_texts_across_splits": len(ad.cross_split_duplicate_texts(splits))}, os.path.join(exp_dir, "splits.json"))
    log(f"run {name} on {device}; trainable parameters {counts['trainable']:,} of {counts['total']:,}")
    log("products per split: " + ", ".join(f"{n} {summ[n]['n_products']}" for n in splits) + "; aspect mentions: " + ", ".join(f"{n} {mentions[n]}" for n in splits))
    at.announce_warnings(warnings, log)
    pad = tokenizer.pad_token_id
    ev = lambda m, items: evaluate_extractor(m, items, tagset=tagset, typed=bool(args.category_tags), pad_id=pad,            # noqa: E731
                                             batch_size=args.batch_size, device=device)
    res = at.fit(model, examples["train"], examples["val"], lambda c: ax.extraction_batch(c, pad), ev, device=device, epochs=args.epochs,
                 batch_size=args.batch_size, accumulation=args.grad_accumulation, lr_head=args.lr_head, lr_encoder=cfg["lr_encoder"],
                 weight_decay=args.weight_decay, schedule=args.schedule, warmup_ratio=args.warmup_ratio, grad_clip=args.grad_clip,
                 selection_metric=args.selection_metric, patience=args.patience, seed=args.seed, exp_dir=exp_dir, config=cfg, log=log)
    at.finish(model, res, ev, examples["val"], None if args.no_test else examples["test"], exp_dir, cfg, warnings, log)
    log(f"wrote {exp_dir}")
    return exp_dir


if __name__ == "__main__":
    main()
