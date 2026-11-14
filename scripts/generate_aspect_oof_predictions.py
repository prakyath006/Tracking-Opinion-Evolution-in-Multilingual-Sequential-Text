"""
Out-of-fold (OOF) aspect-sentiment predictions: the input features of the aspect sequence model.

The sequence model must be fed sentiment probabilities that a sentiment model produced for products it did NOT train on. This script makes them with
product-grouped cross-validation and keeps the final test products out of every model.

Protocol (all splits are by product; a product is never in two places)
  1. TEST products (--test_fraction, default 15%) are set aside first. They are never used to train, validate or choose any model here.
  2. The remaining POOL products are cut into K folds (--folds). For fold k a fresh sentiment model is trained on the pool products outside fold k
     (a share of them, --val_fraction, is held back to choose the checkpoint on validation macro-F1: the held-out fold is never used for selection),
     and it predicts ONLY the products of fold k: those are their OOF probabilities.
  3. The K fold models also predict the test products; the probabilities are averaged over the K models (none of which saw a test product).
Gold sentiment labels are never an input: prediction calls the model with token ids only, the prediction files contain no gold column, and
the sequence trainer uses gold labels only to build targets.

Layout of one run, outputs/aspect_oof/<name>/ (never overwritten):
    config/config.json  config/product_split.json            settings, annotation hashes, warnings; product -> fold / test assignment
    logs/fold_k/        train_log.jsonl, history.json         per-fold sentiment training logs
    metrics/fold_k/     best_validation.json                  per-fold validation metrics (inner validation, not the held-out fold)
    metrics/oof_quality.json                                  OOF accuracy / macro-F1 against gold: REPORTING ONLY, never an input
    checkpoints/fold_k/ best_checkpoint.pt                    only with --keep_fold_checkpoints
    predictions/oof_predictions.csv, predictions/heldout_predictions.csv   probabilities only
    summary.json

REFUSAL: if the data is too small for meaningful cross-validation (aspect_training.cv_blockers) the run stops before writing anything.
--allow_small_data overrides that for a PIPELINE CHECK only; every output is then marked research_result=false.

    python scripts/generate_aspect_oof_predictions.py --annotations data/annotations/<adjudicated>.csv --encoder xlmr --folds 5
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True

import aspect_data as ad                                              # noqa: E402
import aspect_sentiment_model as sm                                   # noqa: E402
import aspect_training as at                                          # noqa: E402
import eval_utils as eu                                               # noqa: E402
import experiment_utils as xu                                         # noqa: E402
import train_aspect_sentiment as tas                                  # noqa: E402

PROB_COLUMNS = [f"prob_{n}" for n in sm.LABELS]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    at.add_common_args(ap, with_splits=False)
    ap.set_defaults(output_root=os.path.join(ROOT, "outputs", "aspect_oof"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test_fraction", type=float, default=0.15, help="products kept out of every model, for the final sequence-model test")
    ap.add_argument("--val_fraction", type=float, default=0.15, help="share of each fold's training products used to choose its checkpoint")
    ap.add_argument("--conditioning", choices=["category", "span", "category_and_span"], default="category_and_span")
    ap.add_argument("--pooling", choices=sm.POOLING, default="cls")
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--selection_metric", choices=tas.SELECTION_METRICS, default="macro_f1")
    ap.add_argument("--keep_fold_checkpoints", action="store_true", help="keep the K fold checkpoints (large when the encoder is fine-tuned)")
    ap.add_argument("--allow_small_data", action="store_true", help="PIPELINE CHECK ONLY: run even though the data is too small; outputs are marked research_result=false")
    return ap.parse_args(argv)


def kfold_products(products, k: int, seed: int):
    """K disjoint, near-equal groups of products covering all of them; deterministic from the set of ids and the seed."""
    ids = sorted(set(products))
    if k < 2:
        raise ValueError("--folds must be at least 2")
    if len(ids) < k:
        raise ValueError(f"{len(ids)} products cannot fill {k} folds")
    order = np.random.default_rng(seed).permutation(len(ids))
    return [sorted(str(x) for x in part) for part in np.array_split(np.array([ids[i] for i in order], dtype=object), k)]


def build_items(messages, tokenizer, conditioning, max_len):
    exs = ad.build_sentiment_examples(messages, level="message_category")
    return list(zip(exs, ad.encode_sentiment_examples(exs, tokenizer, conditioning=conditioning, max_length=max_len)))


@torch.no_grad()
def predict_probs(model, items, pad_id, batch_size, device):
    """Class probabilities [n, 4] with NO labels passed to the model: the 'labels' tensor is removed before the forward call."""
    model.eval()
    out = []
    for chunk in at.iter_batches(list(items), batch_size):
        batch = at.to_device(sm.sentiment_batch([e for _, e in chunk], pad_id), device)
        batch.pop("labels")
        out.append(torch.softmax(model(**batch)["logits"], dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(sm.LABELS)))


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def prediction_rows(items, probs, **fields):
    return [{"example_id": ex.example_id, "message_id": ex.message_id, "sequence_id": ex.sequence_id, "entity_id": ex.entity_id,
             "category": ex.category, **fields, **{c: round(float(v), 8) for c, v in zip(PROB_COLUMNS, p)}} for (ex, _), p in zip(items, probs)]


def main(argv=None, *, encoder_factory=None, tokenizer=None):
    """`encoder_factory` (a function returning a fresh encoder) and `tokenizer` override the pretrained ones, for tests with tiny random models."""
    args = parse_args(argv)
    cfg = at.common_config(args)
    log = at.make_log()
    corpus = at.load_corpus(args.annotations, sequence_types=args.sequence_types, on_invalid="skip" if args.skip_invalid else "raise")
    products = sorted({m.entity_id for m in corpus.messages})
    if any(p == "" for p in products):
        raise ValueError("every message needs a product (entity_id) for product-grouped cross-validation")
    if not 0 < args.test_fraction < 1 or not 0 < args.val_fraction < 1:
        raise ValueError("--test_fraction and --val_fraction must be between 0 and 1")
    n_examples = len(ad.build_sentiment_examples(corpus.messages, level="message_category"))
    blockers = at.cv_blockers(len(products), n_examples, args.folds)
    if blockers and not args.allow_small_data:
        raise at.DataTooSmallError("the data is too small for meaningful cross-validation, so nothing was run: " + "; ".join(blockers)
                                   + ". (--allow_small_data runs a pipeline check whose outputs are marked research_result=false.)")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(xu.ENCODERS.get(args.encoder, args.encoder))
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError("a fast tokenizer is required")

    outer = ad.split_entities(products, (1 - args.test_fraction, args.test_fraction), args.split_seed, ("pool", "test"))
    pool, test = outer["pool"], outer["test"]
    folds = kfold_products(pool, args.folds, args.split_seed)
    by_product = {}
    for m in corpus.messages:
        by_product.setdefault(m.entity_id, []).append(m)
    msgs = lambda prods: [m for p in prods for m in by_product[p]]                                              # noqa: E731
    plan = []
    for k, held in enumerate(folds):
        rest = sorted(set(pool) - set(held))
        inner = ad.split_entities(rest, (1 - args.val_fraction, args.val_fraction), args.split_seed + 1 + k, ("train", "val"))
        data = {"heldout": build_items(msgs(held), tokenizer, args.conditioning, args.max_token_length),
                "train": build_items(msgs(inner["train"]), tokenizer, args.conditioning, args.max_token_length),
                "val": build_items(msgs(inner["val"]), tokenizer, args.conditioning, args.max_token_length)}
        for name in ("train", "val", "heldout"):
            if not data[name]:
                raise ValueError(f"fold {k}: the {name} part has no aspect example; use fewer folds or more data")
        assert not (set(held) & set(inner["train"])) and not (set(held) & set(inner["val"])) and not (set(inner["train"]) & set(inner["val"]))
        assert not (set(held) | set(rest)) & set(test)
        plan.append({"fold": k, "heldout": held, "train": inner["train"], "val": inner["val"], "data": data})
    test_items = build_items(msgs(test), tokenizer, args.conditioning, args.max_token_length)
    warnings = [f"too small for meaningful cross-validation: {b}" for b in blockers]
    identical = {f"fold_{f['fold']}_heldout_vs_rest": len(ad.cross_split_duplicate_texts({"heldout": msgs(f["heldout"]), "rest": msgs(f["train"] + f["val"])}))
                 for f in plan}
    identical["test_vs_pool"] = len(ad.cross_split_duplicate_texts({"test": msgs(test), "pool": msgs(pool)}))
    if any(identical.values()):
        warnings.append("identical review texts occur in different product groups " + str({k: v for k, v in identical.items() if v})
                        + ": product-level grouping does not stop copied reviews, so a small leak into the out-of-fold and test predictions remains")
    if len(corpus.annotators) <= 1:
        warnings.append("the labels come from a single annotator")
    research = not blockers

    name = args.name or at.default_name("oof", cfg, f"{args.conditioning}_{args.pooling}_k{args.folds}")
    exp = xu.prepare_experiment_dir(args.output_root, name)
    sub = {d: os.path.join(exp, d) for d in ("config", "logs", "metrics", "checkpoints", "predictions")}
    for d in sub.values():
        os.makedirs(d, exist_ok=True)
    device = at.pick_device(args.device)
    cfg.update({"name": name, "task": "aspect_oof", "labels": sm.LABELS, "level": "message_category", "folds": args.folds, "device": str(device),
                "annotation_files": corpus.paths, "annotation_sha256": sorted(sha256(p) for p in corpus.paths), "annotators": sorted(corpus.annotators),
                "identical_texts_across_groups": identical, "n_products": len(products), "n_pool_products": len(pool), "n_test_products": len(test), "n_aspect_examples": n_examples,
                "warnings": warnings, "research_result": research, "selection_uses_heldout_fold": False, "selection_uses_test": False,
                "gold_labels_are_model_input": False, "environment": xu.environment_info()})
    xu.write_json(cfg, os.path.join(sub["config"], "config.json"))
    xu.write_json({"split_seed": args.split_seed, "test_fraction": args.test_fraction, "val_fraction": args.val_fraction, "pool": pool, "test": test,
                   "folds": [{k: v for k, v in f.items() if k != "data"} for f in plan]}, os.path.join(sub["config"], "product_split.json"))
    if warnings:
        log("=" * 78)
        log("WARNING: " + ("PIPELINE CHECK ONLY, NOT A RESEARCH RESULT" if not research else "see below"))
        for w in warnings:
            log("  - " + w)
        log("=" * 78)
    log(f"{len(pool)} pool products in {args.folds} folds, {len(test)} test products, {n_examples} aspect examples, device {device}")

    pad = tokenizer.pad_token_id
    oof_rows, test_probs, fold_metrics = [], [], []
    for f in plan:
        k, d = f["fold"], f["data"]
        eu.set_seed(args.seed + k)
        enc = encoder_factory() if encoder_factory is not None else args.encoder
        weights = None
        if args.class_weights != "none":
            weights = xu.class_weight_vector([e.label_id for e, _ in d["train"]], len(sm.LABELS), args.class_weights)
        model = sm.AspectSentimentClassifier(enc, pooling=args.pooling, hidden_dim=args.hidden_dim, dropout=args.dropout, finetune=args.finetune,
                                             finetune_layers=args.finetune_layers, class_weights=weights).to(device)
        fd = {n: os.path.join(sub[n], f"fold_{k}") for n in ("logs", "metrics", "checkpoints")}
        ev = lambda m, its: tas.evaluate_sentiment(m, its, pad_id=pad, batch_size=args.batch_size, device=device)       # noqa: E731
        log(f"fold {k}: train {len(f['train'])} products / {len(d['train'])} examples, val {len(f['val'])} / {len(d['val'])}, held out {len(f['heldout'])} / {len(d['heldout'])}")
        res = at.fit(model, d["train"], d["val"], lambda c: sm.sentiment_batch([e for _, e in c], pad), ev, device=device, epochs=args.epochs,
                     batch_size=args.batch_size, accumulation=args.grad_accumulation, lr_head=args.lr_head, lr_encoder=cfg["lr_encoder"],
                     weight_decay=args.weight_decay, schedule=args.schedule, warmup_ratio=args.warmup_ratio, grad_clip=args.grad_clip,
                     selection_metric=args.selection_metric, patience=args.patience, seed=args.seed + k, exp_dir=exp, config=cfg, log=log, dirs=fd)
        saved = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
        at.load_checkpoint_state(model, saved["model_state_dict"], saved["includes_encoder"])
        p_held = predict_probs(model, d["heldout"], pad, args.batch_size, device)
        test_probs.append(predict_probs(model, test_items, pad, args.batch_size, device))
        src = f"aspect_sentiment_model, fold {k} model, trained without these products"
        oof_rows += prediction_rows(d["heldout"], p_held, role="oof", fold=k, n_models=1, source=src)
        y = [ex.label_id for ex, _ in d["heldout"]]
        m = eu.classification_metrics(y, p_held.argmax(1).tolist(), list(range(len(sm.LABELS))), sm.LABELS)
        fold_metrics.append({"fold": k, "n_examples": len(y), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "best_epoch": res["best_epoch"]})
        if not args.keep_fold_checkpoints:
            shutil.rmtree(fd["checkpoints"], ignore_errors=True)
        del model

    ens = np.mean(np.stack(test_probs), axis=0)
    held_rows = prediction_rows(test_items, ens, role="heldout_ensemble", fold="all", n_models=args.folds,
                                source=f"aspect_sentiment_model, mean of {args.folds} fold models, none of which saw these products")
    pd.DataFrame(oof_rows).to_csv(os.path.join(sub["predictions"], "oof_predictions.csv"), index=False)
    pd.DataFrame(held_rows).to_csv(os.path.join(sub["predictions"], "heldout_predictions.csv"), index=False)
    gold = {ex.example_id: ex.label_id for items in (f["data"]["heldout"] for f in plan) for ex, _ in items}
    y_all = [gold[r["example_id"]] for r in oof_rows]
    p_all = [int(np.argmax([r[c] for c in PROB_COLUMNS])) for r in oof_rows]
    quality = {"NOTE": "reporting only: gold labels are compared with out-of-fold predictions and never fed to any model",
               "oof_pool": eu.classification_metrics(y_all, p_all, list(range(len(sm.LABELS))), sm.LABELS), "folds": fold_metrics,
               "research_result": research}
    xu.write_json(quality, os.path.join(sub["metrics"], "oof_quality.json"))
    xu.write_json({"status": "completed", "name": name, "oof_rows": len(oof_rows), "heldout_rows": len(held_rows), "folds": args.folds,
                   "warnings": warnings, "research_result": research,
                   "oof_accuracy": quality["oof_pool"]["accuracy"], "oof_macro_f1": quality["oof_pool"]["macro_f1"]}, os.path.join(exp, "summary.json"))
    log(f"wrote {exp}")
    return exp


if __name__ == "__main__":
    main()
