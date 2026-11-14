"""
Shared training and evaluation machinery for the aspect-level models (scripts/train_aspect_extractor.py, scripts/train_aspect_sentiment.py).

Reuses, without modifying: aspect_data (reading, validation, product-level splits), experiment_utils (schedulers, run directories, class
weights), eval_utils (seeding, classification metrics). The existing OET model and scripts are not touched.

Guarantees built into this module
  * Splits keep each product entirely inside one split (aspect_data.split_by_entity) and are checked for leakage.
  * Only the training split is used to fit; only the validation split is used to choose a checkpoint, early-stop or step a plateau scheduler.
    `fit` is never given test data; the test split is evaluated once, after training, with the checkpoint chosen on validation.
  * `size_warnings` states plainly when the data is too small for the numbers to mean anything.
"""
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aspect_data as ad                                            # noqa: E402
import eval_utils as eu                                             # noqa: E402
import experiment_utils as xu                                       # noqa: E402

# Heuristic floors, NOT statistical guarantees: below them a metric is anecdotal. They exist so the warning is explicit and testable.
MIN_TOTAL_PRODUCTS = 30
MIN_EVAL_PRODUCTS = 5
MIN_EVAL_ITEMS = 100             # aspect mentions (extraction) or aspect examples (sentiment) in each evaluation split
MIN_EVAL_CLASS_EXAMPLES = 20     # per sentiment class in each evaluation split
MIN_TRAIN_ITEMS = 500


# ───────────────────────────── corpus and splits ─────────────────────────────
@dataclass
class Corpus:
    messages: List[ad.Message]
    unfinished_message_ids: List[str]
    skipped_invalid_message_ids: List[str]
    annotators: Set[str]
    domains: List[str]
    paths: List[str]


def load_corpus(paths: Sequence[str], *, sequence_types: Optional[Sequence[str]] = None, on_invalid: str = "raise") -> Corpus:
    """
    Finished, valid messages from one or more annotation CSVs (read-only). A message id may occur in only one file: two annotators' copies of the
    same messages must be adjudicated into one file first, otherwise the same text would be counted twice. Unfinished messages are never used.
    """
    msgs, seen, unfinished, skipped, annotators = [], {}, [], [], set()
    v = ad._validator()
    for p in paths:
        s = ad.load_annotations(p, on_invalid=on_invalid)
        for m in s.messages:
            if m.message_id in seen:
                raise ValueError(f"message {m.message_id!r} is in both {seen[m.message_id]} and {p}; adjudicate annotator copies into one file first")
            seen[m.message_id] = p
        msgs += s.messages
        unfinished += s.unfinished_message_ids
        skipped += s.skipped_invalid_message_ids
        df = v.read_annotations(p)
        annotators |= set(df.annotator_id[(df.annotator_id != "") & df.annotation_status.isin(v.COMPLETE)])
    if sequence_types:
        msgs = [m for m in msgs if m.sequence_type in set(sequence_types)]
    if not msgs:
        raise ValueError(f"no finished, valid annotations in {list(paths)} ({len(unfinished)} messages are unfinished"
                         + (f"; none of the finished ones has sequence type {list(sequence_types)}" if sequence_types else "") + ")")
    return Corpus(msgs, sorted(unfinished), sorted(skipped), annotators, sorted({m.domain for m in msgs}), list(paths))


def make_splits(messages: Sequence[ad.Message], ratios: Sequence[float] = (0.7, 0.15, 0.15), seed: int = 0,
                names: Sequence[str] = ("train", "val", "test")) -> Dict[str, List[ad.Message]]:
    """Product-level split of messages; raises if a product would be split or a split with a positive ratio would be empty."""
    splits = ad.split_by_entity(messages, ratios, seed, names)
    ad.assert_no_entity_leakage(splits)
    return splits


def split_summary(splits: Dict[str, List[ad.Message]]) -> Dict[str, dict]:
    return {n: {"products": sorted({m.entity_id for m in ms}), "n_products": len({m.entity_id for m in ms}), "n_messages": len(ms),
                "n_mentions": sum(len(m.mentions) for m in ms),
                "messages_by_status": dict(Counter(m.status for m in ms))} for n, ms in splits.items()}


def size_warnings(task: str, stats: Dict[str, dict], *, annotators: Optional[Set[str]] = None, duplicate_texts: int = 0) -> List[str]:
    """
    Plain-language warnings when the data is too small for meaningful evaluation. stats[split] = {"products", "items", optional "class_counts"
    (sentiment class name -> count)}. Thresholds are the heuristic constants at the top of this module.
    """
    w = []
    total = sum(s["products"] for s in stats.values())
    if total < MIN_TOTAL_PRODUCTS:
        w.append(f"only {total} products in total (heuristic floor {MIN_TOTAL_PRODUCTS}): results will depend heavily on which products fall in which split")
    for name in ("val", "test"):
        s = stats.get(name)
        if not s or s["products"] == 0:
            if name == "test":
                w.append("there is no test split, so no held-out result will be reported")
            continue
        if s["products"] < MIN_EVAL_PRODUCTS:
            w.append(f"the {name} split has only {s['products']} product(s) (floor {MIN_EVAL_PRODUCTS}): its metrics are anecdotal")
        if s["items"] < MIN_EVAL_ITEMS:
            w.append(f"the {name} split has only {s['items']} {'aspect mentions' if task == 'extraction' else 'aspect examples'} (floor {MIN_EVAL_ITEMS}): "
                     "metrics are too noisy to compare models")
        if task == "sentiment" and s.get("class_counts") is not None:
            absent = [c for c, n in s["class_counts"].items() if n == 0]
            rare = [f"{c} ({n})" for c, n in s["class_counts"].items() if 0 < n < MIN_EVAL_CLASS_EXAMPLES]
            if absent:
                w.append(f"the {name} split has no example of {absent}: their F1 is 0 by definition and the fixed four-class macro-F1 is pulled down")
            if rare:
                w.append(f"the {name} split has fewer than {MIN_EVAL_CLASS_EXAMPLES} examples of {rare}: per-class F1 is unreliable")
    tr = stats.get("train", {"items": 0})
    if tr["items"] < MIN_TRAIN_ITEMS:
        w.append(f"only {tr['items']} training {'aspect mentions' if task == 'extraction' else 'aspect examples'} (floor {MIN_TRAIN_ITEMS}) "
                 "for an encoder with 100+ million parameters: expect unstable training and overfitting")
    if annotators is not None and len(annotators) <= 1:
        w.append("the labels come from a single annotator: no agreement is known, so label noise cannot be estimated")
    if duplicate_texts:
        w.append(f"{duplicate_texts} identical review text(s) occur in more than one split (different products): a small train/test leak remains")
    if w:
        w.append("treat every number from this run as a pipeline check, not as evidence about model quality")
    return w


# ───────────────────────────── metrics ─────────────────────────────
def _prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if (p + r) else 0.0, "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


def span_scores(gold: Dict[str, Sequence[tuple]], pred: Dict[str, Sequence[tuple]], *, typed: bool) -> dict:
    """
    Span-level precision, recall and F1 (micro over all messages). Spans are (start, end, category-or-None) tuples keyed by message id.
      exact_typed        start, end and category must all match (only when typed)
      exact_untyped      start and end must match
      relaxed_untyped    one-to-one matching at character overlap/union >= 0.5
    With typed spans also per-category scores and their macro F1 (over categories present in the gold or the predictions of the split).
    """
    keys = set(gold) | set(pred)
    ex_t = Counter(); ex_u = Counter(); n_gold_c, n_pred_c = Counter(), Counter()
    tp_t = tp_u = tp_r = ng = npd = 0
    v = ad._validator()
    for k in keys:
        g, p = list(gold.get(k, [])), list(pred.get(k, []))
        ng += len(g); npd += len(p)
        gt, pt = Counter(g), Counter(p)
        tp_t += sum((gt & pt).values())
        gu, pu = Counter((s[0], s[1]) for s in g), Counter((s[0], s[1]) for s in p)
        tp_u += sum((gu & pu).values())
        tp_r += len(v.match_spans([(s[0], s[1]) for s in g], [(s[0], s[1]) for s in p]))
        for s in g:
            n_gold_c[s[2]] += 1
        for s in p:
            n_pred_c[s[2]] += 1
        for c, n in (gt & pt).items():
            ex_t[c[2]] += n
    out = {"exact_untyped": _prf(tp_u, npd, ng), "relaxed_untyped": _prf(tp_r, npd, ng), "exact_typed": None, "by_category": None,
           "macro_f1_by_category": None}
    if typed:
        out["exact_typed"] = _prf(tp_t, npd, ng)
        cats = sorted(c for c in (set(n_gold_c) | set(n_pred_c)) if c is not None)
        out["by_category"] = {c: _prf(ex_t[c], n_pred_c[c], n_gold_c[c]) for c in cats}
        out["macro_f1_by_category"] = float(np.mean([out["by_category"][c]["f1"] for c in cats])) if cats else 0.0
    return out


def token_macro_f1(true: Sequence[int], pred: Sequence[int]) -> float:
    """Macro-F1 over the tags that occur in the gold or the predictions (tags that occur in neither are not counted as zeros)."""
    from sklearn.metrics import precision_recall_fscore_support
    if not len(true):
        return 0.0
    labels = sorted(set(true) | set(pred))
    return float(np.mean(precision_recall_fscore_support(true, pred, labels=labels, zero_division=0)[2]))


# ───────────────────────────── model plumbing ─────────────────────────────
def aspect_param_groups(model: torch.nn.Module, lr_encoder: float, lr_head: float, weight_decay: float) -> List[dict]:
    enc = [p for p in getattr(model, "encoder", torch.nn.Module()).parameters() if p.requires_grad]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    groups = [{"params": head, "lr": lr_head, "weight_decay": weight_decay, "name": "head"}]
    if enc:
        groups.append({"params": enc, "lr": lr_encoder, "weight_decay": weight_decay, "name": "encoder"})
    return groups


def checkpoint_state(model: torch.nn.Module) -> Tuple[Dict, bool]:
    """A frozen encoder is not saved (it is the public pretrained model); any trained encoder weight forces the whole encoder in."""
    if not hasattr(model, "encoder"):                      # a model with no pretrained encoder (the sequence model): save everything
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}, True
    trained = any(p.requires_grad for p in model.encoder.parameters())
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if trained or not k.startswith("encoder.")}, trained


def load_checkpoint_state(model: torch.nn.Module, state: Dict, includes_encoder: bool) -> None:
    res = model.load_state_dict(state, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {res.unexpected_keys[:5]}")
    bad = [k for k in res.missing_keys if includes_encoder or not k.startswith("encoder.")]
    if bad:
        raise RuntimeError(f"missing checkpoint keys: {bad[:5]}")


def to_device(batch: Dict, device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def iter_batches(items: Sequence, batch_size: int, rng: Optional[np.random.Generator] = None):
    order = rng.permutation(len(items)) if rng is not None else np.arange(len(items))
    for i in range(0, len(items), batch_size):
        yield [items[j] for j in order[i:i + batch_size]]


def write_predictions(rows: Sequence[dict], path: str) -> None:
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ───────────────────────────── training ─────────────────────────────
def fit(model, train_items: Sequence, val_items: Sequence, collate: Callable, evaluate: Callable, *, device, epochs: int, batch_size: int,
        accumulation: int, lr_head: float, lr_encoder: Optional[float], weight_decay: float, schedule: str, warmup_ratio: float, grad_clip: float,
        selection_metric: str, patience: int, seed: int, exp_dir: str, config: dict, log, step_fn: Optional[Callable] = None,
        dirs: Optional[Dict[str, str]] = None) -> dict:
    """
    Train on train_items; after each epoch evaluate on val_items ONLY and keep the checkpoint with the best `selection_metric` (higher is better).
    No other data reaches this function. evaluate(model, items) -> (summary dict of floats, full metrics, prediction rows).
    step_fn(model, batch) -> loss tensor (default: model(**batch)["loss"]).
    Writes train_log.jsonl, history.json (dirs["logs"]), best_checkpoint.pt (dirs["checkpoints"]) and best_validation.json (dirs["metrics"]);
    each defaults to exp_dir, so separate directories are optional.
    """
    step_fn = step_fn or (lambda m, b: m(**b)["loss"])
    dirs = dirs or {}
    log_dir, ckpt_dir, met_dir = (dirs.get(k, exp_dir) for k in ("logs", "checkpoints", "metrics"))
    for d in (log_dir, ckpt_dir, met_dir):
        os.makedirs(d, exist_ok=True)
    if not len(train_items):
        raise ValueError("no training examples")
    if not len(val_items):
        raise ValueError("the validation split is empty: there is nothing to select a checkpoint on")
    optimizer = torch.optim.AdamW(aspect_param_groups(model, lr_encoder or 0.0, lr_head, weight_decay))
    n_batches = math.ceil(len(train_items) / batch_size)
    scheduler, per_update = xu.make_scheduler(optimizer, schedule, xu.updates_per_epoch(n_batches, accumulation) * epochs, warmup_ratio, "max")
    rng = np.random.default_rng(seed)
    encoder = getattr(model, "encoder", None)
    encoder_trains = encoder is not None and any(p.requires_grad for p in encoder.parameters())
    params = [p for g in optimizer.param_groups for p in g["params"]]
    best, best_epoch, bad, history = None, 0, 0, []
    ckpt = os.path.join(ckpt_dir, "best_checkpoint.pt")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        if encoder is not None and not encoder_trains:
            encoder.eval()
        optimizer.zero_grad()
        total, updates, batches = 0.0, 0, list(iter_batches(train_items, batch_size, rng))
        for i, chunk in enumerate(batches):
            loss = step_fn(model, to_device(collate(chunk), device))
            (loss / accumulation).backward()
            if (i + 1) % accumulation == 0 or i + 1 == len(batches):
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step(); optimizer.zero_grad(); updates += 1
                if per_update:
                    scheduler.step()
            total += float(loss.detach())
        summary, full, _ = evaluate(model, val_items)
        if selection_metric not in summary:
            raise KeyError(f"selection metric {selection_metric!r} is not one of the validation metrics {sorted(summary)}")
        value = summary[selection_metric]
        if not per_update:
            scheduler.step(value)
        improved = xu.is_improvement(value, best, True)
        row = {"epoch": epoch, "train_loss": total / len(batches), "optimizer_updates": updates, "validation": summary,
               "selection_metric": selection_metric, "selection_value": value, "is_best": improved,
               "lr": {g.get("name", str(k)): g["lr"] for k, g in enumerate(optimizer.param_groups)}, "seconds": round(time.time() - t0, 1)}
        if improved:
            best, best_epoch, bad = value, epoch, 0
            state, includes = checkpoint_state(model)
            torch.save({"model_state_dict": state, "includes_encoder": includes, "epoch": epoch, "config": config,
                        "selection_metric": selection_metric, "selection_value": value, "validation": summary}, ckpt)
            xu.write_json(full, os.path.join(met_dir, "best_validation.json"))
        else:
            bad += 1
        history.append(row)
        with open(os.path.join(log_dir, "train_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        xu.write_json(history, os.path.join(log_dir, "history.json"))
        log(f"epoch {epoch:2d} | train loss {row['train_loss']:.4f} | " + " | ".join(f"{k} {v:.4f}" for k, v in summary.items())
            + f"{'  *' if improved else ''} | {row['seconds']}s")
        if patience and bad >= patience:
            log(f"early stop: no validation improvement for {patience} epochs")
            break
    return {"history": history, "best_epoch": best_epoch, "best_value": best, "checkpoint": ckpt}


def finish(model, fit_result: dict, evaluate: Callable, val_items: Sequence, test_items: Optional[Sequence], exp_dir: str, config: dict,
           warnings: List[str], log) -> dict:
    """
    Reload the checkpoint chosen on validation, write its validation predictions, evaluate the test split ONCE (if there is one),
    write test metrics and predictions separately, and write summary.json.
    """
    saved = torch.load(fit_result["checkpoint"], map_location="cpu", weights_only=False)
    load_checkpoint_state(model, saved["model_state_dict"], saved["includes_encoder"])
    v_summary, v_full, v_rows = evaluate(model, val_items)
    write_predictions(v_rows, os.path.join(exp_dir, "predictions", "validation_predictions.csv"))
    summary = {"status": "completed", "name": config["name"], "task": config["task"], "epochs_run": len(fit_result["history"]),
               "best_epoch": fit_result["best_epoch"], "selection_metric": config["selection_metric"], "validation": v_summary, "test": None,
               "warnings": warnings, "dataset_too_small_for_meaningful_evaluation": bool(warnings)}
    if test_items:
        t_summary, t_full, t_rows = evaluate(model, test_items)
        xu.write_json(t_full, os.path.join(exp_dir, "test_metrics.json"))
        write_predictions(t_rows, os.path.join(exp_dir, "predictions", "test_predictions.csv"))
        summary["test"] = t_summary
        log("test (checkpoint chosen on validation, epoch %d): " % fit_result["best_epoch"] + " | ".join(f"{k} {v:.4f}" for k, v in t_summary.items()))
    xu.write_json(summary, os.path.join(exp_dir, "summary.json"))
    return summary


def announce_warnings(warnings: Sequence[str], log) -> None:
    if warnings:
        log("=" * 78)
        log("WARNING: THE DATASET IS TOO SMALL FOR A MEANINGFUL EVALUATION")
        for w in warnings:
            log("  - " + w)
        log("=" * 78)


# ───────────────────────────── command line and run setup shared by both scripts ─────────────────────────────
def add_common_args(ap, with_splits: bool = True) -> None:
    ap.add_argument("--annotations", nargs="+", required=True, help="completed annotation CSV file(s); finished and valid messages only")
    ap.add_argument("--sequence_types", nargs="+", default=None, help="keep only these sequence types (default: all in the files)")
    ap.add_argument("--skip_invalid", action="store_true", help="drop messages that fail validation instead of stopping")
    ap.add_argument("--encoder", default="mbert", help="mbert, xlmr, or a Hugging Face model name")
    ap.add_argument("--finetune", choices=xu.FINETUNE_MODES, default="full")
    ap.add_argument("--finetune_layers", type=int, default=2, help="top layers to train when --finetune partial")
    ap.add_argument("--max_token_length", type=int, default=128)
    ap.add_argument("--class_weights", choices=xu.WEIGHT_MODES, default="none", help="from the training split only")
    ap.add_argument("--seed", type=int, default=0, help="training seed")
    ap.add_argument("--split_seed", type=int, default=0, help="product split seed; keep it fixed to compare runs on identical splits")
    if with_splits:
        ap.add_argument("--split_ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accumulation", type=int, default=1)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_encoder", type=float, default=None, help="default 2e-5 when the encoder trains")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--schedule", choices=xu.SCHEDULES, default="linear")
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=4, help="epochs without validation improvement; 0 disables early stopping")
    if with_splits:
        ap.add_argument("--no_test", action="store_true", help="make the same splits but do not evaluate the test split")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output_root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "aspect_experiments"))
    ap.add_argument("--name", default=None, help="run directory name (default: built from the settings)")


def common_config(args) -> dict:
    c = vars(args).copy()
    c["lr_encoder_explicit"] = args.lr_encoder is not None
    if args.finetune == "frozen" and args.lr_encoder is not None:
        raise ValueError("--lr_encoder was given but the encoder is frozen")
    if args.finetune != "frozen" and args.lr_encoder is None:
        c["lr_encoder"] = 2e-5
    if args.finetune == "partial" and args.finetune_layers < 1:
        raise ValueError("--finetune partial needs --finetune_layers >= 1")
    if min(args.epochs, args.batch_size, args.grad_accumulation) < 1:
        raise ValueError("epochs, batch_size and grad_accumulation must be at least 1")
    if not 8 <= args.max_token_length <= 512:
        raise ValueError("--max_token_length must be between 8 and 512")
    ratios = getattr(args, "split_ratios", None)
    if ratios is not None and (abs(sum(ratios) - 1.0) > 1e-6 or any(r < 0 for r in ratios) or ratios[0] <= 0 or ratios[1] <= 0):
        raise ValueError("--split_ratios must sum to 1 and give train and val a positive share")
    return c


def pick_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no GPU is available")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def make_log() -> Callable[[str], None]:
    def log(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)
    return log


def default_name(task: str, c: dict, extra: str) -> str:
    ft = {"frozen": "frozen", "partial": f"top{c['finetune_layers']}", "full": "full"}[c["finetune"]]
    return f"{task}_{c['encoder']}_{ft}_{extra}_len{c['max_token_length']}_s{c['seed']}"


# ───────────────────────────── refusing meaningless runs; grouped bootstrap ─────────────────────────────
MIN_PRODUCTS_FOR_CV = 30         # products in the whole corpus before out-of-fold generation is meaningful
MIN_PRODUCTS_PER_FOLD = 3
MIN_EXAMPLES_FOR_CV = 300        # aspect examples in the whole corpus
MIN_TRAIN_SERIES = 50            # aspect series with at least two ranked events, per split
MIN_EVAL_SERIES = 20


class DataTooSmallError(ValueError):
    """The data cannot support the requested cross-validation or training, and `--allow_small_data` was not given."""


def cv_blockers(n_products: int, n_examples: int, folds: int) -> List[str]:
    """Reasons out-of-fold generation would be meaningless (heuristic floors, see the constants above); empty means it may go ahead."""
    b = []
    if n_products < MIN_PRODUCTS_FOR_CV:
        b.append(f"{n_products} products in total (floor {MIN_PRODUCTS_FOR_CV})")
    if n_products // max(folds, 1) < MIN_PRODUCTS_PER_FOLD:
        b.append(f"about {n_products // max(folds, 1)} products per fold with {folds} folds (floor {MIN_PRODUCTS_PER_FOLD})")
    if n_examples < MIN_EXAMPLES_FOR_CV:
        b.append(f"{n_examples} aspect examples in total (floor {MIN_EXAMPLES_FOR_CV})")
    return b


def series_blockers(counts: Dict[str, int], test_expected: bool = True) -> List[str]:
    """Reasons sequence training would give misleading metrics. counts = usable aspect series (two or more ranked events) per split."""
    b = []
    if counts.get("train", 0) < MIN_TRAIN_SERIES:
        b.append(f"{counts.get('train', 0)} usable training series (floor {MIN_TRAIN_SERIES})")
    for name in ("val",) + (("test",) if test_expected else ()):
        if counts.get(name, 0) < MIN_EVAL_SERIES:
            b.append(f"{counts.get(name, 0)} usable {name} series (floor {MIN_EVAL_SERIES})")
    return b


def grouped_paired_bootstrap(y: Sequence[int], pred_a: Sequence[int], pred_b: Sequence[int], groups: Sequence[str], labels: Sequence[int],
                             n_resamples: int = 1000, seed: int = 0) -> Optional[dict]:
    """macro-F1(a) - macro-F1(b), resampling whole groups (products) with the same resample for both. None with fewer than two groups."""
    y, a, b, g = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b), np.asarray(groups)
    uniq = np.unique(g)
    if len(uniq) < 2:
        return None
    K = len(labels)
    assert list(labels) == list(range(K)), "labels must be 0..K-1"

    def f(t, q):                                    # macro-F1 over the fixed classes, zero for a class with no support and no prediction (as sklearn)
        t, q = np.asarray(t), np.asarray(q)
        tp = np.bincount(t[t == q], minlength=K)
        den = np.bincount(t, minlength=K) + np.bincount(q, minlength=K)
        return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))
    idx = [np.flatnonzero(g == u) for u in uniq]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])
        diffs.append(f(y[pick], a[pick]) - f(y[pick], b[pick]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"estimate": float(f(y, a) - f(y, b)), "ci_low": float(lo), "ci_high": float(hi), "resamples": n_resamples, "n_groups": int(len(uniq))}
