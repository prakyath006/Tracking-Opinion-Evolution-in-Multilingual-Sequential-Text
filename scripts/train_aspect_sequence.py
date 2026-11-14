"""
Train the aspect sequence model (transition and trajectory heads) on out-of-fold aspect-sentiment probabilities, and compare it with the
rule-derived baseline.

Needs (1) finished annotations and (2) an out-of-fold run made from THE SAME annotations by scripts/generate_aspect_oof_predictions.py.

What the model sees: per event of one aspect series (one aspect of one product, in time order) the sentiment probabilities PREDICTED for products the
sentiment model did not train on, log time gaps, and the aspect category. Gold sentiment labels are used only to build the targets:
transition (previous ranked event -> this event) and trajectory (whole series), by the approved rules (docs/aspect_annotation_guidelines.md section 11
and the ontology's TransitionType / TrajectoryType; NEUTRAL events are skipped, a series needs two ranked events to have targets).

Splits (by product, never leaking): the TEST products are the ones the OOF run kept out of every sentiment model. The OOF pool is cut into
train / val (--val_ratio) here. Only train is fitted; only val selects the checkpoint; test is evaluated once at the end.
Known residual: the validation products' labels helped train the sentiment models that produced the training products' features (standard stacking);
the test products' labels reached no model at all.

Baselines on the same series: the RULE baseline (the ontology rule applied to the argmax of the same OOF probabilities: what the learned model must beat)
and the MAJORITY baseline (most frequent training target). Reported for model and baselines: accuracy, macro-F1, per-class F1, confusion matrix, for
trajectory (per series) and transition (per event), plus a product-grouped paired bootstrap of the model-minus-rule macro-F1 difference.

REFUSAL: if there are too few usable series (aspect_training.series_blockers), or the OOF run was itself a pipeline check, the script stops before writing
anything. --allow_small_data overrides that for a PIPELINE CHECK only; every output is then marked research_result=false.

Layout of a run, outputs/aspect_sequence/<name>/ (never overwritten): config/, logs/, checkpoints/, metrics/, predictions/, summary.json.

    python scripts/train_aspect_sequence.py --annotations data/annotations/<adjudicated>.csv --oof_dir outputs/aspect_oof/<run>
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.dont_write_bytecode = True

import aspect_data as ad                                              # noqa: E402
import aspect_sequence_model as sq                                    # noqa: E402
import aspect_training as at                                          # noqa: E402
import eval_utils as eu                                               # noqa: E402
import experiment_utils as xu                                         # noqa: E402
import generate_aspect_oof_predictions as oof                         # noqa: E402
from ontology import SentimentState, TrajectoryType, TransitionType   # noqa: E402

SELECTION_METRICS = ("trajectory_macro_f1", "transition_macro_f1", "mean_macro_f1")
TRAJ_LABELS, TRANS_LABELS = list(range(len(sq.TRAJECTORIES))), list(range(len(sq.TRANSITIONS)))
RANKED = {"POSITIVE", "NEGATIVE", "MIXED"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", nargs="+", required=True, help="the same annotation file(s) the OOF run was made from")
    ap.add_argument("--oof_dir", required=True, help="a run directory of generate_aspect_oof_predictions.py")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="share of the OOF pool products used as validation")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accumulation", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--schedule", choices=xu.SCHEDULES, default="constant")
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lstm_layers", type=int, default=2)
    ap.add_argument("--attention_dim", type=int, default=64)
    ap.add_argument("--head_hidden", type=int, default=64)
    ap.add_argument("--category_dim", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--transition_weight", type=float, default=0.5)
    ap.add_argument("--trajectory_weight", type=float, default=1.0)
    ap.add_argument("--class_weights", choices=xu.WEIGHT_MODES, default="none", help="from the training series only")
    ap.add_argument("--selection_metric", choices=SELECTION_METRICS, default="trajectory_macro_f1")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output_root", default=os.path.join(ROOT, "outputs", "aspect_sequence"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--allow_small_data", action="store_true", help="PIPELINE CHECK ONLY: run with too little data; outputs are marked research_result=false")
    return ap.parse_args(argv)


# ───────────────────────────── inputs: OOF probabilities and series ─────────────────────────────
def verify_provenance(split, oof_df, held_df):
    """
    Check the recorded provenance of every probability before it is used. A product's OOF probability must come from the fold that held it out,
    that fold's training and validation products must not include it, and no test product may appear in any fold or in the OOF file.
    """
    pool, test = set(split["pool"]), set(split["test"])
    if pool & test:
        raise sq.LeakageError("a product is both in the pool and in the test set")
    fold_of = {}
    for f in split["folds"]:
        held, train, val = set(f["heldout"]), set(f["train"]), set(f["val"])
        if held & (train | val):
            raise sq.LeakageError(f"fold {f['fold']}: a held-out product is also in that fold's training or validation products")
        if (held | train | val) & test:
            raise sq.LeakageError(f"fold {f['fold']}: a test product appears in the fold")
        for p in held:
            if fold_of.setdefault(p, f["fold"]) != f["fold"]:
                raise sq.LeakageError(f"product {p} is held out in more than one fold")
    if set(fold_of) != pool:
        raise sq.LeakageError("the folds do not cover exactly the pool products")
    bad = [r.example_id for r in oof_df.itertuples() if r.role != "oof" or fold_of.get(r.entity_id) != r.fold]
    if bad:
        raise sq.LeakageError(f"out-of-fold rows whose fold is not the one that held the product out, e.g. {bad[:3]}")
    bad = [r.example_id for r in held_df.itertuples() if r.role != "heldout_ensemble" or r.entity_id not in test]
    if bad:
        raise sq.LeakageError(f"held-out ensemble rows for products that are not test products, e.g. {bad[:3]}")


def read_oof(oof_dir):
    cfg = json.load(open(os.path.join(oof_dir, "config", "config.json"), encoding="utf-8"))
    split = json.load(open(os.path.join(oof_dir, "config", "product_split.json"), encoding="utf-8"))
    probs, roles = {}, {}
    frames = {}
    for f in ("oof_predictions.csv", "heldout_predictions.csv"):
        df = pd.read_csv(os.path.join(oof_dir, "predictions", f))
        frames[f] = df
        if any("gold" in c or c in ("label", "label_id") for c in df.columns):
            raise sq.LeakageError(f"{f} contains a gold-label column; sentiment inputs must be model predictions only")
        for r in df.itertuples():
            if r.example_id in probs:
                raise ValueError(f"example {r.example_id!r} appears twice in the OOF run")
            probs[r.example_id] = [getattr(r, c) for c in oof.PROB_COLUMNS]
            roles[r.example_id] = r.role
    verify_provenance(split, frames["oof_predictions.csv"], frames["heldout_predictions.csv"])
    return cfg, split, probs, roles


def series_for_split(messages, probs, source):
    """Events (predicted probabilities as inputs, gold label only as target) and their series, for the messages of one split."""
    examples = ad.build_sentiment_examples(messages, level="message_category")
    return sq.build_series(sq.events_from_examples(examples, probs, source=source)), examples


def usable_series(series):
    """Series with targets: two or more ranked (POSITIVE / MIXED / NEGATIVE) gold events. The rest are 'insufficient observations'."""
    keep = [s for s in series if sq.series_targets(s)[1] != sq.IGNORE]
    return keep, len(series) - len(keep)


# ───────────────────────────── baselines ─────────────────────────────
def rule_predictions(series):
    """
    The ontology rule on the PREDICTED sentiment of each event (argmax of the same OOF probabilities the model sees).
    Trajectory: TrajectoryType.compute over the predicted-ranked events (STABLE, as the ontology says, when fewer than two).
    Transition at event i: from the previous predicted-ranked event to i; STABLE ("no evidence of change") when event i is not predicted-ranked or
    has no predicted-ranked predecessor. Returns (trajectory id, transition ids per event, trajectory_fallback, transition_fallback flags per event).
    """
    labels = [ad.AspectSentiment(int(np.argmax(e.sentiment_probs))).name for e in series.events]
    ranked = [i for i, l in enumerate(labels) if l in RANKED]
    states = [SentimentState[labels[i]] for i in ranked]
    trans = [TransitionType.STABLE.value] * len(labels)
    fallback = [True] * len(labels)
    for (_, prev), (i, cur) in zip(zip(ranked, states), zip(ranked[1:], states[1:])):
        trans[i] = TransitionType.compute(prev, cur).value
        fallback[i] = False
    return TrajectoryType.compute(states).value, trans, len(states) < 2, fallback


def majority_labels(train_series):
    """Most frequent defined training target for trajectory and for transitions (a valid majority baseline: training labels only)."""
    traj, trans = Counter(), Counter()
    for s in train_series:
        t, j = sq.series_targets(s)
        traj.update([j] if j != -100 else [])
        trans.update(x for x in t if x != -100)
    return (traj.most_common(1)[0][0] if traj else 0), (trans.most_common(1)[0][0] if trans else 0)


# ───────────────────────────── model forward, prediction and metrics ─────────────────────────────
def forward(model, b):
    return model(b["sentiment_probs"], b["time_features"], b["seq_lens"], b["category_ids"], b.get("text_features"))


@torch.no_grad()
def predict_series(model, series, categories, device, batch_size, loss_fn):
    model.eval()
    preds, loss_sum = [], 0.0
    for chunk in at.iter_batches(list(series), batch_size):
        b = at.to_device(sq.collate_series(chunk, categories), device)
        o = forward(model, b)
        loss_sum += float(loss_fn(o, b)["total_loss"]) * len(chunk)
        tr, tj = o["transition_logits"].argmax(-1).cpu().tolist(), o["trajectory_logits"].argmax(-1).cpu().tolist()
        preds += [{"trajectory": tj[i], "transitions": tr[i][:len(s.events)]} for i, s in enumerate(chunk)]
    return preds, loss_sum / max(1, len(series))


def block(y, p, labels, names):
    m = eu.classification_metrics(y, p, labels, names)
    present = [c["f1"] for c in m.get("per_class", {}).values() if c["support"] > 0]
    m["macro_f1_present_classes"] = float(np.mean(present)) if present else 0.0
    return m


def arrays(series, model_preds, majority):
    """Gold and each method's predictions: trajectory per usable series, transitions per event whose gold target is defined."""
    maj_traj, maj_trans = majority
    t = {"y": [], "model": [], "rule": [], "majority": [], "group": []}
    e = {"y": [], "model": [], "rule": [], "majority": [], "group": []}
    rule_traj_fb = rule_trans_fb = 0
    for s, p in zip(series, model_preds):
        gt, gj = sq.series_targets(s)
        rj, rt, rjf, rtf = rule_predictions(s)
        rule_traj_fb += rjf
        t["y"].append(gj); t["model"].append(p["trajectory"]); t["rule"].append(rj); t["majority"].append(maj_traj); t["group"].append(s.key[1])
        for i, g in enumerate(gt):
            if g != -100:
                e["y"].append(g); e["model"].append(p["transitions"][i]); e["rule"].append(rt[i]); e["majority"].append(maj_trans); e["group"].append(s.key[1])
                rule_trans_fb += rtf[i]
    return t, e, rule_traj_fb, rule_trans_fb


def report(series, model_preds, majority, research, seed=0):
    t, e, jf, tf = arrays(series, model_preds, majority)
    out = {"research_result": research, "n_series": len(series), "n_transition_events": len(e["y"])}
    for name in ("model", "rule", "majority"):
        key = {"model": "model", "rule": "rule_baseline", "majority": "majority_baseline"}[name]
        out[key] = {"trajectory": block(t["y"], t[name], TRAJ_LABELS, sq.TRAJECTORIES), "transition": block(e["y"], e[name], TRANS_LABELS, sq.TRANSITIONS)}
    out["rule_baseline_fallbacks"] = {"trajectory_series_with_fewer_than_two_predicted_ranked_events": jf,
                                      "transition_events_without_rule_evidence_defaulted_to_STABLE": tf}
    out["model_minus_rule_macro_f1_product_grouped_bootstrap"] = {
        "trajectory": at.grouped_paired_bootstrap(t["y"], t["model"], t["rule"], t["group"], TRAJ_LABELS, seed=seed),
        "transition": at.grouped_paired_bootstrap(e["y"], e["model"], e["rule"], e["group"], TRANS_LABELS, seed=seed)}
    return out, t, e


def series_rows(split, series, model_preds, t):
    return [{"split": split, "sequence_type": s.key[0], "entity_id": s.key[1], "category": s.key[2], "n_events": len(s.events),
             "gold_trajectory": sq.TRAJECTORIES[t["y"][i]], "model_trajectory": sq.TRAJECTORIES[t["model"][i]],
             "rule_trajectory": sq.TRAJECTORIES[t["rule"][i]], "majority_trajectory": sq.TRAJECTORIES[t["majority"][i]],
             "message_ids": json.dumps([e.message_id for e in s.events])} for i, s in enumerate(series)]


def transition_rows(split, series, model_preds, majority):
    rows = []
    for s, p in zip(series, model_preds):
        gt, _ = sq.series_targets(s)
        _, rt, _, _ = rule_predictions(s)
        for i, g in enumerate(gt):
            if g != -100:
                rows.append({"split": split, "entity_id": s.key[1], "category": s.key[2], "message_id": s.events[i].message_id, "event_index": i,
                             "gold_transition": sq.TRANSITIONS[g], "model_transition": sq.TRANSITIONS[p["transitions"][i]],
                             "rule_transition": sq.TRANSITIONS[rt[i]], "majority_transition": sq.TRANSITIONS[majority[1]]})
    return rows


def main(argv=None):
    args = parse_args(argv)
    log = at.make_log()
    if min(args.epochs, args.batch_size, args.grad_accumulation) < 1 or not 0 < args.val_ratio < 1:
        raise ValueError("epochs, batch_size and grad_accumulation must be at least 1 and --val_ratio between 0 and 1")
    oof_cfg, oof_split, probs, roles = read_oof(args.oof_dir)
    if sorted(oof.sha256(p) for p in args.annotations) != oof_cfg["annotation_sha256"]:
        raise ValueError("the annotation files differ from those the OOF predictions were made from; regenerate the OOF run")
    corpus = at.load_corpus(args.annotations, sequence_types=oof_cfg.get("sequence_types"), on_invalid="skip" if oof_cfg.get("skip_invalid") else "raise")
    if len(corpus.domains) != 1:
        raise ValueError(f"one domain per run, the files contain {corpus.domains}")
    categories = ad.category_ids(corpus.domains[0])
    pool, test = oof_split["pool"], oof_split["test"]
    products = {m.entity_id for m in corpus.messages}
    if products != set(pool) | set(test):
        raise ValueError("the products in the annotations differ from those of the OOF run")

    inner = ad.split_entities(pool, (1 - args.val_ratio, args.val_ratio), args.split_seed, ("train", "val"))
    groups = {"train": inner["train"], "val": inner["val"], "test": test}
    splits = {n: [m for m in corpus.messages if m.entity_id in set(ps)] for n, ps in groups.items()}
    ad.assert_no_entity_leakage(splits)
    ex_ids = {m.message_id: m for m in corpus.messages}
    sources = {"train": "aspect_sentiment_model out-of-fold probabilities (product-grouped K-fold)", "val": "aspect_sentiment_model out-of-fold probabilities (product-grouped K-fold)",
               "test": "aspect_sentiment_model fold-ensemble probabilities (products kept out of every fold model)"}
    all_series, usable, excluded = {}, {}, {}
    for n, msgs in splits.items():
        ser, examples = series_for_split(msgs, probs, sources[n])
        want = "heldout_ensemble" if n == "test" else "oof"
        wrong = [e.example_id for e in examples if roles[e.example_id] != want]
        if wrong:
            raise ValueError(f"{n} examples taken from the wrong prediction source, e.g. {wrong[:3]}")
        all_series[n] = ser
        usable[n], excluded[n] = usable_series(ser)
    extra = set(probs) - {e.example_id for n in splits for e in ad.build_sentiment_examples(splits[n], level="message_category")}
    if extra:
        raise ValueError(f"the OOF run holds predictions for examples that are not in these annotations, e.g. {sorted(extra)[:3]}")
    counts = {n: len(usable[n]) for n in usable}
    blockers = at.series_blockers(counts, test_expected=not args.no_test)
    if not oof_cfg.get("research_result", False):
        blockers.append("the out-of-fold predictions come from a pipeline check (research_result=false)")
    if blockers and not args.allow_small_data:
        raise at.DataTooSmallError("the data cannot support a meaningful sequence-model evaluation, so nothing was trained: " + "; ".join(blockers)
                                   + ". (--allow_small_data runs a pipeline check whose outputs are marked research_result=false.)")
    if not usable["train"] or not usable["val"]:
        raise ValueError("no usable training or validation series (series need two or more ranked events)")
    research = not blockers
    warnings = [f"too small for a meaningful evaluation: {b}" for b in blockers]
    for n in ("val", "test"):
        absent = [sq.TRAJECTORIES[j] for j in TRAJ_LABELS if sum(sq.series_targets(s)[1] == j for s in usable[n]) == 0]
        if usable[n] and absent:
            warnings.append(f"the {n} split has no series of trajectory class {absent}: their F1 is 0 by definition")
    if len(corpus.annotators) <= 1:
        warnings.append("the labels come from a single annotator")

    name = args.name or f"sequence_{os.path.basename(os.path.normpath(args.oof_dir))}_h{args.hidden_dim}_s{args.seed}"
    exp = xu.prepare_experiment_dir(args.output_root, name)
    sub = {d: os.path.join(exp, d) for d in ("config", "logs", "metrics", "checkpoints", "predictions")}
    for d in sub.values():
        os.makedirs(d, exist_ok=True)
    device = at.pick_device(args.device)
    eu.set_seed(args.seed)
    tw = jw = None
    if args.class_weights != "none":
        tj = [sq.series_targets(s) for s in usable["train"]]
        tw = xu.class_weight_vector([x for t, _ in tj for x in t if x != -100], len(sq.TRANSITIONS), args.class_weights)
        jw = xu.class_weight_vector([j for _, j in tj if j != -100], len(sq.TRAJECTORIES), args.class_weights)
    model = sq.AspectSequenceModel(len(categories), category_dim=args.category_dim, hidden_dim=args.hidden_dim, lstm_layers=args.lstm_layers,
                                   attention_dim=args.attention_dim, head_hidden=args.head_hidden, dropout=args.dropout).to(device)
    loss_fn = sq.AspectSequenceLoss(args.transition_weight, args.trajectory_weight, tw, jw).to(device)
    eval_loss = sq.AspectSequenceLoss(args.transition_weight, args.trajectory_weight).to(device)
    cfg = vars(args).copy()
    cfg.update({"name": name, "task": "aspect_sequence", "selection_metric": args.selection_metric, "categories": categories, "parameters": model.parameter_counts(),
                "device": str(device), "oof_run": args.oof_dir, "oof_config_name": oof_cfg["name"], "annotation_sha256": oof_cfg["annotation_sha256"],
                "prediction_sources": sources, "usable_series": counts, "series_without_two_ranked_events": excluded, "warnings": warnings,
                "research_result": research, "gold_labels_are_model_input": False, "selection_uses_test": False, "environment": xu.environment_info()})
    xu.write_json(cfg, os.path.join(sub["config"], "config.json"))
    xu.write_json({"val_ratio": args.val_ratio, "split_seed": args.split_seed, "products": groups,
                   "series": {n: {"all": len(all_series[n]), "usable": counts[n], "excluded_insufficient": excluded[n]} for n in groups}},
                  os.path.join(sub["config"], "splits.json"))
    if warnings:
        log("=" * 78)
        log("WARNING: " + ("PIPELINE CHECK ONLY, NOT A RESEARCH RESULT" if not research else "see below"))
        for w in warnings:
            log("  - " + w)
        log("=" * 78)
    log(f"usable series: " + ", ".join(f"{n} {counts[n]}" for n in counts) + f"; parameters {cfg['parameters']['trainable']:,}")

    def evaluate(m, items):
        preds, loss = predict_series(m, items, categories, device, args.batch_size, eval_loss)
        t, e, _, _ = arrays(items, preds, (0, 0))
        tj, tr = block(t["y"], t["model"], TRAJ_LABELS, sq.TRAJECTORIES), block(e["y"], e["model"], TRANS_LABELS, sq.TRANSITIONS)
        summary = {"trajectory_accuracy": tj.get("accuracy", 0.0), "trajectory_macro_f1": tj.get("macro_f1", 0.0),
                   "transition_accuracy": tr.get("accuracy", 0.0), "transition_macro_f1": tr.get("macro_f1", 0.0), "loss": loss}
        summary["mean_macro_f1"] = (summary["trajectory_macro_f1"] + summary["transition_macro_f1"]) / 2
        return summary, {"summary": summary, "trajectory": tj, "transition": tr}, []

    step = lambda m, b: loss_fn(forward(m, b), b)["total_loss"]                                                        # noqa: E731
    res = at.fit(model, usable["train"], usable["val"], lambda c: sq.collate_series(c, categories), evaluate, device=device, epochs=args.epochs,
                 batch_size=args.batch_size, accumulation=args.grad_accumulation, lr_head=args.lr, lr_encoder=None, weight_decay=args.weight_decay,
                 schedule=args.schedule, warmup_ratio=args.warmup_ratio, grad_clip=args.grad_clip, selection_metric=args.selection_metric,
                 patience=args.patience, seed=args.seed, exp_dir=exp, config=cfg, log=log, step_fn=step,
                 dirs={"logs": sub["logs"], "checkpoints": sub["checkpoints"], "metrics": sub["metrics"]})

    saved = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    at.load_checkpoint_state(model, saved["model_state_dict"], saved["includes_encoder"])
    majority = majority_labels(usable["train"])
    summary = {"status": "completed", "name": name, "best_epoch": res["best_epoch"], "selection_metric": args.selection_metric, "research_result": research,
               "warnings": warnings, "usable_series": counts, "majority_class_ids_from_training": {"trajectory": majority[0], "transition": majority[1]}}
    for split in ("val", "test"):
        if split == "test" and args.no_test:
            continue
        preds, _ = predict_series(model, usable[split], categories, device, args.batch_size, eval_loss)
        rep, t, e = report(usable[split], preds, majority, research, seed=args.seed)
        label = "validation" if split == "val" else "test"
        xu.write_json(rep, os.path.join(sub["metrics"], f"{label}_metrics.json"))
        at.write_predictions(series_rows(split, usable[split], preds, t), os.path.join(sub["predictions"], f"{label}_series_predictions.csv"))
        at.write_predictions(transition_rows(split, usable[split], preds, majority), os.path.join(sub["predictions"], f"{label}_transition_predictions.csv"))
        summary[split] = {m: {"trajectory_macro_f1": rep[m]["trajectory"]["macro_f1"], "trajectory_accuracy": rep[m]["trajectory"]["accuracy"],
                              "transition_macro_f1": rep[m]["transition"]["macro_f1"], "transition_accuracy": rep[m]["transition"]["accuracy"]}
                          for m in ("model", "rule_baseline", "majority_baseline")}
        log(f"{split}: trajectory macro-F1 model {summary[split]['model']['trajectory_macro_f1']:.3f} | rule {summary[split]['rule_baseline']['trajectory_macro_f1']:.3f} "
            f"| majority {summary[split]['majority_baseline']['trajectory_macro_f1']:.3f}" + ("" if research else "   [PIPELINE CHECK, NOT A RESEARCH RESULT]"))
    xu.write_json(summary, os.path.join(exp, "summary.json"))
    log(f"wrote {exp}")
    return exp


if __name__ == "__main__":
    main()
