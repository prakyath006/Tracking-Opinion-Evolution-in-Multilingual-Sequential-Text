"""
Controlled OET experiments for Kaggle GPU runs (Amazon sequences).

This is a NEW entry point. scripts/train.py, src/model.py and outputs/checkpoints/best_model_amazon.pt are unchanged;
the experiment reuses OpinionEvolutionTracker and the unchanged AmazonSequenceDataset split (70/15/15 by user, seed 42,
sequence length 20). Labels are not redefined; only position 0 of the transition target is masked, because it is an
artificial STABLE placeholder (see src/experiment_utils.py).

Each run writes ONE new directory, outputs/experiments/<name>/ (an existing directory is never touched):
    config.json            every setting plus environment (versions, GPU, git commit)
    class_weights.json     weights actually used and the training label counts they came from
    train_log.jsonl        one line per epoch: training losses, validation accuracy and macro-F1 per head, lr
    history.json           the same as one list (rewritten after each epoch, so a crash keeps the progress)
    best_checkpoint.pt     chosen on a VALIDATION metric only
    best_validation.json   full validation metrics (per-class, confusion matrices) of the selected epoch
    test_metrics.json      one test evaluation, run once after training with the selected checkpoint
    predictions/           positions.csv, sequences.csv (same columns as scripts/diagnose_trajectory_rule.py reads)
    summary.json           final status and headline numbers

Examples (see docs/kaggle_experiments.md for the full plan):
    python scripts/train_experiment.py --encoder xlmr --finetune frozen --pooling mean --max_token_length 256 --seed 0
    python scripts/train_experiment.py --encoder xlmr --finetune partial --finetune_layers 2 --lr_encoder 2e-5 --lr_head 1e-3 \
        --batch_size 8 --grad_accumulation 2 --schedule linear --seed 0
"""
import argparse
import json
import logging
import os
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True

import eval_utils as eu                                                      # noqa: E402
import experiment_utils as xu                                                # noqa: E402
from classifier import MultiTaskLoss                                         # noqa: E402
from dataset import AmazonSequenceDataset, sequence_collate_fn               # noqa: E402
from model import OpinionEvolutionTracker                                    # noqa: E402

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("experiment")
log.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
log.addHandler(_h)
log.propagate = False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=None, help="experiment directory name (default: built from the settings)")
    p.add_argument("--output_root", default=os.path.join(ROOT, "outputs", "experiments"))
    p.add_argument("--encoder", choices=sorted(xu.ENCODERS), default="mbert")
    p.add_argument("--finetune", choices=xu.FINETUNE_MODES, default="frozen")
    p.add_argument("--finetune_layers", type=int, default=2, help="top layers to train when --finetune partial")
    p.add_argument("--pooling", choices=xu.POOLING, default="cls", help="cls token or masked mean over real tokens")
    p.add_argument("--max_token_length", type=int, default=128)
    p.add_argument("--class_weights", choices=xu.WEIGHT_MODES, default="full",
                   help="none, sqrt inverse frequency, or full inverse frequency (training labels only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16, help="users per step (before accumulation)")
    p.add_argument("--grad_accumulation", type=int, default=1)
    p.add_argument("--lr_head", type=float, default=1e-3, help="learning rate of LSTM, attention and task heads")
    p.add_argument("--lr_encoder", type=float, default=None, help="learning rate of trained encoder layers")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--schedule", choices=xu.SCHEDULES, default="constant")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--w_sentiment", type=float, default=1.0, help="task loss weights (train.py uses 1.0 / 0.5 / 1.0)")
    p.add_argument("--w_trend", type=float, default=0.5)
    p.add_argument("--w_trajectory", type=float, default=1.0)
    p.add_argument("--selection_metric", choices=xu.SELECTION_METRICS, default="mean_macro_f1",
                   help="validation metric that picks the checkpoint and drives early stopping and the plateau scheduler")
    p.add_argument("--patience", type=int, default=4, help="epochs without validation improvement; 0 disables")
    p.add_argument("--max_seq_len", type=int, default=20, help="reviews per user kept (existing setting; do not change for comparisons)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--no_test", action="store_true", help="skip the final test evaluation")
    p.add_argument("--max_train_users", type=int, default=0, help="SMOKE TEST ONLY: first N users of the train split")
    p.add_argument("--max_eval_users", type=int, default=0, help="SMOKE TEST ONLY: first N users of val and test")
    return p.parse_args(argv)


def make_loader(split, batch_size, shuffle, seed, max_seq_len, limit):
    ds = AmazonSequenceDataset(split=split, max_seq_len=max_seq_len)            # unchanged split
    if limit:
        ds.sequences = ds.sequences[:limit]
    g = torch.Generator().manual_seed(seed)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=sequence_collate_fn,
                          num_workers=0, generator=g if shuffle else None)


def export_predictions(raw, out_dir):
    """positions.csv / sequences.csv: same columns as scripts/export_predictions.py and diagnose_trajectory_rule.py."""
    pos = pd.DataFrame({"user": raw["user_row"], "position": raw["position"], "sent_gold": raw["sent_true"],
                        "sent_pred": raw["sent_pred"], "trend_gold": raw["trend_true"], "trend_pred": raw["trend_pred"]})
    seq = pd.DataFrame({"user": range(len(raw["seq_lens"])), "length": raw["seq_lens"], "traj_gold": raw["traj_true"],
                        "traj_head_pred": raw["traj_pred"]})
    pos.to_csv(os.path.join(out_dir, "predictions", "positions.csv"), index=False)
    seq.to_csv(os.path.join(out_dir, "predictions", "sequences.csv"), index=False)


def main(argv=None):
    a = parse_args(argv)
    config = vars(a).copy()
    config["lr_encoder_explicit"] = a.lr_encoder is not None
    if a.lr_encoder is None:
        config["lr_encoder"] = 2e-5 if a.finetune != "frozen" else None
    xu.validate_config(config)
    name = a.name or xu.default_experiment_name(config)
    exp_dir = xu.prepare_experiment_dir(a.output_root, name)                    # refuses to overwrite
    device = torch.device("cuda" if (a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())) else "cpu")
    if a.device == "cuda" and device.type != "cuda":
        raise SystemExit("--device cuda requested but no GPU is available")
    smoke = bool(a.max_train_users or a.max_eval_users)
    log.info(f"experiment {name} on {device}{'  [SMOKE TEST: user subset]' if smoke else ''}")

    # Data first: AmazonSequenceDataset fixes the global numpy seed to 42 for its split; seeding comes after it.
    train_ds, train_loader = make_loader("train", a.batch_size, True, a.seed, a.max_seq_len, a.max_train_users)
    val_ds, val_loader = make_loader("val", a.batch_size, False, a.seed, a.max_seq_len, a.max_eval_users)
    test_ds, test_loader = (None, None) if a.no_test else make_loader("test", a.batch_size, False, a.seed, a.max_seq_len, a.max_eval_users)
    eu.set_seed(a.seed)

    weights, counts = xu.training_class_weights(train_ds, a.class_weights)
    model = OpinionEvolutionTracker(model_name=xu.ENCODERS[a.encoder], max_token_length=a.max_token_length,
                                    embedding_strategy=a.pooling, freeze_encoder=True, use_cuda=(device.type == "cuda"))
    n_enc = xu.set_encoder_trainable(model.embedding_generator.encoder, a.finetune, a.finetune_layers)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    config.update({"encoder_name": xu.ENCODERS[a.encoder], "trainable_encoder_params": n_enc, "trainable_params": n_train,
                   "total_params": n_total, "smoke_test": smoke, "n_train_users": len(train_ds), "n_val_users": len(val_ds),
                   "n_test_users": len(test_ds) if test_ds else 0, "device": str(device), "environment": xu.environment_info(),
                   "placeholder_transition_masked": True, "selection_uses_test": False})
    xu.write_json(config, os.path.join(exp_dir, "config.json"))
    xu.write_json({"mode": a.class_weights, "training_label_counts": counts,
                   "weights": {k: v.tolist() for k, v in weights.items()} if weights else None},
                  os.path.join(exp_dir, "class_weights.json"))
    log.info(f"trainable parameters: {n_train:,} of {n_total:,} (encoder: {n_enc:,}); weights: {a.class_weights}")

    loss_fn = MultiTaskLoss(
        sentiment_weight=a.w_sentiment, trend_weight=a.w_trend, trajectory_weight=a.w_trajectory,
        sentiment_class_weights=weights["sentiment"] if weights else None,
        trend_class_weights=weights["trend"] if weights else None,
        trajectory_class_weights=weights["trajectory"] if weights else None).to(model.device)
    optimizer = torch.optim.AdamW(xu.build_param_groups(model, config["lr_encoder"] or 0.0, a.lr_head, a.weight_decay))
    total_updates = xu.updates_per_epoch(len(train_loader), a.grad_accumulation) * a.epochs
    _, higher = xu.selection_value({"val_loss": 0.0, a.selection_metric: 0.0}, a.selection_metric)
    scheduler, per_update = xu.make_scheduler(optimizer, a.schedule, total_updates, a.warmup_ratio, "max" if higher else "min")

    best, best_epoch, bad, history = None, 0, 0, []
    ckpt_path = os.path.join(exp_dir, "best_checkpoint.pt")
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        tr = xu.train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, per_update, a.grad_accumulation,
                                a.grad_clip, model.device)
        val, _ = xu.evaluate(model, val_loader, loss_fn, model.device)
        s = xu.flat_summary(val)
        value, higher = xu.selection_value(s, a.selection_metric)
        if scheduler is not None and not per_update:
            scheduler.step(value)
        row = {"epoch": epoch, "train_loss": tr["total"], "train_losses": {k: tr[k] for k in ("sentiment", "trend", "trajectory")},
               "optimizer_updates": tr["optimizer_updates"], "validation": s, "selection_metric": a.selection_metric,
               "selection_value": value, "lr": {g.get("name", str(i)): g["lr"] for i, g in enumerate(optimizer.param_groups)},
               "seconds": round(time.time() - t0, 1)}
        improved = xu.is_improvement(value, best, higher)
        if improved:
            best, best_epoch, bad = value, epoch, 0
            state, includes_encoder = xu.checkpoint_state(model)
            torch.save({"model_state_dict": state, "includes_encoder": includes_encoder, "epoch": epoch, "config": config,
                        "selection_metric": a.selection_metric, "selection_value": value, "validation": s}, ckpt_path)
            xu.write_json(val, os.path.join(exp_dir, "best_validation.json"))
        else:
            bad += 1
        row["is_best"] = improved
        history.append(row)
        with open(os.path.join(exp_dir, "train_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        xu.write_json(history, os.path.join(exp_dir, "history.json"))
        log.info(f"epoch {epoch:2d} | train {tr['total']:.3f} | val sent acc {s['sentiment_accuracy']:.3f} F1 {s['sentiment_macro_f1']:.3f}"
                 f" | trans acc {s['transition_accuracy']:.3f} F1 {s['transition_macro_f1']:.3f}"
                 f" | traj acc {s['trajectory_accuracy']:.3f} F1 {s['trajectory_macro_f1']:.3f}"
                 f" | {a.selection_metric} {value:.4f}{' *' if improved else ''} | {row['seconds']}s")
        if a.patience and bad >= a.patience:
            log.info(f"early stop: no validation improvement for {a.patience} epochs")
            break

    summary = {"status": "completed", "name": name, "epochs_run": len(history), "best_epoch": best_epoch,
               "selection_metric": a.selection_metric, "best_validation": history[best_epoch - 1]["validation"],
               "test": None, "smoke_test": smoke}
    if not a.no_test:
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)             # selected on validation only
        xu.load_checkpoint_state(model, saved["model_state_dict"], saved["includes_encoder"])
        test, raw = xu.evaluate(model, test_loader, loss_fn, model.device)
        xu.write_json(test, os.path.join(exp_dir, "test_metrics.json"))
        export_predictions(raw, exp_dir)
        summary["test"] = xu.flat_summary(test)
        log.info("test (selected checkpoint, epoch %d): sentiment acc %.4f F1 %.4f | transition F1 %.4f | trajectory acc %.4f F1 %.4f"
                 % (best_epoch, test["sentiment"]["accuracy"], test["sentiment"]["macro_f1"], test["transition_observed"]["macro_f1"],
                    test["trajectory"]["accuracy"], test["trajectory"]["macro_f1"]))
    xu.write_json(summary, os.path.join(exp_dir, "summary.json"))
    log.info(f"wrote {exp_dir}")
    return exp_dir


if __name__ == "__main__":
    main()
