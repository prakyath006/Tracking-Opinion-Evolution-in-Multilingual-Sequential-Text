"""
Module 7b — Per-position / per-sequence prediction export and prediction-based statistics.

Phase `export`  : load outputs/checkpoints/best_model_amazon.pt (inference only), run the Amazon test users,
                  save outputs/predictions/amazon_test_positions.csv and amazon_test_sequences.csv.
Phase `analyze` : read those CSVs (no model) and write outputs/metrics/amazon_prediction_stats.{json,md}:
                  trend metrics over observed transitions only, bootstrap 95% intervals over test users
                  for sentiment / trend / trajectory macro-F1, and SCS of the predictions versus the gold sequences.

    python scripts/export_predictions.py --phase export --device cuda --batch_size 16
    python scripts/export_predictions.py --phase analyze
    python scripts/export_predictions.py --phase export --max_sequences 4 --device cpu   # smoke test
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True
import eval_utils as eu                                      # noqa: E402

PRED_DIR = os.path.join(ROOT, "outputs", "predictions")
POS_CSV = os.path.join(PRED_DIR, "amazon_test_positions.csv")
SEQ_CSV = os.path.join(PRED_DIR, "amazon_test_sequences.csv")


def export(args):
    import torch
    from dataset import AmazonSequenceDataset, sequence_collate_fn
    from model import OpinionEvolutionTracker

    ckpt_path = args.checkpoint or os.path.join(ROOT, "outputs", "checkpoints", "best_model_amazon.pt")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    if args.threads:
        torch.set_num_threads(args.threads)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = OpinionEvolutionTracker(
        model_name=a["model_name"], max_token_length=a["max_token_length"], lstm_hidden_dim=a["lstm_hidden"],
        lstm_num_layers=a["lstm_layers"], lstm_dropout=a["dropout"], attention_type=a["attention_type"],
        attention_dim=a["attention_dim"], freeze_encoder=a["freeze_encoder"], use_cuda=(device.type == "cuda"))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)     # stops on any mismatch
    model.eval()

    ds = AmazonSequenceDataset(split="test")
    n = min(len(ds), args.max_sequences) if args.max_sequences else len(ds)
    pos_rows, seq_rows = [], []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            idx = list(range(start, min(n, start + args.batch_size)))
            batch = sequence_collate_fn([ds[i] for i in idx])
            out = model(batch["texts"], seq_lens=batch["seq_lens"], padding_mask=batch["padding_mask"].to(model.device))
            sp = out["sentiment_logits"].argmax(-1).cpu().numpy()
            tp = out["trend_logits"].argmax(-1).cpu().numpy()
            jp = out["trajectory_logits"].argmax(-1).cpu().numpy()
            for b, i in enumerate(idx):
                L = batch["seq_lens"][b]
                for p in range(L):
                    pos_rows.append({"user": i, "position": p, "sent_gold": int(batch["sentiments"][b, p]),
                                     "sent_pred": int(sp[b, p]), "trend_gold": int(batch["trends"][b, p]),
                                     "trend_pred": int(tp[b, p])})
                seq_rows.append({"user": i, "length": L, "traj_gold": int(batch["trajectories"][b]), "traj_pred": int(jp[b])})
            print(f"exported {idx[-1] + 1}/{n} sequences", flush=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    pd.DataFrame(pos_rows).to_csv(POS_CSV, index=False)
    pd.DataFrame(seq_rows).to_csv(SEQ_CSV, index=False)
    print("wrote", POS_CSV, SEQ_CSV)


def analyze(_args):
    pos, seq = pd.read_csv(POS_CSV), pd.read_csv(SEQ_CSV)
    out = {"n_users": int(len(seq)), "n_positions": int(len(pos))}
    s_true, s_pred = pos.sent_gold.values, pos.sent_pred.values
    out["sentiment"] = eu.classification_metrics(s_true, s_pred, [0, 1, 2], eu.SENTIMENT_NAMES[:3])
    out["sentiment"]["macro_f1_ci95_over_users"] = eu.bootstrap_ci(s_true, s_pred, eu.macro_f1_fn([0, 1, 2]), groups=pos.user.values)
    names = ["UPGRADE", "DOWNGRADE", "STABLE"]
    out["trend_all_positions"] = eu.classification_metrics(pos.trend_gold, pos.trend_pred, [0, 1, 2], names)
    m = eu.observed_transition_mask(pos.position.values)
    obs = pos[m]
    out["trend_observed_only"] = eu.classification_metrics(obs.trend_gold, obs.trend_pred, [0, 1, 2], names)
    out["trend_observed_only"]["macro_f1_ci95_over_users"] = eu.bootstrap_ci(
        obs.trend_gold.values, obs.trend_pred.values, eu.macro_f1_fn([0, 1, 2]), groups=obs.user.values)
    tn = ["IMPROVING", "DECLINING", "STABLE", "VOLATILE"]
    out["trajectory"] = eu.classification_metrics(seq.traj_gold, seq.traj_pred, [0, 1, 2, 3], tn)
    out["trajectory"]["macro_f1_ci95_over_users"] = eu.bootstrap_ci(
        seq.traj_gold.values, seq.traj_pred.values, eu.macro_f1_fn([0, 1, 2, 3]), n_resamples=2000)
    gold_seqs = [g.sort_values("position").sent_gold.tolist() for _, g in pos.groupby("user")]
    pred_seqs = [g.sort_values("position").sent_pred.tolist() for _, g in pos.groupby("user")]
    out["scs"] = {"gold": eu.gold_scs(gold_seqs), "predicted": eu.gold_scs(pred_seqs)}
    eu.save_json(out, os.path.join(ROOT, "outputs", "metrics", "amazon_prediction_stats.json"))
    md = ["# Amazon prediction-based statistics", "", f"{out['n_users']} test users, {out['n_positions']} positions.", "",
          "| Measure | Value | 95% CI over users |", "|---|---|---|"]
    for key, label in (("sentiment", "Sentiment macro-F1"), ("trend_observed_only", "Trend macro-F1 (observed transitions only)"),
                       ("trajectory", "Trajectory macro-F1")):
        c = out[key]["macro_f1_ci95_over_users"]
        md.append(f"| {label} | {out[key]['macro_f1']:.4f} | {c['ci_low']:.4f}–{c['ci_high']:.4f} |")
    md.append(f"| Trend macro-F1 (all positions, includes placeholder) | {out['trend_all_positions']['macro_f1']:.4f} | – |")
    md += ["", f"SCS: predicted {out['scs']['predicted']['scs_mean']:.4f}, gold {out['scs']['gold']['scs_mean']:.4f}."]
    with open(os.path.join(ROOT, "outputs", "metrics", "amazon_prediction_stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["export", "analyze"], required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_sequences", type=int, default=0, help="smoke test only")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    export(a) if a.phase == "export" else analyze(a)
