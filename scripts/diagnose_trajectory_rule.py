"""
Diagnostic: does the learned trajectory head beat the ontology rule applied to the model's own predicted sentiments?

Inference only. Loads outputs/checkpoints/best_model_amazon.pt (strict), runs the unchanged Amazon test split,
and writes everything to a NEW directory (nothing existing is read for writing or overwritten):

    outputs/diagnostics/trajectory_rule_vs_head/predictions/{positions,sequences}.csv
    outputs/diagnostics/trajectory_rule_vs_head/metrics/trajectory_rule_vs_head.{json,md}

Phases:  export  (runs the model, writes the two CSVs)
         analyze (reads the CSVs, no model, writes the metrics)
         all     (default)

    python scripts/diagnose_trajectory_rule.py --device cuda          # Kaggle
    python scripts/diagnose_trajectory_rule.py --device cpu           # about 5 to 15 minutes
    python scripts/diagnose_trajectory_rule.py --phase analyze        # re-analyse saved predictions
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True
import eval_utils as eu                                                      # noqa: E402
from ontology import SentimentState, TrajectoryType, TransitionType          # noqa: E402

OUT = os.path.join(ROOT, "outputs", "diagnostics", "trajectory_rule_vs_head")
POS_CSV = os.path.join(OUT, "predictions", "positions.csv")
SEQ_CSV = os.path.join(OUT, "predictions", "sequences.csv")
SENT_NAMES = SentimentState.label_names()
TRANS_NAMES = TransitionType.label_names()
TRAJ_NAMES = TrajectoryType.label_names()
TRAJ_LABELS = list(range(len(TRAJ_NAMES)))
TRANS_LABELS = list(range(len(TRANS_NAMES)))


def export(args):
    import torch
    from dataset import AmazonSequenceDataset, sequence_collate_fn
    from model import OpinionEvolutionTracker

    ckpt_path = os.path.join(ROOT, "outputs", "checkpoints", "best_model_amazon.pt")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = OpinionEvolutionTracker(
        model_name=a["model_name"], max_token_length=a["max_token_length"], lstm_hidden_dim=a["lstm_hidden"],
        lstm_num_layers=a["lstm_layers"], lstm_dropout=a["dropout"], attention_type=a["attention_type"],
        attention_dim=a["attention_dim"], freeze_encoder=a["freeze_encoder"], use_cuda=(device.type == "cuda"))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)             # stops on any mismatch
    model.eval()

    ds = AmazonSequenceDataset(split="test")                                 # same fixed split as training/evaluation
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
                seq_rows.append({"user": i, "length": L, "traj_gold": int(batch["trajectories"][b]), "traj_head_pred": int(jp[b])})
            print(f"exported {idx[-1] + 1}/{n} sequences", flush=True)
    os.makedirs(os.path.dirname(POS_CSV), exist_ok=True)
    pd.DataFrame(pos_rows).to_csv(POS_CSV, index=False)
    pd.DataFrame(seq_rows).to_csv(SEQ_CSV, index=False)
    print("wrote", POS_CSV, SEQ_CSV)


def rule_trajectory(states):
    return TrajectoryType.compute([SentimentState(int(s)) for s in states]).value


def rule_transitions(states):
    """Transition labels derived by the ontology from consecutive predicted sentiments (entry 0 is the placeholder)."""
    st = [SentimentState(int(s)) for s in states]
    return [TransitionType.STABLE.value] + [TransitionType.compute(st[i - 1], st[i]).value for i in range(1, len(st))]


def paired_bootstrap_diff(y, p_a, p_b, labels, n=2000, seed=0):
    """Macro-F1(a) - Macro-F1(b) over resampled sequences, same resample for both."""
    f = eu.macro_f1_fn(labels)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(f(y[i], p_a[i]) - f(y[i], p_b[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"estimate": float(f(y, p_a) - f(y, p_b)), "ci_low": float(lo), "ci_high": float(hi), "resamples": n,
            "unit": "sequences"}


def analyze(_args):
    pos, seq = pd.read_csv(POS_CSV), pd.read_csv(SEQ_CSV)
    pos = pos.sort_values(["user", "position"]).reset_index(drop=True)
    seq = seq.sort_values("user").reset_index(drop=True)
    assert set(pos.user) == set(seq.user) and (pos.groupby("user").size().values == seq.length.values).all()

    # Sanity check of the rule itself: applied to the GOLD sentiments it must reproduce the gold trajectory/transitions.
    gold_states = {u: g.sent_gold.tolist() for u, g in pos.groupby("user")}
    pred_states = {u: g.sent_pred.tolist() for u, g in pos.groupby("user")}
    rule_on_gold = np.array([rule_trajectory(gold_states[u]) for u in seq.user])
    assert (rule_on_gold == seq.traj_gold.values).all(), "rule on gold sentiments does not reproduce gold trajectories"
    rt_gold = np.concatenate([rule_transitions(gold_states[u]) for u in seq.user])
    assert (rt_gold == pos.trend_gold.values).all(), "rule on gold sentiments does not reproduce gold transitions"

    seq["traj_rule_pred"] = [rule_trajectory(pred_states[u]) for u in seq.user]
    pos["trend_rule_pred"] = np.concatenate([rule_transitions(pred_states[u]) for u in seq.user])
    y = seq.traj_gold.values
    head, rule = seq.traj_head_pred.values, seq.traj_rule_pred.values

    out = {
        "n_sequences": int(len(seq)), "n_positions": int(len(pos)), "checkpoint": "outputs/checkpoints/best_model_amazon.pt",
        "split": "AmazonSequenceDataset(split='test'), unchanged",
        "sanity": "rule applied to gold sentiments reproduces 100% of gold trajectories and gold transitions (asserted)",
        "trajectory": {
            "learned_head": eu.classification_metrics(y, head, TRAJ_LABELS, TRAJ_NAMES),
            "rule_from_predicted_sentiments": eu.classification_metrics(y, rule, TRAJ_LABELS, TRAJ_NAMES),
        },
        "trajectory_agreement_head_vs_rule": float((head == rule).mean()),
        "trajectory_accuracy_when_head_and_rule_agree": float((head[head == rule] == y[head == rule]).mean()) if (head == rule).any() else None,
        "trajectory_when_they_disagree": {
            "n": int((head != rule).sum()),
            "head_correct": int(((head == y) & (head != rule)).sum()),
            "rule_correct": int(((rule == y) & (head != rule)).sum()),
            "neither_correct": int(((head != y) & (rule != y) & (head != rule)).sum()),
        },
    }
    for k in ("learned_head", "rule_from_predicted_sentiments"):
        out["trajectory"][k]["macro_f1_ci95_over_sequences"] = eu.bootstrap_ci(y, head if k == "learned_head" else rule,
                                                                               eu.macro_f1_fn(TRAJ_LABELS))
    out["trajectory"]["head_minus_rule_macro_f1"] = paired_bootstrap_diff(y, head, rule, TRAJ_LABELS)
    acc_diff = (head == y).astype(float) - (rule == y).astype(float)
    rng = np.random.default_rng(0)
    d = [acc_diff[rng.integers(0, len(y), len(y))].mean() for _ in range(2000)]
    out["trajectory"]["head_minus_rule_accuracy"] = {"estimate": float(acc_diff.mean()), "ci_low": float(np.percentile(d, 2.5)),
                                                     "ci_high": float(np.percentile(d, 97.5)), "resamples": 2000, "unit": "sequences"}

    m = eu.observed_transition_mask(pos.position.values)                    # positions >= 1 only: placeholder excluded
    obs = pos[m]
    out["transition_observed_only"] = {
        "excluded_placeholder_positions": int((~m).sum()), "n_evaluated": int(m.sum()),
        "learned_head": eu.classification_metrics(obs.trend_gold, obs.trend_pred, TRANS_LABELS, TRANS_NAMES),
        "rule_from_predicted_sentiments": eu.classification_metrics(obs.trend_gold, obs.trend_rule_pred, TRANS_LABELS, TRANS_NAMES),
    }
    out["transition_all_positions_for_reference"] = {
        "learned_head_includes_placeholder": eu.classification_metrics(pos.trend_gold, pos.trend_pred, TRANS_LABELS, TRANS_NAMES)["macro_f1"]}
    out["sentiment_for_reference"] = eu.classification_metrics(pos.sent_gold, pos.sent_pred, [0, 1, 2], SENT_NAMES[:3])
    out["sentiment_for_reference"].pop("confusion_matrix", None)

    eu.save_json(out, os.path.join(OUT, "metrics", "trajectory_rule_vs_head.json"))
    write_md(out)
    print(open(os.path.join(OUT, "metrics", "trajectory_rule_vs_head.md"), encoding="utf-8").read())


def _cm_table(title, cm, names):
    rows = [f"**{title}** (rows = gold, columns = predicted)", "", "| gold \\ pred | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names)]
    rows += [f"| {n} | " + " | ".join(str(v) for v in r) + " |" for n, r in zip(names, cm)]
    return rows + [""]


def write_md(o):
    t, tr = o["trajectory"], o["transition_observed_only"]
    h, r = t["learned_head"], t["rule_from_predicted_sentiments"]
    ci = lambda d: f"{d['ci_low']:.3f} to {d['ci_high']:.3f}"                 # noqa: E731
    md = ["# Diagnostic: learned trajectory head vs ontology rule on predicted sentiments", "",
          f"{o['n_sequences']} Amazon test users, {o['n_positions']} reviews; checkpoint `{o['checkpoint']}`; {o['split']}.",
          "Inference only; no retraining. " + o["sanity"] + ".", "",
          "## Trajectory (per sequence)", "", "| Method | Accuracy | Macro-F1 | 95% CI (sequences) |", "|---|---|---|---|",
          f"| Learned trajectory head | {h['accuracy']:.4f} | {h['macro_f1']:.4f} | {ci(h['macro_f1_ci95_over_sequences'])} |",
          f"| Rule applied to predicted sentiments | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {ci(r['macro_f1_ci95_over_sequences'])} |", "",
          f"Head minus rule: macro-F1 {t['head_minus_rule_macro_f1']['estimate']:+.4f} "
          f"(95% paired bootstrap {ci(t['head_minus_rule_macro_f1'])}); accuracy {t['head_minus_rule_accuracy']['estimate']:+.4f} "
          f"({ci(t['head_minus_rule_accuracy'])}).", "",
          f"The two agree on {o['trajectory_agreement_head_vs_rule'] * 100:.1f}% of sequences. Where they disagree "
          f"(n={o['trajectory_when_they_disagree']['n']}): head correct {o['trajectory_when_they_disagree']['head_correct']}, "
          f"rule correct {o['trajectory_when_they_disagree']['rule_correct']}, neither {o['trajectory_when_they_disagree']['neither_correct']}.", "",
          "### Per-class F1 (trajectory)", "", "| Class | Support | Head F1 | Rule F1 |", "|---|---|---|---|"]
    md += [f"| {n} | {h['per_class'][n]['support']} | {h['per_class'][n]['f1']:.3f} | {r['per_class'][n]['f1']:.3f} |" for n in TRAJ_NAMES]
    md += [""] + _cm_table("Learned head", h["confusion_matrix"], TRAJ_NAMES) + _cm_table("Rule", r["confusion_matrix"], TRAJ_NAMES)
    th, trr = tr["learned_head"], tr["rule_from_predicted_sentiments"]
    md += [f"## Transitions, observed positions only ({tr['n_evaluated']} evaluated; {tr['excluded_placeholder_positions']} placeholder positions excluded)", "",
           "| Method | Accuracy | Macro-F1 |", "|---|---|---|",
           f"| Learned trend head | {th['accuracy']:.4f} | {th['macro_f1']:.4f} |",
           f"| Rule applied to predicted sentiments | {trr['accuracy']:.4f} | {trr['macro_f1']:.4f} |", "",
           f"For reference, the learned trend head scored over all positions including the placeholder: "
           f"{o['transition_all_positions_for_reference']['learned_head_includes_placeholder']:.4f} macro-F1.", "",
           "### Per-class F1 (transitions)", "", "| Class | Support | Head F1 | Rule F1 |", "|---|---|---|---|"]
    md += [f"| {n} | {th['per_class'][n]['support']} | {th['per_class'][n]['f1']:.3f} | {trr['per_class'][n]['f1']:.3f} |" for n in TRANS_NAMES]
    md += [""] + _cm_table("Learned trend head", th["confusion_matrix"], TRANS_NAMES) + _cm_table("Rule", trr["confusion_matrix"], TRANS_NAMES)
    s = o["sentiment_for_reference"]
    md += [f"Sentiment head for reference (3 determinate classes): accuracy {s['accuracy']:.4f}, macro-F1 {s['macro_f1']:.4f}."]
    with open(os.path.join(OUT, "metrics", "trajectory_rule_vs_head.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["export", "analyze", "all"], default="all")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_sequences", type=int, default=0, help="smoke test only")
    ap.add_argument("--out_dir", default=None, help="override the output directory (smoke tests)")
    a = ap.parse_args()
    if a.out_dir:
        OUT = a.out_dir
        POS_CSV, SEQ_CSV = (os.path.join(OUT, "predictions", f) for f in ("positions.csv", "sequences.csv"))
    if os.path.exists(os.path.join(OUT, "predictions", "positions.csv")) and a.phase in ("export", "all") and not a.max_sequences:
        sys.exit(f"{OUT}/predictions already exists; refusing to overwrite. Use --phase analyze or a different --out_dir.")
    if a.phase in ("export", "all"):
        export(a)
    if a.phase in ("analyze", "all"):
        analyze(a)
