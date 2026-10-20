"""
Module 7a — Valid reference scores for the Amazon user histories (labels only: no model, no GPU).

Computes, from the same train/validation/test user split used by scripts/train.py:
  * majority-class baselines chosen from TRAINING labels and scored on the test labels
    (sentiment, trend over all positions, trend over observed transitions only, trajectory);
  * bootstrap 95% intervals over test users for those baselines;
  * the Sequence Consistency Score of the GOLD test sequences;
  * class distributions per split.

Writes outputs/metrics/valid_baselines_amazon.json and .md.

    python scripts/compute_valid_baselines.py [--bootstrap 2000]
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True

import eval_utils as eu                                   # noqa: E402
from dataset import AmazonSequenceDataset                 # noqa: E402
from ontology import SentimentState, TransitionType, TrajectoryType   # noqa: E402

SENT = (eu.DETERMINATE, eu.SENTIMENT_NAMES[:3])           # Amazon has no UNKNOWN label
TREND = ([0, 1, 2], [t.name for t in sorted(TransitionType, key=lambda t: t.value)])
TRAJ = ([0, 1, 2, 3], [t.name for t in sorted(TrajectoryType, key=lambda t: t.value)])


def flatten(ds):
    sent, trend, pos, user, traj = [], [], [], [], []
    for u, s in enumerate(ds.sequences):
        for p, (a, b) in enumerate(zip(s["sentiments"], s["trends"])):
            sent.append(a); trend.append(b); pos.append(p); user.append(u)
        traj.append(s["trajectory"])
    return {k: np.asarray(v) for k, v in dict(sent=sent, trend=trend, pos=pos, user=user, traj=traj).items()}, ds


def dist(labels, names):
    c = np.bincount(labels, minlength=len(names))
    return {n: int(c[i]) for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    data = {}
    for split in ("train", "val", "test"):
        data[split], ds = flatten(AmazonSequenceDataset(split=split))
        data[split]["sequences"] = ds.sequences
    tr, te = data["train"], data["test"]
    out = {"source": "AmazonSequenceDataset split (seed 42, user level, max_seq_len 20)",
           "note": "Majority classes are taken from TRAINING labels; test labels are used only for scoring.",
           "users": {s: len(data[s]["sequences"]) for s in data},
           "positions": {s: int(len(data[s]["sent"])) for s in data}}

    mask_tr, mask_te = eu.observed_transition_mask(tr["pos"]), eu.observed_transition_mask(te["pos"])
    heads = {
        "sentiment (per review)": (tr["sent"], te["sent"], te["user"], *SENT),
        "trend, all positions (includes the position-1 placeholder)": (tr["trend"], te["trend"], te["user"], *TREND),
        "trend, observed transitions only (positions >= 2)": (tr["trend"][mask_tr], te["trend"][mask_te], te["user"][mask_te], *TREND),
        "trajectory (per sequence)": (tr["traj"], te["traj"], np.arange(len(te["traj"])), *TRAJ),
    }
    out["majority_baselines"] = {}
    for name, (ytr, yte, groups, labels, names) in heads.items():
        res = eu.majority_baseline(ytr, yte, labels, names)
        pred = np.full(len(yte), res["majority_class_from_train"])
        res["macro_f1_ci95_over_test_users"] = eu.bootstrap_ci(yte, pred, eu.macro_f1_fn(labels), groups=groups,
                                                               n_resamples=args.bootstrap, seed=0)
        out["majority_baselines"][name] = res

    out["class_distributions"] = {
        s: {"sentiment": dist(data[s]["sent"], SENT[1]), "trend_all_positions": dist(data[s]["trend"], TREND[1]),
            "trend_observed_only": dist(data[s]["trend"][eu.observed_transition_mask(data[s]["pos"])], TREND[1]),
            "trajectory": dist(data[s]["traj"], TRAJ[1])} for s in data}
    out["gold_scs_test"] = eu.gold_scs([s["sentiments"] for s in te["sequences"]])
    eu.save_json(out, os.path.join(ROOT, "outputs", "metrics", "valid_baselines_amazon.json"))

    md = ["# Valid reference scores — Amazon user histories", "",
          "Produced by `scripts/compute_valid_baselines.py` from labels only (no model). Majority classes come from the "
          "training split; test labels are used only for scoring.", "",
          f"Users train/val/test: {out['users']['train']}/{out['users']['val']}/{out['users']['test']}.", "",
          "| Head | Majority class (from train) | Test accuracy | Test macro-F1 | 95% CI over test users | Test n |",
          "|---|---|---|---|---|---|"]
    for name, r in out["majority_baselines"].items():
        names = heads[name][4]
        ci = r["macro_f1_ci95_over_test_users"]
        md.append(f"| {name} | {names[r['majority_class_from_train']]} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
                  f"{ci['ci_low']:.4f}–{ci['ci_high']:.4f} | {r['n']} |")
    g = out["gold_scs_test"]
    md += ["", f"Gold-sequence SCS on the {g['num_sequences']} test users: mean {g['scs_mean']:.4f} (std {g['scs_std']:.4f}).",
           "", "Class distributions per split are in the JSON file."]
    with open(os.path.join(ROOT, "outputs", "metrics", "valid_baselines_amazon.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
