"""Table of every finished experiment under outputs/experiments/ (validation and test headline numbers).
    python scripts/summarize_experiments.py [--root outputs/experiments]   ->  prints and writes <root>/summary_table.csv
Read-only on the experiments; only summary_table.csv is (re)written."""
import argparse
import json
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments"))
a = ap.parse_args()
rows = []
for name in sorted(os.listdir(a.root)):
    f = os.path.join(a.root, name, "summary.json")
    if not os.path.exists(f):
        continue
    s, c = json.load(open(f)), json.load(open(os.path.join(a.root, name, "config.json")))
    v, t = s["best_validation"], s.get("test") or {}
    rows.append({"experiment": name, "smoke": s["smoke_test"], "best_epoch": s["best_epoch"], "selected_on": s["selection_metric"],
                 "val_sent_f1": v["sentiment_macro_f1"], "val_trans_f1": v["transition_macro_f1"], "val_traj_f1": v["trajectory_macro_f1"],
                 "val_mean_f1": v["mean_macro_f1"], "test_sent_acc": t.get("sentiment_accuracy"), "test_sent_f1": t.get("sentiment_macro_f1"),
                 "test_trans_f1": t.get("transition_macro_f1"), "test_traj_acc": t.get("trajectory_accuracy"),
                 "test_traj_f1": t.get("trajectory_macro_f1"), "trainable_M": round(c["trainable_params"] / 1e6, 2), "seed": c["seed"]})
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit(f"no finished experiments under {a.root}")
df.round(4).to_csv(os.path.join(a.root, "summary_table.csv"), index=False)
print(df.round(3).to_string(index=False))
