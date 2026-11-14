"""
Consolidated comparison of finished OET experiments from their ACTUAL summary.json and config.json (read-only).

    python scripts/compare_experiments.py --ids E0 E3c E4 --out outputs/experiments_comparison.md

Each id selects the one directory under --root (default outputs/experiments) whose name starts with "<id>_". E0, E3c and E4 are required; any further id is included only if its summary.json says status "completed" and smoke_test false, otherwise it is listed as not included with the reason.
Numbers are never filled in or estimated: a missing required experiment stops the script. The old Panel 1 checkpoint's own test numbers
(outputs/logs/test_results_amazon.json) appear in a separate section because its protocol differs (not comparable without re-scoring).
One seed per experiment: differences below the seed-to-seed spread (E6) are not conclusions.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = ["sentiment_macro_f1", "transition_macro_f1", "trajectory_macro_f1", "mean_macro_f1"]
TEST = ["sentiment_accuracy", "sentiment_macro_f1", "transition_macro_f1", "trajectory_accuracy", "trajectory_macro_f1"]
REQUIRED = ("E0", "E3c", "E4")


def find(root, i):
    hits = [d for d in sorted(os.listdir(root)) if d.startswith(i + "_") and os.path.isdir(os.path.join(root, d))] if os.path.isdir(root) else []
    if len(hits) > 1:
        raise ValueError(f"{i}: several directories match {hits}")
    return hits[0] if hits else None


def f(x):
    return "n/a" if x is None else f"{x:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(ROOT, "outputs", "experiments"))
    ap.add_argument("--ids", nargs="+", default=["E0", "E3c", "E4"])
    ap.add_argument("--panel1", default=os.path.join(ROOT, "outputs", "logs", "test_results_amazon.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise FileExistsError(a.out)
    used, skipped, missing = {}, {}, []
    for i in a.ids:
        name = find(a.root, i)
        path = os.path.join(a.root, name, "summary.json") if name else None
        if not path or not os.path.exists(path):
            if i in REQUIRED:
                missing.append(f"{i} (no summary.json)")
            else:
                skipped[i] = "no summary.json yet"
            continue
        s = json.load(open(path, encoding="utf-8"))
        why = f"status is {s.get('status')!r}" if s.get("status") != "completed" else "smoke test" if s.get("smoke_test") else None
        if why:
            if i in REQUIRED:
                missing.append(f"{i} ({why})")
            else:
                skipped[i] = why
            continue
        used[i] = (name, s, json.load(open(os.path.join(a.root, name, "config.json"), encoding="utf-8")))
    if missing:
        sys.exit(f"required experiment(s) without a completed summary.json under {a.root}: {missing}. Copy their folders from Kaggle first; nothing is estimated.")
    base = used["E0"][1]
    L = ["# OET experiment comparison (actual summary.json files)", "",
         "One seed each. Selection metric: " + ", ".join(f"{i} {used[i][1]['selection_metric']}" for i in used) + ".", "",
         "## Validation (best epoch)", "| id | experiment | best epoch | trainable M | sentiment F1 | transition F1 | trajectory F1 | mean F1 | delta mean F1 vs E0 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for i, (name, s, c) in used.items():
        v = s["best_validation"]
        L.append(f"| {i} | {name} | {s['best_epoch']} | {c['trainable_params'] / 1e6:.2f} | " + " | ".join(f(v[k]) for k in VAL)
                 + f" | {v['mean_macro_f1'] - base['best_validation']['mean_macro_f1']:+.3f} |")
    L += ["", "## Test (selected checkpoint, scored once)", "| id | sentiment acc | sentiment F1 | transition F1 | trajectory acc | trajectory F1 | delta trajectory F1 vs E0 |",
          "|---|---|---|---|---|---|---|"]
    for i, (name, s, c) in used.items():
        t = s.get("test") or {}
        d = (t["trajectory_macro_f1"] - base["test"]["trajectory_macro_f1"]) if t.get("trajectory_macro_f1") is not None and base.get("test") else None
        L.append(f"| {i} | " + " | ".join(f(t.get(k)) for k in TEST) + f" | {'n/a' if d is None else format(d, '+.3f')} |")
    if skipped:
        L += ["", "Not included: " + "; ".join(f"{k} ({v})" for k, v in skipped.items())]
    if os.path.exists(a.panel1):
        p = json.load(open(a.panel1, encoding="utf-8"))
        L += ["", "## Old Panel 1 Amazon checkpoint (its own logged test numbers; different protocol, NOT comparable to the table above)",
              f"sentiment accuracy {f(p['sentiment']['accuracy'])}, sentiment macro-F1 {f(p['sentiment']['f1_macro'])}, trend macro-F1 {f(p['trend']['f1_macro'])}, "
              f"trajectory accuracy {f(p['trajectory']['accuracy'])}, trajectory macro-F1 {f(p['trajectory']['f1_macro'])}. It scored transition position 0 and chose "
              "epochs on validation loss; re-score it under the new protocol before comparing."]
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
