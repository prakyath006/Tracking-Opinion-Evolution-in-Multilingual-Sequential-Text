"""
=============================================================================
Metrics Compilation — Module 5/9: All Models, In-Domain + Cross-Domain
=============================================================================
Gathers everything scripts/train.py, scripts/train_baselines.py and
scripts/cross_domain_eval.py wrote to outputs/, and compiles a single
comparison table: Accuracy / Precision / Recall / F1 (macro) per task head,
plus SCS, for every model x setting that has been run.

Sources read (all optional -- rows are added only for what actually exists):
  outputs/logs/test_results_<run_id>.json              full model, in-domain
  outputs/logs/baseline_<name>_<run_id>.json            baselines, in-domain
  outputs/cross_domain/cross_domain_results_<lang>.json full model, cross-domain

Output:
  outputs/metrics/results_table.csv
  outputs/metrics/results_table.md
  outputs/metrics/results_table.json

Usage:
    python scripts/compile_metrics.py
=============================================================================
"""

import os
import sys
import json
import glob
import logging
from typing import Dict, List, Optional

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(WORKSPACE_ROOT, "outputs", "logs")
CROSS_DOMAIN_DIR = os.path.join(WORKSPACE_ROOT, "outputs", "cross_domain")
METRICS_DIR = os.path.join(WORKSPACE_ROOT, "outputs", "metrics")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

COLUMNS = [
    "model", "setting", "source", "target", "encoder_finetune_layers",
    "sentiment_accuracy", "sentiment_f1_macro",
    "trend_f1_macro", "trajectory_accuracy", "trajectory_f1_macro",
    "scs_mean",
]


def _get(d: Dict, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k)
    return d if d is not None else default


def row_from_full_model_test_results(run_id: str, data: Dict) -> Dict:
    sent, trend, traj, scs = data.get("sentiment", {}), data.get("trend", {}), data.get("trajectory", {}), data.get("scs", {})
    return {
        "model": "full_model (OpinionEvolutionTracker)",
        "setting": "in-domain",
        "source": run_id, "target": run_id,
        "encoder_finetune_layers": data.get("encoder_finetune_layers"),
        "sentiment_accuracy": sent.get("accuracy"), "sentiment_f1_macro": sent.get("f1_macro"),
        "trend_f1_macro": trend.get("f1_macro"),
        "trajectory_accuracy": traj.get("accuracy"), "trajectory_f1_macro": traj.get("f1_macro"),
        "scs_mean": scs.get("scs_mean"),
    }


def row_from_baseline_result(data: Dict) -> Dict:
    test = data.get("test", {})
    sent, traj, scs = test.get("sentiment", {}), test.get("trajectory", {}), test.get("scs", {})
    return {
        "model": data.get("baseline"),
        "setting": "in-domain",
        "source": data.get("run_id"), "target": data.get("run_id"),
        "encoder_finetune_layers": data.get("encoder_finetune_layers"),
        "sentiment_accuracy": sent.get("accuracy"), "sentiment_f1_macro": sent.get("f1_macro"),
        "trend_f1_macro": None,  # Group A/B baselines don't have a trend head (see train_baselines.py)
        "trajectory_accuracy": traj.get("accuracy"), "trajectory_f1_macro": traj.get("f1_macro"),
        "scs_mean": scs.get("scs_mean"),
    }


def rows_from_cross_domain_result(data: Dict) -> List[Dict]:
    rows = []
    for setup_name, result in data.items():
        source = result.get("source_domain", setup_name.split("_to_")[0])
        target = setup_name.split("_to_")[-1]
        setting = "in-domain" if source.replace("->", "") == target or source == target else "cross-domain"
        # setup_name is like "amazon_to_amazon" or "amazon_to_dravidian_tamil"
        parts = setup_name.split("_to_")
        src, tgt = parts[0], parts[-1]
        setting = "in-domain" if src == tgt else "cross-domain"
        sent, traj, scs = result.get("sentiment", {}), result.get("trajectory", {}), result.get("scs", {})
        rows.append({
            "model": "full_model (OpinionEvolutionTracker)",
            "setting": setting,
            "source": src, "target": tgt,
            "encoder_finetune_layers": data.get("encoder_finetune_layers"),
            "sentiment_accuracy": sent.get("accuracy"), "sentiment_f1_macro": sent.get("f1_macro"),
            "trend_f1_macro": None,  # evaluate_cross_domain() scores sentiment+trajectory only
            "trajectory_accuracy": traj.get("accuracy"), "trajectory_f1_macro": traj.get("f1_macro"),
            "scs_mean": scs.get("scs_mean"),
        })
    return rows


def compile_all() -> List[Dict]:
    rows = []

    # ── Full model, in-domain (per run_id) ──
    for path in sorted(glob.glob(os.path.join(LOGS_DIR, "test_results_*.json"))):
        run_id = os.path.basename(path)[len("test_results_"):-len(".json")]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows.append(row_from_full_model_test_results(run_id, data))
        logger.info(f"Loaded full-model in-domain result: {run_id}")

    # ── Baselines, in-domain ──
    for path in sorted(glob.glob(os.path.join(LOGS_DIR, "baseline_*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows.append(row_from_baseline_result(data))
        logger.info(f"Loaded baseline result: {data.get('baseline')}/{data.get('run_id')}")

    # ── Full model, cross-domain (both directions) ──
    for path in sorted(glob.glob(os.path.join(CROSS_DOMAIN_DIR, "cross_domain_results_*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows.extend(rows_from_cross_domain_result(data))
        logger.info(f"Loaded cross-domain results: {os.path.basename(path)}")

    return rows


def format_markdown(rows: List[Dict]) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "|" + "|".join(["---"] * len(COLUMNS)) + "|"
    lines = [header, sep]
    for row in rows:
        cells = []
        for col in COLUMNS:
            v = row.get(col)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    rows = compile_all()

    if not rows:
        logger.warning(
            "No results found under outputs/logs or outputs/cross_domain. "
            "Run scripts/train.py, scripts/train_baselines.py and "
            "scripts/cross_domain_eval.py first."
        )
        return

    # CSV
    import csv
    csv_path = os.path.join(METRICS_DIR, "results_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved: {csv_path}")

    # JSON
    json_path = os.path.join(METRICS_DIR, "results_table.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # Markdown
    md = format_markdown(rows)
    md_path = os.path.join(METRICS_DIR, "results_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    logger.info(f"Saved: {md_path}")

    print("\n" + md)


if __name__ == "__main__":
    main()
