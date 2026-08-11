"""
=============================================================================
Module 6 — Model Expertise (Strengths & Limitations) Analysis Generator
=============================================================================
Reads outputs/metrics/results_table.json (scripts/compile_metrics.py's
output) and drafts the strengths/limitations comparison between the full
model (OpinionEvolutionTracker) and each baseline, backed by the actual
compiled numbers -- not assumptions.

This cannot produce a real Module 6 write-up until outputs/metrics/
results_table.json holds numbers from models trained on the real encoders
(bert-base-multilingual-cased / xlm-roberta-base) on Colab (see
notebooks/colab_training.ipynb). Every baseline/full-model run in this
repo's development environment used a small substitute encoder
(sentence-transformers/all-MiniLM-L6-v2) purely to prove the code paths
work -- see scripts/sanity_check_pipeline.py's docstring for why -- and
numbers from that substitute are not representative of the real models'
relative strengths. Running this script against them would produce a
document that LOOKS like Module 6 but is not backed by real results.

Usage:
    python scripts/generate_module6_analysis.py
    -> outputs/metrics/module6_analysis.md
=============================================================================
"""

import os
import sys
import json
import logging

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(WORKSPACE_ROOT, "outputs", "metrics")
RESULTS_JSON = os.path.join(METRICS_DIR, "results_table.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FULL_MODEL_NAME = "full_model (OpinionEvolutionTracker)"

BASELINE_STORY = {
    "mbert_sentence": "no sequential modeling (independent per-review mBERT classification)",
    "xlmr_sentence": "no sequential modeling (independent per-review XLM-R classification)",
    "lstm_only": "Bi-LSTM without attention (final hidden state only)",
    "attention_only": "attention without Bi-LSTM (attention over raw embeddings)",
    "textcnn": "no transformer, no sequential modeling (word embeddings + CNN)",
}


def load_rows():
    if not os.path.exists(RESULTS_JSON):
        logger.error(
            f"{RESULTS_JSON} not found. Run scripts/compile_metrics.py first "
            f"(after scripts/train.py and scripts/train_baselines.py have "
            f"produced real results)."
        )
        sys.exit(1)
    with open(RESULTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"


def delta(full, other):
    if full is None or other is None:
        return "N/A"
    d = full - other
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def generate(rows, source: str) -> str:
    in_domain_rows = [r for r in rows if r["setting"] == "in-domain" and r["source"] == source]
    full_row = next((r for r in in_domain_rows if r["model"] == FULL_MODEL_NAME), None)

    lines = [f"## In-domain comparison — trained/tested on `{source}`\n"]

    if full_row is None:
        lines.append(
            f"*No in-domain result for the full model on `{source}` in "
            f"results_table.json -- run `train.py --domain ...` and "
            f"`compile_metrics.py` first.*\n"
        )
        return "\n".join(lines)

    lines.append(
        "| Model | Sentiment F1 | Trajectory F1 | SCS | Δ Sentiment F1 (full - baseline) | Δ Trajectory F1 |"
    )
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| **{FULL_MODEL_NAME}** | {fmt(full_row['sentiment_f1_macro'])} | "
        f"{fmt(full_row['trajectory_f1_macro'])} | {fmt(full_row['scs_mean'])} | - | - |"
    )

    findings = []
    for row in in_domain_rows:
        if row["model"] == FULL_MODEL_NAME:
            continue
        d_sent = delta(full_row["sentiment_f1_macro"], row["sentiment_f1_macro"])
        d_traj = delta(full_row.get("trajectory_f1_macro"), row.get("trajectory_f1_macro"))
        lines.append(
            f"| {row['model']} | {fmt(row['sentiment_f1_macro'])} | "
            f"{fmt(row.get('trajectory_f1_macro'))} | {fmt(row.get('scs_mean'))} | "
            f"{d_sent} | {d_traj} |"
        )
        story = BASELINE_STORY.get(row["model"], "")
        findings.append(f"- **{row['model']}** ({story}): "
                         f"sentiment F1 delta {d_sent}, trajectory F1 delta {d_traj}.")

    lines.append("\n### What the numbers say\n")
    lines.extend(findings)
    lines.append(
        "\n*Fill in the narrative once real numbers are present: which delta is largest "
        "(attention vs Bi-LSTM's individual contribution), whether sentence-level "
        "baselines lag most on trajectory (expected, since they have no sequence "
        "signal at all) or also on sentiment (would suggest the full model's gains "
        "aren't just from sequence modeling), and whether SCS separates the full "
        "model from lstm_only/attention_only more than F1 does (would support SCS "
        "as a distinct, useful metric rather than one F1 already captures).*\n"
    )
    return "\n".join(lines)


def main():
    rows = load_rows()
    sources = sorted({r["source"] for r in rows if r["setting"] == "in-domain"})

    doc = ["# Module 6 — Model Expertise (Strengths & Limitations)\n",
           "Generated by scripts/generate_module6_analysis.py from outputs/metrics/results_table.json.\n"]
    for source in sources:
        doc.append(generate(rows, source))

    out_path = os.path.join(METRICS_DIR, "module6_analysis.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    logger.info(f"Saved: {out_path}")
    print("\n".join(doc))


if __name__ == "__main__":
    main()
