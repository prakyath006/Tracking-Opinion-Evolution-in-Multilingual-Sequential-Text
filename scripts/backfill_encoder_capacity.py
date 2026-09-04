"""
=============================================================================
One-off backfill — record encoder capacity on pre-existing result files
=============================================================================
The 2026-09-04 Kaggle runs predate `encoder_finetune_layers` being written
into outputs/logs/*.json, so the compiled comparison could not tell that the
models were trained with different amounts of trainable encoder capacity —
and outputs/metrics/module6_analysis.md consequently reported deltas between
models that were not comparable.

This script writes the capacity each existing run actually used. It does not
estimate: every value below is either read from the run's own saved config or
fixed by the code path that produced the file.

Provenance of each value
------------------------
full model (test_results_<run>.json)
    Read from outputs/checkpoints/best_model_<run>.pt, which stores the run's
    argparse namespace under "args". Both checkpoints record
    freeze_encoder=True, and src/model.py:145 maps that to finetune_layers=0.
    Independently confirmed: the saved optimizer_state_dict tracks 43 tensors,
    exactly the 43 non-encoder tensors in model_state_dict, so no encoder
    tensor was trainable.

mbert_sentence / xlmr_sentence (Group A)
    src/baselines.py's SentenceLevelTransformer took freeze_encoder=False by
    default and scripts/train_baselines.py never overrode it, so every
    encoder layer was trainable: 12 for both bert-base-multilingual-cased and
    xlm-roberta-base.

lstm_only / attention_only (Group B)
    scripts/train_baselines.py constructed DomainAdaptedEmbeddings with a
    hardcoded finetune_layers=0 and called embedder.eval(), so the encoder was
    frozen: 0.

textcnn
    No transformer encoder. Recorded as null — the metric does not apply.

Runs produced after this commit record the value themselves and are skipped.

Usage:
    python scripts/backfill_encoder_capacity.py [--dry-run]
=============================================================================
"""

import os
import sys
import json
import glob
import argparse

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(WORKSPACE_ROOT, "outputs", "logs")

# baseline name -> (encoder_finetune_layers, provenance)
BASELINE_CAPACITY = {
    "mbert_sentence":  (12, "SentenceLevelTransformer(freeze_encoder=False) default; all 12 mBERT layers trainable"),
    "xlmr_sentence":   (12, "SentenceLevelTransformer(freeze_encoder=False) default; all 12 XLM-R layers trainable"),
    "lstm_only":       (0,  "train_baselines.py built DomainAdaptedEmbeddings(finetune_layers=0) and called embedder.eval()"),
    "attention_only":  (0,  "train_baselines.py built DomainAdaptedEmbeddings(finetune_layers=0) and called embedder.eval()"),
    "textcnn":         (None, "no transformer encoder; metric not applicable"),
}


def full_model_capacity(run_id: str):
    """Read the run's own saved args from its checkpoint. Returns (layers, provenance)."""
    ckpt = os.path.join(WORKSPACE_ROOT, "outputs", "checkpoints", f"best_model_{run_id}.pt")
    if not os.path.exists(ckpt):
        return None, None
    import torch
    try:
        d = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved = d.get("args")
    if saved is None:
        return None, None
    saved = vars(saved) if hasattr(saved, "__dict__") else saved
    if "freeze_encoder" not in saved:
        return None, None
    layers = 0 if saved["freeze_encoder"] else 3
    return layers, (
        f"read from {os.path.relpath(ckpt, WORKSPACE_ROOT)} args: "
        f"freeze_encoder={saved['freeze_encoder']} -> src/model.py:145 finetune_layers={layers}"
    )


def apply(path: str, layers, provenance: str, dry_run: bool) -> str:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "encoder_finetune_layers" in data:
        return "skip (already recorded)"
    data["encoder_finetune_layers"] = layers
    data["encoder_finetune_layers_source"] = "backfilled: " + provenance
    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    return f"set to {layers}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    if not os.path.isdir(LOGS_DIR):
        print(f"No logs directory at {LOGS_DIR} — nothing to do.")
        return

    for path in sorted(glob.glob(os.path.join(LOGS_DIR, "test_results_*.json"))):
        run_id = os.path.basename(path)[len("test_results_"):-len(".json")]
        layers, prov = full_model_capacity(run_id)
        if prov is None:
            print(f"{os.path.basename(path):46s} SKIP — no checkpoint args to read; not guessing")
            continue
        print(f"{os.path.basename(path):46s} {apply(path, layers, prov, args.dry_run)}")

    for path in sorted(glob.glob(os.path.join(LOGS_DIR, "baseline_*.json"))):
        name = os.path.basename(path)[len("baseline_"):-len(".json")]
        match = next((b for b in BASELINE_CAPACITY if name.startswith(b)), None)
        if match is None:
            print(f"{os.path.basename(path):46s} SKIP — unrecognised baseline; not guessing")
            continue
        layers, prov = BASELINE_CAPACITY[match]
        print(f"{os.path.basename(path):46s} {apply(path, layers, prov, args.dry_run)}")

    if args.dry_run:
        print("\n(dry run — nothing written)")


if __name__ == "__main__":
    main()
