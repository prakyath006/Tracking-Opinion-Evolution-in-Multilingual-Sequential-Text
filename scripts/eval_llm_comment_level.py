"""
Module 9 — Instruction-tuned LLM classification (flan-t5), within and across domains, on the same comment-level
test sets and metrics as scripts/eval_comment_level.py.

Controlled prompt (fixed in src/baselines.py::LLMPromptClassifier.PROMPT_TEMPLATE, recorded in the output file):
  "Classify the sentiment of this review or comment. Answer with exactly one word: POSITIVE, NEGATIVE, MIXED, or UNKNOWN.
   Text: {text}\nSentiment:"
Two prediction paths (both reported, never mixed):
  score    constrained label-likelihood ranking over the four label strings (no invalid outputs possible).
  generate real greedy .generate(); the text is parsed by exact label-word match (case-insensitive substring of
           POSITIVE/NEGATIVE/MIXED/UNKNOWN). Anything else is an INVALID output: it is counted as a wrong prediction
           (assigned UNKNOWN) and reported as `invalid_output_rate`.

Settings
  zero_shot        the LLM exactly as released (no parameter is trained).
  calibrated:<S>   the 4-parameter per-class bias of the LLM head (the only trainable parameters at finetune_layers=0,
                   see LLMPromptClassifier) is fitted on the DEV split of source corpus S with the same
                   inverse-frequency weights as the supervised models, then applied unchanged to every target's TEST
                   split. This gives the source -> target (cross-domain / cross-lingual) matrix for the LLM.
Test sets: official test files (Dravidian) / test users (Amazon), duplicates removed, as prepared by
scripts/prepare_comment_level_data.py. Majority baselines are chosen from each corpus's TRAINING labels.

    python scripts/eval_llm_comment_level.py --model google/flan-t5-base --device cuda
    python scripts/eval_llm_comment_level.py --model google/flan-t5-small --max_rows 16 --device cpu   # smoke test
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True
import eval_utils as eu                                         # noqa: E402

CORPORA = ["amazon", "tamil", "malayalam", "kannada"]


def score_texts(model, texts, batch_size):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            out.append((model(texts[i:i + batch_size]) - model.class_bias).float().cpu().numpy())   # raw label log-likelihoods
    return np.concatenate(out) if out else np.zeros((0, 4))


def generate_texts(model, texts, batch_size):
    gen = []
    for i in range(0, len(texts), batch_size):
        gen.extend(model.generate_and_parse(texts[i:i + batch_size])[1])
    labels = model.LABEL_VOCAB
    preds, invalid = [], 0
    for g in gen:
        up = g.strip().upper()
        hit = next((k for k, l in enumerate(labels) if l in up), None)
        if hit is None:
            invalid += 1
            hit = labels.index("UNKNOWN")                   # invalid output counts as a (wrong) UNKNOWN prediction
        preds.append(hit)
    return np.asarray(preds), invalid, gen


def fit_bias(scores, y, steps=300, lr=0.05):
    """Fit the 4-parameter class bias on source dev scores (weighted cross-entropy, deterministic)."""
    import torch
    import torch.nn.functional as F
    counts = np.bincount(y, minlength=4).astype(float)
    w = torch.tensor(np.where(counts > 0, counts.sum() / (4 * np.maximum(counts, 1)), 0.0), dtype=torch.float)
    s, t = torch.tensor(scores, dtype=torch.float), torch.tensor(y)
    b = torch.zeros(4, requires_grad=True)
    opt = torch.optim.Adam([b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(s + b, t, weight=w).backward()
        opt.step()
    return b.detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--targets", nargs="+", default=CORPORA, choices=CORPORA)
    ap.add_argument("--mode", choices=["score", "generate", "both"], default="both")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--max_rows", type=int, default=0, help="smoke test: sample this many rows per file")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    import torch
    import transformers
    from baselines import LLMPromptClassifier
    use_cuda = a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())
    model = LLMPromptClassifier(model_name=a.model, finetune_layers=0, use_cuda=use_cuda).eval()

    data = {c: {s: eu.load_comment_csv(c, s, a.max_rows) for s in ("train", "dev", "test")} for c in a.targets}
    res = {"model": a.model, "library_versions": {"torch": torch.__version__, "transformers": transformers.__version__},
           "device": str(model.device), "smoke_test": bool(a.max_rows), "prompt_template": model.PROMPT_TEMPLATE,
           "label_vocab": model.LABEL_VOCAB,
           "parsing_rule": "generate: case-insensitive substring match against LABEL_VOCAB; no match = invalid output, counted wrong",
           "zero_shot": {}, "calibrated": {}, "generation": {}}

    scores = {c: {s: score_texts(model, data[c][s][0], a.batch_size) for s in ("dev", "test")} for c in a.targets} \
        if a.mode in ("score", "both") else {}
    for c in scores:
        y = np.asarray(data[c]["test"][1])
        maj = eu.majority_baseline(data[c]["train"][1], y[np.isin(y, eu.DETERMINATE)], eu.DETERMINATE, eu.SENTIMENT_NAMES[:3])
        rep = eu.comment_level_report(y, scores[c]["test"].argmax(1), bootstrap=a.bootstrap)
        res["zero_shot"][c] = {"report": rep, "majority_baseline_from_train_3class": maj}
        print(f"zero-shot score   {c:10s} macro-F1(3-class) {rep['primary_3class']['macro_f1']:.4f}", flush=True)
    for src in scores:
        bias = fit_bias(scores[src]["dev"], np.asarray(data[src]["dev"][1]))
        res["calibrated"][src] = {"fitted_class_bias": bias.tolist(), "targets": {}}
        for tgt in scores:
            y = np.asarray(data[tgt]["test"][1])
            rep = eu.comment_level_report(y, (scores[tgt]["test"] + bias).argmax(1), bootstrap=a.bootstrap)
            res["calibrated"][src]["targets"][tgt] = rep
            print(f"calibrated {src:9s}->{tgt:10s} macro-F1(3-class) {rep['primary_3class']['macro_f1']:.4f}", flush=True)
    if a.mode in ("generate", "both"):
        for c in a.targets:
            texts, y = data[c]["test"]
            preds, invalid, gen = generate_texts(model, texts, a.batch_size)
            rep = eu.comment_level_report(np.asarray(y), preds, bootstrap=a.bootstrap)
            res["generation"][c] = {"report": rep, "invalid_output_rate": invalid / max(1, len(texts)), "invalid_outputs": invalid,
                                    "examples_of_generated_text": gen[:10]}
            print(f"generate          {c:10s} macro-F1(3-class) {rep['primary_3class']['macro_f1']:.4f} invalid {invalid}/{len(texts)}", flush=True)
    out = a.output or os.path.join(ROOT, "outputs", "comment_level",
                                   f"llm_{os.path.basename(a.model)}_results{'_smoke' if a.max_rows else ''}.json")
    eu.save_json(res, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
