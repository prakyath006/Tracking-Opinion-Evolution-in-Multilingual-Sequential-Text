# New modules 7–9 — status (2026-10-20)

Three modules added to close the gaps raised in the guide's feedback. **Implemented and smoke-tested locally; the full runs are pending on Kaggle.** Nothing below is a trained-model result except where stated.

| # | Module | Files | Purpose | Status |
|---|---|---|---|---|
| 7 | Evaluation and reproducibility | `src/eval_utils.py`, `scripts/compute_valid_baselines.py`, `scripts/export_predictions.py`, `--seed` in `scripts/train.py` and `scripts/train_baselines.py`, `tests/test_eval_utils.py` | valid majority baselines from training labels, trend metrics over observed transitions only, gold-sequence SCS, prediction export, bootstrap intervals, seeding | 7a **run** (real, label-only numbers); 7b export smoke-tested on 4 users, full export pending |
| 8 | Comment-level multilingual / cross-domain sentiment | `scripts/prepare_comment_level_data.py`, `scripts/eval_comment_level.py` | official DravidianCodeMix splits (no windows), duplicate/leakage control, source→target matrix over Amazon, Tamil, Malayalam, Kannada, 5 seeds, intervals, majority baselines | data preparation **run**; evaluation smoke-tested on 48-row samples, full run pending |
| 9 | LLM classification | `scripts/eval_llm_comment_level.py` (uses `LLMPromptClassifier` in `src/baselines.py`) | fixed-prompt flan-t5, constrained scoring and real generation with parsing and invalid-output rate, zero-shot and source-calibrated cross-domain matrix | smoke-tested with `flan-t5-small` on 12-row samples, full run pending |

Test suite: 38 tests pass (31 ontology + 7 evaluation utilities).

## Results that exist now (no model involved)

**Valid majority baselines, Amazon test users** (class chosen from training labels; `outputs/metrics/valid_baselines_amazon.md`):

| Head | Majority class | Accuracy | Macro-F1 | 95% CI over test users |
|---|---|---|---|---|
| Sentiment | POSITIVE | 0.7523 | 0.2862 | 0.2817–0.2905 |
| Trend, all positions (with placeholder) | STABLE | 0.7576 | 0.2874 | 0.2830–0.2912 |
| Trend, observed transitions only | STABLE | 0.6827 | 0.2705 | 0.2644–0.2761 |
| Trajectory | STABLE | 0.4963 | 0.1658 | 0.1578–0.1729 |

Gold-sequence SCS on the 800 test users: 0.6699 (std 0.3574), against 0.564 for the joint model and 0.893 for Bi-LSTM-only.

**Comment-level data** (`outputs/comment_level/data_report.json`; files in `data/comment_level/`, git-ignored). Official dev/test files, windows not used; dev/test texts seen in training or in dev, and within-split duplicates, are removed:

| Corpus | Train | Dev rows (removed as seen in train) | Test rows (removed as seen in train/dev; duplicates) |
|---|---|---|---|
| Tamil | 33,531 | 4,145 (60) | 4,112 (76; 3) |
| Malayalam | 14,514 | 1,671 (157) | 1,654 (159; 3) |
| Kannada | 5,218 | 615 (41) | 632 (27; 2) |
| Amazon (reviews) | 15,619 | 3,351 (110) | 3,200 (140; 46) |

Training texts with conflicting labels: Tamil 45, Malayalam 94, Kannada 15, Amazon 15.

## Design notes
- Module 8 trains the same head as the `mbert_sentence`/`xlmr_sentence` baselines on cached frozen-encoder [CLS] vectors; this is the same model at much lower cost. The primary metric is 3-class macro-F1 on comments whose gold label is POS/NEG/MIX, so all corpora (including Amazon, which has no UNK) are comparable; four-class results are also stored.
- Module 9 trains no LLM weights. "Calibrated:S" fits only the 4-parameter class bias on S's dev scores (the trainable part of `LLMPromptClassifier`), then applies it to every target's test set.
- Invalid LLM generations are counted as wrong predictions and reported as `invalid_output_rate`.

## Run on Kaggle
Upload `data/comment_level/` (about 15 MB) and, if the code is not pushed to GitHub, `src/`, `scripts/` and `outputs/checkpoints/best_model_amazon.pt` as a private dataset. Then:

```bash
python scripts/eval_comment_level.py --encoder mbert --seeds 0 1 2 3 4 --device cuda
python scripts/eval_comment_level.py --encoder xlmr  --seeds 0 1 2 3 4 --device cuda
python scripts/eval_llm_comment_level.py --model google/flan-t5-base --device cuda
python scripts/export_predictions.py --phase export --device cuda --batch_size 16
python scripts/export_predictions.py --phase analyze
```
Outputs: `outputs/comment_level/*.json`, `outputs/predictions/*.csv`, `outputs/metrics/amazon_prediction_stats.{json,md}`.
For the seeded Amazon retraining use `--seed N` with `scripts/train.py` and `scripts/train_baselines.py` (matched `--epochs`; per-model `--lr`).

## Not done / not claimed
- No multilingual, cross-domain or LLM result exists until the Kaggle runs finish.
- The Amazon retraining with matched baselines and seeds is not done; the existing Amazon results are unchanged.
- Aspect-level trajectories and the combined final-decision head are not implemented.
- Smoke-test numbers (tiny samples) are meaningless and were not saved in the repository.
