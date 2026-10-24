# Modules 7–9 — Team Guide

Shareable summary of the three modules added to the project (pushed to `main`, commits `c0fcade`, `ae0a6fc`, `e12dea5`). It explains what each module does, what already has real numbers, and exactly how to run the rest on Kaggle.

## 1. Why these modules
Our guide's feedback asked for things the project did not yet have: valid baselines, a correct multilingual evaluation, and an LLM classifier evaluated across domains. The old Dravidian sliding-window experiments were invalid (windows overlap, so test comments leak into training, and adjacent comments are unrelated). These modules fix that.

| Module | Name | Fixes |
|---|---|---|
| 7 | Evaluation and reproducibility | missing majority baseline, no seeds, no confidence intervals, trend metrics polluted by an artificial first label, no gold SCS |
| 8 | Comment-level multilingual / cross-domain sentiment | invalid Dravidian evaluation; no source→target (cross-lingual, cross-domain) results |
| 9 | LLM classification | LLM baseline coded but never run; no cross-domain LLM evaluation |

The trajectory task still uses **Amazon only**. Modules 8 and 9 evaluate per-comment sentiment across languages and domains.

## 2. What is new in the repo

| File | Module | What it does |
|---|---|---|
| `src/eval_utils.py` | 7 | seeding, majority baseline from training labels, observed-transition-only metrics, gold SCS, bootstrap confidence intervals, comment-level metrics |
| `tests/test_eval_utils.py` | 7 | 7 unit tests (all 38 tests pass: 31 ontology + 7 new) |
| `scripts/compute_valid_baselines.py` | 7 | label-only reference scores for Amazon (no model, no GPU) |
| `scripts/export_predictions.py` | 7 | runs the saved Amazon checkpoint on the test users, saves per-position predictions, then computes intervals and observed-only trend scores |
| `scripts/train.py`, `scripts/train_baselines.py` | 7 | new optional `--seed N` (default: unseeded, so old runs are unchanged) |
| `scripts/prepare_comment_level_data.py` | 8 | builds comment-level data from the official DravidianCodeMix train/dev/test files and the Amazon split, removes duplicates and leakage |
| `scripts/eval_comment_level.py` | 8 | trains the sentence-level head on each source corpus and tests it on every target corpus (5 seeds) |
| `scripts/eval_llm_comment_level.py` | 9 | flan-t5 with a fixed prompt, scoring and real generation, zero-shot and calibrated cross-domain |
| `docs/new_modules_status.md` | – | status record |

## 3. Results that already exist (no GPU needed)

**Valid majority baselines, Amazon test users** (class picked from the *training* labels; `outputs/metrics/valid_baselines_amazon.md`)

| Head | Majority class | Accuracy | Macro-F1 | 95% CI over test users |
|---|---|---|---|---|
| Sentiment | POSITIVE | 0.7523 | 0.2862 | 0.2817–0.2905 |
| Trend (observed transitions only) | STABLE | 0.6827 | 0.2705 | 0.2644–0.2761 |
| Trajectory | STABLE | 0.4963 | 0.1658 | 0.1578–0.1729 |

Gold-sequence SCS on the 800 test users: **0.670** (the joint model scores 0.564, Bi-LSTM-only 0.893).

**Comment-level data prepared** (`outputs/comment_level/data_report.json`)

| Corpus | Train | Dev | Test |
|---|---|---|---|
| Tamil | 33,531 | 4,145 | 4,112 |
| Malayalam | 14,514 | 1,671 | 1,654 |
| Kannada | 5,218 | 615 | 632 |
| Amazon (reviews) | 15,619 | 3,351 | 3,200 |

Dev/test comments that also occur in training were removed (e.g. 76 in Tamil test, 159 in Malayalam test).

## 4. What is NOT done yet
- Modules 8 and 9 have been **smoke-tested only** on tiny samples. There are **no** multilingual, cross-domain or LLM results until the Kaggle runs finish.
- The Amazon retraining (matched epochs, tuned learning rates, several seeds) has not been done.
- Aspect-level trajectories and a combined final-decision output are not implemented.

## 5. Running it on Kaggle

**Setup (one cell, GPU on, Internet on)**
```python
!git clone https://github.com/prakyath006/Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text.git
%cd Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text
!pip -q install -r requirements.txt
# data: copy from our private Kaggle dataset (adjust the path to match your mount)
!mkdir -p data && cp -r /kaggle/input/datasets/prakyathnandigam/opinion-tracker-data/data/* data/
!ls data
```
`data/` must contain `raw/` (official DravidianCodeMix files) and `preprocessed/`. If `raw/` is missing, upload the folder `data/comment_level/` from the laptop as a dataset instead and skip the prepare step.

**Module 8 — multilingual and cross-domain sentiment**
```python
!python scripts/prepare_comment_level_data.py          # ~15 seconds
!python scripts/eval_comment_level.py --encoder mbert --seeds 0 1 2 3 4 --device cuda
!python scripts/eval_comment_level.py --encoder xlmr  --seeds 0 1 2 3 4 --device cuda
```
Output: `outputs/comment_level/mbert_comment_level_results.json` and `xlmr_comment_level_results.json`. Each source→target pair gives 3-class macro-F1 (mean ± std over seeds), per-class scores, confusion matrix, a bootstrap interval, the majority baseline from the source training labels, and the % drop versus the in-domain score.

**Module 9 — LLM classification**
```python
!python scripts/eval_llm_comment_level.py --model google/flan-t5-base --device cuda
```
Output: `outputs/comment_level/llm_flan-t5-base_results.json` with zero-shot scores, the calibrated source→target matrix, generation results and the invalid-output rate.

**Module 7 — predictions and intervals for the Amazon model**
```python
!mkdir -p outputs/checkpoints
!cp /kaggle/input/<checkpoints-dataset>/best_model_amazon.pt outputs/checkpoints/    # from opinion-tracker-checkpoints
!python scripts/export_predictions.py --phase export --device cuda --batch_size 16
!python scripts/export_predictions.py --phase analyze
```
Output: `outputs/predictions/*.csv` and `outputs/metrics/amazon_prediction_stats.md` (trend over observed transitions only, bootstrap intervals, predicted vs gold SCS).

**Seeded retraining (when we rerun Amazon)**
```python
!python scripts/train.py --domain amazon --freeze_encoder --epochs 10 --lr 0.001 --batch_size 16 --seed 0
```
Use the same `--epochs` for the baselines and set each baseline's `--lr` explicitly, then repeat with `--seed 1 … 4`.

**Download:** zip `outputs/comment_level`, `outputs/predictions` and `outputs/metrics` from the notebook's Output panel and give them to Prakyath. Do not rename the files.

## 6. How to read the results
- **3-class macro-F1** counts only comments whose gold label is POSITIVE / NEGATIVE / MIXED, so all four corpora (Amazon has no UNKNOWN) can be compared. Four-class results are also stored.
- **Majority baseline** is always chosen from training labels.
- **Invalid LLM output** (text that is not one of the four label words) is counted as a wrong prediction and reported as a rate.
- Do not compare the new numbers with the old Tamil window results; those were invalid.

## 7. Common problems
| Problem | Fix |
|---|---|
| `FileNotFoundError ... prepare_comment_level_data.py first` | run the prepare step, or upload `data/comment_level/` |
| `data/raw` missing | upload `data/comment_level/` as a dataset and copy it to `data/comment_level` in the repo |
| CUDA out of memory in the LLM script | add `--batch_size 8` |
| Session timed out | rerun the same command; embeddings are cached in `data/comment_level/cache`, so a rerun is faster |
| Hugging Face download error | switch Internet on in the notebook settings |
| Any error | send Prakyath the full traceback; do not edit the scripts on Kaggle |

---

# Part B — Slide-ready details (for making the PPT)

## 8. One-picture data flow
```
 Amazon Beauty reviews ─┐                        ┌─► Module 7: valid baselines, gold SCS, seeds, intervals
 (user-level split)     │                        │        (Amazon trajectory task, sequence level)
                        ├─► labels via ontology ─┤
 DravidianCodeMix       │   (POS/NEG/MIX/UNK)    └─► Module 8: comment-level source -> target matrix
 official train/dev/test┘   duplicates removed         (Amazon, Tamil, Malayalam, Kannada)
                                                      │
                                                      └─► Module 9: flan-t5 on the same test sets
                                                           (zero-shot + calibrated cross-domain)
```
Earlier modules (1–6): ontology, aspect/WSD tagger, encoder perplexity, sequential model with calibration/SCS, cross-domain analysis, baseline comparison. Modules 7–9 build on them and fix what the guide questioned.

## 9. Module 7 — design details
- **Problem it solves:** earlier "majority" numbers used the test class counts (invalid); no seeds; no intervals; the trend head was scored on an artificial first-position label; gold SCS was never computed.
- **Majority baseline:** most frequent class of the *training* labels, scored on the test labels.
- **Observed-transition trend metric:** drops position 1 of every sequence (the placeholder STABLE), leaving 2,591 of 3,391 test positions.
- **Gold SCS:** SCS of the true sentiment sequences (0.670), so predicted smoothness can be judged against the truth.
- **Bootstrap:** resamples whole test users (2,000 resamples), percentile 95% interval.
- **Seeding:** `--seed` sets python, numpy, torch and cuDNN; default stays unseeded so the old results are unchanged.
- **Prediction export:** one row per review (user, position, gold/pred sentiment, gold/pred trend) and one per user (gold/pred trajectory); the checkpoint is loaded with strict matching and stops on any mismatch.
- **Tested:** 7 unit tests; export smoke-tested on 4 users.

## 10. Module 8 — design details
- **Data:** official DravidianCodeMix files (Tamil, Malayalam, Kannada) and Amazon reviews flattened from the existing user split (val written as dev). No sliding windows, no pseudo-threads.
- **Preprocessing:** drop comments tagged as not in the target language, clean text, map labels through the ontology to POS/NEG/MIX/UNK.
- **Leakage control (dev/test only):** remove empty texts, texts seen in training, test texts seen in dev, and within-split duplicates. Counts are in `data_report.json`.
- **Model:** the same head as the mBERT/XLM-R sentence baselines (768 → 256, LayerNorm, ReLU, dropout 0.3, → 4 classes) on frozen `[CLS]` vectors. Vectors are computed once and cached, so training is fast and equals the frozen-encoder baseline.
- **Training:** AdamW lr 1e-3, weight decay 1e-4, batch 64, up to 30 epochs, early stopping (patience 5) on source dev loss, inverse-frequency class weights from the training split.
- **Evaluation:** every source → every target (16 pairs per encoder), 5 seeds; in-domain = same corpus, transfer = different corpus.
- **Metric:** primary = 3-class macro-F1 on gold POS/NEG/MIX only (comparable across corpora); also 4-class, per-class P/R/F1, confusion matrix, bootstrap interval over comments, majority baseline from source training labels, % degradation versus in-domain.
- **Tested:** run end to end on 48-row samples.

## 11. Module 9 — design details
- **Prompt (fixed, recorded in the output file):** "Classify the sentiment of this review or comment. Answer with exactly one word: POSITIVE, NEGATIVE, MIXED, or UNKNOWN." followed by "Text: <comment>" and "Sentiment:".
- **Two prediction paths, never mixed:** (1) scoring: rank the four label strings by likelihood, so no invalid output is possible; (2) generation: real greedy `.generate()`, parsed by matching a label word; anything else is an *invalid output* (counted wrong, reported as a rate).
- **Settings:** zero-shot (no parameter trained) and *calibrated on S*: only the 4 per-class bias values are fitted on source S's dev scores (Adam, lr 0.05, 300 steps, weighted cross-entropy), then applied unchanged to every target's test set. This gives the LLM a source → target matrix like Module 8.
- **Same test sets and metrics as Module 8**, so supervised and LLM results are directly comparable.
- **Tested:** end to end with `flan-t5-small` on 12-row samples; the real run uses `flan-t5-base`.

## 12. Key messages for the guide
1. We replaced invalid experiments with a leakage-controlled, comment-level protocol on the official splits.
2. We now have valid baselines and uncertainty tooling: majority from training labels, observed-only trend, gold SCS, seeds, bootstrap.
3. LLM classification and cross-domain evaluation are implemented under the same protocol; runs are on Kaggle.
4. Trajectory claims stay Amazon-only; modules 8–9 evaluate per-comment sentiment across languages and domains.
5. Honest status: implemented and smoke-tested; Kaggle results pending.

## 13. Questions the guide may ask
- *"Why not use the old Tamil results?"* Overlapping windows leaked test comments into training (about 56% of test positions) and adjacent comments are unrelated, so they were not histories.
- *"Where are the multilingual results?"* The pipeline and data are ready; the Kaggle run produces them next.
- *"Is the LLM fine-tuned?"* No. Zero-shot, plus an optional 4-parameter bias calibration on the source dev set.
- *"How do you handle LLM outputs that are not a label?"* They count as wrong and are reported as an invalid-output rate.
- *"Does the majority baseline change the old conclusion?"* The values equal the earlier oracle ones because train and test share the same majority class; the baseline is now valid, with intervals.
- *"What does ARR / hallucination mean?"* Still unclear; we need the guide to define them.

## 14. Suggested slide outline (about 8 slides)
1. Where we stand and the guide's gaps
2. Module 7: valid baselines and reproducibility (with the baseline table)
3. Module 8: comment-level multilingual / cross-domain protocol (data counts table, leakage control)
4. Module 9: LLM classification (prompt, two paths, calibration)
5. Real numbers available now (baselines, gold SCS, data counts)
6. Kaggle plan and what the runs will produce
7. Limits and what we do not claim
8. Summary and asks for the guide (define ARR / hallucination, confirm Amazon-only trajectory scope)
