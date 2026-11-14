# Training the aspect-level models

Two new entry points train the components of `docs/aspect_models.md` on **manually annotated** aspect data:

| Script | Trains | Task |
|---|---|---|
| `scripts/train_aspect_extractor.py` | `AspectExtractor` | BIO span tagging of aspects |
| `scripts/train_aspect_sentiment.py` | `AspectSentimentClassifier` | POSITIVE / NEGATIVE / NEUTRAL / MIXED toward one aspect |

Shared machinery is in `src/aspect_training.py`. Nothing in the existing OET model, scripts or checkpoints is used or changed.

**Status.** Both scripts are tested only on synthetic fixtures (tiny random encoders in the unit tests, plus one two-epoch smoke run with the real mBERT on a synthetic file). **No annotated
aspect data exists yet**, so no real run has been made and no result is claimed. The numbers a smoke run prints mean nothing.

## Protocol

1. **Input.** One or more completed annotation CSVs (`--annotations`). Files are read-only and validated with `scripts/validate_aspect_annotations.py`'s rules; a file with errors stops the run (`--skip_invalid` drops
   only the bad messages). Only finished messages (`ANNOTATED` / `ADJUDICATED`) are used; unannotated ones are never treated as "no aspect". A message id may occur in only one file, so two annotators' copies must be
   adjudicated into one file first.
2. **Splits.** Whole products go to train / val / test (`--split_ratios`, default 0.7 / 0.15 / 0.15; `--split_seed`), checked for leakage. The split depends only on the product ids and `--split_seed`, so both scripts and all runs
   with the same seed use identical splits. Identical review texts that still occur in several splits are counted and warned about.
3. **No test leakage.** Only the training split is fitted. Only the validation split chooses the checkpoint, stops early or steps a plateau scheduler. The test split is evaluated once, after training, with the validation-chosen
   checkpoint (`--no_test` skips it but keeps the same splits). Tested: changing only the test products' gold labels leaves the whole training and validation history identical.
4. **Checkpoint selection** on a validation metric (higher is better; `--selection_metric`):
   * extractor: `macro_span_f1` (default: the mean of per-category exact span F1 with `--category_tags`, equal to span F1 with generic tags), `span_f1`, `token_macro_f1`;
   * sentiment: `macro_f1` (default, fixed four classes), `macro_f1_present_classes`.
   Absent classes only rescale `macro_f1`, so both orderings agree on a fixed validation set.

## Metrics

**Extractor.** Span precision, recall and F1, micro over messages, on exact character boundaries (`span_*`), with the category too when `--category_tags` is used. Gold spans are decoded from the gold tags the same way as the
predictions, so boundaries are compared on the same token representation. Also reported: a relaxed F1 (one-to-one match at overlap / union >= 0.5), per-category scores and their macro with category tags, token-level macro-F1, loss.
Messages with an unclear aspect are not scored. Spans lost to truncation and spans that start or end inside a token are counted in `config.json` (`extraction_reports`).

**Sentiment.** Accuracy, macro-F1 (fixed four classes and over the classes that occur), per-class precision, recall, F1 and support, the 4x4 confusion matrix (rows gold, columns predicted), loss.

## Output of one run (`outputs/aspect_experiments/<name>/`, never overwritten)

| File | Content |
|---|---|
| `config.json` | every setting, parameter counts, environment, warnings, extraction report or class counts |
| `splits.json` | the products, messages and mentions of each split; class counts (sentiment) |
| `class_weights.json` | (sentiment) the weights used and the training counts they came from |
| `train_log.jsonl`, `history.json` | per epoch: training loss, validation metrics, learning rates |
| `best_checkpoint.pt` | chosen on validation; a frozen encoder is not saved (small file), a trained one is (about 700 MB mBERT, 1.1 GB XLM-R) |
| `best_validation.json`, `test_metrics.json` | full metrics (per-class, confusion matrix / per-category span scores) |
| `predictions/validation_predictions.csv`, `predictions/test_predictions.csv` | one row per message (extractor: gold and predicted spans) or per aspect example (sentiment: gold, predicted, four probabilities) |
| `summary.json` | headline numbers, selected epoch, warnings |

## Small-data warning

A banner is printed and stored in `config.json` and `summary.json` (`dataset_too_small_for_meaningful_evaluation`) when, by heuristic floors (not statistical guarantees): fewer than 30 products in total, fewer than 5 products or
100 aspect items in the validation or test split, fewer than 20 examples of a sentiment class in an evaluation split (and classes with none), fewer than 500 training items, a single annotator, or identical texts across splits.
The run still goes ahead (the warning is not a block); an empty validation split, or one without any aspect, is an error because no checkpoint could be chosen.

## Options in common

`--encoder mbert|xlmr` (or any Hugging Face name), `--finetune frozen|partial|full` with `--finetune_layers`, separate `--lr_head` and `--lr_encoder` (default 2e-5 when the encoder trains), `--schedule`, `--warmup_ratio`,
`--epochs`, `--batch_size`, `--grad_accumulation`, `--max_token_length`, `--class_weights none|sqrt|full` (training split only), `--seed`, `--patience`, `--sequence_types`, `--device`, `--name`, `--output_root`.
Extractor only: `--category_tags`, `--unclear exclude|mask`, `--overlap error|merge`. Sentiment only: `--level message_category|mention`, `--conditioning category|span|category_and_span`, `--pooling cls|mean`, `--hidden_dim`, `--dropout`.

## Kaggle GPU commands

Prerequisites, none of which exist yet: (1) the **adjudicated annotation CSV(s)** uploaded as a private Kaggle dataset; (2) this code on Kaggle, either by `git clone` after the new files are pushed, or by uploading the code as a dataset;
(3) Internet on (the encoders are downloaded from Hugging Face). Adjust the dataset paths.

```python
# Setup (GPU on, Internet on)
!git clone https://github.com/prakyath006/Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text.git
%cd Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text
!pip -q install -r requirements.txt
ANN = "/kaggle/input/<annotations-dataset>/<adjudicated_annotations>.csv"
!python scripts/validate_aspect_annotations.py {ANN} --strict          # must report no errors and nothing unfinished
!python -m pytest tests/test_aspect_training.py tests/test_aspect_models.py tests/test_aspect_data.py tests/test_validate_aspect_annotations.py -q
```

```python
# Aspect extraction: XLM-R and mBERT, generic tags, full fine-tuning
!python -u scripts/train_aspect_extractor.py --annotations {ANN} --encoder xlmr --finetune full --lr_encoder 2e-5 --lr_head 1e-3 --epochs 10 --batch_size 16 --max_token_length 128 --seed 0 --name ext_xlmr_full_generic_s0
!python -u scripts/train_aspect_extractor.py --annotations {ANN} --encoder mbert --finetune full --lr_encoder 2e-5 --lr_head 1e-3 --epochs 10 --batch_size 16 --max_token_length 128 --seed 0 --name ext_mbert_full_generic_s0

# Aspect extraction with category tags, and a frozen-encoder baseline
!python -u scripts/train_aspect_extractor.py --annotations {ANN} --encoder xlmr --finetune full --category_tags --epochs 10 --batch_size 16 --seed 0 --name ext_xlmr_full_categories_s0
!python -u scripts/train_aspect_extractor.py --annotations {ANN} --encoder xlmr --finetune frozen --lr_head 1e-3 --class_weights sqrt --epochs 20 --batch_size 32 --seed 0 --name ext_xlmr_frozen_generic_s0
```

```python
# Aspect sentiment: XLM-R and mBERT with category and span conditioning, full fine-tuning
!python -u scripts/train_aspect_sentiment.py --annotations {ANN} --encoder xlmr --finetune full --conditioning category_and_span --lr_encoder 2e-5 --lr_head 1e-3 --epochs 10 --batch_size 16 --max_token_length 128 --seed 0 --name sent_xlmr_full_catspan_s0
!python -u scripts/train_aspect_sentiment.py --annotations {ANN} --encoder mbert --finetune full --conditioning category_and_span --lr_encoder 2e-5 --lr_head 1e-3 --epochs 10 --batch_size 16 --max_token_length 128 --seed 0 --name sent_mbert_full_catspan_s0

# Conditioning ablation and a frozen baseline with class weights
!python -u scripts/train_aspect_sentiment.py --annotations {ANN} --encoder xlmr --finetune full --conditioning category --epochs 10 --batch_size 16 --seed 0 --name sent_xlmr_full_cat_s0
!python -u scripts/train_aspect_sentiment.py --annotations {ANN} --encoder xlmr --finetune frozen --class_weights sqrt --lr_head 1e-3 --epochs 20 --batch_size 32 --seed 0 --name sent_xlmr_frozen_catspan_s0
```

```python
# Seeds for the best configuration (same splits because --split_seed stays 0); then download the results without the large checkpoints
for s in (1, 2):
    !python -u scripts/train_aspect_sentiment.py --annotations {ANN} --encoder xlmr --finetune full --conditioning category_and_span --epochs 10 --batch_size 16 --seed {s} --name sent_xlmr_full_catspan_s{s}
!zip -r aspect_experiments.zip outputs/aspect_experiments -x "*best_checkpoint.pt"
```

If a trained-encoder run runs out of GPU memory, halve `--batch_size` and double `--grad_accumulation`. Epoch times and memory are unmeasured.
Choose between configurations on the validation numbers in `summary.json`; the test numbers are for reporting the chosen one.

## Limitations

* Never run on real annotations: the 50-review and 16-review pilots are unannotated. Expect adjustments once real data arrives.
* With the data volumes of a pilot, every split will be tiny and the warning banner will appear; the numbers then check the pipeline, not the models.
* Span scores depend on the tokenizer: spans are compared at token boundaries, and annotated spans that start or end inside a token cannot be matched exactly (they are counted, not hidden).
* Reviews longer than `--max_token_length` lose later aspects (counted, not corrected).
* One training run per configuration unless you loop over seeds; there are no confidence intervals.
* Splits keep products whole but cannot rule out the same reviewer in several products; the annotation files carry no reviewer id.
* The sequence model (`AspectSequenceModel`) has no training script yet; it needs out-of-fold sentiment predictions first.
* Comment-level data (no product id) cannot use these splits.


## The aspect sequence pipeline (out-of-fold sentiment, then the sequence model)

`scripts/generate_aspect_oof_predictions.py` and `scripts/train_aspect_sequence.py` train the third component, `AspectSequenceModel` (`docs/aspect_models.md`). It must be fed sentiment probabilities that a
sentiment model produced for products it did not train on, so the work is done in two steps with product-grouped splits throughout.

```
adjudicated annotations (one file)
   |  1. split products: TEST (kept out of every model)  |  POOL
   |  2. POOL cut into K product folds; for fold k: train a fresh sentiment model on the other pool products
   |     (a share of them selects the checkpoint on validation macro-F1), predict ONLY fold k   --> out-of-fold probabilities
   |     the K fold models also predict the TEST products; the K outputs are averaged            --> held-out probabilities
   v
OOF probabilities + aspect category + time gaps  --> one series per (product, aspect), chronological  --> BiLSTM + temporal attention
   targets from gold labels by the approved rules (NEUTRAL skipped; a series needs two ranked events)  --> transition head, trajectory head
   evaluated against the ontology RULE applied to the same predicted sentiment, and against the majority class
```

Safeguards, each covered by a test with synthetic fixtures (and the key one checked by a mutation test):

* A held-out product's probabilities come from a model whose training never included it: flipping every gold label of a product leaves its own out-of-fold probabilities bit-identical, and flipping a test product's labels changes nothing at all.
* The prediction files hold probabilities only (no gold column) and the model is called without labels. The sequence stage re-checks the recorded fold assignment of every probability (`LeakageError` otherwise).
* Gold labels only build targets (`events_from_examples` rejects a gold-like source and one-hot gold vectors).
* Metrics use only defined targets: padded positions, NEUTRAL events, first events of a series, and series with fewer than two ranked events are excluded. Predictions do not depend on how series are batched.
* Both scripts refuse to run on data that is too small for meaningful cross-validation (heuristic floors in `src/aspect_training.py`) and stop before writing anything; `--allow_small_data` runs a pipeline check whose outputs are marked `research_result=false`.

Layout of a run: `config/ logs/ metrics/ checkpoints/ predictions/ summary.json` under `outputs/aspect_oof/<name>/` and `outputs/aspect_sequence/<name>/` (never overwritten).

Known limitations of the design: (1) validation products' labels helped train the sentiment models that produced the training products' features (standard stacking; test labels reached no model);
(2) training features are single-model out-of-fold probabilities while test features are an average of K models, a mild distribution shift; (3) the sentiment model reads the annotated span and category as input,
and series are grouped by the annotated category, so the result is "given correct aspect identification", not an end-to-end figure; (4) identical review texts in different products are reported, not removed.

### Kaggle commands, in order

Required inputs: the code (git clone after pushing, or an uploaded code dataset); **one adjudicated annotation CSV** (Annotator A and B files merged into one file with one row set per message and status `ADJUDICATED`;
the scripts refuse two copies of the same message); the same file for both steps (the sequence step checks its SHA-256 against the OOF run). GPU on, Internet on.

```python
!git clone https://github.com/prakyath006/Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text.git
%cd Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text
!pip -q install -r requirements.txt
ANN = "/kaggle/input/<annotations-dataset>/<adjudicated_annotations>.csv"
!python scripts/validate_aspect_annotations.py {ANN} --strict
!python -m pytest tests/test_aspect_oof_sequence.py tests/test_aspect_training.py tests/test_aspect_models.py tests/test_aspect_data.py -q

# 1. out-of-fold sentiment probabilities (the longest step: K+ sentiment trainings)
!python -u scripts/generate_aspect_oof_predictions.py --annotations {ANN} --encoder xlmr --finetune full --lr_encoder 2e-5 --lr_head 1e-3 --folds 5 --epochs 10 --batch_size 16 --max_token_length 128 --seed 0 --name oof_xlmr_full_k5

# 2. the sequence model, several seeds on the same out-of-fold run
for s in (0, 1, 2):
    !python -u scripts/train_aspect_sequence.py --annotations {ANN} --oof_dir outputs/aspect_oof/oof_xlmr_full_k5 --epochs 60 --seed {s} --name seq_xlmr_k5_s{s}

!zip -r aspect_pipeline.zip outputs/aspect_oof outputs/aspect_sequence -x "*best_checkpoint.pt"
```

### Blockers with the current pilot (10 products, 50 reviews)

The structure of the pilot (counted from Annotator B's file, without any model) is 10 products, 134 message-and-aspect examples and 69 product-aspect series, of which 29 have two or more ranked events.
Against the floors: out-of-fold generation is refused (10 products < 30; about 2 products per fold with 5 folds < 3; 134 examples < 300) and, even with the outer split, sequence training would have about 18 / 8 / 3
usable train / validation / test series against floors of 50 / 20 / 20. Both scripts therefore stop on the pilot. `--allow_small_data` exists for pipeline checks only. The pilot cannot support this stage; it needs roughly three
times as many products at the same annotation density (the floors are heuristic, not statistical guarantees).
