# Tracking Opinion Evolution in Multilingual Sequential Text

A multi-task framework that reads a person's reviews **in chronological order** and
predicts not just how they feel, but **how their opinion changed and where it turned**.

Standard sentiment analysis classifies each review in isolation. This project models
the sequence: per-review sentiment, the transition between consecutive reviews, and
the overall trajectory — across **two structurally different domains** (English
e-commerce timelines and code-mixed Dravidian social-media threads).

---

## What it does

Given one user's reviews in order:

```
1. "Product looks great, very happy with the purchase"
2. "Still working fine after two weeks"
3. "Started having problems, quality dropped"
4. "Completely stopped working, very disappointed"
```

the model returns three things at once:

| Output | Head | Example |
|---|---|---|
| Sentiment per review | 4 classes | POSITIVE → NEGATIVE → NEGATIVE → NEGATIVE |
| Transition between reviews | 3 classes | STABLE, DOWNGRADE, DOWNGRADE, DOWNGRADE |
| Overall trajectory | 4 classes | **DECLINING** |

plus attention weights identifying the turning point (`0.004, 0.156, 0.333, 0.507`
— review 4).

---

## Architecture

```
["Great product", "Still fine", "Broke down", "Very disappointed"]
        |
   [MultilingualTokenizer]              src/tokenization.py
        |
   [mBERT]  -- frozen by default --     src/embeddings.py    177,853,440 params
        |   4 x 768
   [Bi-LSTM]  2 layers, bidirectional   src/bilstm.py          2,828,032 params
        |
   [Self-Attention]                     src/attention.py          65,792 params
        |
   [3 Multi-Task Heads]                 src/classifier.py        199,179 params
```

**180,946,443 total parameters; 3,093,003 trainable** in the released runs
(mBERT frozen via `--freeze_encoder`, which is the default). `--no_freeze_encoder`
unfreezes the top 3 encoder layers.

Assembled in [`src/model.py`](src/model.py) — read `forward()` to follow the whole flow.

---

## Datasets

Two domains, deliberately chosen to be structurally different.

**Domain 1 — Amazon Beauty** (e-commerce, real user timelines)

| Split | Sequences |
|---|---|
| train | 3,726 |
| val | 798 |
| test | 800 |

**Domain 2 — DravidianCodeMix-2020** (social media, sliding-window pseudo-threads)

| Language | Rows | Train seq | Test seq | Ontology coverage |
|---|---|---|---|---|
| Tamil | 33,531 | 7,823 | 1,677 | 83.5% |
| Malayalam | 14,514 | 3,385 | 727 | 64.3% 🚩 |
| Kannada | 5,218 | 1,216 | 262 | 87.0% |

Only **Tamil** is trained. Malayalam and Kannada are deliberately **held out as unseen
languages**, which is what makes the zero-shot transfer result below possible.
Malayalam is flagged because over a third of its comments are labelled
`unknown_state` by the original annotators.

`data/` is gitignored. See [`scripts/`](scripts/) for the builders.

---

## Results

All numbers below are read from `outputs/metrics/` and regenerate with
`scripts/compile_metrics.py`. **Every model was trained with 0 trainable encoder
layers**, so the comparison is capacity-matched.

### In-domain — sentiment F1 (macro)

| Model | Amazon | Tamil | Trajectory head? |
|---|---|---|---|
| xlmr_sentence | **0.6188** | 0.3669 | ✗ |
| **Full model (OET)** | 0.5329 | 0.3930 | ✓ |
| mbert_sentence | 0.4837 | **0.4198** | ✗ |
| attention_only *(ablation)* | 0.4715 | 0.3920 | ✓ |
| lstm_only *(ablation)* | 0.4377 | 0.2933 | ✓ |
| textcnn | 0.4271 | 0.3514 | ✗ |

The sentence-level baselines answer **one** of the three questions — they produce no
transitions, no trajectory, and no SCS. XLM-R's Amazon win is consistent with its far
lower perplexity on English (see Module 3).

### Ablation — both components earn their place

| Comparison | Amazon Δ F1 | Tamil Δ F1 | Amazon Δ trajectory F1 |
|---|---|---|---|
| vs `lstm_only` (no attention) | **+0.095** | **+0.100** | +0.122 |
| vs `attention_only` (no Bi-LSTM) | **+0.061** | +0.001 | **+0.217** |

### Cross-domain transfer (preliminary; sequence-level, single run)

| Trained on | Tested on | Sentiment F1 | % of in-domain retained |
|---|---|---|---|
| **Tamil** | **Kannada** *(never trained on)* | 0.2903 | **74%** |
| **Tamil** | **Malayalam** *(never trained on)* | 0.2878 | **73%** |
| Tamil | Amazon | 0.2505 | 64% |
| Amazon | Kannada | 0.2214 | 42% |
| Amazon | Tamil | 0.2300 | 43% |
| Amazon | Malayalam | 0.2038 | 38% |

In this single run the Tamil model retains ~73–74% of its in-domain score on two
languages it never saw, and the Amazon model retains 38–43%. These Dravidian
sequences are sliding windows over independent comments and may overlap between
train and test, so the comparison is indicative only; Module 8 re-evaluates transfer
at comment level with leakage control.

### Module 3 — encoder perplexity (why this corpus is hard)

| Domain | mBERT | XLM-R |
|---|---|---|
| amazon_beauty (English) | 19.35 | **6.38** |
| dravidian_malayalam | **78.06** | 111.51 |
| dravidian_kannada | **107.67** | 216.79 |
| dravidian_tamil | 151.97 | **118.11** |

Code-mixed Dravidian text is **6–34× harder** for both encoders than English
e-commerce text. Neither encoder wins outright.

### Module 4 — calibration

| | Amazon | Tamil |
|---|---|---|
| Sentiment ECE | 0.0888 | 0.0588 |
| Trend ECE | 0.0994 | 0.1860 |
| Trajectory ECE | 0.1013 | 0.0302 |
| SCS mean | 0.5636 | 0.3062 |
| SCS reliability ratio | 1.67 | 1.33 |

**Known limitation:** the Tamil trajectory head predicts only STABLE or VOLATILE —
zero IMPROVING and zero DECLINING predictions — which explains its 0.2678 F1.

---

## The six modules

| Module | What it is | Code | Result |
|---|---|---|---|
| 1 | Structural ontology (closed vocabulary across domains) | `src/ontology.py`, `src/ontology_eval.py` | `outputs/metrics/module1_ontology.md` |
| 2 | Word sense disambiguation for code-mixed aspects | `src/wsd.py` | `outputs/metrics/module2_wsd_results.md` |
| 3 | Encoder comparison via MLM perplexity | `src/mlm_perplexity_eval.py` | `outputs/metrics/module3_bert_perplexity.md` |
| 4 | Confidence, entropy, ECE, SCS reliability | `src/confidence_eval.py` | `outputs/metrics/module4_sequential_model_<run_id>.md` |
| 5 | Cross-domain fuzzy typicality | `src/fuzzy_domain_score.py` | `outputs/fuzzy_domain_scores.csv` |
| 6 | Baseline comparison and analysis | `scripts/generate_module6_analysis.py` | `outputs/metrics/module6_analysis.md` |

**Module 2 status:** the disambiguator resolves 121 of 682 ambiguous instances
(17.74%) over 144,973 words. The remaining 82% are **abstentions, not errors** — short
code-mixed comments often carry no disambiguating context within the ±5-token window.
**No sense-disambiguation accuracy is reported**, because no gold sense labels exist
for this corpus; the MCS comparison measures share of high-confidence predictions, not
accuracy. See [`docs/module2_wsd_status.md`](docs/module2_wsd_status.md).

---

## Progress since Panel 1

Everything below is backed by files in this repository; nothing is claimed that has no result file or test.

| Component | Status | Where |
|---|---|---|
| OET with XLM-R and partial fine-tuning (E0 to E4 experiment framework) | implemented and unit-tested; **results not in this repository yet** | `scripts/train_experiment.py`, `src/experiment_utils.py`, `docs/kaggle_experiments.md` |
| DravidianCodeMix Tamil, Malayalam, Kannada, comment-level sentiment with XLM-R | **trained and evaluated** on Kaggle (frozen encoder, 5 seeds, leakage-controlled official splits, heads saved) | `scripts/dravidian_evaluation.py`, `outputs/dravidian_evaluation/`, `docs/dravidian_evaluation.md` |
| Aspect-level annotation pipeline (guidelines v0.2, strict validator, agreement report, adjudication package) | implemented; 250 reviews double-annotated | `docs/aspect_annotation_guidelines.md`, `scripts/validate_aspect_annotations.py`, `outputs/annotation_waves/` |
| Aspect-conditioned sentiment (XLM-R, top 2 layers) | **trained once** on 459 agreed-only examples (pipeline result, see limits) | `src/aspect_sentiment_model.py`, `scripts/train_aspect_sentiment.py`, `outputs/aspect_experiments/` |
| Aspect extraction (BIO tagging) | implemented and unit-tested; **not trained** (data below the size floors) | `src/aspect_extractor.py`, `src/aspect_data.py`, `scripts/train_aspect_extractor.py` |
| Aspect chronological sequence / trajectory model | implemented and unit-tested; **not trained** (series floors not met) | `src/aspect_sequence_model.py`, `scripts/train_aspect_sequence.py` |
| Product-level splits, leakage guards, data-readiness gates | implemented, tested, checked on the real data | `src/aspect_training.py`, `scripts/aspect_readiness.py` |
| Shared Amazon + Dravidian multitask encoder | **design only**, no code | `docs/shared_xlmr_multitask_design.md` |

**DravidianCodeMix with XLM-R (comment-level; mean over 5 seeds).** Each comment is independent and
has no timestamp, author or thread, so this is *not* a temporal result.

| Setup, test language | Accuracy | 4-class macro-F1 | Majority-class macro-F1 |
|---|---|---|---|
| Tamil model | 0.467 | 0.392 ± 0.007 | 0.187 |
| Malayalam model | 0.498 | 0.433 ± 0.008 | 0.150 |
| Kannada model | 0.495 | 0.437 ± 0.016 | 0.173 |
| Joint model, on Tamil / Malayalam / Kannada | 0.471 / 0.464 / 0.498 | 0.397 / 0.380 / 0.399 | same |

MIXED is the hardest class (F1 0.09 to 0.29). One joint head is no better than per-language heads
and is worse for Malayalam and Kannada. These numbers use a different protocol from the Panel 1
Tamil row above (official splits with overlaps removed, not sliding windows), so they are not a
like-for-like improvement over it.

**Aspect-conditioned sentiment.** One seed: validation accuracy 0.784 / macro-F1 0.332, test accuracy
0.851 / macro-F1 0.388. Only POSITIVE and NEGATIVE are learned (NEUTRAL and MIXED score 0 F1), and the
validation and test sets (74 and 67 examples) are below the project's 100-example floor, so treat it as a
pipeline result.

**Annotation agreement** (two annotators, spans matched at overlap 0.5): Wave 1, 195 reviews, span F1 0.76,
category kappa 0.91, sentiment kappa 0.98, span-type kappa 0.82; pilot, 48 reviews, span F1 0.65, category
kappa 0.78. The 394 Wave 1 disagreements are archived, not adjudicated.

**Data-readiness gates.** The pipeline refuses to train when the data is too small, and the thresholds were
not lowered: aspect extraction was skipped (6 validation and 8 test mentions against floors of 100) and the
aspect trajectory model is blocked (17 test series against a floor of 20 on the default product split).

**Not claimed:** any temporal opinion-evolution result for Tamil, Malayalam or Kannada, a trained aspect
extractor or aspect trajectory model, a trained joint Amazon + Dravidian model, gold-standard aspect labels,
or an improvement of E4 over the Panel 1 model (no E4 result is in this repository).

One notebook, [`notebooks/kaggle_full_training_pipeline.ipynb`](notebooks/kaggle_full_training_pipeline.ipynb),
runs the aspect readiness gate, the aspect sentiment training and the five-seed Dravidian evaluation on a
Kaggle GPU. Nothing here is trained on a laptop.

---

## Modules 7–9 — evaluation, multilingual and LLM

| Module | What it is | Code | Status |
|---|---|---|---|
| 7 | Valid majority baselines, observed-transition trend metrics, gold-sequence SCS, bootstrap intervals, seeding, prediction export | `src/eval_utils.py`, `scripts/compute_valid_baselines.py`, `scripts/export_predictions.py` | baselines **run** (`outputs/metrics/valid_baselines_amazon.md`); full prediction export pending |
| 8 | Comment-level multilingual and cross-domain sentiment (official DravidianCodeMix splits, leakage control, 5 seeds) | `scripts/prepare_comment_level_data.py`, `scripts/eval_comment_level.py` | data prepared; the in-language XLM-R comment-level run is **complete** (5 seeds, see Progress since Panel 1); the cross-domain transfer run is still pending (GPU) |
| 9 | Instruction-tuned LLM (flan-t5) classification, zero-shot and source-calibrated | `scripts/eval_llm_comment_level.py` | smoke-tested, full run pending (GPU) |

The in-language XLM-R results for the three Dravidian languages exist (see Progress since Panel 1); no
cross-domain comment-level or LLM result exists yet. Details and Kaggle
commands: [`docs/new_modules_status.md`](docs/new_modules_status.md).

**Validity notes.** The Amazon task is a rating-derived sentiment trajectory over a user's
chronological reviews, which mostly cover different products. The Dravidian sequence results above
come from sliding windows over independent comments (not real histories) and may be affected by
train/test text overlap; treat them as preliminary. Module 8 now provides comment-level, leakage-controlled
results for the three languages (see Progress since Panel 1). Single training runs, no confidence intervals yet for the model results.

---

## Research gaps addressed

| Gap | Status |
|---|---|
| 1 — Transformers ignore sequential opinion evolution | ⚠️ Addressed on Amazon (single run, ablation shows both components contribute); not yet shown with intervals or seeds |
| 4 — Poor cross-domain generalization | ⚠️ Partial — preliminary transfer results; leakage-controlled in-language comment-level evaluation done, cross-domain comment-level run pending |
| 2 — Inconsistent multilingual/code-mixed embeddings | ⚠️ Partial — XLM-R evaluated on Tamil, Malayalam and Kannada at comment level (frozen encoder, 5 seeds); partial fine-tuning is implemented but its results are not in this repository |
| 5 — Lack of aspect-level opinion evolution | ⚠️ Partial — aspect extraction, aspect-conditioned sentiment and an aspect-level sequence model are implemented and tested; only the sentiment model is trained (small data); extraction and trajectory are blocked by data-size floors |
| 3 — Heavy dependence on supervised learning | ❌ Scoped as future work |

---

## Usage

### Install

Requires Python 3.10+ (developed on 3.14; CI uses 3.11). Training and full evaluation need a GPU;
tests and the label/metric utilities run on CPU.

```bash
git clone https://github.com/prakyath006/Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text.git
cd Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q                             # sanity check: data-dependent tests skip without data/
```

### Data and checkpoints

`data/` and `outputs/checkpoints/` are not in the repository (size and dataset licences).
They are provided as the Kaggle datasets `prakyathnandigam/opinion-tracker-data` and
`prakyathnandigam/opinion-tracker-checkpoints`. Download them and place the contents in
`data/` and `outputs/checkpoints/`, or rebuild the data with the scripts in `scripts/`
(`build_amazon_sequences.py`, `prepare_comment_level_data.py`). Raw datasets (Amazon Reviews 2023,
DravidianCodeMix) keep their own licences. Without the checkpoints the demo cannot run live
inference, but the pages that read `outputs/metrics/` still work.

### Train

```bash
python scripts/train.py --domain amazon --epochs 20 --batch_size 16
python scripts/train.py --domain dravidian --language tamil --epochs 20 --batch_size 16
```

Checkpoints are written to `outputs/checkpoints/best_model_<run_id>.pt`.
Add `--no_freeze_encoder` to fine-tune the top 3 encoder layers.

### Baselines and ablation

```bash
python scripts/train_baselines.py --baseline all --domain amazon \
    --encoder_finetune_layers 0 --lr 1e-3
```

`--encoder_finetune_layers` must match the full model's setting, or the comparison
measures training budget rather than architecture.

### Evaluate

```bash
python scripts/cross_domain_eval.py            # all transfer directions
python src/ontology_eval.py                    # Module 1
python scripts/evaluate_wsd.py                 # Module 2
python src/mlm_perplexity_eval.py              # Module 3
python src/confidence_eval.py --domain amazon  # Module 4
python src/fuzzy_domain_score.py               # Module 5
```

### Compile reports

```bash
python scripts/compile_metrics.py
python scripts/generate_module6_analysis.py
python scripts/generate_module_by_baseline_matrix.py
```

### Interactive demo

**Live demo:** https://tracking-opinion-evolution-in-multilingual-sequential-text.streamlit.app/ (free hosting; it may take a minute to wake, and the first prediction downloads the model files).

Run locally:

```bash
python -m streamlit run web_demo/app.py
```

Four sections (Analyze, Insights, How it works, About), including a playground where you
can enter your own review sequence and watch the trajectory being predicted live. It needs
the checkpoints in `outputs/checkpoints/` (hosted on Kaggle, see Datasets). **Every figure is read
from `outputs/` at page load** — where a result has not been produced, the page says
so rather than showing a placeholder.

### Tests

```bash
python -m pytest tests/ -q      # 303 tests (ontology, evaluation utilities, aspect pipeline, annotation tooling, Dravidian evaluation); data-dependent tests skip without data/
```

### GPU

Training needs a GPU. [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb)
is a ready-to-run entry point for Colab or Kaggle.

---

## Project structure

```
src/                 model, ontology, WSD, baselines, evaluation utilities
scripts/             data building, training, evaluation, reporting
tests/               ontology, evaluation, aspect pipeline, annotation tooling and Dravidian evaluation tests
notebooks/           EDA, Colab training and the Kaggle full training pipeline
web_demo/            Streamlit dashboard
docs/                status documents, literature comparison, defect register
ontology/            aspect taxonomy and WSD lexicons
outputs/             results: metrics, Dravidian evaluation, aspect experiments, annotation outputs (checkpoints are gitignored)
```

Notable documents:

- [`docs/defect_register.md`](docs/defect_register.md) — every defect found in the
  2026-09-04 audit, what it affected, and the evidence
- [`docs/capacity_matched_comparison.md`](docs/capacity_matched_comparison.md) — why
  the baseline comparison was re-run at equal encoder capacity
- [`docs/module2_wsd_status.md`](docs/module2_wsd_status.md) — WSD status and its
  open limitation
- [`docs/literature_comparison.md`](docs/literature_comparison.md) — comparison
  against the surveyed papers
- [`docs/dravidian_evaluation.md`](docs/dravidian_evaluation.md) — the XLM-R comment-level protocol and results
- [`docs/aspect_annotation_guidelines.md`](docs/aspect_annotation_guidelines.md) and [`docs/aspect_models.md`](docs/aspect_models.md) — aspect annotation rules and models
- [`docs/annotation_requirements_and_batch_02.md`](docs/annotation_requirements_and_batch_02.md) — training floors, measured yield and the next annotation batch
- [`docs/shared_xlmr_multitask_design.md`](docs/shared_xlmr_multitask_design.md) — design (not implemented) of a shared Amazon + Dravidian encoder

---

## Known limitations

- The Panel 1 runs **freeze the encoder**; XLM-R partial fine-tuning (E4) is implemented and tested, but no
  E4 result is in this repository.
- The **Tamil trajectory head** never predicts IMPROVING or DECLINING.
- **No WSD accuracy figure** — gold sense labels do not exist for this corpus.
- **Malayalam labels** are 35.7% `unknown_state`; treat that domain's sentiment labels
  with caution.
- Adding a new dataset currently requires code changes, not just configuration — the
  model and ontology are domain-agnostic, but the data loaders are not yet registry-driven.
- **LLM baseline** (Module 9) is implemented but its full run is pending; no LLM result is reported.
- Only the Dravidian XLM-R evaluation is repeated over 5 seeds; the Amazon OET and aspect results are single runs.
- The aspect extraction and aspect trajectory models are **not trained**: the agreed-only annotations are below the
  project's size floors (500 / 100 / 100 mentions, 50 / 20 / 20 series) and the floors were not lowered.
- The aspect-conditioned sentiment model is a pipeline result only (tiny validation and test sets, NEUTRAL and MIXED not learned).
- Dravidian data has no time order, so no temporal opinion-evolution claim is made for Tamil, Malayalam or Kannada.

---

## References

- B. R. Chakravarthi et al., "Corpus Creation for Sentiment Analysis in Code-Mixed
  Tamil-English Text," 2020
- B. R. Chakravarthi et al., "A Sentiment Analysis Dataset for Code-Mixed
  Malayalam-English," 2020
- DravidianCodeMix Shared Task @ FIRE 2020
- McAuley-Lab, Amazon-Reviews-2023

---

## Authors

Kanishkhan J, Prakyath Nandigam, Yaswanth J, Hemanth G; guide: Krishna Priya G.
Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore.

## License

MIT, see [`LICENSE`](LICENSE).
