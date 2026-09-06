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

### Cross-domain transfer — the headline finding

| Trained on | Tested on | Sentiment F1 | % of in-domain retained |
|---|---|---|---|
| **Tamil** | **Kannada** *(never trained on)* | 0.2903 | **74%** |
| **Tamil** | **Malayalam** *(never trained on)* | 0.2878 | **73%** |
| Tamil | Amazon | 0.2505 | 64% |
| Amazon | Kannada | 0.2214 | 42% |
| Amazon | Tamil | 0.2300 | 43% |
| Amazon | Malayalam | 0.2038 | 38% |

**Training on code-mixed text produces a model that generalises far better.** The
Tamil model retains ~73–74% of its performance on two languages it never saw; the
Amazon model retains 38–43%. Cross-lingual transfer within the Dravidian family also
beats cross-domain transfer (0.2878/0.2903 vs 0.2505).

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

## Research gaps addressed

| Gap | Status |
|---|---|
| 1 — Transformers ignore sequential opinion evolution | ✅ Solved; ablation confirms both components contribute |
| 4 — Poor cross-domain generalization | ✅ Solved; 8 transfer pairs, zero-shot to 2 unseen languages |
| 2 — Inconsistent multilingual/code-mixed embeddings | ⚠️ Partial — released runs use a **frozen** encoder; `--no_freeze_encoder` is supported but not yet reported |
| 5 — Lack of aspect-level opinion evolution | ⚠️ Half — aspects are tagged corpus-wide, but no per-aspect trajectory head exists yet |
| 3 — Heavy dependence on supervised learning | ❌ Scoped as future work |

---

## Usage

### Install

```bash
pip install -r requirements.txt
```

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

```bash
python -m streamlit run web_demo/app.py
```

Eight pages covering every module, plus a playground where you can enter your own
review sequence and watch the trajectory being predicted live. **Every figure is read
from `outputs/` at page load** — where a result has not been produced, the page says
so rather than showing a placeholder.

### Tests

```bash
python -m pytest tests/ -q      # 31 ontology consistency tests
```

### GPU

Training needs a GPU. [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb)
is a ready-to-run entry point for Colab or Kaggle.

---

## Project structure

```
src/                 16 modules — model, ontology, WSD, baselines, evaluation
scripts/             19 scripts — data building, training, evaluation, reporting
tests/               ontology consistency suite
notebooks/           EDA and Colab training
web_demo/            Streamlit dashboard
docs/                status documents, literature comparison, defect register
ontology/            aspect taxonomy and WSD lexicons
outputs/             results (gitignored except metrics/*.md)
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

---

## Known limitations

- The released runs **freeze the encoder**; the fine-tuned variant is supported but not
  yet reported.
- The **Tamil trajectory head** never predicts IMPROVING or DECLINING.
- **No WSD accuracy figure** — gold sense labels do not exist for this corpus.
- **Malayalam labels** are 35.7% `unknown_state`; treat that domain's sentiment labels
  with caution.
- Adding a new dataset currently requires code changes, not just configuration — the
  model and ontology are domain-agnostic, but the data loaders are not yet registry-driven.
- **No LLM baseline** has been implemented.

---

## References

- B. R. Chakravarthi et al., "Corpus Creation for Sentiment Analysis in Code-Mixed
  Tamil-English Text," 2020
- B. R. Chakravarthi et al., "A Sentiment Analysis Dataset for Code-Mixed
  Malayalam-English," 2020
- DravidianCodeMix Shared Task @ FIRE 2020
- McAuley-Lab, Amazon-Reviews-2023

---

## Author

**Prakyath Nandigam**
