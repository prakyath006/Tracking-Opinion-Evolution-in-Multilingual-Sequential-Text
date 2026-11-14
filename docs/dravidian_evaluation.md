# DravidianCodeMix multilingual evaluation (comment-level sentiment, XLM-R)

**Status: run completed on Kaggle (Tesla T4, XLM-R base frozen, seeds 0 to 4); results are in `outputs/dravidian_evaluation/`** (one `seed_<k>/` folder per seed with `config/`, `metrics/`, `predictions/`, `checkpoints/` and `summary.md`, plus `aggregate_over_seeds.json`; the full Kaggle collection, logs and summary report are in `outputs/kaggle_final_training/`). The run came from `notebooks/kaggle_full_training_pipeline.ipynb`.

Test-split results, mean over 5 seeds (4-class macro-F1 +- std; majority-class baseline in brackets): Tamil 0.392 +- 0.007 (0.187), Malayalam 0.433 +- 0.008 (0.150), Kannada 0.437 +- 0.016 (0.173); joint model on Tamil 0.397 (0.187), Malayalam 0.380 (0.150), Kannada 0.399 (0.173). MIXED is the hardest class (F1 0.09 to 0.29). The joint head is no better for Tamil and worse for Malayalam and Kannada than the per-language heads (frozen encoder only).

**Scope.** This is a comment-level sentiment experiment. DravidianCodeMix rows are independent YouTube comments with no timestamps, authors, threads or ordering; none were invented and no sequence is built. It is separate from the Amazon temporal (OET) experiments and says nothing about opinion evolution over time. Multilingual temporal evolution has not been demonstrated.

## The data (inspected)

Official DravidianCodeMix files in `data/raw/`; the project preprocessing (`src/preprocessing.py`) drops the `not-<language>` rows (Tamil 1,687 train / 189 dev / 211 test; Malayalam 1,180 / 119 / 146; Kannada 918 / 111 / 107) and maps labels through the ontology: `Positive` to POSITIVE, `Negative` to NEGATIVE, `Mixed_feelings` to MIXED, `unknown_state` to UNKNOWN. `scripts/prepare_comment_level_data.py` then removes dev/test comments whose text also occurs in an earlier split and duplicates inside dev or test, and writes `data/comment_level/<language>_<split>.csv` (the files used here; counts in `outputs/comment_level/data_report.json`).

| Language (code-mixed with English) | train | dev | test | test POS / NEG / MIXED / UNKNOWN | removed from dev / test as seen in an earlier split |
|---|---|---|---|---|---|
| Tamil | 33,531 | 4,145 | 4,112 | 2,451 / 544 / 499 / 618 | 60 / 76 |
| Malayalam | 14,514 | 1,671 | 1,654 | 708 / 259 / 111 / 576 | 157 / 159 |
| Kannada | 5,218 | 615 | 632 | 335 / 142 / 72 / 83 | 41 / 27 |

Train class shares: Tamil 59% POSITIVE, Malayalam 44%, Kannada 54%; MIXED is the rarest class everywhere (Malayalam 6%). Classes are imbalanced, so macro-F1 and per-class F1 are reported next to accuracy, plus a majority-class baseline.

## Protocol (`scripts/dravidian_evaluation.py`)

* **Reused:** the leakage-controlled splits above; the frozen-encoder embedding cache, sentence head and training loop of `scripts/eval_comment_level.py`; metrics from `src/eval_utils.py`. The original OET model, annotation files and existing results are not touched.
* **Model:** frozen `xlm-roberta-base` [CLS] vectors, a small head trained with inverse-frequency class weights from the training split, early stopping on dev loss. Test is only scored.
* **Setups:** one model per language (Tamil, Malayalam, Kannada), each scored on its own test split, and one joint model trained on the three training splits and scored on each language's test split separately. Languages are never pooled in a metric.
* **Metrics, per setup and test split (mean and standard deviation over seeds 0 to 4):** accuracy and macro-F1 over the four ontology classes, macro-F1 over POSITIVE/NEGATIVE/MIXED only (comparable with other corpora), per-class precision, recall and F1, confusion matrices, a bootstrap 95% interval over comments, the majority-class baseline.
* **Subsets:** each test split is also scored by the script of the comment, derived from its characters (native script, romanised, mixed script); subsets with fewer than 30 comments are marked unreliable.
* **Leakage:** the script recomputes exact normalised-text overlap between train, dev and test inside each language and stops if it is not 0; it also reports train-versus-other-language-test overlap. The encoder is frozen, so the test split cannot influence it; early stopping uses dev only.
* **Outputs (new directory, refuses to overwrite):** `outputs/dravidian_evaluation/config/` (config with versions and file hashes, data report with class distributions and leakage checks), `metrics/` (one JSON per setup and test split, all seeds), `predictions/` (one CSV per setup and test split with the text, gold label and every seed's prediction), `checkpoints/` (one `<setup>_seed<k>.pt` per language model and for the joint model: the trained head chosen on dev loss, its settings, class weights and input-file hashes; the frozen XLM-R is not stored, `load_head` and `predict_texts` in the script reload it), and `summary.md`.
* **Tests:** `tests/test_dravidian_evaluation.py` (4 tests on synthetic comments and random embeddings; no model is run). A 64-comment smoke run of the whole pipeline with mBERT worked; its numbers are meaningless and are not reported.

## Run it on Kaggle (GPU, internet on)

The bundle `dravidian_kaggle_bundle_v2.zip` (2.4 MB; it replaces the first bundle, which did not save checkpoints) holds the three code files and the nine split files with their repository paths. Upload it as a Kaggle dataset, then in a notebook:

```
!mkdir -p /kaggle/working/repo && cd /kaggle/working/repo && unzip -q /kaggle/input/<your-dataset>/dravidian_kaggle_bundle_v2.zip
!pip -q install transformers scikit-learn pandas
%cd /kaggle/working/repo
!python scripts/dravidian_evaluation.py --encoder xlmr --seeds 0 1 2 3 4 --device cuda --out_dir /kaggle/working/dravidian_evaluation
```

Expected cost: about 66,000 comments embedded once (a few minutes on a T4), then small heads. Download `/kaggle/working/dravidian_evaluation/` and place it at `outputs/dravidian_evaluation/` in the project (the folder must be empty or absent). To check the setup first, add `--max_rows 64 --epochs 2 --seeds 0 --out_dir /kaggle/working/smoke` (a smoke test, not a result). Local alternative: the same command with `--device cpu` and `--encoder_path <local folder with the weights>`, which takes roughly an hour and is not recommended on a laptop.

## Limitations

* Code-mixed YouTube comments about films; labels are the dataset's own and noisy (some texts carry conflicting labels, see the data report).
* Tamil, Malayalam and Kannada differ in size and script mix, so scores across languages are not directly comparable.
* A frozen encoder with a small head is a baseline; fine-tuning XLM-R would score higher and is a separate experiment.
* Removing dev/test comments seen in training makes the test splits slightly smaller than the official files and cannot be compared one-to-one with published DravidianCodeMix scores.
* Nothing here supports a claim about opinion evolution, trajectories or time.
