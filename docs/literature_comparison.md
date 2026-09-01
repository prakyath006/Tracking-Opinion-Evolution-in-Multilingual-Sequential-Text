# Literature Comparison

Grounded only in papers actually present in this project's research materials
(`Project Phase/Research Papers/`) and the project's own gap analysis
(`Project Phase/Research Gaps.docx`). No number below was estimated, guessed,
or pulled from outside this project's cited materials — where a paper does
not report something comparable, that is stated directly rather than
papered over.

**Source note on how these numbers were found**: `Research Gaps.docx` (the
project's literature review write-up) discusses five research gaps in
qualitative terms — it does not cite specific papers by title inline, and
reports no specific accuracy/F1 numbers anywhere. The actual reported
numbers below come from reading the primary papers themselves in
`Research Papers/`, which is the only place in the project's materials that
states them.

## Papers reviewed and their relevance

Of 14 PDFs in `Research Papers/`, the following were judged relevant to this
project's actual claims (multilingual / code-mixed sentiment and opinion
mining) and are cited below:

| Paper | Venue | Relevance |
|---|---|---|
| Kavitha et al., "A fine-grained multi-lingual opinion mining method on social media texts using multi-scale fused features..." | Data & Knowledge Engineering 161 (2026) 102524 | Multilingual, social-media, sentence-level opinion mining |
| Rashid et al., "BERT-KAN: Enhancing bilingual sentiment analysis in bangladeshi E-commerce through fine-tuned large language models" | (journal not stated in extracted text) | Code-mixed (Bengali-English), E-commerce, mBERT-based — closest architectural match to this project's encoder choice |
| Provaa et al., "Multilingual sentiment analysis in e-commerce customer reviews using GPT and deep learning-based weighted-ensemble model" | Intl. J. Cognitive Computing in Engineering | Same platform as this project's Domain 1 — trained/tested on the Multilingual Amazon Reviews Corpus (MARC) |
| Ansari et al., "PolyModNet: Advanced positional encodings and ethical bias mitigation in adaptive multimodal fusion for multilingual language understanding" | — | Multilingual NLU benchmark including sentiment analysis; broader scope (also image/translation), included as a secondary reference point |

**Excluded from the rows below, with reasons stated rather than silently
dropped:**
- Al-Zoubi et al., "A hybrid TwinSVM-HHO model for multilingual spam review
  detection using sentiment features and pre-trained embeddings" — reports
  accuracy for **spam detection** (binary spam/not-spam), using sentiment as
  an auxiliary feature, not for sentiment classification itself. Its
  92.97%/89.03%/80.36%/85.09% (Arabic/English/Spanish/multilingual) numbers
  measure a different target variable than any row below and would misstate
  the comparison if presented as a sentiment-accuracy figure.
- Four Word Sense Disambiguation papers (EnhancedBERT WSD, GlossGPT, "System
  Fusion Based on WordNet WSD", "Improving selection of synsets from
  WordNet...") and two lexical-complexity-adjacent papers ("Efficient and
  scalable masked word prediction using concept formation", "Text-to-text
  generative approach for enhanced complex word identification") — relevant
  only to the aspect-level/WSD extension, which is explicitly paused pending
  guide confirmation (see project constraints). Not used for comparison here.
- Three papers with no connection to this project's subject matter
  (federated learning for smart-grid energy management, generative AI in
  finance, semantic enrichment for BIM/building-energy simulation) — appear
  to be from an unrelated course/project sharing the same folder. Excluded
  entirely.

---

## Comparison Table

| Module | Our Metric | Prior Work Paper | Prior Work's Reported Number | Our Number | Gap/Claim |
|---|---|---|---|---|---|
| Sentiment Classification | Accuracy | Kavitha et al. (MRFPO-MARCLA) | 95.12% accuracy (multilingual social-media opinion mining, sentence-level, single-domain) | **TBD** — pending Colab training run with the real encoder (bert-base-multilingual-cased); see `notebooks/colab_training.ipynb` | Comparison pending. Cannot claim superiority or inferiority until a real number exists. Architecturally different: their model has no sequential/cross-domain component. |
| Sentiment Classification | Precision / Recall / F1 | Rashid et al. (BERT-KAN) | Best config: precision 95.3%, recall 97.0%, F1 96.1% (Bengali-English code-mixed E-commerce, mBERT-base encoder) | **TBD** — pending Colab training run | Comparison pending. This is the closest architectural match in the surveyed literature (same base encoder family, code-mixed, E-commerce), so it is the most meaningful benchmark once Our Number exists — flagged here for that future comparison, not claimed against yet. |
| Sentiment Classification | Accuracy | Provaa et al. (T5-CapsNet ensemble) | 97.56% accuracy on the Multilingual Amazon Reviews Corpus (MARC) | **TBD** — pending Colab training run | Comparison pending. Same platform (Amazon) as this project's Domain 1, different architecture (T5-CapsNet ensemble vs. mBERT+Bi-LSTM+Attention) and different task framing (single-review classification vs. this project's per-review + sequence-level trajectory). |
| Sentiment Classification (secondary reference) | Accuracy | Ansari et al. (PolyModNet) | 85.71% accuracy in sentiment analysis (broader multimodal NLU benchmark, not e-commerce-specific) | **TBD** — pending Colab training run | Included as a lower-scoring secondary data point for range context; not the primary comparison target since PolyModNet's sentiment task is one part of a much broader multimodal benchmark. |
| Trajectory / Trend Classification | Accuracy, F1 (macro) per class (`UPGRADE`/`DOWNGRADE`/`STABLE`, `IMPROVING`/`DECLINING`/`STABLE`/`VOLATILE`) | **None found in surveyed literature** | Not reported — no paper reviewed for this project performs sequence-level trajectory or pairwise trend classification. All three sentiment-classification papers above (Kavitha et al., Rashid et al., Provaa et al.) classify each review/comment independently. This absence is exactly the "Limitation of Transformer Models in Capturing Sequential Opinion Evolution" identified as Research Gap 1 in this project's own gap analysis. | **TBD** — pending Colab training run | No directly comparable prior baseline exists in the surveyed literature. The project's own 5-baseline ablation study (`scripts/train_baselines.py`: `mbert_sentence`, `xlmr_sentence`, `lstm_only`, `attention_only`, `textcnn`) is therefore the primary comparison mechanism for this module, not literature — it isolates what sequential modeling (Bi-LSTM) and attention each individually contribute, which the surveyed literature cannot speak to. |
| Ontology (coverage, consistency, structure) | Coverage %, consistency (conflict count), depth/breadth/coupling | **None found in surveyed literature** | Not reported — none of the papers reviewed define or evaluate a label-taxonomy ontology as a distinct artifact; all treat sentiment labels as given, not as something to be evaluated for coverage or cross-domain consistency in their own right. | **Real, computed** (`src/ontology_eval.py`, run against real preprocessed data): coverage — amazon_beauty 100.0% (39,427/39,427; no UNKNOWN branch by construction), dravidian_tamil 83.5% (27,988/33,531), dravidian_kannada 87.0% (4,542/5,218), dravidian_malayalam 64.3% (9,335/14,514, flagged low — 35.7% `unknown_state`). Consistency: 0 conflicts across 7 pooled raw labels (fully consistent). Structure: depth 4, breadth 15, coupling 12 files. | No prior work to be "better than" here — there is nothing numeric to compare against, so none is claimed. The contribution being claimed is architectural, not a beaten benchmark: a single formal ontology (`SentimentState`/`TransitionType`/`TrajectoryType`/`DOMAIN_CONFIGS`) shared across e-commerce and social-media domains, evaluated on its own terms (not just via downstream model accuracy) — which the surveyed literature does not do at all. |
| Cross-Domain Generalization | F1 (macro), in-domain vs. cross-domain, % degradation | Kavitha et al., Rashid et al., Provaa et al. (implicitly, by absence) | Not reported as a comparison — all three train and evaluate within a single dataset/domain; none report train-on-X/test-on-Y transfer results. This matches Research Gap 4 in this project's own gap analysis ("Poor Cross-Domain Generalization" — "most existing studies evaluate their proposed models using the same dataset on which the models are trained"). | **TBD** — pending Colab training + `scripts/cross_domain_eval.py` run (both directions: Amazon→Dravidian and Dravidian→Amazon are both implemented, see prior session's Step 8 commit) | The claim here is methodological, not a beaten number: performing cross-domain evaluation at all is the contribution the surveyed literature's own gap supports, independent of what the eventual degradation percentage turns out to be. |
| Sequence Consistency Score (SCS) | SCS (mean, 0-1 range) | **None found in surveyed literature** | Not reported — no paper reviewed defines a metric for how consistent a model's predictions are across a sequence of reviews; all evaluate per-item classification quality only (accuracy/F1/precision/recall). | **TBD** — implemented and unit-exercised (`src/evaluation.py::sequence_consistency_score`), but only meaningful once real trained-model predictions exist to score | This is a methodological contribution: SCS has no prior equivalent in the surveyed literature. The claim is that it captures something F1 cannot (sequential coherence), not that it beats a number — there is no prior SCS-like number to beat. |

---

## What "TBD" means here

Every "TBD" in this table is blocked on the same thing: running
`notebooks/colab_training.ipynb` with the real encoder
(`bert-base-multilingual-cased` / `xlm-roberta-base`), which could not be
downloaded in this development environment (large binary transfers stall
here — see `scripts/sanity_check_pipeline.py`'s docstring). Every piece of
code needed to produce these numbers has been built and verified to run
correctly end-to-end using a small substitute encoder on real data — only
the actual training run is outstanding.

## Explicitly out of scope for this comparison

- **Word Sense Disambiguation / aspect-level opinion evolution** — paused
  pending guide confirmation (per this project's own scoping decision);
  the four WSD papers in `Research Papers/` were reviewed but not compared
  against, consistent with that pause.
- **"Sharpe ratio"** — not implemented; the term's intended meaning in this
  project's context is unconfirmed with the guide and is not addressed here.
