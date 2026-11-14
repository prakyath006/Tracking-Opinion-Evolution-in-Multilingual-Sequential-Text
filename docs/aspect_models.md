# Aspect-level model components (Phase 2)

Three independent PyTorch components for aspect-level opinion evolution. They live beside, and do not change, the existing OET model (`src/model.py`) and its checkpoints.
**Nothing here has been trained or evaluated, and no performance is claimed.** The tests use tiny randomly initialised encoders and synthetic fixtures.

```
review text --[1 AspectExtractor]--> aspect spans (what is mentioned, where)
(aspect text, review) --[2 AspectSentimentClassifier]--> POSITIVE / NEGATIVE / NEUTRAL / MIXED for that aspect, per review
events of ONE aspect of ONE product, in time --[3 AspectSequenceModel]--> transition per event, trajectory per series
```

The components do not import each other. They are connected only by plain data (spans, probability vectors) and by `src/aspect_data.py`. A shared helper module,
`src/aspect_encoders.py`, loads the encoder, sets how much of it trains, pools, and counts parameters.

## 1. `src/aspect_extractor.py`: `AspectExtractor`

Multilingual transformer token classifier (mBERT or XLM-R) with a dropout and linear layer over the last hidden states, giving one BIO tag per token.

| | |
|---|---|
| Input | `input_ids`, `attention_mask` [batch, tokens]; optional `token_type_ids`, `labels` (-100 ignored) |
| Output | `logits` [batch, tokens, tags]; `loss` if labels are given (a batch with no valid label gives 0, not NaN) |
| Tags | `aspect_data.bio_tagset()`: `O, B-ASP, I-ASP`, or `bio_tagset(categories)`: `O, B-<c>, I-<c>, ...` (29 tags for the 14 product categories) |
| Encoder | `"mbert"`, `"xlmr"`, any Hugging Face name, or a module; `finetune` = `frozen`, `partial` (top n layers) or `full` (default) |
| Helpers | `bio_to_spans` (tags and token offsets to character spans), `extraction_batch` (pads `ExtractionExample`s), `predict_spans(texts, tokenizer)` |

`bio_to_spans` reads a span from a B tag over the I tags of the same category; an I tag with no open span starts one; special tokens never belong to a span. Several aspects in one review and
reviews with no aspect are decoded correctly (round trip checked against the gold labels of the fixtures).

## 2. `src/aspect_sentiment_model.py`: `AspectSentimentClassifier`

Aspect-conditioned sentiment from a sentence pair (aspect text, review text) built by `aspect_data.encode_sentiment_examples`. The encoder reads the aspect first, so one review gives a **different
output per aspect**; the model never produces a review-level sentiment. Output classes are `aspect_data.AspectSentiment` (POSITIVE 0, NEGATIVE 1, NEUTRAL 2, MIXED 3), not the ontology's labels.

| | |
|---|---|
| Input | `input_ids`, `attention_mask` (and `token_type_ids` for BERT-style tokenizers) [batch, tokens], one row per (aspect, review) pair; optional `labels` |
| Output | `logits` [batch, 4]; `loss`; with `return_embedding=True` also the pooled vector [batch, hidden] |
| Pooling | `cls` or masked `mean` (padding is excluded) |
| Head | the project's existing `ClassificationHead` (Linear, LayerNorm, ReLU, Dropout, Linear) |
| Helpers | `sentiment_batch` (pads encoded pairs), `predict_for_aspects(review, aspect_texts, tokenizer)` giving [n_aspects, 4] probabilities, and [0, 4] when the review has no aspect |

## 3. `src/aspect_sequence_model.py`: `AspectSequenceModel`

Models one **series**: all events of one aspect category of one product, in time order. `build_series` groups by `(sequence_type, entity_id, category)`, so aspects are never mixed in a trajectory, product
timelines and same-reviewer histories are never mixed, and an aspect that never occurs gives no series. `collate_series` re-checks this and the time order, and the model processes every series of a batch independently
(tested: rewriting another series does not change a series' output; padding does not either).

```
event features = [ predicted sentiment probabilities (4) | log(1+days since previous event) , log(1+days since first event) | category embedding (8) | optional text vector ]
   --> BiLSTM (2 layers, reused BiLSTMEncoder) --> state per event --+--> transition head  : UPGRADE / DOWNGRADE / STABLE   (per event, after the first)
                                                                     +--> temporal attention (reused SelfAttention) --> trajectory head : IMPROVING / DECLINING / STABLE / VOLATILE (per series)
```

* **No gold labels as input.** `forward(sentiment_probs, time_features, seq_lens, category_ids, text_features)` has no label parameter. The probabilities must come from the sentiment model:
  `events_from_examples` needs a probability vector for every example, a `source` naming the predictor (not "gold"), and raises `LeakageError` if every vector is exactly the one-hot of its own gold label (a best-effort guard, not a proof).
* **Gold labels build targets only** (`series_targets`), through the ontology's `TransitionType` and `TrajectoryType`. Following the guidelines only POSITIVE, MIXED and NEGATIVE are ranked: a NEUTRAL event has target -100,
  a transition compares an event with the previous ranked one, and a trajectory needs two ranked events (else -100, "insufficient observations").
* **Time gaps** enter as log-scaled inputs. The model has no continuous-time component.
* `AspectSequenceLoss` ignores -100 targets and gives 0 (not NaN) when nothing is defined.

## Trainable parameters

Counted with random weights from the real configurations (`python scripts/aspect_model_summary.py`; no weights downloaded, nothing trained).

| Model | Encoder mode | Total | Trainable | of which encoder |
|---|---|---|---|---|
| Extractor, mBERT, generic tags | frozen / top 2 layers / full | 177,265,155 | 2,307 / 14,178,051 / 177,265,155 | 0 / 14,175,744 / 177,262,848 |
| Extractor, mBERT, 29 category tags | frozen / top 2 / full | 177,285,149 | 22,301 / 14,198,045 / 177,285,149 | 0 / 14,175,744 / 177,262,848 |
| Extractor, XLM-R, generic tags | frozen / top 2 / full | 277,455,363 | 2,307 / 14,178,051 / 277,455,363 | 0 / 14,175,744 / 277,453,056 |
| Extractor, XLM-R, 29 category tags | frozen / top 2 / full | 277,475,357 | 22,301 / 14,198,045 / 277,475,357 | 0 / 14,175,744 / 277,453,056 |
| Sentiment classifier, mBERT | frozen / top 2 / full | 177,461,252 | 198,404 / 14,374,148 / 177,461,252 | 0 / 14,175,744 / 177,262,848 |
| Sentiment classifier, XLM-R | frozen / top 2 / full | 277,651,460 | 198,404 / 14,374,148 / 277,651,460 | 0 / 14,175,744 / 277,453,056 |
| Sequence model (14 categories, defaults) | n/a (no pretrained encoder) | 712,375 | 712,375 | 0 |
| Sequence model with 768-d text vectors | n/a | 810,679 | 810,679 | 0 |

The pooler is omitted from the encoders (the aspect models never use it), so these totals are slightly below a stock encoder's.

## What remains before real training

1. **Data.** No annotated aspect data exists: the 50-review pilot and the 16-review pilot are unannotated. Everything above has run only on fixtures.
2. **Out-of-fold sentiment predictions.** The sequence model must be trained on probabilities that the sentiment model produced for reviews it did *not* train on (cross-validation by product). In-sample
   predictions are over-confident and gold labels are forbidden. That pipeline is not written.
3. **Training and evaluation code.** There are no training loops, datasets or collators for the three models beyond the batch helpers, no learning-rate or seed handling (the existing `experiment_utils` could be reused), and no checkpointing.
4. **Evaluation protocol.** Span-level F1, aspect-sentiment macro-F1, and transition and trajectory macro-F1 on aspect series, on entity-disjoint splits, with majority baselines and with the **rule-derived trajectory from predicted aspect sentiment**
   as the comparison the learned sequence model has to beat (for the original OET the learned head added only a small, uncertain gain over that rule).
5. **End-to-end inference.** Extractor to spans, spans to sentiment, sentiment to series, series to trajectories is not wired together or evaluated as a pipeline.
6. **Series sizes.** How many aspect series have at least two ranked events is unknown until annotation exists; it may be small. Long product timelines (hundreds of reviews) also need a time-window rule (open decision in the guidelines).
7. **Real-encoder checks.** Forward passes were tested with tiny random encoders; the real mBERT and XLM-R models were used only for parameter counts and tokenizer alignment. XLM-R weights are not on the laptop (download on Kaggle),
   and GPU memory and speed are unmeasured.
8. **Choices not yet tested for value**: category-specific versus generic tags, `cls` versus `mean` pooling, the optional text vector and the category embedding.
