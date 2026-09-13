# Final Comparison — Full Model vs. All Baselines

Every number below is copied from an already-committed, already-verified
report — `outputs/metrics/results_table.md` (accuracy, F1 macro, SCS) and
`outputs/metrics/module4_sequential_model_{amazon,dravidian_tamil}.md`
(precision/recall macro, full model only). None is estimated, rounded
favourably, or re-derived. Where a number was never recorded for a model, the
cell says **N/A** rather than a guess — this applies to precision/recall for
every baseline (never computed in any committed run) and to every cell for
the LLM-based baseline (never implemented or run; see the note at the bottom).

All 6 baselines requested by the panel are listed. 5 are architectural
baselines with real, matched-capacity results (0 trainable encoder layers
throughout, verified in `docs/defect_register.md` D1). The 6th, an LLM-based
baseline, could not be added in this pass — see "LLM-based baseline" below.

---

## Amazon (in-domain, sentiment head)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | SCS |
|---|---|---|---|---|---|
| **Full Model (OET)** | 0.6700 | 0.5308 | 0.5758 | 0.5329 | 0.5636 |
| xlmr_sentence | **0.7800** | N/A | N/A | **0.6188** | N/A — no sequence output |
| attention_only | 0.5756 | N/A | N/A | 0.4715 | 0.5198 |
| lstm_only | 0.5665 | N/A | N/A | 0.4377 | **0.8925** |
| mbert_sentence | 0.5727 | N/A | N/A | 0.4837 | N/A — no sequence output |
| textcnn | 0.5391 | N/A | N/A | 0.4271 | N/A — no sequence output |
| LLM-based (flan-t5-base, not run) | N/A | N/A | N/A | N/A | N/A |

**% improvement of proposed model over the best baseline, Amazon:**

| Metric | Full Model | Best baseline | Improvement |
|---|---|---|---|
| Accuracy | 0.6700 | 0.7800 (xlmr_sentence) | **-14.10%** (loses) |
| F1 (Macro) | 0.5329 | 0.6188 (xlmr_sentence) | **-13.88%** (loses) |
| SCS | 0.5636 | 0.8925 (lstm_only) | **-36.85%** (loses on raw SCS — see caveat below) |
| Precision / Recall (Macro) | 0.5308 / 0.5758 | N/A (never recorded for any baseline) | Not computable |

On Amazon, the full model does **not** win sentiment accuracy or F1 against
`xlmr_sentence`, even at matched (0-layer) encoder capacity. This is a real,
reproducible result (see `docs/capacity_matched_comparison.md`), not an
artifact of the capacity bug that was fixed earlier.

---

## Dravidian Tamil (in-domain, sentiment head)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | SCS |
|---|---|---|---|---|---|
| **Full Model (OET)** | 0.5134 | 0.4095 | 0.4257 | 0.3930 | 0.3062 |
| mbert_sentence | **0.5219** | N/A | N/A | **0.4198** | N/A — no sequence output |
| attention_only | 0.4860 | N/A | N/A | 0.3920 | 0.3087 |
| xlmr_sentence | 0.4256 | N/A | N/A | 0.3669 | N/A — no sequence output |
| textcnn | 0.4032 | N/A | N/A | 0.3514 | N/A — no sequence output |
| lstm_only | 0.3863 | N/A | N/A | 0.2933 | **0.6585** |
| LLM-based (flan-t5-base, not run) | N/A | N/A | N/A | N/A | N/A |

**% improvement of proposed model over the best baseline, Tamil:**

| Metric | Full Model | Best baseline | Improvement |
|---|---|---|---|
| Accuracy | 0.5134 | 0.5219 (mbert_sentence) | **-1.63%** (loses) |
| F1 (Macro) | 0.3930 | 0.4198 (mbert_sentence) | **-6.38%** (loses) |
| SCS | 0.3062 | 0.6585 (lstm_only) | **-53.50%** (loses on raw SCS — see caveat below) |
| Precision / Recall (Macro) | 0.4095 / 0.4257 | N/A (never recorded for any baseline) | Not computable |

On Tamil, the full model loses sentiment accuracy and F1 to `mbert_sentence`
by a smaller margin than on Amazon, but still loses. `attention_only` is
essentially tied on F1 (0.3920 vs 0.3930, a 0.0010 gap).

---

## Where the full model actually wins: sequence-level heads

Sentiment accuracy/F1 is the one metric every baseline can attempt, which is
why it's the headline comparison above — and on it, the full model does not
win outright. Its real, defensible advantage is structural: it is the only
architecture (besides the two sequence ablations) that predicts a trend and a
trajectory at all, and the only one that beats both sequence ablations on
trajectory:

| Trajectory F1 (Macro) | Amazon | Tamil |
|---|---|---|
| **Full Model (OET)** | **0.5471** | **0.2678** |
| lstm_only | 0.4250 | 0.2624 |
| attention_only | 0.3307 | 0.2347 |
| mbert_sentence / xlmr_sentence / textcnn | N/A — no trajectory head | N/A — no trajectory head |

% improvement over best baseline: Amazon **+28.73%** (0.5471 vs 0.4250),
Tamil **+2.06%** (0.2678 vs 0.2624). Both real wins.

**Trend F1** (Amazon 0.5749, Tamil 0.4450) has no baseline to compare against
at all — none of the 5 architectural baselines implement a trend/transition
head (see `outputs/metrics/module_by_baseline_comparison.md`), so this is not
a "win," it is a capability no baseline attempts.

---

## SCS caveat — read alongside F1, not in isolation

`lstm_only`'s Amazon SCS (0.8925) and Tamil SCS (0.6585) are the highest of
any model, including the full model — but `lstm_only` also has the lowest or
near-lowest sentiment F1 in both domains (0.4377 Amazon, 0.2933 Tamil, the
worst Tamil F1 of any model tested). Sequence Consistency Score rewards a
model for predicting the *same* label across a whole review sequence,
regardless of whether that label is correct; a model that (near-)collapses to
one dominant class scores a high SCS trivially. This is a known reading of
the metric (`docs/step_b2_b3_encoder_sequential_comparison.md`'s "Methodological
Contributions" section), and it is why the SCS "improvement" rows above are
reported honestly as losses for the full model rather than omitted — but they
should not be read as "lstm_only produces more useful sequential predictions
than the full model." The full model's trajectory F1 (which requires getting
the actual class right across the sequence, not just repeating one) beats
`lstm_only` in both domains, which is the metric that actually penalizes
trivial collapse.

---

## LLM-based baseline

**Not implemented in this pass.** Adding `flan-t5-base` (or a comparable
model) as a 6th `BASELINE_REGISTRY` entry and evaluating it on both domains
requires either a GPU (this environment is CPU-only; fine-tuning or even
zero/few-shot inference with a ~250M-1B parameter seq2seq model over the full
Amazon/Tamil test sets is not practical on CPU in this session) or API access
to a hosted model, neither of which is available here. This is recorded as
open item **O3** in `docs/defect_register.md` and remains genuinely
unresolved — no placeholder numbers are given for it above.

---

## Domains not included

Malayalam and Kannada are not included in this table. No full-model
in-domain training run exists for either language in the committed results
(`outputs/metrics/results_table.md` has no `dravidian_malayalam` or
`dravidian_kannada` in-domain rows — only Amazon→Malayalam/Kannada and
Tamil→Malayalam/Kannada *cross-domain transfer* rows exist, which measure
something different: how well a model trained on one domain generalizes to
another, not that domain's own baseline comparison). Extending this table to
those languages needs a full training run (all 6 models) on each, which is
GPU work not available in this pass.
