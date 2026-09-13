# Final Report — Tracking Opinion Evolution in Multilingual Sequential Text

Structured around the five research gaps this project set out to address
(`README.md`'s "Research gaps addressed" table). Every number below traces to
a real file in `outputs/metrics/` — none is estimated or reused from a stale
prior claim. Where a gap's evidence is still incomplete, that is stated
directly rather than rounded up. No dedicated "final report" file existed
before this one (checked: no `FINAL*`, `*final*`, `*summary*` file anywhere
in the repo) — this is a new document, not an update to something that
existed.

---

## Gap 1 — Transformers ignore sequential opinion evolution

**Status: Solved**, with the evidence now including a genuine head-to-head at
matched training capacity, not just an internal ablation.

The full model's trajectory classification head beats both single-component
ablations in both domains tested:

| Trajectory F1 (Macro) | Amazon | Tamil |
|---|---|---|
| **Full Model (Bi-LSTM + Attention)** | **0.5471** | **0.2678** |
| lstm_only (Bi-LSTM, no attention) | 0.4250 | 0.2624 |
| attention_only (attention, no Bi-LSTM) | 0.3307 | 0.2347 |

(`outputs/metrics/results_table.md`, `outputs/metrics/module6_analysis.md`)

Both components measurably help: dropping attention costs 0.1221 F1 on
Amazon and 0.0054 on Tamil; dropping the Bi-LSTM costs 0.2164 F1 on Amazon
and 0.0331 on Tamil. Neither ablation reaches the full combination in either
domain — this is the project's clearest, least caveated result.

On raw single-review sentiment classification, however, the full model does
**not** universally win — `xlmr_sentence` beats it on Amazon sentiment F1
(0.6188 vs 0.5329) and `mbert_sentence` beats it on Tamil (0.4198 vs 0.3930),
both at the same matched (0 trainable encoder layers) capacity
(`docs/capacity_matched_comparison.md`). The claim for Gap 1 is specifically
about *sequential* prediction quality (trajectory), where the full model's
architecture is designed to help and does; it is not a claim that adding
Bi-LSTM+attention improves every metric.

---

## Gap 2 — Inconsistent multilingual/code-mixed embeddings

**Status: Partial.** Module 3 measures the underlying problem directly via
masked-language-model perplexity — how much harder each encoder finds
code-mixed Dravidian text than English:

| MLM Perplexity | Kannada | Malayalam | Tamil | Amazon (English) |
|---|---|---|---|---|
| mBERT | 107.67 | 78.06 | 151.97 | 19.35 |
| XLM-R | 216.79 | 111.51 | 118.11 | 6.38 |

(`outputs/metrics/module3_bert_perplexity.md`)

Code-mixed Dravidian text is 6-34x harder for both encoders than English —
this is a real, measured confirmation of the gap this project set out to
address, not an assumption. mBERT wins on Kannada and Malayalam; XLM-R wins
on Tamil and (by a wide margin) English.

What remains partial: the released, reported runs all use a **frozen**
encoder (`--freeze_encoder`, 0 trainable layers) for the full model, so no
domain-adapted fine-tuning of the embeddings themselves has actually been
measured yet. `--no_freeze_encoder` (3 trainable top layers) is implemented
and was fixed for a real gradient bug (D2 in `docs/defect_register.md`), but
no run at that setting has been reported. Closing this fully means running
and reporting that configuration, which needs GPU time not spent on it yet.

---

## Gap 3 — Heavy dependence on supervised learning

**Status: In progress**, newly moved from "scoped as future work" by this
session's addition of the LLM-based baseline (`llm_prompt`,
`google/flan-t5-base`, `src/baselines.py`).

At its matched-capacity default (`--encoder_finetune_layers 0`), this
baseline requires **no task-specific supervised fine-tuning at all** — it is
prompted to generate a sentiment label directly, using only a 4-parameter
per-class calibration term trained on top of the frozen pretrained LLM
(verified locally: `trainable_params=4`, see the commit that added it). This
is a genuine, if small, step toward reducing dependence on supervised
training data, and is the first concrete artifact addressing this gap.

**No accuracy/F1 number exists for it yet** — the implementation is verified
correct via a local smoke test (`flan-t5-small`, real Amazon review text,
`forward()`'s scores and real `.generate()` agreed on all 6 samples tested),
but the actual evaluation on the real Amazon/Tamil test sets has not been
run: `flan-t5-base` could not be downloaded in this CPU-only environment
(timed out after 90s). `notebooks/colab_training.ipynb` section 8c is the
ready-to-run GPU cell; see `docs/panel_review_2_status.md` for the exact
remaining step. This gap should be reported as "in progress with a concrete
next step," not solved and not still purely future work.

---

## Gap 4 — Poor cross-domain generalization

**Status: Solved** — this project not only performs cross-domain evaluation
(the methodological gap identified in the literature review,
`docs/literature_comparison.md`) but reports real degradation numbers for
all 8 transfer directions:

| Transfer | Sentiment F1 (in-domain → cross-domain) | % Degradation |
|---|---|---|
| Amazon → Tamil | 0.5329 → 0.2300 | 56.8% |
| Amazon → Malayalam | 0.5329 → 0.2038 | 61.8% |
| Amazon → Kannada | 0.5329 → 0.2214 | 58.5% |
| Tamil → Amazon | 0.3930 → 0.2505 | 36.2% |
| Tamil → Malayalam | 0.3930 → 0.2878 | 26.8% |
| Tamil → Kannada | 0.3930 → 0.2903 | 26.1% |

(`outputs/metrics/module5_cross_domain.md`)

The clearest finding: the **Tamil-trained model degrades far less**
transferring to other Dravidian languages (26-27%) than the **Amazon-trained
model** degrades transferring to any Dravidian language (57-62%) — code-mixed
Dravidian-to-Dravidian transfer is much more forgiving than English-to-code-mixed
transfer, which is itself a useful, specific finding beyond "cross-domain
degrades performance." F1 stability ratio (mean/std across all 8 setups) is
highest for trend (4.74) and lowest for trajectory (2.46), meaning trend
predictions are the most consistent across domain shifts of the three heads.

---

## Gap 5 — Lack of aspect-level opinion evolution

**Status: Half solved**, unchanged by this session (this gap's own
scope — a per-aspect trajectory head — was not part of this session's
assigned work).

Module 2 tags aspects corpus-wide across 7 categories (4,997 aspect words of
144,973 total) and resolves 121 of 682 genuinely ambiguous instances
(17.74%) — real disambiguation, not just tagging, verified against the
lexicon (`outputs/metrics/module2_wsd_results.md`). What is still missing is
a trajectory head scoped *per aspect* rather than per whole review sequence
— aspects are identified, but their individual evolution over a sequence is
not yet modeled or scored separately from the sequence-level trajectory head
Gap 1 addresses.

**One number for this module is still open, not closed by this session
either:** WSD sense-disambiguation accuracy. `outputs/wsd_human_annotation_sample.csv`
(45 real annotation rows from the 33-sample review set) is prepared and
ready, but 0 of 45 rows are filled in as of this report — no team member has
annotated it yet. No accuracy figure should be cited until that happens.

---

## Summary table

| Gap | Status | Real evidence exists? |
|---|---|---|
| 1 — Sequential opinion evolution | ✅ Solved | Yes — trajectory F1 beats both ablations, both domains |
| 2 — Multilingual/code-mixed embeddings | ⚠️ Partial | Yes for the problem (perplexity gap measured); no for the fix (fine-tuned-encoder run not reported) |
| 3 — Supervised-learning dependence | 🔄 In progress | Implementation + smoke test yes; evaluation numbers no (Colab-pending) |
| 4 — Cross-domain generalization | ✅ Solved | Yes — 8 transfer directions, real degradation %, real stability ratios |
| 5 — Aspect-level opinion evolution | ⚠️ Half | Yes for aspect tagging; no for per-aspect trajectory; WSD accuracy still open |

Modules 1, 3, 4, 5, 6 are complete with real numbers. Module 2 is complete
except its accuracy figure. The LLM baseline is implemented and correct but
unevaluated. The balanced 100-per-class comparison is explicitly out of
scope for this report (a teammate's assigned task) and is not represented
above.
