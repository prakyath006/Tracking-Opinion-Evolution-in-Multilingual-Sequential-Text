# Final Status — 2026-09-13

Plain-language summary. Every claim below is checked against a real file in
this repository as of this date — see the linked file for the underlying
numbers.

## The 6 modules

| Module | Complete with real numbers? |
|---|---|
| 1 — Ontology | ✅ Yes — [`outputs/metrics/module1_ontology.md`](metrics/module1_ontology.md) |
| 2 — WSD | ✅ Yes, **except accuracy** — see below |
| 3 — Encoder perplexity | ✅ Yes — [`outputs/metrics/module3_bert_perplexity.md`](metrics/module3_bert_perplexity.md) |
| 4 — Confidence/ECE/SCS | ✅ Yes — [`outputs/metrics/module4_sequential_model_amazon.md`](metrics/module4_sequential_model_amazon.md), [`..._dravidian_tamil.md`](metrics/module4_sequential_model_dravidian_tamil.md) |
| 5 — Cross-domain fuzzy typicality | ✅ Yes — [`outputs/metrics/module5_cross_domain.md`](metrics/module5_cross_domain.md) |
| 6 — Baseline comparison | ✅ Yes — [`outputs/metrics/module6_analysis.md`](metrics/module6_analysis.md), [`outputs/metrics/final_baseline_vs_proposed_comparison.md`](metrics/final_baseline_vs_proposed_comparison.md) |

**All 6 modules are complete with real, verified numbers.**

## LLM baseline (Panel Comment 4 / O3)

**Status: implemented and smoke-tested, Colab run pending. Not "done."**

- Code: `LLMPromptClassifier` in `src/baselines.py`, registered as `llm_prompt`
  in `BASELINE_REGISTRY` (`google/flan-t5-base` by default), wired into
  `scripts/train_baselines.py`'s existing training/eval loop.
- Verified correct via a real local smoke test (not an official result):
  `google/flan-t5-small` against 6 real Amazon review texts — the
  label-likelihood scoring path and real `.generate()` path produced
  identical predictions on all 6, and a real training step produced a real
  gradient with no crash (4 trainable params at the matched-capacity
  default).
- **`google/flan-t5-base` was never downloaded or run in this environment** —
  the ~1GB download timed out after 90 seconds (CPU-only environment). No
  accuracy/F1/confusion-matrix number exists for this baseline anywhere.
- Next step: run `notebooks/colab_training.ipynb` section 8c on a GPU, both
  domains. That alone closes this item — `scripts/compile_metrics.py` picks
  up the result automatically, no further code needed.

## WSD accuracy (O2)

**Status: still open.** `outputs/wsd_human_annotation_sample.csv` (45 real
annotation rows from the real 33-sample review set) is prepared and ready,
but **0 of 45 rows are filled in** as of this report — checked directly
against the file, not assumed. No accuracy number is computed or claimed.
Closing this needs a team member to fill in the `correct_sense` and
`is_correct` columns by hand; nothing further can be automated.

## Balanced 100-per-class comparison

**Explicitly out of scope for this report.** This is a teammate's assigned
task and was not touched here — no status claim is made about it one way or
the other in this document. See whichever status document that teammate
maintains for its actual state.

## One-line summary

Modules 1-6: done. WSD accuracy: waiting on a human, not on more code. LLM
baseline: code is real and verified, results are one GPU run away. Balanced
comparison: not this report's job.
