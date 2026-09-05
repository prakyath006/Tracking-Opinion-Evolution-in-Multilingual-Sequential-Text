# Defect Register — audit of 2026-09-04

One place to see every defect found in this project, what it affected, and its
status. Ordered by severity. Everything marked FIXED was verified, not assumed.

---

## Severity 1 — invalidated a result or a research claim

### D1. Models were compared at unequal trainable encoder capacity — FIXED
`mbert_sentence` and `xlmr_sentence` trained all 12 encoder layers (~178M
params) because `SentenceLevelTransformer` defaulted to `freeze_encoder=False`.
The full model trained 0 (`freeze_encoder=True`). `module6_analysis.md` printed
the resulting gaps (−0.33 sentiment F1 on Tamil) as architectural evidence.

*Verified from* `outputs/checkpoints/best_model_*.pt`, which stores each run's
argparse namespace, cross-checked against its `optimizer_state_dict` (43 tracked
tensors = exactly the 43 non-encoder tensors).

**Fix:** `--encoder_finetune_layers` (default 0) applied uniformly to every
baseline; `SentenceLevelTransformer` gained `finetune_layers` mirroring
`DomainAdaptedEmbeddings`. **Requires a GPU re-run of 4 baselines to close.**

### D2. The full model's encoder could never be fine-tuned — FIXED
`OpinionEvolutionTracker.encode_texts()` wrapped the encoder in an
unconditional `torch.no_grad()`, so `--no_freeze_encoder` unfroze 3 layers that
then received **no gradient**. The "own embeddings / domain-adapted
fine-tuning" claim (guide Module 1b) could not have worked.

Not triggered in the 2026-09-04 runs, which used `--freeze_encoder`, but it
would have silently wasted any run that tried the opposite.

**Fix:** `torch.set_grad_enabled(self.training and encoder_is_trainable)`.
*Verified:* `freeze_encoder=True` → 0 trainable, `requires_grad=False`;
`freeze_encoder=False` → 5,471,232 trainable, `requires_grad=True`.

### D3. Group B baselines could never fine-tune either — FIXED
`encode_sequence_batch()` in `train_baselines.py` was hardcoded to
`torch.no_grad()`. Same class of bug as D2.

**Fix:** gated on `enable_grad`; the embedder now joins the optimizer and train
mode, and grad clipping covers both parameter sets.

### D4. WSD's disambiguation logic was dead code — FIXED
All 173 lexicon surface forms mapped to exactly one aspect, so
`is_ambiguous()` was always `False` and `ContextWindowDisambiguator` never ran.
Measured `ambiguous_words = 0`; the reported "100% resolution" was `0/0`.
Module 2 was an aspect tagger, not a disambiguator.

**Fix:** three forms (`mass`, `punch`, `climax`) now carry the multiple senses
they demonstrably have, established by collocation analysis over all 53,263
rows. Now 682 ambiguous instances, 121 resolved (17.74%).
See `docs/module2_wsd_status.md`.

---

## Severity 2 — wrong numbers or claims in a deliverable

### D5. Half of all `hit` matches were false positives — FIXED
604 of 1,200 occurrences (50.3%) are followed by `like`/`likes` — the YouTube
idiom "fans hit like", not a box-office reference. All were tagged
`box_office_collection`.

**Fix:** `NON_ASPECT_COLLOCATIONS` guard. `box_office_collection` fell 402 → 327.

### D6. Three lexicon entries were unreachable — FIXED
`box office`, `first look`, `background music` are stored with spaces, but
`process()` only looked up single tokens, so they could never match.

**Fix:** bigram lookup before unigram lookup.

### D7. Six factual errors in the B5 module-by-baseline matrix — FIXED
The deliverable the guide explicitly asked for stated: full model "top 3 layers
fine-tuned" (actually 0); "~21.8M trainable" (actually 3,093,003); mBERT
Sentence "frozen or full fine-tune" (actually all 12); TextCNN "Static
Pretrained Word Embeddings (GloVe / FastText 300d)" (actually a randomly
initialised `nn.Embedding`, no pretrained vectors at all); kernel sizes
"3, 4, 5" (actually 2, 3, 4, 5); vocabulary "corpus-derived" (actually ≤20,000
whitespace tokens).

**Fix:** corrected in the generator, not the output, then regenerated.

### D8. Compiled results table contained duplicate rows — FIXED
`cross_domain_eval.py` scores each source domain against itself, so its
`amazon_to_amazon` and `tamil_to_tamil` entries duplicated the in-domain rows
from `test_results_*.json` — 20 rows where 18 were distinct, one copy carrying
capacity metadata and one not.

**Fix:** de-duplication in `compile_all()`, keeping the row that records
capacity.

---

## Severity 3 — code existed but had never been executed

### D9. Steps A5 and A6 were never run — FIXED
Committed but never executed, so Module 2 had no measured numbers and
`module2_wsd_planned.md` still claimed "NOT BUILT". Both now run.

### D10. Modules 1, 3, 4 and 5 metric scripts were never run — PARTIALLY FIXED
`ontology_eval.py`, `mlm_perplexity_eval.py`, `confidence_eval.py` and
`fuzzy_domain_score.py` all had `__main__` entry points and declared output
paths, none of which existed.

- Module 1 → `outputs/metrics/module1_ontology.md` + `outputs/ontology_evaluation_report.md` — **done**
- Modules 3, 4, 5 — **running locally on CPU**; slow but not blocked.

### D11. `ontology_eval.py`'s full report was unreachable — FIXED
`generate_ontology_eval_report()` existed but `__main__` only called
`generate_module1_report()`, so `outputs/ontology_evaluation_report.md` was
never written.

**Fix:** `__main__` now generates both.

---

## Severity 4 — housekeeping

### D12. Report generators crashed on Windows consoles — FIXED
`generate_module6_analysis.py` and `ontology_eval.py` raised
`UnicodeEncodeError` echoing Δ / ✅ to a cp1252 console. The files were always
written correctly; only the console echo failed, but it made the scripts exit
non-zero. Both now degrade gracefully.

### D13. Capacity was not recorded anywhere — FIXED
Nothing in the results recorded how much encoder capacity a run used, which is
why D1 went unnoticed. `train.py` and `train_baselines.py` now write
`encoder_finetune_layers` and `trainable_params`; `compile_metrics.py` carries
the column; `generate_module6_analysis.py` refuses to present a delta between
models of differing capacity without flagging it.

### D14. `.pytest_cache/` was not gitignored — FIXED

---

## Still open — not defects, but outstanding work

| # | Item | Blocked on |
|---|---|---|
| O1 | Re-run 4 baselines at matched capacity (closes D1) | Kaggle GPU, ~3h, in progress |
| O2 | WSD sense-disambiguation **accuracy** | 100-sample human-review set needs annotating; no gold labels exist |
| O3 | LLM-based baseline (listed as a panel requirement) | Not coded; needs a model choice and API access |
| O4 | Modules 3/4/5 reports | Running locally on CPU |
| O5 | `README.md` describes only the August preprocessing stage | Nothing — just needs writing |

One known inconsistency, documented rather than fixed:
`WordSenseDisambiguator.get_coverage_stats()` looks up single tokens directly,
so `coverage_pct` does not reflect the bigram matching (D6) or the collocation
guard (D5), while `aspect_distribution` (via `process()`) does. Ambiguity and
resolution counts are unaffected.

---

## Summary

**14 defects found, 13 fully fixed.** D1 is fixed in code and needs the GPU
re-run to close in the results. All 31 ontology consistency tests pass.

The two that would most have hurt at a review: **D1**, which made the headline
comparison say a plain baseline beat the proposed model, and **D7**, which
claimed pretrained embeddings the CNN baseline never used.
