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
`DomainAdaptedEmbeddings`.

**Closed 2026-09-05** (commit `b3b4f98`): the 4 affected baselines were
re-run at matched (0-layer) capacity and `outputs/metrics/results_table.md` /
`module6_analysis.md` were regenerated with the real post-re-run numbers.
*Verified by this audit* (2026-09-13) by diffing `results_table.md`'s git
history: `fa43658` (2026-09-04) shows `mbert_sentence`/`xlmr_sentence` still
at `encoder_finetune_layers=12` with their original (invalidated) F1 scores
(Amazon 0.6419/0.6768, Tamil 0.7206/0.6626); the very next commit, `b3b4f98`
(2026-09-05), replaces both with `encoder_finetune_layers=0` and materially
different, lower F1 scores (Amazon 0.4837/0.6188, Tamil 0.4198/0.3669) —
consistent with a genuinely re-run frozen-encoder baseline, not a relabeling.
This closes O1 below; the "Still open" and `docs/capacity_matched_comparison.md`
entries describing it as pending were stale and are corrected in this pass.

**Result post-re-run, matched capacity (0 encoder layers) throughout:** the
full model still wins sentiment F1 on Tamil (0.3930) against `xlmr_sentence`
(0.3669), but **loses** on Tamil to `mbert_sentence` (0.4198) and **loses** on
Amazon to `xlmr_sentence` (0.6188 vs 0.5329). These sentence-level losses are
real at matched capacity — they are no longer explained by D1's budget
confound, and should be reported as genuine findings, not fixed further.

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

### D10. Modules 1, 3, 4 and 5 metric scripts were never run — FIXED
`ontology_eval.py`, `mlm_perplexity_eval.py`, `confidence_eval.py` and
`fuzzy_domain_score.py` all had `__main__` entry points and declared output
paths, none of which existed.

- Module 1 → `outputs/metrics/module1_ontology.md` + `outputs/ontology_evaluation_report.md` — **done**
- Module 3 → `outputs/metrics/module3_bert_perplexity.md` — **done**
- Module 4 → `outputs/metrics/module4_sequential_model_{amazon,dravidian_tamil}.md` — **done**
- Module 5 → `outputs/metrics/module5_cross_domain.md` — **done** (2026-09-13, commit `aa249fa`; see O4 below)

*Updated 2026-09-13: this row previously said Modules 3/4/5 were "running
locally on CPU; slow but not blocked" — all three had in fact finished and
their reports were sitting in `outputs/metrics/` already. Corrected against
the real files, not assumed.*

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

## Investigated — Amazon SCS discrepancy (train.py log vs Module 4 report)

A panel review flagged that `train.py`'s own end-of-training console log for
the Amazon run reported **SCS = 0.5893 (std 0.3517)**, while the committed
Module 4 report (`outputs/metrics/module4_sequential_model_amazon.md`, via
`src/confidence_eval.py`) reports **SCS = 0.5636**.

**What was checked:** `train.py` (`test_results["scs"]`, line ~458) and
`confidence_eval.py` (line ~412) both call the *same* function,
`sequence_consistency_score()` in `src/evaluation.py` — a pure, deterministic
function of the model's predicted label sequences (mean/std of per-sequence
flip counts; order-independent). Both load the checkpoint from the same path
convention (`outputs/checkpoints/best_model_amazon.pt`) and both evaluate the
`test` split of `get_amazon_dataloader()`, which uses a fixed `random_seed=42`
train/val/test permutation — so for a *fixed* checkpoint and a *fixed* data
file, the two code paths are mathematically guaranteed to produce an
identical number. This rules out a genuine calculation difference between the
two SCS implementations — there is only one implementation.

**Update, 2026-09-13:** `outputs/logs/training_log_amazon.json` and
`outputs/logs/test_results_amazon.json` are now tracked (commit `b75fab1`),
so this could be re-checked against the real artifacts rather than reasoned
about in their absence.

`test_results_amazon.json` records `scs_mean: 0.5635779...` — matches the
Module 4 report's 0.5636 to four decimal places, confirming that figure is
exactly this file's test-set SCS, not a re-derived or rounded number.

`training_log_amazon.json` is a 10-epoch list of `val_scs` (validation-set
SCS logged after each epoch, not test-set SCS). Its values range 0.5424–0.6196
across epochs, and epoch 10 (the last one, and the one `train.py`'s console
would have printed right before loading the best checkpoint for final test
evaluation) is **0.5885** — close to the reported 0.5893, but not an exact
match, and no `val_scs_std` is logged per epoch at all, so the paired "std
0.3517" cannot be matched against this file either (`test_results_amazon.json`'s
own `scs_std` is 0.3383, also not 0.3517). No exact 0.5893/0.3517 pairing
exists anywhere in the now-tracked logs.

**Most likely explanation, updated:** 0.5893 is very likely the **last
epoch's validation SCS** (0.5885, off by 0.0008 — plausibly a transcription
or rounding slip when the figure was quoted from console output rather than
read back from a saved file) misremembered or mislabeled as the **test** SCS,
rather than evidence of two different trained models or a stale checkpoint as
originally hypothesized. Validation SCS and test SCS are computed by the same
function over different data splits (val vs. test), so they are expected to
differ — this alone fully explains a gap of this size without needing to
invoke a retrain or a changed dataset.

**Which number is correct:** treat **0.5636** (`test_results_amazon.json`,
the Module 4 report, and the figure the README cites) as ground truth for
"Amazon test-set SCS." 0.5893 is real (or very close to real) but describes
validation-epoch-10 SCS, a different quantity, and should not be cited as the
test SCS. This is not a fixed defect (no bug was found in either code path) —
it is recorded here because the discrepancy was a val/test mismatch that
using the now-tracked logs could confirm directly.

The original recommendation to commit `outputs/logs/*.json` (made when this
section was first written, before those files were tracked) has since been
acted on — see commit `b75fab1` — which is exactly what made this update
possible.

---

## Still open — not defects, but outstanding work

| # | Item | Status |
|---|---|---|
| O1 | Re-run 4 baselines at matched capacity (closes D1) | **DONE** (2026-09-05, commit `b3b4f98`) — see D1 above |
| O2 | WSD sense-disambiguation **accuracy** | Still open — no gold labels exist yet, so no accuracy figure can be computed. The 33-sample human-review set (`outputs/metrics/module2_wsd_results.json`'s `sample_annotations`, 45 word-level annotations) is now exported to `outputs/wsd_human_annotation_sample.csv` (2026-09-13) with empty `correct_sense`/`is_correct` columns, ready for a team member to fill in by hand. **Do not mark this closed until that file comes back annotated and an accuracy number is computed from it** — exporting the sample is not the same as measuring accuracy. |
| O3 | LLM-based baseline (listed as a panel requirement) | Still open — not coded; needs a model choice, and a GPU or API access to actually run it |
| O4 | Module 5 report (cross-domain fuzzy typicality) | **DONE** (2026-09-13, commit `aa249fa`) — `outputs/cross_domain/cross_domain_results_tamil.json` and `outputs/fuzzy_domain_scores.csv` were generated on GPU and tracked (commit `b75fab1`), then `generate_module5_report()` ran against them for real. `outputs/metrics/module5_cross_domain.md` now reports in-domain vs. cross-domain F1 and % degradation for all 8 setups, F1 stability ratios per head, and fuzzy typicality scores per domain — no "Pending" sections remain. All of Modules 1/3/4/5/6 are done. |
| O5 | `README.md` describes only the August preprocessing stage | **DONE** — rewritten 2026-09-12 (commit `201f1e5`) |

Note on how O1 *and* O4 stayed misreported after the underlying work was
already done: O1's fix landed on 2026-09-05 but this table still said
"pending" until a 2026-09-13 pass diffed the actual commit history. O4 landed
on 2026-09-13 (commit `aa249fa`) but this table still said "Still open" in
the very next commit that same day, because updating the register isn't part
of what closing a module does automatically. Same root cause both times: this
register is not self-updating. Check it against `outputs/metrics/*.json`/`.md`
(and, now that they're tracked, `outputs/logs/`, `outputs/cross_domain/`)
before citing a status from it, the same way this document already asks of
everything else.

One known inconsistency, documented rather than fixed:
`WordSenseDisambiguator.get_coverage_stats()` looks up single tokens directly,
so `coverage_pct` does not reflect the bigram matching (D6) or the collocation
guard (D5), while `aspect_distribution` (via `process()`) does. Ambiguity and
resolution counts are unaffected.

---

## Summary

**14 defects found, all 14 fully fixed** (D1 closed 2026-09-05 by the GPU
re-run, verified 2026-09-13 against git history — see D1 above). All 31
ontology consistency tests pass.

Module status as of 2026-09-13: **Modules 1, 2 (except accuracy), 3, 4, 5, 6
are all complete**, each with a real, populated report in `outputs/metrics/`.
The one number still missing project-wide is Module 2's WSD sense
**accuracy** (O2) — the sample is now ready for human annotation
(`outputs/wsd_human_annotation_sample.csv`), but no one has filled it in yet,
so accuracy stays unmeasured until they do. O3 (LLM-based baseline) remains
uncoded, blocked on GPU/API access.

The two defects that would most have hurt at a review: **D1**, which made the
headline comparison say a plain baseline beat the proposed model, and **D7**,
which claimed pretrained embeddings the CNN baseline never used. The one
process lesson worth carrying forward: **O1 and O4 both sat "closed in the
data but open in this document" for a week or more** — closing a module in
the actual output files and updating this register are two different steps,
and only re-checking this file against `outputs/` directly (not against its
own prior claims) caught the drift both times.
