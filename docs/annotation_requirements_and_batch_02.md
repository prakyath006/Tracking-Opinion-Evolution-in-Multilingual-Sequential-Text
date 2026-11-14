# Training requirements and annotation batch 02

Why this exists: the 50-review pilot (`outputs/annotation_consolidation/pilot_batch_01/`) is far below the floors the aspect training pipeline enforces. This page states the floors, what the pilot has, what is still needed, and the batch that closes the gap. **Nothing is trained and no floor is lowered.**

## 1. The floors (from `src/aspect_training.py`, unchanged)

| Floor | Value | Effect |
|---|---|---|
| Products in the whole corpus for out-of-fold generation (`MIN_PRODUCTS_FOR_CV`) | 30 | **refuses** (`cv_blockers`) |
| Products per fold (`MIN_PRODUCTS_PER_FOLD`, 5 folds by default) | 3, so at least 15 | refuses |
| Aspect examples in the whole corpus (`MIN_EXAMPLES_FOR_CV`) | 300 | refuses |
| Usable aspect series (two or more ranked events) in train (`MIN_TRAIN_SERIES`) | 50 | **refuses** sequence training (`series_blockers`) |
| Usable series in val and in test (`MIN_EVAL_SERIES`) | 20 each | refuses |
| Products in the whole corpus (`MIN_TOTAL_PRODUCTS`) | 30 | warns |
| Products per evaluation split (`MIN_EVAL_PRODUCTS`) | 5 | warns |
| Items per evaluation split (`MIN_EVAL_ITEMS`) | 100 | warns |
| Examples per sentiment class per evaluation split (`MIN_EVAL_CLASS_EXAMPLES`) | 20 | warns |
| Training items (`MIN_TRAIN_ITEMS`) | 500 | warns |

An example is one (review, category) pair; a series is one (product, category) with at least two ranked (`POSITIVE`, `NEGATIVE`, `MIXED`) reviews. Products are split 70/15/15 whole, so the evaluation splits hold few products.

## 2. What the pilot has

| | Training eligible now (`yes`) | Plus `pending_policy` | All provisional rows |
|---|---|---|---|
| Reviews / products | 39 / 10 | 49 / 10 | 50 / 10 |
| Examples (review, category) | 61 | 121 | 130 |
| Usable series | 14 | 29 | 30 |

The pilot has 5 reviews per product, which is why it yields about 3 series per product. It fails the products floor (10 of 30), the examples floor (61 to 130 of 300) and the series floors (14 to 30 of 90).

## 3. How much more is needed

Rates measured on the pilot: 2.6 examples per review, about 16% `NEGATIVE`, and per-category ranked-review rates (`GENERAL` 0.62, `EFFECTIVENESS` 0.56, `OTHER_ASPECT` 0.26, `FRAGRANCE` 0.16 ... `PACKAGING` 0.04). Expected usable series per product with k reviews, treating reviews as independent: 3.0 (k=5), 4.5 (k=8), **5.4 (k=10)**, 6.2 (k=12), 8.6 (k=20). These are estimates from 50 reviews and will move; re-measure on wave 1.

| Target | Additional product timelines | Additional annotated reviews | Resulting corpus |
|---|---|---|---|
| **Hard floors only** (refusals clear, warnings remain) | at least 20 (30 products) | about 200 (20 products x 10), giving 650 examples and about 137 series; train examples about 455 (< 500, warns), eval splits 4 to 5 products (warns) | passes `cv_blockers` and `series_blockers` narrowly: val and test about 25 series each against a floor of 20 |
| **Recommended: floors and warnings clear** | **40 (50 products)** | **400 (40 x 10)**, giving about 1,170 examples, about 245 series (train about 170, val and test about 37 each), 7 to 8 products per evaluation split, about 28 `NEGATIVE` examples per evaluation split | no warning expected |

So: at least **about 180 to 240 more annotated examples and 20 more product timelines** to clear the refusals (300 examples minus the 121 eligible-or-pending or the 61 eligible now), and **about 1,040 more examples from 40 more product timelines** for a clean run. Only signed-off rows count, so the examples must be annotated, adjudicated and (for policy-dependent rows) released by the D1 to D11 approvals. Plan for 85% of rows being usable, which puts the recommended target at about 475 reviews if the yield is lower; wave 1 will tell.

## 4. Batch 02

`data/annotations/pilot_batch_02_product_timelines.csv` (git-ignored like batch 01) and `pilot_batch_02_manifest.json`; generator and independent check: `scripts/make_batch_02.py`. 400 real reviews, 40 product timelines x 10, annotation columns blank, same 22 columns and sequence definition as batch 01.

* **Not only popular, long-lived products.** Four popularity strata by the number of kept reviews: SMALL 12 to 19 (10 products), MEDIUM 20 to 49 (12), LARGE 50 to 199 (12), XLARGE 200 or more (6). Median kept reviews per product: 14, 25, 62, 283 (batch 01: 26, with one product of 564).
* **Broader aspect categories.** The draw is round-robin over each product's distinctive category (the category its reviews mention most, relative to the corpus, by taxonomy alias matching). Result: 13 different distinctive categories among 40 products (for example `DURABILITY`, `PACKAGING`, `SKIN_HAIR_REACTION` 4 each; `DELIVERY_SERVICE`, `SIZE_QUANTITY`, `AUTHENTICITY` 3 each). Alias matching is only a sampling aid; no label was created and no text was judged. The overall lexical category mix of the 400 reviews stays close to the corpus (for example `PACKAGING` 14% against 4% in batch 01), so the gain is in product variety, not in a skewed mix.
* **One deliberate deviation.** SMALL products use at least 12 reviews over 180 days (batch 01: 20 reviews over 365 days). It is flagged in the manifest and needs the team's approval; the other 30 products meet the original rule. If the team refuses it, replace the 10 SMALL products by re-running with another stratum quota.
* **Excluded:** every product of batch 01 and of the 16-record pilot. Message ids and products do not overlap (checked).
* **Two waves of 20 products,** alternating inside each stratum, so each wave is a balanced half. **Annotate wave 1 first** (about 200 reviews), re-measure the rates in section 3 and decide whether wave 2 is needed.
* **Verification:** `python scripts/make_batch_02.py --verify` re-reads the raw file and checks every review (line, product, timestamps, text, length, 10 distinct reviewers per product, strictly increasing timestamps, strata counts, no overlap with earlier batches). It passed.

## 5. Workflow for the batch

1. Two annotators label the same reviews independently (`docs/aspect_annotation_guidelines.md`, v0.2 until v0.3 is approved), starting with wave 1. Check each file with `python scripts/validate_aspect_annotations.py FILE.csv --reference data/annotations/pilot_batch_02_product_timelines.csv`, and compare with `--compare`.
2. Adjudicate the disagreements with `scripts/prepare_adjudication_package.py`, recording the anonymized adjudicator ID on each row (individual `adjudicator_N`, or `team_adjudicators_1_to_4` for collective decisions).
3. Consolidate with `scripts/build_provisional_annotations.py`. It is written for batch 01: its rejected-span audit table and low-confidence lists are batch-specific, so batch 02 needs its own audit table (or an adapted copy) before the same tiers and `training_eligible` logic are used.
4. Train only on rows with `training_eligible=yes` after the D1 to D11 decisions are approved and the strict validator passes.
5. Re-run `python scripts/aspect_model_summary.py` and the pipelines' own floor checks, which will refuse again if the corpus is still too small.
