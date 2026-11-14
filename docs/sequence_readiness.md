# Sequence readiness: which data supports chronological aspect-level evaluation

All numbers come from the project's own files and are reproducible with `python scripts/sequence_readiness.py`
(read-only on the data; full output in `outputs/diagnostics/sequence_readiness/readiness_stats.json`).
Nothing was simulated, resampled or annotated. Product-review timelines and social-media threads are reported separately.

## Research units (decided 2026-11-14)

| Unit | Role | Definition | Data available now |
|---|---|---|---|
| **Product timeline** | **primary e-commerce unit** | reviews of one product from potentially different reviewers, ordered by real timestamp | 9,271 products with at least 10 reviews, 450 with at least 100; pilot batch 01 of 50 reviews in 10 timelines |
| Same reviewer, same product | **secondary experiment** | one reviewer's repeated reviews of one product | 195 sequences of 2+ reviews, 9 of 3+, 5 of 3+ spanning a day or more; pilot of 16 reviews |
| Reviewer timeline across products | legacy (the OET sequences) | one reviewer's reviews of different products | 5,324 reviewers, 23,736 reviews; 98.2% of reviewers never repeat a product |
| DravidianCodeMix comments | **auxiliary** comment-level sentiment dataset | single comments, no order | 53,263 comments; **0** chronological threads |
| Social-media discussion thread | not available | comments with reply links and timestamps | none; needs newly collected data |

## Bottom line

* **Product timelines are the only data that is chronological, tied to one entity and large enough for aspect-level evaluation.** Timestamps are real and nearly always distinct. They answer "how does opinion about a product's aspect drift?", not "did a person change their mind?".
* **Same-reviewer, same-product** histories are the right construct for one person's change, but only 9 sequences of three or more reviews exist locally, so they stay a pilot-sized secondary experiment.
* **DravidianCodeMix** has no time, author, id or reply information. It remains an auxiliary comment-level sentiment dataset, and no temporal or discussion claim is made from it.
* **No aspect-level annotation exists anywhere in the project yet**, so nothing here can be *evaluated* at aspect level until the pilot (`docs/pilot_batch_sampling.md`) and the annotation that follows.

### Limitations of product timelines
* Neighbouring reviews are by **different reviewers** with different standards, so a change between reviews is population drift, not a person's change; reviewers who review early or late may differ systematically.
* The timestamp is when the review was **written**, not when the product was used; `parent_asin` merges product variants.
* The OET model was trained on reviewer timelines with ratings-derived trajectory labels. **It has not been trained or evaluated on product timelines**, so its trajectory output is not a valid measure of product drift, and the aspect-associated sentiment projection remains unvalidated against any human labels.
* The local file has no product titles or categories, so the products cannot be described, and one aspect vocabulary may fit some product types better than others.
* A product's full timeline can contain hundreds of reviews (the pilot's longest has 564); how to turn that into time windows is undecided.
* The sample drawn for the pilot is not representative (section 1.C and `docs/pilot_batch_sampling.md`).

## 1. Amazon All Beauty (product reviews)

Local raw file `data/raw/All_Beauty.jsonl`: 701,528 reviews, 631,986 reviewers, 112,565 products (by `parent_asin`), from 2000-11-01 to 2023-09-09.
Every review has a millisecond timestamp, a title, a text, a rating, a product id, a reviewer id and a verified-purchase flag.
The reviewer ids are the dataset's anonymised ids.

The OET sequences use a subset: `scripts/download_and_build_amazon_sequences.py` parses at most 250,000 records, which is consistent with the 5,324 reviewers in the
sequences against 9,159 reviewers with at least three reviews in the full local file. The product-timeline and same-reviewer counts below use the full local file.

### A. Reviewer timelines across products (existing OET sequences)

| Measure | Value |
|---|---|
| Reviewers / reviews | 5,324 / 23,736 |
| Sequence length 3 / 4 / 5 / 6 / 7 / 8 / 9 / 10 | 3,202 / 1,045 / 368 / 213 / 120 / 92 / 52 / 25 |
| Length at least 5 / 10 / 20 | 1,077 / 232 / 77 |
| Order equals timestamp order | 100% of reviewers |
| Consecutive review pairs | 18,412; 215 share an identical timestamp |
| Pairs posted within one day | 14.3%; median gap 65 days |
| Reviewers whose reviews are all different products | 98.2%; only 96 reviewers repeat a product |

Chronology is sound. The limit is the entity: opinion about an aspect (a scent, a bottle) in review 2 and review 3 concerns different products.
What this timeline can honestly show is a reviewer's pattern across products (for example whether they keep complaining about packaging), not a change of opinion about something.

### B. Same reviewer, same product

Filters, chosen to remove duplicate postings: at least 10 words per review; reviews with identical text for the same reviewer and product dropped; reviews with identical timestamp dropped.

| Measure | Value |
|---|---|
| Reviewer and product pairs with at least 2 reviews, before filters | 6,437 |
| After filters: at least 2 / 3 / 4 reviews | **195 / 9 / 1** (400 reviews in the 195 pairs) |
| Of the 195: spanning at least 1 day / at least 30 days | 124 / 90; median span 18 days |
| Of the 9 with at least 3 reviews: spanning at least 1 day / at least 30 days / posted within one day | 5 / 4 / 4 |
| Pairs whose star rating changes between reviews | 26.7% |
| Pairs where a keyword pre-screen category appears in at least 2 reviews | 105 (an upper-bound hint, section 4) |

Almost all of the 6,437 raw pairs are the same text posted more than once: dropping repeated texts alone leaves 280 pairs (the identical-timestamp rule removes nothing further), and the 10-word minimum takes it to 195. These are not repeated opinions, so they are not sequences.
The filtered data has real elapsed time and one entity, so it is the right construct, but it cannot support a statistical evaluation.
The 5 sequences that have at least 3 reviews and span a day or more (16 messages) are the pilot of this **secondary experiment**, in `data/annotations/aspect_annotation_template.csv`.
The other 4 were posted within one day and are not offered as evidence of change over time. The primary pilot is a separate file (section 1.C).

### C. Product timelines (different reviewers, one product)

Rule: reviews of at least 10 words, one review per reviewer per product, ordered by timestamp. Timestamps are distinct for 99.4% of reviews of a product on average.

| Products with at least | 10 | 20 | 50 | 100 |
|---|---|---|---|---|
| Number of products | 9,271 | 4,073 | 1,208 | 450 |
| Reviews they hold | 300,034 | 230,235 | 145,476 | 93,749 |
| Median span in years | 2.2 | 2.8 | 3.9 | 5.0 |

Inside the OET subset only 137 products have 10 or more reviews, so this view needs the full file rather than the existing sequences.
A product timeline answers "has opinion about this product's scent, packaging or price shifted?" It does not answer "did this person change their mind?"

**Pilot batch 01** takes 50 of these reviews: 10 products, 5 reviews each at spread-out ranks of the product's chronological order, from the 3,626 products with at least 20 reviews (10 to 300 words, one per reviewer) spanning at least a year.
Procedure, selected products and checks: `docs/pilot_batch_sampling.md`; file: `data/annotations/pilot_batch_01_product_timelines.csv`, all annotation columns blank.

## 2. DravidianCodeMix (social media)

| Language | Train / dev / test comments (after cleaning) |
|---|---|
| Tamil | 33,531 / 4,145 / 4,112 |
| Malayalam | 14,514 / 1,671 / 1,654 |
| Kannada | 5,218 / 615 / 632 |

What the raw files contain: for sentiment, `text;label;`; for offensive-language, `text<TAB>label<TAB>`. In the six training files checked, the extra field is empty in all but 2 lines. There is **no timestamp,
no author, no comment or video id and no parent-comment field**. The row order is not documented as chronological, so it cannot be treated as one.
In the raw sentiment training files 733 of 57,050 lines contain an `@name` mention, but a mention names a person, not a comment, and does not identify which comment is
answered, so no reply tree can be rebuilt from it.

Consequences:

* **Valid chronological discussion threads: 0.** The earlier sliding-window "sequences" over these files are not threads (they mix unrelated comments and overlap between splits) and
  are not used for anything here.
* Single comments can still be annotated for aspect and aspect sentiment (non-sequential), and the existing WSD component already tags 31.85% of Tamil, 28.16% of Malayalam and 18.34% of Kannada rows with at least one aspect
  (`docs/module2_wsd_status.md`). That supports annotating *comments*, not tracking opinion over time.
* Product timelines and discussion threads are different constructs and are not combined.

## 3. Language

The Amazon file has no language field. It is the English-language All Beauty dataset, so `language` is set to `en` for all rows; this is a property of the dataset, not a per-review detection.
DravidianCodeMix files are per language (`ta`, `ml`, `kn`).

## 4. Keyword pre-screen (indicative only)

To judge where annotation is worth the effort, the alias lists in `ontology/product_aspect_taxonomy.json` were matched against the review texts. This is **not** annotation, not an aspect detector,
and not a measurement of anything about accuracy. It overstates true aspect mentions because some aliases are generic words.

| Population | Pre-screen result |
|---|---|
| Reviewer timelines (23,736 reviews) | at least one specific alias in 77.6% of reviews; 3,267 reviewers have a category matched in at least 2 reviews; 1,451 in at least 3 |
| Same reviewer, same product | 105 of 195 pairs have a category matched in at least 2 reviews |
| Product timelines (products with at least 10 reviews) | 3,744 products have a category matched in at least 10 reviews; 1,383 in at least 20; 10,756 product and category pairs with at least 10 |

The true numbers of usable sequences are known only after annotation.

## 5. What each construct can support

| Question | Data to use | Ready? |
|---|---|---|
| Does aspect-associated sentiment from OET agree with human aspect sentiment? | any annotated messages (not sequences) | needs annotation only |
| How does opinion about one aspect of one product drift over time? | product timelines (C), **primary** | needs annotation; pilot batch drawn |
| Does one reviewer's opinion about one product change? | same reviewer, same product (B), **secondary** | data too small for evaluation; pilot only |
| Does a reviewer's pattern across products have a stable aspect profile? | reviewer timelines (A) | weak construct; entity differs per review |
| How does opinion move inside a social-media discussion? | discussion threads | **no data** |
| Aspect sentiment of Dravidian comments (not over time) | comment-level files | needs annotation; data exists |

## 6. Remaining data requirements

1. **Aspect-level gold annotations**, double-annotated for agreement (`docs/aspect_annotation_guidelines.md`). None exist. Both pilots are unannotated: batch 01 (10 product timelines, 50 reviews) and the same-reviewer template (5 sequences, 16 reviews).
2. **A larger set of genuine same-product, same-reviewer sequences** if per-person evolution is to be evaluated. The full local Amazon Beauty file yields 9 sequences of at least three reviews. Whether other categories or sources yield more has not been checked.
3. **Social-media discussion threads with structure** (comment id, parent id, publication time, anonymised author id, video id) if opinion evolution in discussions is to be studied. This must be newly collected under the platform's terms; the current data cannot provide it.
4. **Sequence construct: decided.** Primary = product timelines; secondary = same reviewer, same product; auxiliary = DravidianCodeMix comments. Still needed: a rule for cutting long product timelines into time windows.
5. **Remaining decisions** (guidelines section 12): time windows for long timelines, whether `NEUTRAL` enters the ontology, adjudication, and the scale-up size after the pilot.
6. **Full-file extraction.** The pilot draws from the full local file (701,528 reviews), not the 250,000-record subset behind the OET sequences. A complete product-timeline dataset beyond the pilot has not been built.

## Reproduce

```
python scripts/sequence_readiness.py          # statistics and the 16-review same-reviewer template
python scripts/make_pilot_batch.py            # pilot batch 01 (refuses to overwrite)
python scripts/make_pilot_batch.py --verify   # check the batch against the raw file
```
Writes `outputs/diagnostics/sequence_readiness/readiness_stats.json` and regenerates `data/annotations/aspect_annotation_template.csv` (unannotated rows only).
Note that `data/` is git-ignored, so the template is not tracked by git unless an exception is added.
