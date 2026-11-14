# Pilot batch 01: sampling procedure

A batch of **50 real Amazon All Beauty reviews from 10 product timelines**, with every annotation column left blank, for the first pilot of aspect-level annotation (`docs/aspect_annotation_guidelines.md`). File: `data/annotations/pilot_batch_01_product_timelines.csv`; manifest: `data/annotations/pilot_batch_01_manifest.json`; generator and independent check: `scripts/make_pilot_batch.py`.

Nothing in the batch is labelled. No rating, keyword, language model or model prediction was used to choose products or reviews, and none is stored in the file. The 16-record same-reviewer pilot (`data/annotations/aspect_annotation_template.csv`) is a separate file and was not changed (its SHA-256 is recorded in the manifest and checked by the verification).

## Sequence definition

A **product timeline** (`sequence_type = PRODUCT_TIMELINE`) is the set of reviews of one product (`parent_asin`) written by different reviewers, ordered by the real review timestamp from the dataset. The 5 reviews in each sequence here are a sample of that product's full timeline, taken at spread-out ranks (below), so `position` counts within the sample, not within the whole product history.

## Procedure

1. **Source:** the full local file `data/raw/All_Beauty.jsonl` (701,528 reviews), not the 250,000-record subset behind the OET sequences.
2. **Review filter:** keep reviews of 10 to 300 words (very short reviews rarely carry an aspect with a sentiment; very long ones are a heavy annotation burden). Sort by timestamp, then file line.
3. **Duplicates, per product:** keep each reviewer's earliest review only; drop later reviews with the same text; drop reviews sharing a timestamp. This gives timelines of distinct reviewers, distinct texts and strictly increasing timestamps.
4. **Eligible products:** at least 20 remaining reviews spanning at least 365 days, and not already in the 16-record same-reviewer pilot. **3,626 products** are eligible.
5. **Product draw:** sort eligible product ids alphabetically and draw 10 without replacement with `numpy.random.default_rng(20261114)`.
6. **Review selection inside a product:** with n eligible reviews in time order, take 5 at the evenly spaced ranks `1 + round(linspace(0, n-1, 5))` (NumPy rounding, halves to the even number): the first, the last and three between. This covers the product's whole lifetime, is deterministic and does not look at the content.
7. **Output:** the template's 22 columns (`sequence_id`, `message_id`, timestamps, `language`, `title`, `text`, ...) with `annotation_status = NOT_STARTED` and all annotation columns empty.

## Selected sequences

| Sequence | Product (`parent_asin`) | Eligible reviews | Product timeline span (days) | Selected ranks | First selected | Last selected |
|---|---|---|---|---|---|---|
| AMZ-PT-57bf9faec0 | B000FEF1V4 | 564 | 4,350 | 1, 142, 283, 423, 564 | 2006-10-11 | 2018-09-09 |
| AMZ-PT-aa36f30b18 | B00JZKAUV0 | 22 | 1,118 | 1, 6, 11, 17, 22 | 2015-06-18 | 2018-07-10 |
| AMZ-PT-b56493ff93 | B00O6ZZCEY | 28 | 2,243 | 1, 8, 15, 21, 28 | 2015-07-27 | 2021-09-16 |
| AMZ-PT-315b388c09 | B011MTR5QO | 21 | 1,567 | 1, 6, 11, 16, 21 | 2015-10-21 | 2020-02-03 |
| AMZ-PT-bef9ec9ce8 | B01IPXUIDC | 61 | 1,912 | 1, 16, 31, 46, 61 | 2017-06-06 | 2022-09-01 |
| AMZ-PT-9d83815fa3 | B0753JSKQZ | 142 | 2,057 | 1, 36, 71, 107, 142 | 2018-01-10 | 2023-08-28 |
| AMZ-PT-d2c103c2a9 | B07V2Y4Z5B | 31 | 659 | 1, 9, 16, 23, 31 | 2019-09-01 | 2021-06-21 |
| AMZ-PT-5a0c70b73d | B07W81JVBC | 22 | 960 | 1, 6, 11, 17, 22 | 2019-11-25 | 2022-07-13 |
| AMZ-PT-17951c2744 | B08YK8B89C | 20 | 3,860 | 1, 6, 11, 15, 20 | 2010-12-06 | 2021-07-01 |
| AMZ-PT-9603bd0891 | B0974GWS7Z | 26 | 511 | 1, 7, 13, 20, 26 | 2021-09-30 | 2023-02-23 |

Totals: 50 reviews, 10 sequences, 50 distinct message ids, 10 distinct products. Word counts range from 10 to 216.

## Verification (all passed)

`python scripts/make_pilot_batch.py --verify` re-reads the raw file and checks, for every selected review: it exists at the line number in its `message_id`; its product, timestamp (milliseconds and UTC text), title and text equal the raw record; word count is inside the rule. For every sequence: one product only, strictly increasing timestamps, positions 1 to 5, five different reviewers. For the file: 50 rows in 10 sequences, columns identical to the template, unique message ids, all annotation columns blank, no overlap in message or product with the 16-record pilot, and the 16-record template unchanged since generation. A separate re-check with plain JSON parsing (no pandas) found 50 of 50 matches and 0 mismatches.

## Limitations

* **Not a representative sample.** Eligible products are the 3,626 with at least 20 filtered reviews and a one-year span, so popular, long-lived products are over-represented; 10 products cannot represent the catalogue. The batch tests the annotation guidelines, not the research hypotheses.
* **A sample of a timeline, not the timeline.** Five reviews per product, spread over up to years, show too little to say anything about opinion drift. Whole timelines are annotated only if the pilot succeeds.
* **Different reviewers.** Neighbouring reviews are by different people with different standards, so a change between two reviews is not one person's change of opinion.
* **Review time is not use time.** The timestamp is when the review was written; a review can describe a purchase made much earlier, and `parent_asin` merges product variants (sizes, scents).
* **Product type unknown.** The local file has no product titles or categories, so the 10 products cannot be described (shampoo, nail polish, perfume) and the aspect categories in `docs/aspect_taxonomy.md` may fit some better than others.
* **Filters remove content.** Reviews under 10 or over 300 words, a reviewer's later reviews, and repeated texts are excluded, and `language` is `en` by dataset, not detected.
* **Verified purchase and helpful votes are not used**; both are in the raw file if a later sampling rule wants them.

Regenerating is refused if the batch file exists, so annotations cannot be overwritten by accident. To redraw, move the file away first.
