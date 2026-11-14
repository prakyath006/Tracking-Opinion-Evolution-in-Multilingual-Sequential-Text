# Validating aspect annotations

`scripts/validate_aspect_annotations.py` checks completed annotation CSV files against the schema in `docs/aspect_annotation_guidelines.md` (v0.2),
reports statistics, and measures agreement between two annotators.

It is **read-only and never infers anything**. It does not fill, correct, normalise or complete an annotation (a lowercase `positive` is reported, not changed), it
never writes next to an input file, and it refuses a `--json` path that equals an input. The files in `data/annotations/` are only read.

## Usage

```
python scripts/validate_aspect_annotations.py FILE.csv                     # validate, then statistics
python scripts/validate_aspect_annotations.py FILE.csv --strict            # unfinished messages also fail
python scripts/validate_aspect_annotations.py FILE.csv --reference BATCH.csv
python scripts/validate_aspect_annotations.py A.csv --compare B.csv        # validate both, then agreement
python scripts/validate_aspect_annotations.py A.csv --compare B.csv --json report.json
```

| Exit code | Meaning |
|---|---|
| 0 | no errors (and, with `--strict`, nothing unfinished) |
| 1 | at least one error, or unfinished or missing messages under `--strict`; an unreadable header |
| 2 | bad usage (for example `--json` pointing at an input file) |

**Reference batch.** The validator looks for the unannotated batch the file was made from (`pilot_batch_01_product_timelines.csv` or `aspect_annotation_template.csv` in
`data/annotations/`) by matching message ids, or you give it with `--reference`. With a reference it also checks that the prepared columns were not edited and that
every message came back. Without one it warns (`REFERENCE_NOT_FOUND`) and skips those two checks.

## What is checked

| # | Check | Issue codes (ERROR unless noted) |
|---|---|---|
| 1 | Labels belong to the schema: `aspect_status`, `sentiment`, `span_type`, `annotation_status`, `sequence_type`; no stray whitespace; header has exactly the 22 columns; `source` is a known dataset | `BAD_LABEL`, `WHITESPACE`, `BAD_HEADER`, `UNKNOWN_SOURCE`, `MISSING_STATUS`, `MISSING_ID` |
| 2 | Spans match the text: offsets are whole numbers, `0 <= start < end <= len(text)`, and `text[start:end] == aspect_span_text` (the message shows what the offsets actually select) | `SPAN_NOT_INTEGER`, `SPAN_RANGE`, `SPAN_MISMATCH`, `MISSING_SPAN` |
| 3 | `NO_ASPECT` and `UNCLEAR_ASPECT` rows have no sentiment, span or category | `SENTIMENT_ON_NON_ASPECT`, `FIELDS_ON_NON_ASPECT` |
| 4 | Every finished `ASPECT_PRESENT` row has a category valid for its domain and a sentiment. Domain follows `source`: Amazon uses `ontology/product_aspect_taxonomy.json`, DravidianCodeMix uses `ontology/aspect_taxonomy.json`; `GENERAL` and `OTHER_ASPECT` are valid in both | `BAD_CATEGORY`, `MISSING_CATEGORY`, `MISSING_SENTIMENT` |
| 5 | Duplicates and incomplete annotation: identical rows, same span with different labels, overlapping spans with one category (WARNING), a message with both a no/unclear-aspect row and aspect rows, more than one no/unclear-aspect row, rows of one message that disagree on prepared columns, unfinished messages (WARNING), reference messages missing from the file (WARNING) | `DUPLICATE_ROW`, `CONFLICTING_DUPLICATE`, `OVERLAPPING_SAME_CATEGORY`, `MIXED_NO_ASPECT_AND_ASPECT`, `MULTIPLE_NO_ASPECT`, `MESSAGE_ROWS_DISAGREE`, `UNFINISHED_MESSAGES`, `MESSAGE_NOT_IN_FILE`, `MIXED_STATUSES` (WARNING) |
| 6 | Annotation status is consistent with content (table below); the prepared columns equal the reference batch; timestamps follow `position` | `STATUS_CONTENT_MISMATCH`, `MISSING_ASPECT_STATUS`, `MISSING_ANNOTATOR`, `CONTENT_WITHOUT_ASPECT_STATUS`, `NEEDS_REVIEW_NO_NOTES` (WARNING), `PREPARED_COLUMN_CHANGED`, `UNKNOWN_MESSAGE`, `CHRONOLOGY`, `BAD_PREPARED_VALUE` |

### How `annotation_status` is handled

| Status | Rule | Counted as |
|---|---|---|
| `NOT_STARTED` | every annotation column must be empty (content here is `STATUS_CONTENT_MISMATCH`) | unfinished |
| `IN_PROGRESS` | partly filled rows are allowed; whatever is filled is still checked for valid labels, offsets and categories | unfinished |
| `ANNOTATED` | must be fully valid; `aspect_status` and `annotator_id` required; a message with nothing to mark must be `NO_ASPECT`, never blank | finished |
| `NEEDS_REVIEW` | an `ASPECT_PRESENT` row may have an empty sentiment (undecidable polarity) but still needs its span and category; `annotator_id` required; notes recommended | unfinished |
| `ADJUDICATED` | same rules as `ANNOTATED` | finished |

A message is **finished and valid** only if every one of its rows has status `ANNOTATED` or `ADJUDICATED` and none has an error. Statistics and agreement use only those messages.
An unannotated row is never treated as `NO_ASPECT`.

## Statistics

Printed per file and stored under `statistics` in the JSON report, over finished and valid messages only:

* rows by `annotation_status`, annotators, message-level aspect status (`ASPECT_PRESENT`, `NO_ASPECT`, `UNCLEAR_ASPECT`);
* aspect rows by category, by sentiment, and the category by sentiment table;
* per `sequence_type`, and per sequence (product timeline): messages, finished messages, messages with, without and with an unclear aspect, aspect rows, sentiment and category counts.

Sequence types are never merged into one total.

## Agreement between two annotators (`--compare`)

Both files are validated first. Only messages that are finished and valid in **both** are compared; the rest are counted and left out (`excluded_not_finished_or_invalid_in_a_or_b`,
`only_in_a`, `only_in_b`). Messages whose prepared columns differ between the files are excluded too. The report warns if the two files share an `annotator_id`.

Following the guidelines (section 10), results are given **per `sequence_type` and language group and are never pooled**:

| Measure | Definition |
|---|---|
| Aspect status per message | one label per message: `ASPECT_PRESENT` if any aspect row, otherwise `NO_ASPECT` or `UNCLEAR_ASPECT`; raw agreement and Cohen's kappa |
| Span matching | spans of one message are matched one-to-one, best overlap first, when overlap divided by union is at least 0.5; reports spans in A, in B, matched, only-A, only-B, and span F1 = 2 x matched / (A + B) |
| Category | on matched spans: raw agreement and kappa |
| Sentiment | on matched spans where both annotators gave a sentiment: raw agreement and kappa; spans lacking a sentiment in either file are counted separately |

The number of items (`n`) is printed next to every kappa. Kappa is `n/a` when it is undefined: no items, or both annotators used a single label throughout.
With a pilot of 50 or 16 messages the figures are very uncertain; they are for improving the guidelines, not for claiming annotation quality.

## What it does not do

* It does not judge whether an annotation is *right*, only whether it is well-formed and consistent. Correctness comes from agreement and adjudication.
* It does not compute agreement against a model, rating or WSD output, and it does not read ratings.
* It does not check the span of a reply against its parent, or that a sentiment fits the words; those are annotator judgement.
* It does not merge several rows of one message into an aspect value, or compute trends; that is later code (guidelines section 11).

## Tests

`tests/test_validate_aspect_annotations.py` (45 tests). Its CSV rows are fixtures built in memory or in a temporary folder to exercise the checks; they are not project data and
are not saved in the repository. One test reads the unannotated pilot batch read-only and confirms 0 errors, 0 finished messages and that the file is unchanged.

```
python -m pytest tests/test_validate_aspect_annotations.py -q
```
