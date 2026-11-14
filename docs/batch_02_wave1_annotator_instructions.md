# Batch 02, Wave 1: annotator instructions

200 real Amazon reviews, 20 product timelines (10 reviews each). Two people annotate **independently**; do not look at each other's file or discuss labels until both are validated.

## Files (one per annotator, identical and blank)

| Who | Annotate this file (copy it first) | Save your finished file as |
|---|---|---|
| Annotator 1 | `data/annotations/batch_02_wave1_annotator_1.csv` | `data/annotations/batch_02_wave1_annotator_1_DONE.csv` |
| Annotator 2 | `data/annotations/batch_02_wave1_annotator_2.csv` | `data/annotations/batch_02_wave1_annotator_2_DONE.csv` |

`data/annotations/batch_02_wave1_reference.csv` is the untouched copy the validator compares with. Never edit it or the blank originals. Open the CSV as UTF-8 (Excel: import as "UTF-8", keep all columns as text; do not let it reformat numbers).

## Rules: guidelines v0.2 only

Use `docs/aspect_annotation_guidelines.md` (v0.2) exactly. **Do not use** `aspect_annotation_guidelines_v0.3_DRAFT.md` (not approved) or the decision sheet's recommendations. Taxonomy: `docs/aspect_taxonomy.md`, only its 14 categories plus `GENERAL` and `OTHER_ASPECT` (an `OTHER_ASPECT` row needs the facet written in `notes`).

Fill, per aspect mention, one row (columns 13 to 22; columns 1 to 12 are fixed, never edit them):

| Column | Value |
|---|---|
| `aspect_status` | `ASPECT_PRESENT`, `NO_ASPECT` (review says nothing about the product) or `UNCLEAR_ASPECT` |
| `aspect_span_text`, `span_start`, `span_end` | the shortest phrase naming the aspect (section 5); offsets are 0-based, end exclusive, counted in `text` (the raw review including `<br />`); the span must equal `text[start:end]` exactly |
| `span_type` | `EXPLICIT` (aspect word present) or `IMPLICIT` (only evaluates a facet, section 5.3) |
| `aspect_category` | one category id from the taxonomy |
| `sentiment` | `POSITIVE`, `NEGATIVE`, `NEUTRAL` or `MIXED` toward that aspect (section 6); empty for `NO_ASPECT` and `UNCLEAR_ASPECT` |
| `annotation_status` | `ANNOTATED` when finished; `NEEDS_REVIEW` for a genuine doubt (write why in `notes`; sentiment may stay empty) |
| `annotator_id` | **your own anonymized ID, the same on every row; it must differ from the other annotator's** |
| `notes` | optional; required for `OTHER_ASPECT` (facet), pronoun referents, `COMPARISON` |

A review with several mentions gets several rows (copy columns 1 to 12 onto each). A review with no aspect gets one `NO_ASPECT` row. Every review must end up `ANNOTATED` or `NEEDS_REVIEW`; none may stay `NOT_STARTED` or `IN_PROGRESS`.

Do not "fix" the wording of a rule that seems wrong: annotate by v0.2 and put the doubt in `notes`. Disagreements are settled later by adjudication.

## Validate before you hand in (exact commands, run from the project root)

Use the normal validator with `--reference`. **Do not use `--strict`**: strict mode fails on every `NEEDS_REVIEW` message, and genuine doubts must be allowed (it is reserved for the final adjudicated dataset).

Annotator 1:
```
python scripts/validate_aspect_annotations.py data/annotations/batch_02_wave1_annotator_1_DONE.csv --reference data/annotations/batch_02_wave1_reference.csv
```
Annotator 2: the same command with `annotator_2_DONE`.

Hand in only when:
1. the output says `errors: 0` and the last line says `RESULT: no errors` (fix every ERROR line it prints); and
2. you report two numbers from the output with your file: the **NEEDS_REVIEW count** (the `annotation_status rows: {...}` line) and the **unfinished messages** (`unfinished: N` on the `errors:` line; it should be 0 apart from your NEEDS_REVIEW messages. `NOT_STARTED` or `IN_PROGRESS` rows left over mean you are not done).

A `UNFINISHED_MESSAGES` warning that counts only your NEEDS_REVIEW messages is fine.

**Unsure cases:** when the aspect is clear but you cannot decide the sentiment (sarcasm, garbled text), keep the span and category, leave `sentiment` empty, set `annotation_status` to `NEEDS_REVIEW` and explain in `notes` (v0.2 section 6). Do not force a category or sentiment just to make the validator happy.

## After both files are in (coordinator only)

```
python scripts/validate_aspect_annotations.py data/annotations/batch_02_wave1_annotator_1_DONE.csv --compare data/annotations/batch_02_wave1_annotator_2_DONE.csv --reference data/annotations/batch_02_wave1_reference.csv
python scripts/wave_report.py --a data/annotations/batch_02_wave1_annotator_1_DONE.csv --b data/annotations/batch_02_wave1_annotator_2_DONE.csv --out_dir outputs/annotation_waves/batch_02_wave1
```
`wave_report.py` validates both files (normal mode; strict validation is kept for the final adjudicated dataset), reports span F1, aspect_status, category, sentiment and span_type agreement and all unresolved cases, writes the adjudication package (`outputs/annotation_waves/batch_02_wave1/adjudication_package/`), measures the real yield (products, examples, usable series) and recommends whether to run Wave 2. Open `report.md`.

## Pending team approval

Five of the 20 products (50 reviews) come from the **SMALL stratum**, sampled with a relaxed rule (at least 12 reviews over 180 days instead of 20 over 365). The team has not approved it. Annotate them like the rest; the report lists the yield with and without them, and they are dropped from the counts if the deviation is rejected. Their ids are in `data/annotations/batch_02_wave1_manifest.json` under `PENDING_TEAM_APPROVAL`.
