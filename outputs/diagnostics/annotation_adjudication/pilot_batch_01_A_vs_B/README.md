# Adjudication package (to be filled in by the team)

Nothing here was decided by a script. No annotator was chosen, no label was inferred, no annotation was merged. The `adjudicated_*`, `decision_reason` and
`adjudicator_id` columns are **blank on purpose**; the originals are untouched.

## How to use `adjudication_items.csv` / `.xlsx`
* One row per item to settle. Rows with the same `conflict_group` are the two sides of one boundary conflict (an A span and a B span that partly overlap): settle the group once.
* `in_agreement_statistics = no` marks messages that were not finished in both files (they are outside the agreement figures, listed so the whole message can be settled).
* `guideline_topics` tags the kind of question (span boundary, referring expression, OTHER_ASPECT, GENERAL, implicit aspect, NEUTRAL vs polarity, UNSURE note...). The tags come from
  observable fields only; they do not say who is right.
* For an unmatched span the other annotator's columns show the nearest overlapping span (see `other_side_shows`) or are empty.
* Fill `adjudicated_span_text` with `adjudicated_span_start`/`_end` (0-based character offsets in `review_text`, end exclusive) when the span itself is decided, `adjudicated_category` (a product-taxonomy id
  or OTHER_ASPECT), `adjudicated_sentiment` (POSITIVE, NEGATIVE, NEUTRAL or MIXED) and a short `decision_reason`. Leave the span columns empty for an item you decide to drop and say so in the reason.
* The completed file is **not yet** an annotation file: a later, separate step turns the decisions into the 22-column format and validates it.

## Columns
`item_type`, `conflict_group`, `guideline_topics`: see above. `message_id`, `sequence_id`, `product_id`, `position`, `timestamp_utc`/`timestamp_ms`, `review_title`, `review_text`: copied from the original batch
(the title is context only; spans point into `review_text`). `match_kind`, `overlap_iou`: how the two spans relate. `A_*` / `B_*`: each annotator's span text, offsets, span type, category, sentiment, annotation status and notes.
