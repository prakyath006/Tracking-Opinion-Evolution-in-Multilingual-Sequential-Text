> **PROVISIONAL DRAFT. NOT APPROVED. NOT THE OFFICIAL GUIDELINES.**
> The official guidelines remain **v0.2** (`docs/aspect_annotation_guidelines.md`, unchanged and the current approved baseline).
> Part A below is v0.2 verbatim. Part B lists the amendments proposed for v0.3. No team approval of any Part B item is on record
> (the decision sheet's "Approve?" column is empty), so every row of Part B is either a *clarification supported by v0.2 text*
> (applied as working practice in the provisional annotation file) or *PENDING REVIEW* (not applied as a rule).
> The 27 offset decisions were made by four human team members (anonymized IDs `adjudicator_1` to `adjudicator_4`). The 162-item adjudication workbook was reviewed by the same four members collectively (no row-to-member assignment is recorded); its own ID field still reads `claude_adjudicator_1`, so the consolidated records use `team_adjudicators_1_to_4` and keep the original value in the provenance file.

# Part A: v0.2 text (unchanged except this title)

# Aspect annotation guidelines (v0.3 PROVISIONAL DRAFT)

Manual annotation of **aspect spans and the sentiment expressed toward each aspect**, message by message, inside chronological sequences.
This produces the first gold data for aspect-level opinion evolution. No such gold data exists in the project yet: the current
aspect-associated sentiment (`src/aspect_sentiment.py`) only copies a whole-message sentiment onto detected aspects and has never been
checked against human labels.

**Status: v0.2, the definitions below were fixed on 2026-11-14 for pilot batch 01.** They have not been tested on real annotators. Expect to revise them after the pilot (section 10) and keep the version number in every annotation file.

Companion documents: `docs/aspect_taxonomy.md` (categories and aliases), `docs/sequence_readiness.md` (which sequences exist),
`docs/pilot_batch_sampling.md` (how the pilot batch was drawn), and the files you fill in: `data/annotations/pilot_batch_01_product_timelines.csv` (primary, 50 reviews) and `data/annotations/aspect_annotation_template.csv` (secondary, 16 reviews).

## 1. What you annotate

For every message of a sequence, mark each place where the text talks about an aspect of the thing being discussed, name the
aspect category, and give the sentiment expressed toward that aspect **in that message**.

* Annotate only what the message's own words say. Do not infer from earlier or later messages.
* You see the whole sequence in time order so you can understand references ("same here", "the new bottle"), but each message is judged from its own text.
* You do **not** see, and must not look up, star ratings, the datasets' sentiment labels, or any model or WSD output. Seeing them would make the gold data
  agree with the thing it is meant to check.

## 2. Sequence types and their roles

| `sequence_type` | Role | Domain | What it is | Status of data |
|---|---|---|---|---|
| `PRODUCT_TIMELINE` | **primary** | product reviews | reviews of **one product** by potentially different reviewers, ordered by real timestamp | pilot batch 01: 50 reviews in 10 timelines |
| `SAME_REVIEWER_SAME_PRODUCT` | **secondary experiment** | product reviews | one reviewer's repeated reviews of one product | pilot: 16 reviews in 5 sequences; only 9 such sequences of 3+ reviews exist locally |
| `USER_TIMELINE` | legacy | product reviews | one reviewer's reviews of different products (the sequences OET was trained on) | not annotated; entity differs from review to review |
| `COMMENT_LEVEL` | **auxiliary** | social media | a single DravidianCodeMix comment, one message per sequence | no aspect annotation prepared; no order or thread exists |
| `DISCUSSION_THREAD` | not available | social media | comments in one thread with reply links and timestamps | needs newly collected data |

Rules:

* Keep the types in separate files and separate agreement and result tables. Never pool them.
* A **product timeline** measures how opinion about a *product* drifts over time across the people who reviewed it. It is not one person changing their mind, and a change between two
  neighbouring reviews can come from two reviewers holding different standards.
* The **same-reviewer** experiment is the only one where one person's opinion about one entity can change; it is a secondary experiment because the data is too small to evaluate on.
* DravidianCodeMix remains a comment-level sentiment dataset until genuine thread data (comment id, parent id, timestamp) is obtained. There is no sequence, so no temporal claim is made from it.

## 3. The file

One row per **aspect mention**. A message with several aspects has several rows that repeat the message columns. A message with no
aspect has one row. `data/annotations/aspect_annotation_template.csv` already contains the message columns for the pilot sequences; you fill
the right-hand columns. Save as UTF-8 (the template has a byte-order mark so Excel opens it correctly).

| Column | Filled by | Meaning |
|---|---|---|
| `sequence_id` | prepared | identifier of the sequence |
| `sequence_type` | prepared | one of the values in section 2 |
| `source` | prepared | dataset (`amazon_all_beauty_2023`, `dravidiancodemix_youtube`) |
| `entity_id` | prepared | product id for reviews; video id for threads; empty if the data has none |
| `message_id` | prepared | identifier of the message; unique within the file |
| `position` | prepared | 1-based place in the chronological order |
| `timestamp_ms`, `timestamp_utc` | prepared | real timestamp from the data; never edited |
| `language` | prepared | `en`, `ta`, `ml`, `kn`: the dataset's language, not a per-message detection |
| `title` | prepared | review title, shown as context; **spans never point into the title** |
| `text` | prepared | the message; spans are measured in this column only |
| `parent_message_id` | prepared | for discussion threads, the message replied to; empty for reviews |
| `aspect_status` | annotator | `ASPECT_PRESENT`, `NO_ASPECT` or `UNCLEAR_ASPECT` (section 4) |
| `aspect_span_text` | annotator | the words that name the aspect, copied exactly |
| `span_start`, `span_end` | annotator | 0-based character offsets in `text`, end exclusive, so `text[span_start:span_end] == aspect_span_text` |
| `span_type` | annotator | `EXPLICIT` (the aspect is named) or `IMPLICIT` (only the evaluation is there) |
| `aspect_category` | annotator | a category id from `docs/aspect_taxonomy.md` for this domain |
| `sentiment` | annotator | `POSITIVE`, `NEGATIVE`, `NEUTRAL` or `MIXED` (section 6) |
| `annotation_status` | annotator | workflow state (section 9) |
| `annotator_id` | annotator | your short id |
| `notes` | annotator | free text; required in the cases named below |

If a message has no aspect or an unclear one, leave `aspect_span_text`, `span_start`, `span_end`, `span_type`, `aspect_category` and `sentiment` empty.

## 4. Is there an aspect? (`aspect_status`)

| Value | Use when | Sentiment |
|---|---|---|
| `ASPECT_PRESENT` | the message names or clearly implies at least one facet of the entity, including `GENERAL` | filled (section 6) |
| `NO_ASPECT` | the message says nothing about the entity: a greeting, a question to another person, "first", "like if you agree", an agreement with no content ("same bro") | empty |
| `UNCLEAR_ASPECT` | there is probably an aspect but you cannot tell which: truncated text, a word that fits two categories and the sentence does not decide, text you cannot read | empty |

This is the rule that keeps three different situations apart: **no aspect** (`NO_ASPECT`), **an aspect you cannot identify** (`UNCLEAR_ASPECT`),
and **an aspect mentioned without an opinion** (`ASPECT_PRESENT` with `NEUTRAL`). Do not use `NEUTRAL` for the first two.

## 5. Spans

1. Mark the **shortest** phrase that names the aspect: `scent`, `the cap`, `price`, `background score`. Not the whole sentence.
2. Mark the aspect, not the opinion word: in "the scent is lovely" the span is `scent`, not `lovely`.
3. **Implicit aspects.** When there is no aspect word but the sentence evaluates a facet ("worth every penny", "broke in a week"), mark the
   shortest evaluative phrase, set `span_type` to `IMPLICIT`, and choose the category the phrase is about (`PRICE_VALUE`, `DURABILITY`).
4. **One row per mention.** If the same aspect is mentioned twice in a message, make two rows. Combining mentions into one aspect-per-message value is done later by code.
5. **Pronouns** ("it smells great"): mark the pronoun only if no named mention is nearby, and write the referred aspect in `notes`. Prefer the named mention.
6. The span must lie inside `text`. Copy it exactly (case, spelling, script, punctuation). Offsets count characters, so an emoji or a native-script
   character is one position.

## 6. Sentiment toward the aspect

Judge the sentiment **expressed toward this aspect**, not the message as a whole. A message can be negative overall and still be positive about one aspect.
Only the direction is annotated, not its strength: "good" and "amazing" are both `POSITIVE`.

| Label | Definition |
|---|---|
| `POSITIVE` | the message evaluates the aspect favourably |
| `NEGATIVE` | the message evaluates the aspect unfavourably |
| `NEUTRAL` | the aspect is present and the message gives **no evaluation** of it (a description, a fact, a question) or says explicitly that it is average ("okay", "fine", "average", "nothing special") |
| `MIXED` | **one aspect** receives both a favourable and an unfavourable evaluation in the same mention or short passage ("smells good at first, then turns sour") |

The four sentiment labels and the two missing-aspect statuses are different things:

| Situation | `aspect_status` | `sentiment` |
|---|---|---|
| aspect named or implied, favourable | `ASPECT_PRESENT` | `POSITIVE` |
| aspect named or implied, unfavourable | `ASPECT_PRESENT` | `NEGATIVE` |
| aspect named, described or asked about, not evaluated; or explicitly average | `ASPECT_PRESENT` | `NEUTRAL` |
| one aspect both praised and criticised | `ASPECT_PRESENT` | `MIXED` |
| the product or film as a whole is evaluated | `ASPECT_PRESENT`, category `GENERAL` | by polarity |
| no facet of the entity is mentioned | `NO_ASPECT` | empty |
| a facet is probably there but you cannot tell which | `UNCLEAR_ASPECT` | empty |
| the aspect is clear but the polarity cannot be decided (sarcasm, garbled text) | `ASPECT_PRESENT` | empty, and `annotation_status` = `NEEDS_REVIEW` |

Decision rules:

* **MIXED is about one aspect.** Praise for one aspect and criticism of another is two rows (`POSITIVE` and `NEGATIVE`), never `MIXED`.
* **MIXED needs two evaluations.** "Okay" alone is `NEUTRAL`; "decent" or "good enough" is `POSITIVE`; "good but not great" has a favourable and an unfavourable part, so `MIXED`.
* **NEUTRAL needs a present aspect.** No aspect is `NO_ASPECT`, an unreadable aspect is `UNCLEAR_ASPECT`. Never use `NEUTRAL` for either.
* **Absence statements** are judged by the stance toward the aspect: "no irritation" is `POSITIVE` for `SKIN_HAIR_REACTION`; "no scent" in a product described as scented is `NEGATIVE`; "unscented" as a plain description is `NEUTRAL`.
* **Expectations:** "smaller than expected", "not as described" are `NEGATIVE`; "exactly as described" is `POSITIVE`.
* **Negation and contrast** belong to the evaluation: "not bad" is mild praise (`POSITIVE`), "not good" is `NEGATIVE`. If you cannot decide, leave `sentiment` empty and use `NEEDS_REVIEW`.
* **Sarcasm and irony:** annotate the intended meaning if the sentence makes it obvious; otherwise `NEEDS_REVIEW`.
* **Questions and hypotheticals** ("is the scent strong?", "if the battery were better") state no sentiment: `NEUTRAL` if the aspect is named, otherwise `NO_ASPECT`.
* **Comparisons with another product or film:** annotate only the entity being reviewed and write `COMPARISON` in `notes`.
* **Hearsay:** annotate the author's own stance; if the author only reports someone else's opinion, `NEUTRAL`.
* **Emoji and punctuation** are evidence only when attached to the aspect ("the scent 😍").
* **Abusive language** is annotated like any other text; abuse aimed at a person rather than an aspect is `NO_ASPECT`.

## 7. Domain notes

**Product reviews (English).** Categories in `docs/aspect_taxonomy.md` section 2. The title is context only.
* *Product timelines:* each review is annotated on its own text. The neighbouring reviews are by other people; never carry an aspect, an opinion or a "still" over from them.
* *Same-reviewer sequences:* do not copy an earlier review's aspect into a later one. A later review that says only "still love it" is `GENERAL`, `POSITIVE`.
* Text that only describes the purchase ("bought for my wife") is `NO_ASPECT` unless an aspect is named.
* One product id can cover several variants (scents, sizes). If a reviewer says which variant, write it in `notes`.

**Social-media comments (Tamil, Malayalam, Kannada, code-mixed with English).** Categories in section 3 of the taxonomy. No aspect annotation has been prepared for these yet.
* Write the span exactly as typed, in whatever script or romanisation the comment uses; spelling varies (`bgm`/`BGM`, `paattu`/`paatu`/`pattu`) and all variants are valid.
* Mixed-script comments are annotated normally. If you cannot read a comment, ask a reader of that language; do not guess. If nobody can, `UNCLEAR_ASPECT`.
* Intensifiers such as `mass`, `semma`, `marana` take the aspect they modify ("mass trailer": `trailer_teaser`); a bare "mass" with no head word is `UNCLEAR_ASPECT`
  unless the sentence is about the actor's fans or appeal.
* A reply that only agrees with its parent ("same bro", "100% true") is `NO_ASPECT`; write `REPLY_TO_PARENT` in `notes`. Do not copy the parent's aspect.
* Comments about the audience or the platform ("first comment", "subscribe", "who is here after 2023") are `NO_ASPECT`.

## 8. Time

* Order is fixed by the data: timestamp, then position. Never reorder.
* In pilot batch 01 every timestamp inside a sequence is different. In other data several messages can share a timestamp; their order is the prepared `position`, and no meaning is to be read into it.
* Use no later message to interpret an earlier one, so the opinion trend is not built into the labels.
* In product timelines the five reviews of a sequence are spread over the product's life (months to years); the time between them is real and is not a sign of any one reviewer's change.

## 9. Workflow and status values

`annotation_status`: `NOT_STARTED` (as prepared) → `IN_PROGRESS` → `ANNOTATED` → `NEEDS_REVIEW` (you are unsure; say why in `notes`) → `ADJUDICATED` (a second pass settled it).
A row left `NOT_STARTED` has not been seen by anyone and must never be counted as `NO_ASPECT`.

## 10. Quality control

* Two annotators label the same pilot sequences independently, then compare. A third person adjudicates disagreements, and the rule changes that follow go into the guidelines.
* Agreement is reported for: message-level `aspect_status`; span matches (two spans match if their character overlap divided by their union is at least 0.5);
  category and sentiment on matched spans, using Cohen's kappa, with the counts next to every kappa.
* Report agreement per `sequence_type` and per language; do not pool them.
* `scripts/validate_aspect_annotations.py` checks every returned file and computes these agreement figures (`docs/aspect_annotation_validation.md`). Run it before counting anything.
* A number of annotated messages is not a result until the agreement is known. Gold labels from a single annotator are labelled as such.

## 11. How the labels connect to the rest of the project

* **Ontology.** `POSITIVE`, `NEGATIVE` and `MIXED` have the same names as the ontology's `SentimentState` values (`src/ontology.py`). **Decision: `NEUTRAL` stays an annotation-only label.** The ontology, `SentimentState`
  and the model heads are not changed. The ontology's `UNKNOWN` ("could not be determined") is not an annotation label: an undecidable polarity is an empty `sentiment` with `NEEDS_REVIEW`.
* **Comparison with OET.** OET's message-level sentiment is compared with aspect labels only on rows labelled `POSITIVE`, `NEGATIVE` or `MIXED`. `NEUTRAL` rows and rows with an empty `sentiment` are reported separately and left out of that comparison.
* **One value per (sequence, category, message).** Several rows for the same category in one message are combined by code: if all are `NEUTRAL` the value is `NEUTRAL`; otherwise `NEUTRAL` rows are ignored,
  and the value is `MIXED` if any row is `MIXED` or both `POSITIVE` and `NEGATIVE` occur, else the single polarity.
* **Per-aspect trend.** The series for a category is its (sequence, message) values in time order. Trends use the ontology's order NEGATIVE < MIXED < POSITIVE, skip `NEUTRAL` values (as the ontology skips `UNKNOWN`), and are computed only
  with at least two ranked values; otherwise the result is "insufficient observations for trend".
* **Interpretation by sequence type.** A trend over a product timeline is drift in what reviewers say about the product. A trend over a same-reviewer sequence is one person's change. They are reported in separate tables.

## 12. Decisions

Fixed for pilot batch 01 (2026-11-14), to be revisited after the pilot:

1. **Primary research unit:** product timelines. **Secondary experiment:** same-reviewer, same-product histories. **Auxiliary:** DravidianCodeMix comment-level sentiment.
2. **`NEUTRAL`:** annotation-only label, not added to the ontology (section 11).
3. **Product-timeline rule:** one review per reviewer per product, 10 to 300 words, distinct text and timestamp, product with at least 20 such reviews spanning at least 365 days.
4. **Same-reviewer rule:** at least 10 words, distinct text and timestamp, at least 3 reviews spanning at least one day.
5. **Pilot size and protocol:** 50 reviews in 10 product timelines plus the 16 same-reviewer reviews; two annotators label every pilot row independently (section 10).
6. **Aggregation and trend rules:** section 11.

Still open:

* How to cut a long product timeline (hundreds of reviews) into time windows for aspect trends.
* Whether `NEUTRAL` should enter the ontology if aspect-level models are later trained.
* Who adjudicates, and any new categories or rules the pilot disagreements show are needed.
* The scale-up size, which depends on the pilot's agreement.

## 13. Illustrative sentences

These sentences were **written for this guideline**. They are not taken from any dataset, and they have no annotation in any data file.

| Sentence | Rows |
|---|---|
| "The scent is lovely but the cap leaks." | `scent`: `FRAGRANCE`, `POSITIVE`; `cap`: `PACKAGING`, `NEGATIVE` |
| "Smells great at first, then it turns sour." | `Smells`: `FRAGRANCE`, `MIXED` (one aspect, both evaluations) |
| "It comes in a 8 oz bottle." | `8 oz bottle`: `SIZE_QUANTITY`, `NEUTRAL` (named, not evaluated) |
| "Worth every penny." | `Worth every penny`: `PRICE_VALUE`, `POSITIVE`, `IMPLICIT` |
| "Bought it for my wife." | `NO_ASPECT` (no facet of the product is discussed) |
| "I love this product." | `this product`: `GENERAL`, `POSITIVE` |
| "Trailer super, bgm mass" | `Trailer`: `trailer_teaser`, `POSITIVE`; `bgm`: `music_bgm`, `POSITIVE` |
| "same bro" (a reply) | `NO_ASPECT`, notes `REPLY_TO_PARENT` |

Counting a row as an example here does not make it data; the first real annotations will come from the pilot sequences in the template.


---

# Part B: proposed v0.3 amendments

Status key. **SUPPORTED** = restates what v0.2 already says; used in the provisional file as working practice. **PENDING** = a new or changed rule; not approved, not applied as a rule.
**OPEN** = no defensible decision exists; stays unresolved. Decision numbers refer to `docs/annotation_guidelines_v0.3_decision_sheet.md`.

| # | Amendment | Status | Basis | Decision |
|---|---|---|---|---|
| B1 | An opinion word or predicate of an aspect mention that is already annotated is not a second row. | SUPPORTED | v0.2 5.2 (aspect, not opinion word), 5.4 (one row per mention) | D3 |
| B2 | An explicit span excludes negation, degree words and comparison standards; the negation is carried by the sentiment label. | SUPPORTED | v0.2 6 ("negation and contrast belong to the evaluation") | D1 |
| B3 | An implicit span is the shortest complete evaluative phrase and keeps its negation or predicate. | SUPPORTED | v0.2 5.3 | D1 |
| B4 | `span_type` test: `EXPLICIT` if the span contains the facet or product noun, `IMPLICIT` if it only evaluates a facet. | SUPPORTED (sharpens 5.3) | v0.2 5.3 | D10 |
| B5 | A hypothetical or a question that names the aspect is `NEUTRAL`; reported (hearsay) opinion is `NEUTRAL`. | SUPPORTED | v0.2 6 | D4 |
| B6 | A described, unevaluated aspect is `NEUTRAL`. | SUPPORTED | v0.2 6 | D6 |
| B7 | A pronoun span is valid when no named mention is nearby (referent in `notes`); `this product` is a valid whole-product span. | SUPPORTED | v0.2 5.5, 13 | D2 |
| B8 | `OTHER_ASPECT` rows must name the facet in `notes`. | SUPPORTED | taxonomy principle 3 | D5 |
| B9 | Garbled or undecidable polarity: empty sentiment, `NEEDS_REVIEW`. | SUPPORTED | v0.2 6 | D8 |
| B10 | Adjudicated rows carry provenance: `annotator_id` records the adjudicator's anonymized ID (`adjudicator_N` for individual decisions, `team_adjudicators_1_to_4` for collective ones); flagged, low-confidence, restored and unresolved rows stay `NEEDS_REVIEW`. | SUPPORTED (workflow) | v0.2 9, 10 | D7 |
| B11 | Gold is built from agreed spans plus distinct adjudicated spans only; rejected spans stay in an audit trail. | SUPPORTED | v0.2 5.4, 10 | D11 |
| B12 | The determiner is optional in an explicit span; prefer the form without it (`cap`, not `the cap`). | **PENDING** | contradicts the v0.2 5.1 example `the cap` | D1 |
| B13 | A warning from the reviewer's own experience ("may come open") is evaluative, not a hypothetical. | **PENDING** | v0.2 6 treats hypotheticals as `NEUTRAL` | D4 |
| B14 | Coordinated aspects ("soft and light") become one row per aspect. | **PENDING** | v0.2 is silent | D3 |
| B15 | An evaluation of the reviewer's own workaround is not a product evaluation (proposal Q3). | **PENDING** | v0.2 is silent | D5 |
| B16 | Fragrance-mediated effects ("boosts my mood") belong to `FRAGRANCE` (proposal Q4). | **PENDING** | v0.2 is silent | D5 |
| B17 | Availability ("out of stock", "carrying") is `DELIVERY_SERVICE` (proposal P5). | **PENDING** | taxonomy definition does not list it | D5 |
| B18 | New categories (`BUILD_QUALITY`, `PHYSICAL_PROPERTIES`, `FEATURES_ACCESSORIES`). | **PENDING; no taxonomy change made** | see `docs/aspect_taxonomy_proposed_additions_DRAFT.md` | D5 |
| B19 | Adjudication identity: an anonymized adjudicator ID (individual or collective) is required on every `ADJUDICATED` row. | SUPPORTED (done in the consolidation) | v0.2 9, 10 | D7 |
| B20 | ADJ-109 ("It doesn't excellent job"): no defensible polarity. | **OPEN** | v0.2 6 | D8 |
| B21 | The seven low-confidence groups stay `NEEDS_REVIEW`; every human-adjudicated row that depends on an unapproved decision is `ADJUDICATED` but `training_eligible=pending_policy` until that decision is approved. | **PENDING** | decision sheet D1 to D6, D9, D10 | D1 to D6, D9, D10 |
