# Annotation guidelines v0.3: decision sheet for team approval

**Status: DRAFT. Nothing in this sheet has been applied.** The official guidelines (`aspect_annotation_guidelines.md`, v0.2), the taxonomy, the annotator files and `adjudication_items_FINAL.xlsx` are unchanged, and no gold CSV exists. Each recommendation becomes policy only after a human ticks it in the last column of the summary table at the end.

**Source of the conflicts.** The workbook's adjudicator did not have the guidelines or taxonomy (rules sheet A1) and worked from rules it inferred (S1 to S8, N1, N2, N4, N5, C1, C2). Its adjudicator ID is the placeholder `claude_adjudicator_1` on all 162 rows. Item IDs below are `ADJ-` numbers from the workbook; for agreed pairs the review is named by the last six characters of its `message_id`.

**Workbook figures used here.** 162 items from 41 of the 50 reviews. 110 retained rows, which are 84 distinct spans. 51 rejected rows. ADJ-109 unresolved.

---

## D1. Span boundaries and negation

- **Official (§5.1, §5.2, §6).** Mark the *shortest* phrase naming the aspect (example `the cap`), not the opinion word. Negation and contrast belong to the *evaluation*, not the span. §5 is silent on negation, degree words and modals inside an explicit span.
- **Conflicting decisions.**
  - S1 widens spans to "facet words plus complements". 24 of 82 kept unmatched spans match neither annotator's text.
  - S2 removes determiners, negation, degree words, modals and comparison standards, so `the cap` becomes `cap`.
  - Two retained implicit spans keep negation: ADJ-040 "not buy again", ADJ-086 "for years with no problems".
  - ADJ-135 dropped both negation and the comparison standard from "not as thick".
- **Affected.**
  - S1: 53 rows including ADJ-028 and ADJ-034.
  - S2: 35 rows.
  - N1: ADJ-010, 040, 130, 146.
  - N5: ADJ-117, 135, 140.
  - Negation mentioned in reasons: ADJ-040, 041, 066, 073, 096, 098, 100, 106, 107, 108, 121, 128, 130, 135, 140, 143, 146, 158.
- **Recommended policy.**
  1. Explicit span means the shortest aspect-naming phrase. A determiner is optional; prefer the form without it. Amend the §5.1 example from `the cap` to `cap`.
  2. Never put degree words, modals or comparison standards in a span.
  3. Negation stays out of an explicit span and in the sentiment label.
  4. An implicit span is the shortest *complete evaluative phrase*, so it keeps its negation or predicate ("not buy again", "didn't last a month").
  5. A comparison goes in `notes` as `COMPARISON`.
  6. Drop S1's "complements": no widening beyond the shortest phrase.
  7. Re-check the 24 spans that match neither annotator's text.

## D2. Pronoun and whole-product opinions

- **Official (§5.5, §13).** A pronoun is marked only if no named mention is nearby, and the referent goes in `notes`. §13 gives "I love this product." the span `this product`, category `GENERAL`.
- **Conflicting decisions.** S4 says a bare pronoun is not a valid span. Four rows were rejected on it and eight retained. The retained rows rely on S4+N4 or typo readings.
- **Affected.**
  - Rejected under S4: ADJ-047, 058, 083, 095.
  - Retained under S4: ADJ-009, 038, 059, 077, 089, 112, 157, 161.
  - Agreed pairs where one annotator used `this product` or `this X` and the other `product` or `X`: 029600, 432787, 099263, 699573, 102847 (see D9).
- **Recommended policy.**
  1. Keep §5.5 as written. A pronoun span is valid when no nearby named mention exists, with the referent in `notes`.
  2. If a nearby phrase names the product, annotate that phrase.
  3. Whole-product opinions are `GENERAL` with the span `product` or `this product`, one row per mention.
  4. Revisit ADJ-047, 058, 083 and 095, which were rejected only because of S4.
  5. ADJ-157 ("live this" as "love this") stays under D6, because it relies on a typo reading.

## D3. Multiple mentions and implicit aspects

- **Official (§5.3, §5.4, taxonomy GENERAL aliases).** One row per mention. An implicit aspect gets the shortest evaluative phrase, `span_type=IMPLICIT`. "Buy again" and "recommend" are `GENERAL` aliases.
- **Conflicting decisions.**
  - S5 rejects repeated or elaborating mentions ("evaluate once"). That drops mentions the guideline says to keep.
  - S3 splits coordinated phrases, which the guidelines do not cover.
  - S8 treats repurchase and recommendation as implicit `GENERAL`, which agrees with the taxonomy.
- **Affected.**
  - S5 rejected 27: ADJ-004, 012, 014, 017, 020, 022, 028, 033, 034, 043, 049, 050, 054, 065, 071, 076, 091, 093, 094, 097, 107, 123, 125, 127, 128, 139, 162.
  - S3: ADJ-015, 023, 024, 044, 079, 081, 084, 115, 116, 120, 121, 129, 159.
  - S8: ADJ-026, 032, 040, 154, 160.
- **Recommended policy.**
  1. §5.4 stands: a distinct mention gets its own row.
  2. Treat a repeat as a duplicate only when it has the *same offsets*. A different mention of the same aspect is kept.
  3. A wider span that merely elaborates an already marked mention (for example "Great product" over `product`) is a boundary variant, handled under D1.
  4. Re-open the 27 S5 rejections and keep any that are separate mentions.
  5. Add to the guidelines that coordinated aspects ("soft and light") become one row per aspect.
  6. Adopt S8 as written.

## D4. Hypothetical, hearsay and comparison sentiment

- **Official (§6).**
  - Questions and hypotheticals are `NEUTRAL` if the aspect is named, otherwise `NO_ASPECT`.
  - Hearsay is `NEUTRAL` unless it is the author's own stance.
  - A comparison annotates only the reviewed entity, with `COMPARISON` in `notes`.
- **Conflicting decisions.**
  - S7 rejects conditionals tied to the reader's preference, and keeps warnings that restate the reviewer's own problem as `NEGATIVE`.
  - N4 counts recipients' or family members' reported reactions as the author's evaluation.
  - Comparison notes are never recorded in the workbook.
- **Affected.**
  - S7: ADJ-003 and ADJ-124 (rejected, so they would not be `NEUTRAL` spans); ADJ-066 and ADJ-106 (kept as `NEGATIVE`).
  - N4: ADJ-059.
  - COMPARISON cases: ADJ-007, 079, 092, 093, 135.
  - Agreed pairs with COMPARISON notes: 4.
- **Recommended policy.**
  1. Follow §6 as written. A hypothetical naming an aspect is `NEUTRAL` and kept (ADJ-003, 124).
  2. A *warning about the reviewer's own experience* ("may come open", from a bottle that did) is evaluative and `NEGATIVE`. Add this exception to §6 and keep ADJ-066 and 106 with a note.
  3. Reported reactions: `NEUTRAL` unless the author endorses them ("she loves it, so do I" is `POSITIVE`).
  4. Change ADJ-059 to `NEUTRAL`, or reject it if the author does not endorse.
  5. Carry every `COMPARISON` note into gold.

## D5. OTHER_ASPECT and proposed new categories

- **Official (taxonomy).** `OTHER_ASPECT` needs the facet named in `notes` (principle 3). A new category needs a version bump and a changelog (§1.6). It also changes the category list and the BIO tag set.
- **Conflicting decisions.** 18 retained rows (17 distinct spans) use `OTHER_ASPECT`, mostly because proposals P2, P3 and P4 are unapproved. The workbook has no notes column, so no facet is recorded.
- **Affected.**
  - P2 BUILD_QUALITY: ADJ-069, 092, 114.
  - P3 PHYSICAL_PROPERTIES: ADJ-061, 062, 070, 073, 108.
  - P4 FEATURES_ACCESSORIES: ADJ-016, 056, 057, 066, 067, 072, 074, 126.
  - P5 DELIVERY_SERVICE scope or rename (to SELLER_SERVICE_AVAILABILITY): ADJ-029, 136.
  - P6 speed or time-saving rule: no row cites it.
  - Q3 (reject a reviewer's workaround): ADJ-075.
  - Q4 (fragrance-mediated effects go to `FRAGRANCE`): logic behind rejecting ADJ-049, not cited by tag.
  - `OTHER_ASPECT` with no proposal cited: ADJ-143, 144.
  - Agreed `OTHER_ASPECT` pairs: 5 (noise, loud ×2, manufacturing, battery operated).
- **Recommended policy.**
  1. Do *not* add categories for v0.3 on a 50-review pilot. Keep the 14 categories and `OTHER_ASPECT` with a mandatory facet note.
  2. Record P2 to P4 as candidate categories for v0.4 once at least 20 `OTHER_ASPECT` examples are collected.
  3. P5: keep the id `DELIVERY_SERVICE`, because a rename breaks existing files. Extend its definition to cover availability and the "carrying" case (ADJ-136).
  4. P6: not needed.
  5. Q3: adopt it as a guideline sentence, because an evaluation of a workaround is not an evaluation of the product.
  6. Q4: adopt it, and review ADJ-049.
  7. Write the facet into `notes` for all 22 `OTHER_ASPECT` spans (17 adjudicated, 5 agreed).

## D6. NEUTRAL and NEEDS_REVIEW handling

- **Official (§6, §9).**
  - `NEUTRAL` is for a present aspect with no evaluation, or an explicit "average".
  - Garbled or undecidable polarity gives an empty `sentiment` with `annotation_status=NEEDS_REVIEW`.
  - `NEUTRAL` is annotation-only and is not in the ontology.
- **Conflicting decisions.**
  - S6 rejects background, purpose and reviewer-need mentions, but §6 makes described aspects valid `NEUTRAL` spans.
  - 7 kept spans are `NEUTRAL`.
  - ADJ-117 (N5) is `NEEDS_REVIEW`. ADJ-109 is unresolved.
  - The seven low-confidence groups below are not resolved.
- **Affected.**
  - S6 rejected 20: ADJ-002, 041, 075, 082, 095, 099, 101, 102, 103, 104, 111, 118, 123, 137, 138, 145, 148, 149, 150, 155.
  - Four of these were `NEUTRAL` descriptions that the guideline would keep: ADJ-118 "case", ADJ-138 "1.7 oz size", ADJ-149 "irritated", ADJ-150 "infected".
  - Kept `NEUTRAL`: ADJ-035, 036, 057, 067, 072, 074, 158.
  - Low-confidence groups: ADJ-035/036 ("lasts"), 059, 066, 115, 135, 157, 037.
- **Recommended policy.**
  1. Keep §6 for `NEUTRAL`. Re-open the 20 S6 rejections and keep any that describe a named aspect (at least ADJ-118, 138, 149, 150).
  2. Drop S6 for pure background with no facet, which becomes `NO_ASPECT`.
  3. Low-confidence rows go in gold only with `annotation_status=NEEDS_REVIEW` until a human confirms them. The proposed decisions:
     - ADJ-035/036 `NEUTRAL`: confirm.
     - ADJ-059: change to `NEUTRAL` (D4).
     - ADJ-066 `NEGATIVE`: confirm under the D4 exception.
     - ADJ-115 `PACKAGING`: check the source.
     - ADJ-135: restore the negation and comparison, `NEUTRAL` or `NEGATIVE` with a `COMPARISON` note.
     - ADJ-157: keep as "love this" only if the team accepts typo repair; otherwise `NEEDS_REVIEW`.
     - ADJ-037: confirm `POSITIVE`, or `NEUTRAL` if the polarity is not stated.
  4. `NEEDS_REVIEW` rows never enter agreement statistics or training.

## D7. Span types, notes and adjudicator identity

- **Official (§3, §9).** The file has `span_type`, `notes` and `annotation_status` columns. Every row names its annotator.
- **Conflicting decisions.**
  - The workbook has no adjudicated `span_type`, `notes`, `annotation_status` or annotator convention.
  - All 162 rows carry `claude_adjudicator_1`, which A2 says to replace.
  - 59 of 84 distinct adjudicated spans equal an A or B span, so a type could be inherited. 25 relocated or new spans have none.
- **Affected.** All 162 rows (identity). 25 spans without a type. 22 `OTHER_ASPECT` spans needing facet notes. 42 agreed pairs carrying notes.
- **Recommended policy.**
  1. A named human adjudicator replaces the placeholder in all 162 rows and takes ownership of the decisions. Until then every adjudicated row stays `annotation_status=NEEDS_REVIEW`.
  2. Adjudicated rows get `annotator_id` = the human's ID, and `annotation_status=ADJUDICATED` once confirmed.
  3. Agreed rows keep `ANNOTATED`, with `annotator_id` noting the two original annotators.
  4. Span type: inherit from the matching annotator span; for the 25 new or relocated spans the adjudicator sets it by the D1 rule.
  5. Notes are carried over from the original rows (agreed: 42; adjudicated: from the source row), with the facet written for every `OTHER_ASPECT`.

## D8. The unresolved ADJ-109

- **Official (§6).** If the aspect is clear but the polarity cannot be decided (garbled text), `sentiment` is empty and the row is `NEEDS_REVIEW`.
- **Conflicting decision.** The workbook leaves ADJ-109 empty and marks it UNRESOLVED. Its reason offers `POSITIVE` if the text is a typo, or exclusion.
- **Affected.** ADJ-109: review `L378073`, span "excellent job" [79:92], `EFFECTIVENESS`, from the garbled "It doesn't excellent job".
- **Recommended policy.**
  1. Check the source review text if it is available.
  2. If it cannot be resolved, keep the row with empty `sentiment` and `NEEDS_REVIEW`, excluded from training and agreement.
  3. Do not guess `POSITIVE`.

## D9. The 27 agreed spans with different offsets

- **Official (§5.1, §10).** The shortest aspect-naming phrase; matching in agreement statistics is by IoU of at least 0.5.
- **Conflicting decisions.** Annotators matched on category and sentiment but not on boundaries, and the workbook does not decide them. Two matched pairs would overlap other gold spans depending on the choice (077526, 473638).
- **Recommended policy.**
  1. Choose the **narrower existing span** of the two. Do not invent new text.
  2. For implicit spans, keep the span that still carries the evaluation (negation or predicate); flagged in the table.
  3. A human reviews the flagged pairs.

| Review | A span | B span | Proposed pick | Flag |
|---|---|---|---|---|
| 029600 | this product | product | B | |
| 403870 | by far one of the best | one of the best | B | |
| 432787 | this product | product | B | |
| 099263 | this cream | cream | B | |
| 306537 | extremely functional | functional | B | |
| 679989 | manufacturing | lousy manufacturing | A | |
| 645328 | the real thing | real thing | B | |
| 699573 | this fragrance | fragrance | B | |
| 378032 | shipping | fast shipping | A | |
| 102847 | this tail | Love this tail | A | |
| 248083 | gave out after 10 years of hard use | 10 years of hard use | A | narrower drops the predicate |
| 158906 | last a month | didn't last a month | B | narrower drops the negation |
| 244720 | feel stiffer | stiffer | B | check polarity cue |
| 689441 | used and abused | clearly used and abused | A | |
| 506321 | keeps the flair ups less intense | flair ups less intense | B | check predicate |
| 096457 | product | Great product | A | |
| 495166 | greasy | Not greasy | A | negation (explicit aspect) |
| 077526 | firmer, more youthful | more youthful | B | overlaps 'firmer' |
| 372698 | functions like a barrier-repair cream | barrier-repair cream | B | check predicate |
| 227957 | much softer | softer | B | |
| 227957 | healing | healing cream | A | |
| 169305 | continue to purchase it | continue to purchase | B | |
| 476052 | smell | no smell | A | negation (explicit aspect) |
| 473638 | bigger | a lot bigger | A | |
| 473638 | tight and less puffy | less puffy | B | overlaps 'tight' |
| 410085 | for sore muscles | sore muscles | B | |
| 633318 | helps smoothen my skin | smoothen my skin | B | check predicate |

## D10. The 21 agreed spans with different span types

- **Official (§3, §5.3).** `span_type` is `EXPLICIT` when the aspect is named, `IMPLICIT` when the phrase only evaluates a facet. §10 does not require agreement on type.
- **Conflicting decisions.** A and B disagree on type for 21 agreed pairs: 10 with identical offsets, 11 where the boundaries also differ.
- **Affected.**
  - Same offsets (10):

    | Review | Span | A | B |
    |---|---|---|---|
    | 586811 | 95% mess free | IMPLICIT | EXPLICIT |
    | 161788 | loud | IMPLICIT | EXPLICIT |
    | 362529 | loud | IMPLICIT | EXPLICIT |
    | 306537 | getting you clean | EXPLICIT | IMPLICIT |
    | 102847 | Not long enough | IMPLICIT | EXPLICIT |
    | 236867 | filled in my hair | IMPLICIT | EXPLICIT |
    | 236867 | hair filled in | IMPLICIT | EXPLICIT |
    | 688447 | mostly empty | IMPLICIT | EXPLICIT |
    | 372698 | slippery | IMPLICIT | EXPLICIT |
    | 169305 | cleared it up | IMPLICIT | EXPLICIT |
  - Different offsets (11): 029600, 403870, 306537 (functional), 102847 (this tail), 248083, 244720, 077526, 372698 (barrier-repair), 227957, 473638, 410085.
- **Recommended policy.**
  1. Do not resolve by majority, and never default to A.
  2. Apply one test: if the span names a facet (`loud`, `slippery`), the type is `EXPLICIT`. If it only evaluates one ("mostly empty", "cleared it up"), it is `IMPLICIT`.
  3. For the 11 pairs where boundaries differ, set the type after D9 chooses the span, from that span.
  4. A human confirms the ten same-offset pairs, using the test.
  5. Do not use type disagreement as a reason to drop an agreed pair.

## D11. Dropping rejected spans and deduplicating the 26 repeated rows

- **Official (§5.4, §10).** One row per mention. Original annotator rows are not edited. Nothing says how adjudicated rejection is represented in gold.
- **Conflicting decisions.**
  - 51 original rows are rejected (44 from B only, 7 from A only), and each cites a rule. Several rules are in dispute (S4, S5, S6, S7), so some rejections will be reversed.
  - 110 retained rows are 84 distinct spans, with 26 repeat rows in 23 groups, all marked "count once", and no conflicting labels.
  - ADJ-033's rejected `product` [161:168] overlaps the kept 'continue to purchase this product' [135:168].
- **Affected.**
  - Rejected: ADJ-002, 003, 004, 012, 014, 017, 020, 022, 028, 033, 034, 041, 043, 047, 049, 050, 054, 058, 065, 071, 075, 076, 082, 083, 091, 093, 094, 095, 097, 099, 101, 102, 103, 104, 107, 111, 118, 123, 124, 125, 127, 128, 137, 138, 139, 145, 148, 149, 150, 155, 162.
  - Repeat rows (the second and later row of each group): ADJ-006, 008, 019, 025, 027, 031, 036, 039, 045, 046, 053, 064, 078, 080, 087, 088, 090, 113, 122, 131, 133, 134, 141, 144, 147, 152.
  - First rows of the groups: ADJ-005, 007, 018, 023, 026, 030, 035, 038, 044, 052, 063, 077, 079, 086, 089, 112, 120, 130, 132, 140, 143, 146, 151.
- **Recommended policy.**
  1. Build gold only from *agreed spans plus distinct adjudicated spans*, never from A ∪ B, so a rejected span cannot return.
  2. A span is distinct by `(message_id, start, end)`, so the 26 repeats count once. Keep the first row of each group, and keep the rest in a provenance log.
  3. Rejected spans stay out of gold, but are listed in a rejection log with the rule, so the decision is auditable.
  4. After D2, D3 and D6 decide, re-apply the rejected list: any row whose rule was overturned is restored.
  5. A rejection that overlaps a kept span (ADJ-033) is confirmed by a human.
  6. Before the gold CSV is written, check that no gold span matches a rejected `(message_id, start, end)`.

---

## Summary: approval table

| # | Decision | Recommendation | Approve? |
|---|---|---|---|
| D1 | Boundaries and negation | Shortest aspect phrase; negation in the label; implicit spans keep their evaluation; `the cap` becomes `cap` | |
| D2 | Pronouns and whole-product | Keep §5.5; `this product` is valid; revisit ADJ-047, 058, 083, 095 | |
| D3 | Multiple mentions and implicit | Keep §5.4; only same-offset repeats are duplicates; re-open 27 S5 rejections | |
| D4 | Hypothetical, hearsay, comparison | Follow §6 plus a "reviewer's own experience" exception; ADJ-059 becomes `NEUTRAL`; keep COMPARISON notes | |
| D5 | OTHER_ASPECT and new categories | No new categories now; facet note on 22 spans; P2 to P4 to v0.4; P5 widen definition; adopt Q3 and Q4 | |
| D6 | NEUTRAL and NEEDS_REVIEW | Keep S6 only for facet-less background; re-open 20 S6 rejections; low-confidence rows stay `NEEDS_REVIEW` | |
| D7 | Span type, notes, identity | Human adjudicator ID on all 162 rows; type inherited or set by D1; carry notes | |
| D8 | ADJ-109 | Keep with empty sentiment and `NEEDS_REVIEW`; do not guess | |
| D9 | 27 offset pairs | Narrower existing span; human reviews the flagged pairs | |
| D10 | 21 span-type pairs | Facet-named is `EXPLICIT`, otherwise `IMPLICIT`; human confirms the ten same-offset pairs | |
| D11 | Rejected and repeated rows | Build from agreed plus distinct adjudicated; rejection log; check no rejected span reaches gold | |

**Order of work after approval:** approve D1 to D6, then update the guidelines to v0.3 and re-open the affected rows, then D7 to D11, and only then build the gold CSV.
