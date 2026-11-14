# Guideline changes the adjudication needs (proposals, nothing applied)

Evidence: the pilot batch (50 reviews, 10 product timelines) annotated independently by A and B, compared with `scripts/validate_aspect_annotations.py --compare` and itemised in
`adjudication_items.csv` (162 items; 158 inside the agreement statistics). The current text is `docs/aspect_annotation_guidelines.md` v0.2.

These are **proposals for the team to decide**. No annotator is treated as right: where the two practices differ, each option is labelled with the practice it corresponds to, and the draft wording is
a candidate for a vote. Nothing in `docs/aspect_annotation_guidelines.md` or the taxonomy has been changed.

## 1. Span boundaries (57 items tagged `span_boundary`: 55 partial overlaps below the 0.5 threshold, 2 matched spans below IoU 1)

What the data shows (the section 5 rules say "mark the shortest phrase that names the aspect, not the opinion word"; for implicit aspects "the shortest evaluative phrase"):

* Of the 55 partial-overlap items, in 35 the A span contains the nearest B span, in 14 the B span contains the A span, and in 6 they cross (an item is one annotator's span with its nearest counterpart, so a conflict can appear as two items). Typical pairs:
  "takes me much less time to trim my beard" / "much less time"; "softer and has less wrinkles" / "softer" and "less wrinkles"; "compact way to keep them together" / "keep them together".
* **Negation.** Both annotators include a negation in some spans (A: 7 spans such as "will not buy again", "Not long enough"; B: 14 such as "didn't work", "no smell") and leave it out in others: for the same sentence A has "work" and B has "didn't work"; A has "solve the bleeding finger tips" and B has "didn't solve". The guidelines treat negation as part of the evaluation but say nothing about the span.
* **Determiners and modifiers.** "the pin" / "pin"; "Love this" / "this"; "lasts" / "lasts about a month"; "swing door on the collector" / "swing door".
* **Coordination.** One span over "softer and has less wrinkles" or two spans.
* **Clause versus head.** "cleared up within the week" / "cleared up".

Decisions needed:
1. Does the span include negation and modals ("didn't work")? Neither annotator is consistent on it today.
2. Are determiners, quantities and time expressions ("the", "about a month", "within the week") inside the span?
3. A coordinated phrase: one span, or one span per coordinated aspect or evaluation?
4. For implicit aspects, the evaluative predicate only, or predicate plus its object ("cuts much better" / "cuts")?
5. Maximum span length (A: 22 spans of five or more words; B: 9).
6. Should agreement keep the exact-boundary rule at overlap/union >= 0.5, or add a tolerance for determiners and punctuation?

## 2. Pronouns and referring expressions (22 items; unmatched: 16 from A, 6 from B)

Evidence: A has 19 pronoun-like spans (12 with the referent in notes) and B has 4 (3 with notes). A's unmatched spans include "it", "It", "this", "This one", "product", "This trimmer", "this fragrance", "This stuff";
B's include "Love it", "love it", "loved it", "Love this", "this", "product". The same sentence is often covered by `it` (A) and `Love it` (B). Section 5 rule 5 says to mark a pronoun only when no named mention is nearby and to put the referent in notes,
but not what the span is for a whole-product opinion.

Decisions needed:
1. For a `GENERAL` opinion, which words are the span: the referring expression ("it", "this trimmer"), the evaluative phrase ("love it"), or the evaluative word alone ("great")?
2. May a bare pronoun be a span at all (the current rule allows it with a note)? Should the referent be required in notes or in a column?
3. Is "product", "this one", "this stuff" an aspect span (`GENERAL`) or just a reference?
4. Is `GENERAL` annotated when specific aspects are already annotated in the same sentence? (`GENERAL` rows: A 43, B 29; 24 of A's 51 unmatched spans are `GENERAL`.)

## 3. `OTHER_ASPECT` (25 items; A used it for 26 rows, B for 7; 16 of the 23 category disagreements are A's `OTHER_ASPECT` against a taxonomy category)

Facets named in the notes of A's 26 `OTHER_ASPECT` rows, and what B did with the matched span (B has 7 `OTHER_ASPECT` rows of its own: noise 3, manufacturing, battery operated, "quality", residue / transfer):

| Facet (from A's notes) | A rows | Where B put the matched span |
|---|---|---|
| noise level ("noise", "loud") | 4 | `OTHER_ASPECT` for the three matched spans (both annotators wrote that noise is not in the taxonomy); one A span unmatched |
| manufacturing / build quality ("manufacturing", "well made") | 3 | `OTHER_ASPECT` ("lousy manufacturing") for one, `DURABILITY` for the two "well made" spans |
| adjustability, flexibility, shape retention, rigidity ("able to adjust", "bendable", "flexible", "stays shaped", "does not Bend") | 5 | `EASE_OF_USE` for three, `DURABILITY` for two |
| trim settings | 1 | `EASE_OF_USE` |
| attachment and included parts ("safety pin", "pin", "ribbon tie", "ribbon ties", "come open", "held well", "the pin") | 7 | `PACKAGING` for five, `EFFECTIVENESS` for "held well", one unmatched |
| battery compartment | 1 | `PACKAGING` |
| weight ("heavy") | 1 | `TEXTURE` |
| power source ("battery operated") | 1 | `OTHER_ASPECT` |
| sharpness (a `NEEDS_REVIEW` span), SPF ("sunblock"), availability | 3 | `EFFECTIVENESS`, `EFFECTIVENESS`, unmatched |

The two annotators already agree that noise and power source need `OTHER_ASPECT` (or a new category). B itself uses `DURABILITY` for "well made" and `OTHER_ASPECT` for "lousy manufacturing", and A labels the review-level "quality" span
`GENERAL` where B uses `OTHER_ASPECT`. Two of A's rows have no facet in the notes.

The taxonomy defines `OTHER_ASPECT` as a facet "no category covers" and promotes frequent cases to new categories, but does not say how wide the existing categories are (`PACKAGING`: the container only, or any included part?
`DURABILITY`: lifespan only, or build quality? `EASE_OF_USE`: handling only, or adjustability?).

Decisions needed:
1. Widen existing definitions, for example `PACKAGING` to included parts and attachments, `DURABILITY` to build quality and material, `EASE_OF_USE` to adjustability and flexibility (this is the practice B shows).
2. Or add categories (candidates the notes support: noise, build quality, attachment / included components, weight, power source) and keep the narrow definitions (this follows A's use of `OTHER_ASPECT`). Adding categories changes the taxonomy version and the model's tag set.
3. A promotion threshold: how many `OTHER_ASPECT` uses of one facet trigger a new category (noise: both annotators independently used `OTHER_ASPECT` for the same three spans, A for a fourth).
4. Is the product type itself ("sunblock") an aspect, `GENERAL`, or no span?
5. `OTHER_ASPECT` must always carry the facet in notes (2 of A's rows have none).

## 4. Implicit aspects (83 items tagged `implicit_aspect`; A 73 of 181 rows implicit (40%), B 62 of 212 (29%))

Evidence: B has 19 one-word implicit spans ("great", "easy", "best", "clean", "convenient", "glowing"), A has 8; A's implicit spans are longer clauses ("would certainly order again", "bent over backwards to apologize").
Eight matched items differ in span type. B often annotates both a bare evaluative word and the explicit phrase carrying the same evaluation ("clean", "easy to clean", "swing door" in one review). Rule 5.3 says: when there is no aspect word
but the sentence evaluates a facet, mark the shortest evaluative phrase and set `IMPLICIT`.

Decisions needed:
1. Does a bare evaluative word ("great", "easy") get its own span when an explicit aspect is already annotated nearby, or is one evaluation one span?
2. Are single adjectives allowed as implicit spans, and what is the category when the word alone is ambiguous ("clean")?
3. Definition of `EXPLICIT` versus `IMPLICIT` so that `span_type` is reproducible (for example "the span contains the word that names the facet").
4. Should every aspect mention be annotated even when it carries no evaluation (`NEUTRAL`: B's "tube" and "case"; A's "scent"), or only evaluated ones? This drives the 78 items where one annotator has no span at all.
5. Handling garbled or ungrammatical text ("It doesn't excellent job", a `NEEDS_REVIEW` case).

## 5. Related points the comparison exposed

* **Sentiment (3 items).** Hypothetical warnings ("breakouts" in a warning to others: A `NEUTRAL`, B `NEGATIVE`), comparisons ("not as thick": A `POSITIVE`, B `NEUTRAL`, both noting COMPARISON) and a requirement ("freeze for many hours or overnight":
  A `NEUTRAL` with an UNSURE note, B `NEGATIVE`). Section 6 says hypotheticals are `NEUTRAL` only when the aspect is named and gives no rule for comparisons that imply a judgement or for burdens.
* **UNSURE versus `NEEDS_REVIEW`.** A wrote UNSURE in the notes of 23 `ANNOTATED` rows but marked only 2 rows `NEEDS_REVIEW`. Decide what makes a row `NEEDS_REVIEW` (empty sentiment only, or any doubt) and whether UNSURE notes are allowed.
* **Review status of a message.** Both `NEEDS_REVIEW` messages have other rows `ANNOTATED`; the validator flags this as `MIXED_STATUSES`. Decide whether a message may carry mixed statuses.

## Unresolved decisions at a glance

| # | Topic | Decision | Items affected (of 162) |
|---|---|---|---|
| 1 | Span boundaries | negation, determiners and qualifiers, coordination, predicate versus object, maximum length, tolerance in agreement | 57 |
| 2 | Pronouns | span for a whole-product opinion, bare pronouns, "product", `GENERAL` alongside specific aspects | 22 |
| 3 | `OTHER_ASPECT` | widen definitions or add categories, promotion threshold, product type, notes required | 25 |
| 4 | Implicit aspects | one evaluation one span, single adjectives, `EXPLICIT` / `IMPLICIT` definition, annotate unevaluated mentions | 83 (78 with no counterpart span) |
| 5 | Sentiment edge cases | hypotheticals, comparisons, requirements | 3 |
| 6 | `NEEDS_REVIEW` and UNSURE | when to use, mixed statuses in one message, garbled text | 2 (+ 22 tagged UNSURE) |

The tags overlap, so the counts do not add up to 162. Settling decisions 1 to 4 first would resolve most items by rule; decisions 5 and 6 and the content of each remaining item need a human judgement.
