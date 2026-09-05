# Module 2 — Word Sense Disambiguation: Implemented and Exercised

**Status: BUILT (Steps A1–A6) and the disambiguation path now actually runs.**
The defect recorded in the previous version of this document — that
`ContextWindowDisambiguator` was dead code — is closed. Two further bugs found
while closing it are also fixed and described below.

Guide confirmed WSD on 2026-09-03; implementation and this evaluation followed.

## What was built

| Step | Artifact |
|------|----------|
| A1 | `ontology/aspect_taxonomy.json` — 7 aspect categories derived from real corpus frequency |
| A2 | `ontology/wsd_lexicon.json` — IndoWordNet synsets (pyiwn: Tamil 25,419 / Malayalam 30,140 / Kannada 22,042) |
| A3 | `ontology/wsd_lexicon_full.json` — + indic-transliteration Latin forms and manual code-mixed variants |
| A4 | `src/wsd.py` — `WSDLexicon`, `ContextWindowDisambiguator`, `WordSenseDisambiguator`, `MostCommonSenseBaseline` |
| A5 | `outputs/metrics/module2_wsd_results.{md,json}` — coverage, aspect distribution, MCS comparison, human-review sample |
| A6 | `scripts/integrate_wsd.py` → `outputs/wsd/*_wsd_annotated.csv` — standalone annotation, not a dependency of the sentiment pipeline |

## Defect 1 (closed) — the disambiguation path never fired

Originally every one of the 173 lexicon surface forms mapped to exactly one
aspect, so `WSDLexicon.is_ambiguous()` returned `False` for every corpus token
and the contextual disambiguator was never invoked. Measured
`ambiguous_words = 0` across all three languages, and the reported "100%
resolution" was `0/0`.

**Fix:** three surface forms now carry the multiple aspect senses they
demonstrably have in this corpus. Each was established by immediate-collocation
analysis plus ±5-token co-occurrence over all 53,263 rows — not assumed:

| Form | Senses | Corpus evidence |
|---|---|---|
| `mass` | fan_stardom, trailer_teaser, music_bgm, dialogue | n=3,510. Acts as an intensifier that takes the aspect of what it modifies: followed by `trailer`(97), `dialogue`(43), `bgm`(40). Preceded by aspect-free Tamil intensifiers `marana`(541), `semma`(174), `pakka`(150), `kola`(103). Another aspect's vocabulary occurs within ±5 tokens in ~23% of uses. |
| `punch` | dialogue, hero_character | n=75. `punch dialogue`/`dialogues`(15) is speech; `superman punch`(9) and `mass punch`(3) are an action beat. |
| `climax` | story_screenplay, dialogue | n=96. `climax dialog`(4) + `dialogue`(3) is the spoken line; `climax scene`(5) and `climax la`(17) are the plot point. |

**Result — the disambiguator now runs on real input:**

| Language | Aspect words | Ambiguous | Resolved | Resolution rate |
|---|---|---|---|---|
| Tamil | 2,161 | 484 | 96 | 19.83% |
| Malayalam | 1,768 | 190 | 22 | 11.58% |
| Kannada | 1,068 | 8 | 3 | 37.50% |
| **Overall** | **4,997** | **682** | **121** | **17.74%** |

The ~18% resolution rate is itself a finding, not a failure: in 82% of ambiguous
instances no other aspect vocabulary appears within the ±5-token window, so the
disambiguator correctly abstains rather than guessing. Short code-mixed YouTube
comments frequently carry too little context to disambiguate — worth stating
directly, since it motivates a wider window or an embedding-based approach as
future work.

## Defect 2 (fixed) — multi-word lexicon entries were unreachable

`process()` looked up one token at a time, so the three multi-word forms in the
lexicon — `box office`, `first look`, `background music` — could never match.
They were dead entries: "box office collection is huge" matched only
`collection`, and "the first look poster" matched neither word.

**Fix:** `process()` now tries a bigram before each unigram and consumes both
tokens on a hit. Verified: `box office` → box_office_collection, `first look` →
trailer_teaser, `background music` → music_bgm.

## Defect 3 (fixed) — half of all `hit` matches were false positives

Of the 1,200 occurrences of `hit`, **604 (50.3%) are followed by `like` or
`likes`** — the YouTube call-to-action "fans hit like", which has nothing to do
with box-office performance. All 1,200 were being tagged
`box_office_collection`.

**Fix:** `NON_ASPECT_COLLOCATIONS` in `src/wsd.py` cancels an aspect reading when
the following token makes it a non-aspect idiom. `fans hit like` no longer tags
`hit`; `padam super hit` still does. This moved `box_office_collection` from 402
to 327 in the 15,000-text sample.

## Aspect distribution after all three fixes

| Aspect | Before | After |
|---|---|---|
| music_bgm | 1,182 | 1,727 |
| trailer_teaser | 1,057 | 1,127 |
| fan_stardom | 1,547 | 920 |
| hero_character | 368 | 382 |
| box_office_collection | 402 | 327 |
| dialogue | 228 | 230 |
| story_screenplay | 213 | 210 |

`fan_stardom` falls by 40% because `mass` is now distributed by context instead
of being assigned to it wholesale, and `box_office_collection` falls because of
the `hit like` guard. Both moves are corrections, not losses.

Corpus-wide annotation (Step A6) covers 53,263 rows: Tamil 31.85%, Malayalam
28.16%, Kannada 18.34% carry at least one aspect.

## Still open — no accuracy figure yet

Section 3 of `module2_wsd_results.md` compares WSD against the Most-Common-Sense
baseline on *share of high-confidence predictions* (89.85% vs 98.51%), **not on
accuracy**. No gold sense labels exist for this corpus, so accuracy-over-MCS —
the metric the plan specified, and the standard WSD protocol — remains
uncomputed. Do not quote those two percentages as accuracy; on this proxy a
model that always predicts confidently scores best, which is why MCS "wins" it.

The human-review sample written by Step A5 (`sample_annotations`, every entry
`human_verified: false`) is the input for the real measurement. Annotating it is
the one remaining task before Module 2 can report a WSD accuracy number.

One known inconsistency to be aware of: `get_coverage_stats()` still looks up
single tokens directly, so `coverage_pct` does not reflect the bigram matching
or the collocation guard, while `aspect_distribution` (computed via `process()`)
does. The ambiguity and resolution counts are unaffected.

## Literature comparison, once accuracy exists

The four WSD papers reviewed for this project and currently excluded from
`docs/literature_comparison.md`'s active table become the comparison points at
that stage: EnhancedBERT (Arabic WSD), GlossGPT (few-shot CoT WSD), System
Fusion Based on WordNet WSD, and Improving selection of synsets from WordNet.

## Relationship to Research Gap 5

Gap 5 ("Lack of Aspect-Level Opinion Evolution") needs to know which aspect a
sentence concerns. `outputs/wsd/*_wsd_annotated.csv` now carries that per row,
with `mass` resolved by context rather than collapsed into one category — so
Gap 5 work is unblocked and its input is more accurate than before.
