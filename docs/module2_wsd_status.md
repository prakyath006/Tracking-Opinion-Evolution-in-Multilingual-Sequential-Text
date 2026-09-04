# Module 2 — Word Sense Disambiguation: Implemented, With One Open Defect

**Status: BUILT (Steps A1–A6 complete and committed), but the disambiguation
path is currently unexercised.** See "Open defect" below before citing this
module as a completed WSD contribution.

This document supersedes the status section of `module2_wsd_planned.md`,
which was written while Module 2 was still paused pending guide confirmation.
Guide confirmed WSD on 2026-09-03, and implementation followed.

## What was built

| Step | Artifact | Status |
|------|----------|--------|
| A1 | `ontology/aspect_taxonomy.json` — 7 aspect categories derived from real corpus frequency | Done |
| A2 | `ontology/wsd_lexicon.json` — IndoWordNet synsets (pyiwn: Tamil 25,419 / Malayalam 30,140 / Kannada 22,042) | Done |
| A3 | `ontology/wsd_lexicon_full.json` — + indic-transliteration Latin forms and manual code-mixed variants | Done |
| A4 | `src/wsd.py` — `WSDLexicon`, `ContextWindowDisambiguator`, `WordSenseDisambiguator`, `MostCommonSenseBaseline` | Done |
| A5 | `outputs/metrics/module2_wsd_results.{md,json}` — coverage, aspect distribution, MCS comparison, 100-sample human-review set | Done (run 2026-09-04) |
| A6 | `scripts/integrate_wsd.py` → `outputs/wsd/*_wsd_annotated.csv` — standalone annotation pass, not a hard dependency of the sentiment pipeline | Done |

## Measured results (real corpus, 15,000 texts)

Aspect coverage, i.e. share of tokens matching a lexicon surface form:

| Language | Total words | Aspect words | Coverage |
|----------|------------|-------------|----------|
| Tamil | 51,021 | 2,161 | 4.24% |
| Malayalam | 52,070 | 1,768 | 3.40% |
| Kannada | 41,882 | 1,068 | 2.55% |
| **Overall** | **144,973** | **4,997** | **3.45%** |

Aspect distribution is dominated by film-domain categories, consistent with
the DravidianCodeMix corpus being sourced from YouTube film comments:
`fan_stardom` (1,547), `music_bgm` (1,182), `trailer_teaser` (1,057),
`box_office_collection` (402), `hero_character` (368), `dialogue` (228),
`story_screenplay` (213).

## Open defect — the disambiguation path never fires

`outputs/metrics/module2_wsd_results.md` reports `ambiguous_words = 0` and
`resolved_words = 0` for all three languages. This is not a data property; it
is a property of how the lexicon was constructed.

All 173 distinct surface forms in `wsd_lexicon_full.json` map to **exactly one**
aspect category each — the categories partition the vocabulary with no overlap.
`WSDLexicon.is_ambiguous()` (`src/wsd.py:91`) defines a word as ambiguous iff it
maps to more than one category, so it returns `False` for every token in the
corpus, and `ContextWindowDisambiguator` is consequently never invoked.

Two consequences for how this module may honestly be described:

1. **What Module 2 currently is:** a multilingual, code-mix-aware *aspect
   tagger*. That is a real and working artifact, and its coverage numbers above
   are real.
2. **What Module 2 is not yet:** a *word sense disambiguator*. The contextual
   disambiguation logic exists and is unit-testable, but no corpus input
   reaches it, so it contributes nothing measurable.

The reported "resolution rate = 100%" is `0/0` returned as `100.0` by the
guard at `src/wsd.py:282` and must not be cited as a performance figure.

Likewise, Section 3 of the results file compares WSD against the Most-Common-
Sense baseline on *proportion of high-confidence predictions* (100.0% vs
98.52%), not on accuracy. No gold sense labels exist for this corpus, so
accuracy-over-MCS — the metric the original plan specified, and the standard
WSD protocol — is still uncomputed. The 100-sample human-review set written by
Step A5 (`human_verified: false` on every entry) is the input for that check
and has not yet been annotated.

## What would close the defect

Both items are prerequisites to claiming Module 2 as a WSD contribution:

1. **Introduce genuine polysemy into the lexicon.** Identify surface forms that
   legitimately belong to more than one aspect in this corpus and let them map
   to multiple categories. Real candidates visible in the data: `mass` (fan
   reaction vs. crowd scene), `hit` (box-office performance vs. song quality),
   `padam`/`movie` (the film as a whole vs. a specific screening), `sound`
   (BGM vs. audio mix quality). Until at least some forms are polysemous,
   `is_ambiguous()` cannot return `True` and A4's logic stays dead code.
2. **Annotate the 100-sample human-review set** so sense-disambiguation accuracy
   and improvement-over-MCS can be computed as originally specified.

## Literature comparison, once the defect is closed

The four WSD papers reviewed for this project and currently excluded from
`docs/literature_comparison.md`'s active table become the natural comparison
points at that stage:

- "EnhancedBERT: A feature-rich ensemble model for Arabic word sense
  disambiguation with statistical analysis and optimized data collection"
- "GlossGPT: GPT for Word Sense Disambiguation using Few-shot Chain-of-Thought
  Prompting"
- "System Fusion Based on WordNet Word Sense Disambiguation"
- "Improving selection of synsets from WordNet for domain-specific..."

## Relationship to Research Gap 5

Gap 5 ("Lack of Aspect-Level Opinion Evolution") needs to know which aspect a
sentence concerns. The aspect tagger delivered here supplies exactly that
signal, and `outputs/wsd/*_wsd_annotated.csv` carries it per row — so Gap 5
work is unblocked by what exists today, independently of the disambiguation
defect above.
