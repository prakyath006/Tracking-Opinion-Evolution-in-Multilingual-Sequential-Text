# Module 2 — Word Sense Disambiguation: Planned, Not Implemented

**Status: NOT BUILT.** This document describes what Module 2 would measure
if implemented. No WSD logic exists anywhere in this repository. This is
intentional and explicitly scoped — Word Sense Disambiguation and the
aspect-level ontology extension were paused pending guide confirmation on
scope and timeline, and stay paused in this task.

## Why this exists as a document rather than code

The per-module metrics framework (Modules 1, 3, 4, 5, 6) gives every
implemented module its own dedicated metric set rather than generic shared
metrics. Module 2 has no implementation to measure yet, so instead of
skipping it, this documents what its metrics *would* be — so the shape of
the eventual work is decided ahead of time, without writing WSD logic before
the guide has confirmed it should happen at all.

## Relevant literature already in this project's materials

Four papers in `Project Phase/Research Papers/` are specifically about WSD
and were reviewed for this project but excluded from
`docs/literature_comparison.md`'s active comparison table (noted there
explicitly) because Module 2 isn't built:

- "EnhancedBERT: A feature-rich ensemble model for Arabic word sense
  disambiguation with statistical analysis and optimized data collection"
- "GlossGPT: GPT for Word Sense Disambiguation using Few-shot
  Chain-of-Thought Prompting"
- "System Fusion Based on WordNet Word Sense Disambiguation"
- "Improving selection of synsets from WordNet for domain-specific..."

These would be the natural comparison points once Module 2 is built and has
real numbers — the same way `docs/literature_comparison.md` compares
Module 1/3/4/5/6 work against the sentiment/opinion-mining papers already
reviewed.

## What problem Module 2 would address

This project's ontology (`src/ontology.py`) currently maps whole reviews/
comments to a sentiment state. It has no mechanism for resolving which
*sense* of an ambiguous word is meant in context — e.g., whether "cool" in
a beauty-product review means "temperature" or "stylish approval", which
changes how that word should contribute to sentiment. Research Gap 5 in this
project's own gap analysis (`Project Phase/Research Gaps.docx`) — "Lack of
Aspect-Level Opinion Evolution" — is adjacent to this: aspect-level tracking
needs to know which aspect (camera, battery, delivery, ...) a sentence is
about, which itself often requires resolving word sense first (e.g. "battery"
in a phone review vs. a car review). WSD would be a prerequisite building
block for that aspect-level work, not a replacement for it.

## Metrics Module 2 would report, once built

### 1. Sense-Disambiguation Accuracy
Accuracy of the WSD component's predicted word sense against a
manually-labeled sample of ambiguous words in context — the standard WSD
evaluation protocol (matching how the reviewed WSD papers evaluate their own
systems). Requires: a manually-labeled gold sample does not exist yet in
this project and would need to be created (or an existing sense-annotated
corpus adopted) before this metric could be computed.

### 2. Coverage
% of ambiguous words in the corpus (i.e., words with more than one sense in
the reference sense inventory, e.g. WordNet) that the WSD component actually
resolves to a specific sense, as opposed to abstaining or falling back to a
default. Mirrors Module 1's coverage metric in spirit (fraction of inputs
that resolve to a real answer vs. fall through to "unknown"), applied to
word senses instead of sentiment labels.

### 3. Improvement Over Most-Common-Sense Baseline
The standard WSD baseline is "always predict the most frequent sense of a
word" (MFS), which is a surprisingly strong baseline in practice. Module 2's
number here would be accuracy-over-MFS-baseline — the actual contribution
of doing disambiguation at all, rather than raw accuracy alone (raw accuracy
can look high while barely beating the trivial baseline).

## What is explicitly NOT decided yet

- Which sense inventory (WordNet, BabelNet, a custom code-mixed inventory —
  code-mixed Tamil/Malayalam/Kannada-English text has no off-the-shelf
  multilingual sense inventory with the same coverage WordNet has for
  English) to use.
- Whether disambiguation happens per-language or requires a cross-lingual
  approach, given this project's code-mixed domains.
- Timeline and scope relative to the rest of the project's remaining work.

These require guide confirmation before any implementation work starts, per
the task constraint this document was written under.
