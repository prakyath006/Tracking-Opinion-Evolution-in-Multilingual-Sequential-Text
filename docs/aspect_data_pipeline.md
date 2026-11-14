# Aspect-level data pipeline (`src/aspect_data.py`)

Phase 2 preparation. Turns completed annotation CSVs (`docs/aspect_annotation_guidelines.md`) into data for aspect extraction and aspect-conditioned
sentiment. **Nothing is trained and no real annotation exists yet**: the real pilot files are unannotated, so the module has been exercised only on
synthetic fixtures (`tests/test_aspect_data.py`).

## Flow

```
annotation CSV --load_annotations--> finished, valid Messages --+--> build_extraction_examples --> BIO labels (per token)
 (read-only, validator rules)                                   +--> build_sentiment_examples  --> one example per aspect
                                                                +--> group_by_sequence         --> product timelines, in time order
examples --split_by_entity--> train / val / test (a product is never split)
```

| Function | What it does |
|---|---|
| `load_annotations(path, on_invalid="raise"\|"skip")` | opens the file read-only and runs `scripts/validate_aspect_annotations.py` (labels, categories per domain, offsets, status, duplicates). Returns finished messages (`ANNOTATED`/`ADJUDICATED`, no errors), plus the ids of unfinished and skipped messages. Unfinished messages are never treated as `NO_ASPECT`. |
| `bio_encode(text, mentions, tokenizer, ...)` | character spans to token BIO labels with a fast tokenizer's offsets |
| `build_extraction_examples(messages, tokenizer, ...)` | one BIO example per message |
| `build_sentiment_examples(messages, level=...)` | aspect-conditioned sentiment examples |
| `encode_sentiment_examples(examples, tokenizer, conditioning=...)` | sentence-pair token ids (aspect text, review text) |
| `group_by_sequence(items)` / `category_series(examples)` | product timelines in chronological order; per-category label series |
| `split_by_entity(items, ratios, seed)`, `assert_no_entity_leakage`, `cross_split_duplicate_texts` | entity-disjoint splits and their checks |

## Label spaces

* **Aspect sentiment** is `AspectSentiment`: POSITIVE 0, NEGATIVE 1, NEUTRAL 2, MIXED 3. It is **not** the ontology's `SentimentState` (POSITIVE 0, NEGATIVE 1, MIXED 2, UNKNOWN 3), whose
  definition is unchanged. `to_ontology_name()` converts by name and returns `None` for NEUTRAL. Never index ontology tensors with these ids.
* **BIO tags**: generic `O, B-ASP, I-ASP`, or category-specific `O, B-<category>, I-<category>, ...` (the tag's index is its label id). Special tokens, and masked messages, get `-100`.
* **Categories** come from `ontology/product_aspect_taxonomy.json` or `ontology/aspect_taxonomy.json` by the row's `source`, plus `GENERAL` and `OTHER_ASPECT`.

## Rules

**BIO.** A token belongs to a span if its character offsets overlap it. The first token is `B`, the rest `I`, so a word split into subwords is `B I I`. Adjacent spans give `B B`.
Offsets are read from the tokenizer, not guessed: this matters because mBERT offsets start at the word and XLM-R tokens carry a leading `▁` while their offsets do not.
A span starting or ending inside a token is still labelled on the tokens it touches and is listed in `partial_boundary_mentions`. A span lost to truncation is listed in
`truncated_mentions`. Overlapping spans raise `SpanOverlapError` by default; `overlap="merge"` (generic tags only) merges them.

**Messages without an aspect.** `NO_ASPECT`: all tokens `O` (a real negative example). `UNCLEAR_ASPECT`: excluded by default, or kept with every label `-100` (`unclear="mask"`); never all-`O`, because an aspect may be present.

**Several aspects and mixed sentiment.** One example per (message, category): different categories give separate examples with separate labels (praise for one and criticism of another is never MIXED).
For several mentions of one category the label is the guideline rule (section 11): all NEUTRAL gives NEUTRAL; otherwise NEUTRAL is ignored, and the value is MIXED if any mention is MIXED or
POSITIVE and NEGATIVE both occur. `level="mention"` keeps each mention's own label instead.

**Conditioning.** The pair is (aspect text, review text) with `conditioning` = `category` (the category name), `span` (the annotated text) or `category_and_span`. Only the review text is truncated.

**Timelines.** Sequences are grouped by `sequence_id`, ordered by (timestamp, position), and refused if positions and timestamps disagree. Sequence types are not mixed unless asked.

**Splits.** Whole entities (products) are assigned to splits by a seeded shuffle of the sorted ids: deterministic, independent of item order, every non-empty split gets at least one product.
An empty `entity_id` raises (comment-level data has no product, so it needs another grouping rule).

## Limitations

* Tested on synthetic fixtures and on the mBERT and XLM-R tokenizers only. Real annotations will show cases the fixtures do not (implicit spans, pronouns, long reviews); expect fixes after the pilot.
* **Truncation.** Reviews of up to 300 words are cut at the token limit, which can drop later aspects; the report counts them. A longer limit or sliding windows may be needed.
* **Implicit spans** (`span_type = IMPLICIT`, e.g. "worth every penny") are labelled like explicit ones; an extraction model has to learn to tag evaluative phrases.
* **MIXED** from aggregation is derived, not annotated by a person.
* **Leakage.** Entity-disjoint splits do not rule out the same reviewer or identical review texts appearing in different products. The annotation files hold no reviewer id, so reviewer leakage cannot be checked;
  identical texts are reported by `cross_split_duplicate_texts`, not removed.
* Ten products give splits of one or two products each; any result from them would be anecdotal.
* The module imports the validator from `scripts/` so the rules stay in one place; moving either file needs both updated.
* Datasets are plain lists of dataclasses and dicts. No torch `Dataset`, collator or training code exists yet.
