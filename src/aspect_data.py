"""
Aspect-level data pipeline (Phase 2 preparation).

Turns manually annotated aspect CSV files (docs/aspect_annotation_guidelines.md) into training data for two future tasks:
  * aspect extraction      : token-level BIO labels aligned to a fast tokenizer's character offsets (mBERT and XLM-R);
  * aspect-conditioned sentiment : one example per aspect, labelled POSITIVE / NEGATIVE / NEUTRAL / MIXED.
It also groups examples by product timeline in chronological order and provides splits that keep a product entirely in one split.

Principles
  * Read-only: annotation files are opened for reading only; nothing is written, completed or inferred.
  * Validation is not re-implemented: the rules of scripts/validate_aspect_annotations.py (labels, categories per domain, offsets,
    status handling, duplicates) are used as they are, so the two cannot drift apart.
  * Only FINISHED messages (status ANNOTATED or ADJUDICATED, no errors) become data. Unannotated or unfinished messages are never
    treated as "no aspect".
  * NEUTRAL is an annotation-only label (guidelines section 11). `AspectSentiment` is its own enum with its own ids; the ontology's
    SentimentState is not changed and NEUTRAL has no ontology counterpart.
  * No torch dependency and no model: tokenizers only.
"""
import importlib
import json
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGNORE_INDEX = -100                                           # label of special tokens, padding and masked messages


def _validator():
    """The annotation validator lives in scripts/; it is imported, not copied."""
    scripts = os.path.join(ROOT, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("validate_aspect_annotations")


# ───────────────────────────── label spaces ─────────────────────────────
class AspectSentiment(Enum):
    """
    Sentiment toward an aspect. Deliberately NOT the ontology's SentimentState: it contains NEUTRAL, which the ontology does not,
    and its ids are its own. Convert by NAME (`to_ontology_name`), never by id.
    """
    POSITIVE = 0
    NEGATIVE = 1
    NEUTRAL = 2
    MIXED = 3

    @classmethod
    def label_names(cls) -> List[str]:
        return [m.name for m in cls]

    @classmethod
    def from_label(cls, label: str) -> "AspectSentiment":
        try:
            return cls[label]
        except KeyError:
            raise ValueError(f"{label!r} is not an aspect sentiment label; expected one of {cls.label_names()}") from None

    def to_ontology_name(self) -> Optional[str]:
        """Name of the ontology SentimentState with the same meaning, or None for NEUTRAL (no counterpart)."""
        return None if self is AspectSentiment.NEUTRAL else self.name


def category_ids(domain: str, root: str = ROOT) -> List[str]:
    """Aspect category ids of a domain ('product' or 'social') in taxonomy order, then GENERAL and OTHER_ASPECT if not already there."""
    if domain == "product":
        tax = json.load(open(os.path.join(root, "ontology", "product_aspect_taxonomy.json"), encoding="utf-8"))
    elif domain == "social":
        tax = json.load(open(os.path.join(root, "ontology", "aspect_taxonomy.json"), encoding="utf-8"))
    else:
        raise ValueError(f"unknown domain {domain!r}; expected 'product' or 'social'")
    ids = [c["id"] for c in tax["categories"]]
    return ids + [x for x in ("GENERAL", "OTHER_ASPECT") if x not in ids]


def bio_tagset(categories: Optional[Sequence[str]] = None) -> List[str]:
    """['O', 'B-ASP', 'I-ASP'], or with categories ['O', 'B-<c1>', 'I-<c1>', ...]. The index of a tag is its label id."""
    if not categories:
        return ["O", "B-ASP", "I-ASP"]
    return ["O"] + [f"{p}-{c}" for c in categories for p in ("B", "I")]


class AnnotationError(ValueError):
    """Annotation file is not usable. `issues` holds the validator's findings."""

    def __init__(self, message, issues=None):
        super().__init__(message)
        self.issues = issues or []


class SpanOverlapError(ValueError):
    """Two aspect spans of one message share tokens and the overlap policy is 'error'."""


# ───────────────────────────── records ─────────────────────────────
@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    text: str
    category: str
    sentiment: str
    span_type: str


@dataclass
class Message:
    sequence_id: str
    sequence_type: str
    entity_id: str
    message_id: str
    position: int
    timestamp_ms: int
    language: str
    domain: str
    text: str
    title: str
    status: str                                          # ASPECT_PRESENT, NO_ASPECT or UNCLEAR_ASPECT
    mentions: Tuple[Mention, ...] = ()


@dataclass
class AnnotationSet:
    path: str
    messages: List[Message]
    unfinished_message_ids: List[str]                    # not ANNOTATED/ADJUDICATED yet; never treated as NO_ASPECT
    skipped_invalid_message_ids: List[str]               # only with on_invalid='skip'
    issues: List[dict] = field(default_factory=list)     # the validator's warnings (errors raise or are skipped)

    @property
    def domains(self) -> List[str]:
        return sorted({m.domain for m in self.messages})


# ───────────────────────────── reading ─────────────────────────────
def load_annotations(path: str, *, reference: Optional[str] = None, on_invalid: str = "raise", root: str = ROOT) -> AnnotationSet:
    """
    Read an annotation CSV (read-only), validate it with the project's validator, and return its finished, valid messages.
    on_invalid='raise' stops with AnnotationError if the file has any error; 'skip' drops only the messages that have errors.
    Unfinished messages are always left out and listed in `unfinished_message_ids`.
    """
    if on_invalid not in ("raise", "skip"):
        raise ValueError("on_invalid must be 'raise' or 'skip'")
    v = _validator()
    df = v.read_annotations(path)
    header = v.header_issues(df)
    if header:
        raise AnnotationError("invalid header: " + "; ".join(i["detail"] for i in header), header)
    res = v.validate(df, v.load_taxonomies(root), v.find_reference(df, root, reference))
    errors = [i for i in res["issues"] if i["severity"] == v.ERROR]
    bad = {df.at[i, "message_id"] for i in res["error_rows"]}
    if errors and on_invalid == "raise":
        raise AnnotationError(f"{len(errors)} validation errors in {path}; first: {errors[0]['code']} {errors[0]['detail']}", errors)
    complete = res["complete_messages"]
    messages = []
    for mid, g in df.groupby("message_id", sort=False):
        if mid not in complete:
            continue
        first = g.iloc[0]
        rows = g[g.aspect_status == "ASPECT_PRESENT"]
        status = "ASPECT_PRESENT" if len(rows) else first.aspect_status
        mentions = tuple(sorted((Mention(int(r.span_start), int(r.span_end), r.aspect_span_text, r.aspect_category, r.sentiment, r.span_type)
                                 for r in rows.itertuples()), key=lambda m: (m.start, m.end, m.category)))
        messages.append(Message(first.sequence_id, first.sequence_type, first.entity_id, mid, int(first.position), int(first.timestamp_ms),
                                first.language, v.domain_of(first.source), first.text, first.title, status, mentions))
    return AnnotationSet(path, messages, sorted(res["unfinished_messages"] - bad), sorted(bad) if on_invalid == "skip" else [],
                         [i for i in res["issues"] if i["severity"] == v.WARNING])


# ───────────────────────────── timelines ─────────────────────────────
def _order_key(item):
    return (item.timestamp_ms, item.position, getattr(item, "category_order", 0), getattr(item, "example_id", ""))


def group_by_sequence(items: Iterable, sequence_type: Optional[str] = None) -> "OrderedDict[str, list]":
    """
    Group messages or examples by sequence (product timeline), each group in chronological order (timestamp, then position).
    Sequences are ordered by their first timestamp, then id. Raises AnnotationError if a sequence's positions and timestamps disagree,
    because that would mean the order was changed. Sequence types are never mixed unless sequence_type is None and the data mixes them.
    """
    groups = defaultdict(list)
    for it in items:
        if sequence_type is None or it.sequence_type == sequence_type:
            groups[it.sequence_id].append(it)
    out = OrderedDict()
    for sid in sorted(groups, key=lambda s: (min(i.timestamp_ms for i in groups[s]), s)):
        ordered = sorted(groups[sid], key=_order_key)
        by_position = sorted(groups[sid], key=lambda i: (i.position, getattr(i, "category_order", 0), getattr(i, "example_id", "")))
        if [i.timestamp_ms for i in ordered] != [i.timestamp_ms for i in by_position]:
            raise AnnotationError(f"sequence {sid}: timestamps are not in the order of position")
        out[sid] = ordered
    return out


# ───────────────────────────── BIO labels for aspect extraction ─────────────────────────────
@dataclass
class BioEncoding:
    input_ids: List[int]
    attention_mask: List[int]
    offsets: List[Tuple[int, int]]                       # character offsets of every token; (0, 0) for special tokens
    special_tokens_mask: List[int]
    tags: List[Optional[str]]                            # None for special tokens
    labels: List[int]                                    # tag ids, IGNORE_INDEX for special tokens (or everything if masked)
    tagset: List[str]
    truncated_mentions: List[Mention] = field(default_factory=list)       # no token left after truncation
    unaligned_mentions: List[Mention] = field(default_factory=list)       # inside the text but no token overlaps (should not happen)
    partial_boundary_mentions: List[Mention] = field(default_factory=list)   # span starts or ends inside a token


def _require_fast(tokenizer):
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError("a fast tokenizer is required (character offsets); load it with AutoTokenizer.from_pretrained(..., use_fast=True)")


def _boundary_ok(text, a, b):
    lo, hi = min(a, b), max(a, b)
    return text[lo:hi].strip() == ""                     # equal, or differing only by whitespace (XLM-R tokens may carry the space)


def bio_encode(text: str, mentions: Sequence[Mention], tokenizer, *, max_length: int = 128, categories: Optional[Sequence[str]] = None,
               overlap: str = "error", mask_all: bool = False) -> BioEncoding:
    """
    Token-level BIO labels for one message.

    A token belongs to a span if its character offsets overlap the span. The first token of a span gets B, the others I (so a word split
    into several subwords is B then I, I), everything else O, and special tokens (and, with mask_all, every token) IGNORE_INDEX.
    categories=None gives the generic tags O/B-ASP/I-ASP; passing the category list gives B-<category>/I-<category>.
    overlap: 'error' raises SpanOverlapError if two spans share a token; 'merge' (generic tags only) treats overlapping spans as one.
    Spans cut off by truncation are reported in truncated_mentions; spans starting or ending inside a token in partial_boundary_mentions.
    """
    _require_fast(tokenizer)
    if overlap not in ("error", "merge"):
        raise ValueError("overlap must be 'error' or 'merge'")
    if overlap == "merge" and categories:
        raise ValueError("merging overlapping spans is only defined for the generic tag set")
    tagset = bio_tagset(categories)
    tag_id = {t: i for i, t in enumerate(tagset)}
    enc = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True, truncation=True, max_length=max_length)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    special = list(enc["special_tokens_mask"])
    real = [i for i, (sp, (s, e)) in enumerate(zip(special, offsets)) if not sp and e > s]
    last_end = max((offsets[i][1] for i in real), default=0)
    tags: List[Optional[str]] = [None if sp else "O" for sp in special]
    truncated, unaligned, partial = [], [], []
    covered_by = {}                                      # token index -> index of the span that owns it
    spans_tokens = []
    for k, m in enumerate(mentions):
        toks = [i for i in real if offsets[i][0] < m.end and offsets[i][1] > m.start]
        if not toks:
            (truncated if m.start >= last_end else unaligned).append(m)
            continue
        if not (_boundary_ok(text, offsets[toks[0]][0], m.start) and _boundary_ok(text, offsets[toks[-1]][1], m.end)):
            partial.append(m)
        spans_tokens.append((m, toks))
    if overlap == "merge":
        merged = []                                      # union of token sets that share tokens
        for m, toks in sorted(spans_tokens, key=lambda x: x[1][0]):
            if merged and set(toks) & set(merged[-1][1]):
                merged[-1] = (merged[-1][0], sorted(set(merged[-1][1]) | set(toks)))
            else:
                merged.append((m, toks))
        spans_tokens = merged
    for m, toks in spans_tokens:
        for n, i in enumerate(toks):
            if i in covered_by:
                raise SpanOverlapError(f"spans {covered_by[i].text!r} and {m.text!r} share a token; use overlap='merge' or fix the annotation")
            covered_by[i] = m
            suffix = f"-{m.category}" if categories else "-ASP"
            if f"B{suffix}" not in tag_id:
                raise ValueError(f"category {m.category!r} is not in the category list given for the tag set")
            tags[i] = ("B" if n == 0 else "I") + suffix
    labels = [IGNORE_INDEX if (t is None or mask_all) else tag_id[t] for t in tags]
    return BioEncoding(list(enc["input_ids"]), list(enc["attention_mask"]), offsets, special, tags, labels, tagset,
                       truncated, unaligned, partial)


@dataclass
class ExtractionExample:
    sequence_id: str
    sequence_type: str
    entity_id: str
    message_id: str
    position: int
    timestamp_ms: int
    language: str
    status: str
    text: str
    encoding: BioEncoding

    @property
    def category_order(self):
        return 0

    example_id = property(lambda self: self.message_id)


def build_extraction_examples(messages: Sequence[Message], tokenizer, *, max_length: int = 128, categories: Optional[Sequence[str]] = None,
                              unclear: str = "exclude", overlap: str = "error") -> Tuple[List[ExtractionExample], Dict[str, int]]:
    """
    One BIO example per message.
      ASPECT_PRESENT : spans become B/I tags.
      NO_ASPECT      : all tokens O (a real negative example).
      UNCLEAR_ASPECT : 'exclude' drops the message; 'mask' keeps it with every label IGNORE_INDEX. It is never all-O, because an
                       unclear aspect may well be present.
    Returns (examples in the order given, counts).
    """
    if unclear not in ("exclude", "mask"):
        raise ValueError("unclear must be 'exclude' or 'mask'")
    out, rep = [], defaultdict(int)
    for m in messages:
        rep["messages"] += 1
        if m.status == "UNCLEAR_ASPECT" and unclear == "exclude":
            rep["unclear_excluded"] += 1
            continue
        enc = bio_encode(m.text, m.mentions, tokenizer, max_length=max_length, categories=categories, overlap=overlap,
                         mask_all=m.status == "UNCLEAR_ASPECT")
        rep["unclear_masked"] += m.status == "UNCLEAR_ASPECT"
        rep["no_aspect_all_O"] += m.status == "NO_ASPECT"
        rep["mentions"] += len(m.mentions)
        rep["truncated_mentions"] += len(enc.truncated_mentions)
        rep["unaligned_mentions"] += len(enc.unaligned_mentions)
        rep["partial_boundary_mentions"] += len(enc.partial_boundary_mentions)
        out.append(ExtractionExample(m.sequence_id, m.sequence_type, m.entity_id, m.message_id, m.position, m.timestamp_ms, m.language,
                                     m.status, m.text, enc))
        rep["examples"] += 1
    return out, dict(rep)


# ───────────────────────────── aspect-conditioned sentiment ─────────────────────────────
def aggregate_polarity(labels: Sequence[str]) -> str:
    """
    One value for several mentions of the same category in one message (guidelines section 11).
    All NEUTRAL gives NEUTRAL. Otherwise NEUTRAL mentions are ignored, and the value is MIXED if any mention is MIXED or both POSITIVE and
    NEGATIVE occur, else the single polarity.
    """
    names = [AspectSentiment.from_label(x).name for x in labels]
    if not names:
        raise ValueError("no sentiment labels to aggregate")
    rest = {n for n in names if n != "NEUTRAL"}
    if not rest:
        return "NEUTRAL"
    if "MIXED" in rest or rest >= {"POSITIVE", "NEGATIVE"}:
        return "MIXED"
    return next(iter(rest))


@dataclass
class SentimentExample:
    example_id: str
    sequence_id: str
    sequence_type: str
    entity_id: str
    message_id: str
    position: int
    timestamp_ms: int
    language: str
    category: str
    category_order: int
    text: str
    spans: Tuple[Tuple[int, int], ...]
    span_texts: Tuple[str, ...]
    label: str                                           # an AspectSentiment name
    label_id: int
    mention_labels: Tuple[str, ...]
    level: str


def category_phrase(category: str) -> str:
    return category.lower().replace("_", " ")


def build_sentiment_examples(messages: Sequence[Message], *, level: str = "message_category") -> List[SentimentExample]:
    """
    Aspect-conditioned sentiment examples from finished messages.
      level='message_category' : one example per (message, category), label = aggregate_polarity of that category's mentions; this is
                                 the unit later per-aspect timelines use.
      level='mention'          : one example per mention with its own label.
    Several aspects in one review give several examples, each with its own label (praise for one and criticism of another is never
    MIXED). Messages with no aspect or an unclear aspect give no example.
    """
    if level not in ("message_category", "mention"):
        raise ValueError("level must be 'message_category' or 'mention'")
    out = []
    domain_cats = {}
    for m in messages:
        if m.status != "ASPECT_PRESENT":
            continue
        cats = domain_cats.setdefault(m.domain, category_ids(m.domain))
        if level == "mention":
            groups = [((x.category,), [x]) for x in m.mentions]
        else:
            by_cat = OrderedDict()
            for x in m.mentions:
                by_cat.setdefault(x.category, []).append(x)
            groups = [((c,), xs) for c, xs in sorted(by_cat.items(), key=lambda kv: cats.index(kv[0]))]
        for (cat,), xs in groups:
            labels = tuple(x.sentiment for x in xs)
            label = labels[0] if level == "mention" else aggregate_polarity(labels)
            eid = f"{m.message_id}|{cat}" if level == "message_category" else f"{m.message_id}|{xs[0].start}-{xs[0].end}|{cat}"
            out.append(SentimentExample(eid, m.sequence_id, m.sequence_type, m.entity_id, m.message_id, m.position, m.timestamp_ms,
                                        m.language, cat, cats.index(cat), m.text, tuple((x.start, x.end) for x in xs),
                                        tuple(x.text for x in xs), label, AspectSentiment[label].value, labels, level))
    return out


def conditioning_text(ex: SentimentExample, conditioning: str = "category") -> str:
    """The aspect side of the sentence pair: the category name, the annotated span text(s), or both."""
    spans = " | ".join(ex.span_texts)
    if conditioning == "category":
        return category_phrase(ex.category)
    if conditioning == "span":
        return spans
    if conditioning == "category_and_span":
        return f"{category_phrase(ex.category)}: {spans}"
    raise ValueError("conditioning must be 'category', 'span' or 'category_and_span'")


def encode_sentiment_examples(examples: Sequence[SentimentExample], tokenizer, *, conditioning: str = "category", max_length: int = 128) -> List[dict]:
    """
    Sentence-pair encodings (aspect text, review text) for mBERT or XLM-R. Only the review text is truncated, so the aspect is never
    cut. Returns plain dicts: input_ids, attention_mask, token_type_ids (when the tokenizer has them), label, example_id.
    """
    _require_fast(tokenizer)
    out = []
    for ex in examples:
        enc = tokenizer(conditioning_text(ex, conditioning), ex.text, truncation="only_second", max_length=max_length)
        d = {k: list(enc[k]) for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
        d.update(label=ex.label_id, example_id=ex.example_id)
        out.append(d)
    return out


def category_series(sequence_examples: Sequence[SentimentExample]) -> Dict[str, List[dict]]:
    """For one sequence in chronological order: category -> [{message_id, position, timestamp_ms, label}], keeping time order."""
    out = defaultdict(list)
    for ex in sorted(sequence_examples, key=_order_key):
        out[ex.category].append({"message_id": ex.message_id, "position": ex.position, "timestamp_ms": ex.timestamp_ms, "label": ex.label})
    return dict(out)


# ───────────────────────────── leakage-safe splits ─────────────────────────────
def _split_sizes(n: int, ratios: Sequence[float]) -> List[int]:
    positive = [i for i, r in enumerate(ratios) if r > 0]
    if n < len(positive):
        raise ValueError(f"{n} entities cannot fill {len(positive)} non-empty splits")
    raw = [n * r for r in ratios]
    sizes = [max(1, int(np.floor(x))) if r > 0 else 0 for x, r in zip(raw, ratios)]
    while sum(sizes) < n:                                # give leftovers to the largest fractional parts
        i = max(positive, key=lambda j: (raw[j] - sizes[j], -j))
        sizes[i] += 1
    while sum(sizes) > n:                                # the 1-per-split minimum overshot: take from the largest split
        i = max((j for j in positive if sizes[j] > 1), key=lambda j: (sizes[j], -j))
        sizes[i] -= 1
    return sizes


def split_entities(entity_ids: Iterable[str], ratios: Sequence[float] = (0.7, 0.15, 0.15), seed: int = 0,
                   names: Sequence[str] = ("train", "val", "test")) -> Dict[str, List[str]]:
    """
    Assign whole entities (products) to splits. Deterministic: depends only on the set of ids, the ratios and the seed, not on the order or
    number of items. Every split with a positive ratio gets at least one entity.
    """
    if len(ratios) != len(names) or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must be non-negative, match the split names and sum to 1")
    ids = sorted(set(entity_ids))
    if any(e == "" or e is None for e in ids):
        raise ValueError("every item needs a non-empty entity_id; without it a leakage-safe split is impossible")
    order = np.random.default_rng(seed).permutation(len(ids))
    shuffled = [ids[i] for i in order]
    out, start = {}, 0
    for name, size in zip(names, _split_sizes(len(ids), ratios)):
        out[name] = sorted(shuffled[start:start + size])
        start += size
    return out


def split_by_entity(items: Sequence, ratios: Sequence[float] = (0.7, 0.15, 0.15), seed: int = 0,
                    names: Sequence[str] = ("train", "val", "test"), entity_of: Callable = lambda x: x.entity_id) -> Dict[str, list]:
    """Split items (messages, examples, sequences) so that all items of one entity land in the same split. Item order is kept."""
    items = list(items)
    assign = {e: n for n, es in split_entities([entity_of(i) for i in items], ratios, seed, names).items() for e in es}
    out = {n: [] for n in names}
    for it in items:
        out[assign[entity_of(it)]].append(it)
    assert_no_entity_leakage(out, entity_of)
    return out


def assert_no_entity_leakage(splits: Dict[str, list], entity_of: Callable = lambda x: x.entity_id) -> None:
    """Raises ValueError if any entity appears in more than one split."""
    seen = {}
    for name, items in splits.items():
        for it in items:
            e = entity_of(it)
            if seen.setdefault(e, name) != name:
                raise ValueError(f"entity {e!r} appears in both {seen[e]!r} and {name!r}")


def cross_split_duplicate_texts(splits: Dict[str, list], text_of: Callable = lambda x: x.text) -> Dict[str, List[str]]:
    """
    Texts that occur (ignoring case and whitespace) in more than one split, with the splits they occur in. Entity-disjoint splits can
    still leak through identical review texts posted for different products; this reports them. It does not remove anything.
    """
    where = defaultdict(set)
    for name, items in splits.items():
        for it in items:
            where[" ".join(text_of(it).lower().split())].add(name)
    return {t: sorted(s) for t, s in where.items() if len(s) > 1 and t}
