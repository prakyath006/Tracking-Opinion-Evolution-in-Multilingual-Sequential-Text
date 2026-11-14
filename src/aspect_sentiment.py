"""
Aspect-associated sentiment projection (rule-based).

WHAT THIS IS
    OET predicts ONE sentiment per message. The WSD component tags aspect words in a message.
    This module attaches the message-level sentiment to every aspect detected in that message
    and aggregates those observations per aspect over the chronological sequence.

WHAT THIS IS NOT
    It is not aspect-specific sentiment classification. "Great music, weak story" predicted
    POSITIVE yields a POSITIVE observation for both music and story. No model here was trained
    or evaluated on aspect-level sentiment labels.

Pure functions, no model and no file access; labels are the ontology's SentimentState names.
"""
from typing import Dict, List, Optional, Sequence

from ontology import SentimentState, TrajectoryType

METHOD = "aspect-associated sentiment projection (rule-based)"
LIMITATION = (
    "The sentiment shown for an aspect is the sentiment predicted for the whole message in which the "
    "aspect was mentioned. It is not a sentiment expressed specifically toward that aspect, and no "
    "aspect-level sentiment model was trained or evaluated."
)
INSUFFICIENT = "insufficient observations for trend"

_DETERMINATE = ("POSITIVE", "NEGATIVE", "MIXED")           # UNKNOWN carries no ordinal position (ontology.py)
_VALID = set(SentimentState.label_names())


def tag_messages(messages: Sequence[str], wsd) -> List[List[Dict]]:
    """
    Run the WSD component on each message and return one list of tag dicts per message.

    A tag is unresolved when the surface form has several aspect senses and the context window did not
    give one of them a majority (overlap score <= 0.5). This is the same threshold as
    WordSenseDisambiguator.get_coverage_stats(). For those, wsd.process() still returns a candidate
    (an arbitrary one); it is dropped here so that no aspect is invented.
    """
    out = []
    for text in messages:
        tags = []
        for word, aspect, score in wsd.process(text or ""):
            resolved = aspect != "unknown" and not (wsd.lexicon.is_ambiguous(word) and score <= 0.5)
            tags.append({
                "word": word,
                "aspect": aspect if resolved else None,
                "resolved": resolved,
                "wsd_overlap_score": float(score),         # share of context overlap, not a probability
                "candidates": sorted({a for a, _ in wsd.lexicon.lookup(word)}),
            })
        out.append(tags)
    return out


def _as_tag(tag) -> Dict:
    """Accept the dicts from tag_messages() or the raw (word, aspect, score) tuples of wsd.process()."""
    if isinstance(tag, dict):
        aspect = tag.get("aspect")
        resolved = tag.get("resolved", True) and aspect not in (None, "", "unknown")
        return {"word": tag.get("word"), "aspect": aspect if resolved else None, "resolved": resolved,
                "wsd_overlap_score": tag.get("wsd_overlap_score"), "candidates": tag.get("candidates", [])}
    word, aspect, score = tag
    resolved = aspect not in (None, "", "unknown")
    return {"word": word, "aspect": aspect if resolved else None, "resolved": resolved,
            "wsd_overlap_score": float(score), "candidates": []}


def associated_trend(sentiments: Sequence[str]) -> Optional[str]:
    """
    Trend of an aspect's associated sentiments, in chronological order.

    Only POSITIVE/NEGATIVE/MIXED observations are used. With fewer than two of them there is no
    transition to look at and None is returned. Otherwise the ontology's own trajectory rule is applied
    (TrajectoryType.compute: ordinal rank NEGATIVE < MIXED < POSITIVE, each consecutive pair is an
    UPGRADE, DOWNGRADE or STABLE step, and the step counts decide IMPROVING / DECLINING / STABLE /
    VOLATILE). Examples: POS, MIX, NEG -> DECLINING; NEG, POS -> IMPROVING; POS, POS, POS -> STABLE.
    """
    states = [SentimentState[s] for s in sentiments if s in _DETERMINATE]
    if len(states) < 2:
        return None
    return TrajectoryType.compute(states).name


def aspect_sentiment_projection(messages: Sequence[str], sentiments: Optional[Sequence[Optional[str]]],
                                aspect_tags: Optional[Sequence[Sequence]],
                                sentiment_confidences: Optional[Sequence[Optional[float]]] = None) -> Dict:
    """
    messages               chronological texts
    sentiments             one ontology label name per message (None = no prediction for that message);
                           None for the whole argument = no sentiment model was run
    aspect_tags            per message, the tags from tag_messages() or wsd.process(); None = WSD not run
    sentiment_confidences  optional per-message model score (e.g. softmax of the predicted class);
                           passed through unchanged, never combined with WSD scores
    """
    n = len(messages)
    sentiments = list(sentiments) if sentiments is not None else [None] * n
    aspect_tags = list(aspect_tags) if aspect_tags is not None else [[] for _ in range(n)]
    if len(sentiments) != n or len(aspect_tags) != n:
        raise ValueError("messages, sentiments and aspect_tags must have the same length")
    if sentiment_confidences is not None and len(sentiment_confidences) != n:
        raise ValueError("sentiment_confidences must have one entry per message")
    for s in sentiments:
        if s is not None and s not in _VALID:
            raise ValueError(f"invalid sentiment label {s!r}; expected one of {sorted(_VALID)}")

    aspects: Dict[str, Dict] = {}
    unresolved, timeline = [], []
    for i in range(n):
        per_message: Dict[str, Dict] = {}                   # one observation per (aspect, message)
        for tag in map(_as_tag, aspect_tags[i] or []):
            if not tag["resolved"]:
                unresolved.append({"message_index": i + 1, "word": tag["word"], "candidates": tag["candidates"]})
                continue
            obs = per_message.setdefault(tag["aspect"], {"message_index": i + 1, "sentiment": sentiments[i],
                                                         "words": [], "wsd_overlap_scores": []})
            obs["words"].append(tag["word"])
            obs["wsd_overlap_scores"].append(tag["wsd_overlap_score"])
            if sentiment_confidences is not None:
                obs["sentiment_confidence"] = sentiment_confidences[i]
        for aspect, obs in per_message.items():
            aspects.setdefault(aspect, {"aspect": aspect, "observations": []})["observations"].append(obs)
        timeline.append({"message_index": i + 1, "sentiment": sentiments[i], "aspects": sorted(per_message)})

    for a in aspects.values():
        obs = a["observations"]
        labels = [o["sentiment"] for o in obs if o["sentiment"] is not None]
        a["occurrences"] = sum(len(o["words"]) for o in obs)          # aspect words found
        a["n_messages"] = len(obs)                                    # messages mentioning the aspect
        a["counts"] = {name: labels.count(name) for name in SentimentState.label_names()}
        det = {k: a["counts"][k] for k in _DETERMINATE}
        top = max(det.values()) if det else 0
        leaders = [k for k, v in det.items() if v == top and v > 0]
        # Tendency = the single most frequent determinate label; a tie or no label gives no tendency.
        a["tendency"] = leaders[0] if len(leaders) == 1 else None
        a["tendency_share"] = round(top / sum(det.values()), 4) if len(leaders) == 1 else None
        a["trend"] = associated_trend(labels)
        a["trend_note"] = None if a["trend"] else INSUFFICIENT
        a["evidence_coverage"] = round(len(obs) / n, 4) if n else 0.0  # share of messages mentioning it

    with_aspect = sum(1 for t in timeline if t["aspects"])
    ordered = dict(sorted(aspects.items(), key=lambda kv: (-kv[1]["n_messages"], -kv[1]["occurrences"], kv[0])))
    return {
        "method": METHOD,
        "limitation": LIMITATION,
        "n_messages": n,
        "n_messages_with_aspect": with_aspect,
        "n_messages_without_aspect": n - with_aspect,
        "message_coverage": round(with_aspect / n, 4) if n else 0.0,
        "sentiment_available": any(s is not None for s in sentiments),
        "aspects": ordered,
        "unresolved": unresolved,
        "timeline": timeline,
    }
