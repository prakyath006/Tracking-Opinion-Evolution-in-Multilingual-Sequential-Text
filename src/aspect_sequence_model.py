"""
Aspect-level sequence model: how the opinion about ONE aspect of ONE product moves over time.

Component 3 of 3 of the aspect-level architecture (independent of the extractor and the sentiment model; the existing OET model is not used,
though its BiLSTMEncoder, SelfAttention and ClassificationHead layers are reused).

Unit of modelling: a SERIES = all events of one aspect category of one product, in time order. Series of different aspects, and of different
sequence types, are never merged: `build_series` groups by (sequence_type, entity_id, category), `collate_series` re-checks it, and the
model treats every series of a batch independently.

An event is one review's mention of the aspect. Its model INPUTS are
    * the class probabilities PREDICTED by the aspect sentiment model (4 values; never gold labels),
    * time-gap features: log(1 + days since the previous event of the series) and log(1 + days since its first event),
    * the aspect category (a learned embedding, identity only) and optionally a text vector from the sentiment model.
Gold sentiment labels are used ONLY to build the training targets (transition and trajectory), through `series_targets`; `forward` has no
parameter that can carry them.

    [probs | time | category] --BiLSTM--> state per event --+--> transition head  (per event: UPGRADE / DOWNGRADE / STABLE)
                                                            +--> temporal attention --> trajectory head (per series: IMPROVING / DECLINING / STABLE / VOLATILE)

Targets follow the guidelines (section 11): only POSITIVE, MIXED and NEGATIVE are ranked (NEGATIVE < MIXED < POSITIVE, via the ontology's
TransitionType and TrajectoryType). NEUTRAL events are skipped: their own target is -100, a transition compares an event with the previous
ranked one, and a trajectory needs at least two ranked events (otherwise -100, "insufficient observations").

NOT TRAINED: nothing here has been trained, and no performance is claimed.
"""
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aspect_data import AspectSentiment, SentimentExample                                  # noqa: E402
from aspect_encoders import count_parameters                                                # noqa: E402
from attention import SelfAttention                                                         # noqa: E402  (existing, reused)
from bilstm import BiLSTMEncoder                                                            # noqa: E402  (existing, reused)
from classifier import ClassificationHead                                                   # noqa: E402  (existing, reused)
from ontology import SentimentState, TrajectoryType, TransitionType                         # noqa: E402

IGNORE = -100
NUM_SENTIMENT = len(AspectSentiment)
TIME_DIM = 2
TRANSITIONS = TransitionType.label_names()
TRAJECTORIES = TrajectoryType.label_names()
_MS_PER_DAY = 86_400_000.0
_NOT_PREDICTED = {"gold", "label", "labels", "annotation", "annotations", "ground_truth", "truth"}


class LeakageError(ValueError):
    """Gold labels were about to be used as model input."""


# ───────────────────────────── events and series ─────────────────────────────
@dataclass
class AspectEvent:
    sequence_type: str
    entity_id: str
    category: str
    message_id: str
    position: int
    timestamp_ms: int
    sentiment_probs: Sequence[float]                      # PREDICTED by the sentiment model; the only sentiment information the model sees
    gold_label: Optional[str] = None                      # target construction only; never an input
    text_features: Optional[Sequence[float]] = None


@dataclass
class AspectSeries:
    key: Tuple[str, str, str]                             # (sequence_type, entity_id, category)
    events: List[AspectEvent] = field(default_factory=list)


def events_from_examples(examples: Sequence[SentimentExample], probs: Dict[str, Sequence[float]], *, source: str) -> List[AspectEvent]:
    """
    Events from aspect-sentiment examples (level='message_category') and the probabilities the sentiment MODEL predicted for them.
    `source` names the predictor (for example "aspect_sentiment_model fold 2") and may not say gold or labels. Every example needs an entry in
    `probs`. The example's gold label is carried only as the target. As a best-effort guard it raises LeakageError if every probability vector is
    exactly the one-hot of its own gold label, which is what feeding gold labels in would look like.
    """
    if not source or not source.strip() or source.strip().lower() in _NOT_PREDICTED:
        raise LeakageError(f"source {source!r} does not name a predictor; sentiment inputs must be model predictions, never gold labels")
    events = []
    for ex in examples:
        if ex.level != "message_category":
            raise ValueError("aspect series need level='message_category' examples (one value per message and category)")
        if ex.example_id not in probs:
            raise KeyError(f"no predicted probabilities for example {ex.example_id!r}")
        p = [float(x) for x in probs[ex.example_id]]
        if len(p) != NUM_SENTIMENT or any((not math.isfinite(x)) or x < 0 for x in p) or abs(sum(p) - 1.0) > 1e-3:
            raise ValueError(f"example {ex.example_id!r}: probabilities must be {NUM_SENTIMENT} non-negative numbers summing to 1, got {p}")
        events.append(AspectEvent(ex.sequence_type, ex.entity_id, ex.category, ex.message_id, ex.position, ex.timestamp_ms, p, ex.label))
    if events and all(max(abs(a - (1.0 if i == AspectSentiment[e.gold_label].value else 0.0)) for i, a in enumerate(e.sentiment_probs)) < 1e-9
                      for e in events):
        raise LeakageError("every probability vector equals the one-hot of its own gold label: gold labels appear to be used as input")
    return events


def build_series(events: Sequence[AspectEvent]) -> List[AspectSeries]:
    """
    One series per (sequence_type, entity_id, category), each in chronological order (timestamp, position, message id).
    Different aspects of a product give different series; an aspect that never occurs gives no series.
    """
    groups: Dict[Tuple[str, str, str], List[AspectEvent]] = {}
    for e in events:
        groups.setdefault((e.sequence_type, e.entity_id, e.category), []).append(e)
    return [AspectSeries(k, sorted(v, key=lambda e: (e.timestamp_ms, e.position, e.message_id))) for k, v in sorted(groups.items())]


def time_features(timestamps_ms: Sequence[int]) -> np.ndarray:
    """[n, 2]: log(1 + days since the previous event) and log(1 + days since the first event). Timestamps must not decrease."""
    t = np.asarray(timestamps_ms, dtype=np.float64)
    if len(t) and (np.diff(t) < 0).any():
        raise ValueError("timestamps must be in chronological order")
    gap = np.concatenate([[0.0], np.diff(t)]) / _MS_PER_DAY if len(t) else t
    since = (t - t[0]) / _MS_PER_DAY if len(t) else t
    return np.stack([np.log1p(gap), np.log1p(since)], axis=1).astype(np.float32)


_RANKED = {"POSITIVE", "NEGATIVE", "MIXED"}


def series_targets(series: AspectSeries) -> Tuple[List[int], int]:
    """
    Gold targets from the events' gold labels (None labels, e.g. at inference, give all -100).
    transitions[i]: the ontology transition from the previous RANKED event to event i, or -100 (first ranked event, NEUTRAL event, no labels).
    trajectory: the ontology trajectory of the ranked events, or -100 with fewer than two of them.
    """
    n = len(series.events)
    transitions = [IGNORE] * n
    ranked = [(i, SentimentState[e.gold_label]) for i, e in enumerate(series.events) if e.gold_label in _RANKED]
    for (_, prev), (i, cur) in zip(ranked, ranked[1:]):
        transitions[i] = TransitionType.compute(prev, cur).value
    trajectory = TrajectoryType.compute([s for _, s in ranked]).value if len(ranked) >= 2 else IGNORE
    return transitions, trajectory


def collate_series(series_list: Sequence[AspectSeries], categories: Sequence[str]) -> Dict[str, object]:
    """
    Pad series into tensors. Keys: sentiment_probs [B,T,4], time_features [B,T,2], seq_lens [B], category_ids [B], text_features [B,T,D] if
    present (these go into the model), and transition_targets [B,T], trajectory_targets [B] (-100 on padding and where undefined; used by the
    loss only). Re-checks that each series holds one product and one aspect category in chronological order.
    """
    if not series_list:
        raise ValueError("no series")
    cat_id = {c: i for i, c in enumerate(categories)}
    B, T = len(series_list), max(len(s.events) for s in series_list)
    probs = torch.zeros(B, T, NUM_SENTIMENT)
    times = torch.zeros(B, T, TIME_DIM)
    trans = torch.full((B, T), IGNORE, dtype=torch.long)
    traj = torch.full((B,), IGNORE, dtype=torch.long)
    lens, cats, keys = [], [], []
    dim = {len(e.text_features) for s in series_list for e in s.events if e.text_features is not None}
    text = torch.zeros(B, T, next(iter(dim))) if len(dim) == 1 else None
    if len(dim) > 1 or (dim and any(e.text_features is None for s in series_list for e in s.events)):
        raise ValueError("text_features must be present with one size on every event, or on none")
    for b, s in enumerate(series_list):
        if not s.events:
            raise ValueError(f"series {s.key} has no events")
        if {(e.sequence_type, e.entity_id, e.category) for e in s.events} != {s.key}:
            raise ValueError(f"series {s.key} mixes products, aspects or sequence types")
        if s.key[2] not in cat_id:
            raise ValueError(f"unknown aspect category {s.key[2]!r}")
        n = len(s.events)
        probs[b, :n] = torch.tensor([list(map(float, e.sentiment_probs)) for e in s.events])
        times[b, :n] = torch.from_numpy(time_features([e.timestamp_ms for e in s.events]))
        if text is not None:
            text[b, :n] = torch.tensor([list(map(float, e.text_features)) for e in s.events])
        t, j = series_targets(s)
        trans[b, :n] = torch.tensor(t)
        traj[b] = j
        lens.append(n); cats.append(cat_id[s.key[2]]); keys.append(s.key)
    out = {"sentiment_probs": probs, "time_features": times, "seq_lens": torch.tensor(lens), "category_ids": torch.tensor(cats),
           "transition_targets": trans, "trajectory_targets": traj, "keys": keys,
           "message_ids": [[e.message_id for e in s.events] for s in series_list]}
    if text is not None:
        out["text_features"] = text
    return out


# ───────────────────────────── model ─────────────────────────────
class AspectSequenceModel(nn.Module):
    """
    BiLSTM over the events of one aspect series, temporal (additive) attention, and two heads. See the module docstring.
    num_categories : size of the category list used by collate_series (aspect_data.category_ids(domain)).
    text_dim       : size of optional per-event text vectors (0 = none).
    """

    def __init__(self, num_categories: int, *, category_dim: int = 8, text_dim: int = 0, hidden_dim: int = 128, lstm_layers: int = 2,
                 attention_dim: int = 64, head_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.num_categories, self.text_dim, self.category_dim = num_categories, text_dim, category_dim
        self.category_embedding = nn.Embedding(num_categories, category_dim) if category_dim > 0 else None
        in_dim = NUM_SENTIMENT + TIME_DIM + category_dim + text_dim
        self.bilstm = BiLSTMEncoder(in_dim, hidden_dim, lstm_layers, dropout, bidirectional=True)
        out_dim = self.bilstm.get_output_dim()
        self.attention = SelfAttention(out_dim, attention_dim, dropout=0.1)
        self.transition_head = ClassificationHead(out_dim, head_hidden, len(TRANSITIONS), dropout)
        self.trajectory_head = ClassificationHead(out_dim, head_hidden, len(TRAJECTORIES), dropout)

    def forward(self, sentiment_probs: torch.Tensor, time_features: torch.Tensor, seq_lens: torch.Tensor, category_ids: torch.Tensor,
                text_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        sentiment_probs [B,T,4] (predicted), time_features [B,T,2], seq_lens [B], category_ids [B], text_features [B,T,text_dim].
        Returns transition_logits [B,T,3] (meaningful at positions after the first and inside seq_lens), trajectory_logits [B,4],
        attention_weights [B,T] (0 on padding). Every series in the batch is processed independently.
        """
        B, T, _ = sentiment_probs.shape
        if sentiment_probs.size(-1) != NUM_SENTIMENT or time_features.shape != (B, T, TIME_DIM):
            raise ValueError("unexpected input shapes")
        parts = [sentiment_probs, time_features]
        if self.category_embedding is not None:
            parts.append(self.category_embedding(category_ids).unsqueeze(1).expand(-1, T, -1))
        if self.text_dim:
            if text_features is None or text_features.size(-1) != self.text_dim:
                raise ValueError(f"text_features of size {self.text_dim} are required")
            parts.append(text_features)
        lens = seq_lens.to("cpu").tolist()
        mask = torch.arange(T, device=sentiment_probs.device).unsqueeze(0) < seq_lens.to(sentiment_probs.device).unsqueeze(1)
        hidden, _ = self.bilstm(torch.cat(parts, dim=-1), lens)
        context, weights = self.attention(hidden, mask)
        return {"transition_logits": self.transition_head(hidden), "trajectory_logits": self.trajectory_head(context),
                "attention_weights": weights}

    def parameter_counts(self) -> Dict[str, int]:
        return count_parameters(self)


class AspectSequenceLoss(nn.Module):
    """Weighted sum of the transition and trajectory cross-entropies; -100 targets are ignored and an all-ignored batch gives zero, not NaN."""

    def __init__(self, transition_weight: float = 0.5, trajectory_weight: float = 1.0, transition_class_weights: Optional[torch.Tensor] = None,
                 trajectory_class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.wt, self.wj = transition_weight, trajectory_weight
        self.register_buffer("tw", transition_class_weights if transition_class_weights is not None else None)
        self.register_buffer("jw", trajectory_class_weights if trajectory_class_weights is not None else None)

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        tl, jl = out["transition_logits"], out["trajectory_logits"]
        tt, jt = batch["transition_targets"].to(tl.device), batch["trajectory_targets"].to(tl.device)
        zero = tl.sum() * 0.0
        t = F.cross_entropy(tl.reshape(-1, tl.size(-1)), tt.reshape(-1), weight=self.tw, ignore_index=IGNORE) if bool((tt != IGNORE).any()) else zero
        j = F.cross_entropy(jl, jt, weight=self.jw, ignore_index=IGNORE) if bool((jt != IGNORE).any()) else zero
        return {"total_loss": self.wt * t + self.wj * j, "transition_loss": t, "trajectory_loss": j}


@torch.no_grad()
def decode_predictions(out: Dict[str, torch.Tensor], seq_lens: Sequence[int]) -> List[dict]:
    """Per series: transition names per event (None for the first event, which has no predecessor), trajectory name, attention weights."""
    tr = out["transition_logits"].argmax(-1).tolist()
    tj = out["trajectory_logits"].argmax(-1).tolist()
    att = out["attention_weights"].tolist()
    return [{"transitions": [None] + [TRANSITIONS[i] for i in tr[b][1:n]], "trajectory": TRAJECTORIES[tj[b]], "attention": att[b][:n]}
            for b, n in enumerate(seq_lens)]
