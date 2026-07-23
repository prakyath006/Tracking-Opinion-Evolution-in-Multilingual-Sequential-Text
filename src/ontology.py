"""
=============================================================================
Module 1a — Structural Ontology for Opinion Evolution
=============================================================================
Defines the formal structural ontology for tracking opinion evolution
in multilingual sequential text. This ontology provides:

  1. Sentiment State Taxonomy — All possible opinion states
  2. Transition Taxonomy     — Valid transitions between states
  3. Trajectory Taxonomy     — Sequence-level evolution patterns
  4. Code-Mix Taxonomy       — Linguistic categories for code-mixed text
  5. Domain Ontology         — Domain-specific entity structures

The ontology serves as the knowledge backbone that guides the embedding
layer, the functional layer, and the evaluation framework.

Reference: Guide Module 1 (Structural Ontology & Own Embeddings)
=============================================================================
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Sentiment State Taxonomy
# ──────────────────────────────────────────────────────────────────────────────

class SentimentState(Enum):
    """
    Defines all possible sentiment/opinion states in the ontology.
    These states apply uniformly across domains and languages.
    """
    POSITIVE = 0
    NEGATIVE = 1
    MIXED = 2
    UNKNOWN = 3

    @classmethod
    def from_label(cls, label: str) -> "SentimentState":
        """Map raw dataset labels to ontology states."""
        label_lower = label.lower().strip()
        mapping = {
            "positive": cls.POSITIVE,
            "negative": cls.NEGATIVE,
            "mixed_feelings": cls.MIXED,
            "mixed feelings": cls.MIXED,
            "unknown_state": cls.UNKNOWN,
            "unknown state": cls.UNKNOWN,
            "not-tamil": cls.UNKNOWN,
            "not-malayalam": cls.UNKNOWN,
            "not-kannada": cls.UNKNOWN,
        }
        return mapping.get(label_lower, cls.UNKNOWN)

    @classmethod
    def from_rating(cls, rating: float) -> "SentimentState":
        """Map numeric ratings (1-5) to ontology states."""
        if rating >= 4.0:
            return cls.POSITIVE
        elif rating <= 2.0:
            return cls.NEGATIVE
        else:
            return cls.MIXED

    @classmethod
    def num_classes(cls) -> int:
        return len(cls)

    @classmethod
    def label_names(cls) -> List[str]:
        return [s.name for s in cls]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Transition Taxonomy
# ──────────────────────────────────────────────────────────────────────────────

class TransitionType(Enum):
    """
    Defines valid pairwise transitions between consecutive sentiment states.
    Used to track how opinion changes from one text to the next in a sequence.
    """
    UPGRADE = 0       # Sentiment improved (e.g., Negative → Positive)
    DOWNGRADE = 1     # Sentiment declined (e.g., Positive → Negative)
    STABLE = 2        # Sentiment stayed the same

    @classmethod
    def compute(cls, prev: SentimentState, curr: SentimentState) -> "TransitionType":
        """Determine the transition type between two consecutive states."""
        if prev == curr:
            return cls.STABLE
        # Positive > Mixed > Negative > Unknown (ordering)
        order = {
            SentimentState.POSITIVE: 3,
            SentimentState.MIXED: 2,
            SentimentState.NEGATIVE: 1,
            SentimentState.UNKNOWN: 0,
        }
        if order[curr] > order[prev]:
            return cls.UPGRADE
        else:
            return cls.DOWNGRADE

    @classmethod
    def num_classes(cls) -> int:
        return len(cls)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Trajectory Taxonomy
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryType(Enum):
    """
    Defines sequence-level opinion evolution patterns.
    Computed over an entire sequence of sentiment states.
    """
    IMPROVING = 0     # Overall trend is positive (e.g., Neg → Mixed → Pos)
    DECLINING = 1     # Overall trend is negative (e.g., Pos → Mixed → Neg)
    STABLE = 2        # Sentiment stays consistent throughout
    VOLATILE = 3      # Frequent oscillations with no clear trend

    @classmethod
    def compute(cls, states: List[SentimentState]) -> "TrajectoryType":
        """
        Determine the trajectory type for a sequence of sentiment states.
        Uses transition counting to classify the overall pattern.
        """
        if len(states) < 2:
            return cls.STABLE

        upgrades = 0
        downgrades = 0
        stables = 0

        for i in range(1, len(states)):
            t = TransitionType.compute(states[i - 1], states[i])
            if t == TransitionType.UPGRADE:
                upgrades += 1
            elif t == TransitionType.DOWNGRADE:
                downgrades += 1
            else:
                stables += 1

        total_changes = upgrades + downgrades
        total_transitions = len(states) - 1

        # Mostly stable
        if stables >= total_transitions * 0.7:
            return cls.STABLE
        # Both upgrades and downgrades → volatile
        if upgrades > 0 and downgrades > 0:
            change_ratio = min(upgrades, downgrades) / max(upgrades, downgrades)
            if change_ratio > 0.4:
                return cls.VOLATILE
        # Mostly upgrades
        if upgrades > downgrades:
            return cls.IMPROVING
        # Mostly downgrades
        return cls.DECLINING

    @classmethod
    def num_classes(cls) -> int:
        return len(cls)

    @classmethod
    def label_names(cls) -> List[str]:
        return [t.name for t in cls]


# ──────────────────────────────────────────────────────────────────────────────
# 4. Code-Mix Taxonomy
# ──────────────────────────────────────────────────────────────────────────────

class ScriptType(Enum):
    """Script categories for code-mixed text."""
    LATIN = "latin"           # English / romanized Dravidian
    TAMIL = "tamil"           # Tamil script
    MALAYALAM = "malayalam"   # Malayalam script
    KANNADA = "kannada"       # Kannada script
    MIXED = "mixed"           # Multiple scripts in one text
    OTHER = "other"           # Emojis, numbers, symbols


@dataclass
class CodeMixProfile:
    """
    Represents the code-mixing characteristics of a text sample.
    Code-Mix Index (CMI) measures the degree of language mixing.
    """
    dominant_script: ScriptType
    code_mix_index: float = 0.0  # 0 = monolingual, 100 = fully mixed
    scripts_present: List[ScriptType] = field(default_factory=list)
    is_code_mixed: bool = False

    @staticmethod
    def compute_cmi(native_chars: int, foreign_chars: int) -> float:
        """
        Compute Code-Mix Index.
        CMI = (foreign_chars / total_chars) * 100
        """
        total = native_chars + foreign_chars
        if total == 0:
            return 0.0
        return (foreign_chars / total) * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# 5. Domain Ontology
# ──────────────────────────────────────────────────────────────────────────────

class DomainType(Enum):
    """Supported domains for opinion tracking."""
    ECOMMERCE = "ecommerce"       # Amazon product reviews
    SOCIAL_MEDIA = "social_media" # YouTube/social media comments


@dataclass
class DomainConfig:
    """
    Configuration for a specific domain within the ontology.
    Defines how data from each domain maps to the universal sentiment states.
    """
    domain_type: DomainType
    languages: List[str]
    has_sequential_data: bool
    sequence_source: str  # e.g., "user_timeline", "sliding_window"
    label_mapping: Dict[str, SentimentState] = field(default_factory=dict)


# Pre-defined domain configurations
DOMAIN_CONFIGS = {
    "amazon_beauty": DomainConfig(
        domain_type=DomainType.ECOMMERCE,
        languages=["english"],
        has_sequential_data=True,
        sequence_source="user_timeline",
        label_mapping={
            "positive": SentimentState.POSITIVE,
            "negative": SentimentState.NEGATIVE,
            "neutral": SentimentState.MIXED,
        },
    ),
    "dravidian_tamil": DomainConfig(
        domain_type=DomainType.SOCIAL_MEDIA,
        languages=["tamil", "english"],
        has_sequential_data=False,
        sequence_source="sliding_window",
        label_mapping={
            "Positive": SentimentState.POSITIVE,
            "Negative": SentimentState.NEGATIVE,
            "Mixed_feelings": SentimentState.MIXED,
            "unknown_state": SentimentState.UNKNOWN,
        },
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 6. Ontology Registry & Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_ontology_summary() -> Dict:
    """
    Returns a complete summary of the structural ontology.
    Useful for documentation, validation, and model configuration.
    """
    return {
        "sentiment_states": {
            "classes": SentimentState.num_classes(),
            "labels": SentimentState.label_names(),
        },
        "transitions": {
            "classes": TransitionType.num_classes(),
            "labels": [t.name for t in TransitionType],
        },
        "trajectories": {
            "classes": TrajectoryType.num_classes(),
            "labels": TrajectoryType.label_names(),
        },
        "supported_scripts": [s.value for s in ScriptType],
        "supported_domains": list(DOMAIN_CONFIGS.keys()),
    }


def map_labels_to_ontology(
    labels: List[str],
    domain: str,
) -> List[SentimentState]:
    """
    Map raw dataset labels to ontology sentiment states using
    the domain-specific label mapping.

    Parameters
    ----------
    labels : List[str]
        Raw labels from the dataset.
    domain : str
        Domain key (e.g., 'dravidian_tamil', 'amazon_beauty').

    Returns
    -------
    List[SentimentState]
        Mapped ontology states.
    """
    config = DOMAIN_CONFIGS.get(domain)
    if config is None:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Available: {list(DOMAIN_CONFIGS.keys())}"
        )

    mapped = []
    for label in labels:
        if label in config.label_mapping:
            mapped.append(config.label_mapping[label])
        else:
            mapped.append(SentimentState.from_label(label))

    return mapped


def compute_sequence_trajectory(
    labels: List[str],
    domain: str,
) -> TrajectoryType:
    """
    Compute the trajectory type for a sequence of raw labels.

    Parameters
    ----------
    labels : List[str]
        Raw sentiment labels in temporal order.
    domain : str
        Domain key.

    Returns
    -------
    TrajectoryType
        The computed trajectory.
    """
    states = map_labels_to_ontology(labels, domain)
    return TrajectoryType.compute(states)


# ──────────────────────────────────────────────────────────────────────────────
# Module initialization log
# ──────────────────────────────────────────────────────────────────────────────

logger.info(
    f"Ontology loaded: {SentimentState.num_classes()} sentiment states, "
    f"{TransitionType.num_classes()} transition types, "
    f"{TrajectoryType.num_classes()} trajectory types, "
    f"{len(DOMAIN_CONFIGS)} domains"
)
