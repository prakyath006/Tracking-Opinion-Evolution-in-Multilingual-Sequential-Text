"""
=============================================================================
Ontology Consistency Tests
=============================================================================
Guards the invariant established by the Module 1a wiring: src/ontology.py is
the single source of truth for every label taxonomy in the pipeline.

These tests fail loudly if any module drifts back to hand-maintained label
data. Specifically they assert:

  (a) dataset.py's numeric label encoding matches SentimentState for every
      domain — including preprocessing.py's label_encoded cache and, when the
      real preprocessed CSVs are present, their actual label strings.
  (b) classifier.py's (and model.py's) class counts equal the ontology's
      num_classes() for each taxonomy.
  (c) evaluation.py's label name lists equal the ontology's label_names()
      for each taxonomy.

Run with pytest:
    pytest tests/test_ontology_consistency.py -v

or standalone (no pytest required):
    python tests/test_ontology_consistency.py
=============================================================================
"""

import os
import sys
import glob
import inspect
import tempfile

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PREPROCESSED_DIR = os.path.join(REPO_ROOT, "data", "preprocessed")

from ontology import (                                    # noqa: E402
    DOMAIN_CONFIGS,
    SentimentState,
    TransitionType,
    TrajectoryType,
    map_labels_to_ontology,
)
import preprocessing                                      # noqa: E402
import dataset                                            # noqa: E402
import evaluation                                         # noqa: E402
from classifier import MultiTaskClassifier                # noqa: E402


DRAVIDIAN_DOMAINS = [d for d in DOMAIN_CONFIGS if d.startswith("dravidian_")]

# Language -> preprocessed-CSV filename prefix, mirroring dataset.py.
LANG_FILE_PREFIX = {"tamil": "tamil", "malayalam": "mal", "kannada": "kannada"}


# ═══════════════════════════════════════════════════════════════════════════
# Taxonomy shape — encoded ids must be usable as list indices
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "taxonomy", [SentimentState, TransitionType, TrajectoryType]
)
def test_enum_values_are_contiguous_from_zero(taxonomy):
    """
    Every taxonomy must encode as 0..n-1 with no gaps.

    The whole pipeline relies on this: labels are fed to CrossEntropyLoss as
    class indices, and label_names()[i] is used as the display name for
    encoded class i.
    """
    values = [member.value for member in taxonomy]
    assert values == list(range(len(values))), (
        f"{taxonomy.__name__} values {values} are not contiguous from 0; "
        f"they are used directly as classifier class indices."
    )


@pytest.mark.parametrize(
    "taxonomy", [SentimentState, TransitionType, TrajectoryType]
)
def test_label_names_are_ordered_by_value(taxonomy):
    """label_names()[i] must name the member whose encoded value is i."""
    names = taxonomy.label_names()
    assert len(names) == taxonomy.num_classes()
    for member in taxonomy:
        assert names[member.value] == member.name, (
            f"{taxonomy.__name__}.label_names()[{member.value}] is "
            f"{names[member.value]!r}, expected {member.name!r}. Report and "
            f"confusion-matrix labels would be misattributed."
        )


# ═══════════════════════════════════════════════════════════════════════════
# (a) dataset.py's numeric label encoding matches SentimentState
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("domain", sorted(DOMAIN_CONFIGS))
def test_domain_label_mapping_targets_valid_states(domain):
    """Every domain's label_mapping must map onto real SentimentState members."""
    config = DOMAIN_CONFIGS[domain]
    assert config.label_mapping, f"{domain} has an empty label_mapping"
    for raw_label, state in config.label_mapping.items():
        assert isinstance(state, SentimentState), (
            f"{domain} maps {raw_label!r} to {state!r}, "
            f"which is not a SentimentState"
        )


@pytest.mark.parametrize("domain", sorted(DRAVIDIAN_DOMAINS))
def test_preprocessing_encoding_matches_ontology(domain):
    """
    preprocessing.SENTIMENT_LABELS writes the label_encoded column that
    dataset.py used to read directly. Its integers must equal the encoded
    value of the SentimentState the ontology maps the same string to.
    """
    for raw_label, state in DOMAIN_CONFIGS[domain].label_mapping.items():
        assert raw_label in preprocessing.SENTIMENT_LABELS, (
            f"{domain} maps {raw_label!r} but preprocessing.SENTIMENT_LABELS "
            f"cannot encode it, so label_encoded would be NaN."
        )
        assert preprocessing.SENTIMENT_LABELS[raw_label] == state.value, (
            f"Encoding drift for {raw_label!r} in {domain}: "
            f"preprocessing.py encodes it as "
            f"{preprocessing.SENTIMENT_LABELS[raw_label]}, but the ontology "
            f"maps it to {state.name}={state.value}."
        )


def test_non_target_language_labels_resolve_to_unknown():
    """
    'not-<Language>' rows are dropped by preprocessing by default, but if they
    are kept (filter_mode='keep'), map_labels_to_ontology must still resolve
    them rather than guessing a sentiment.
    """
    not_lang = sorted(preprocessing.LanguageFilter.NOT_LANG_LABELS)
    for domain in DRAVIDIAN_DOMAINS:
        states = map_labels_to_ontology(not_lang, domain)
        assert all(s is SentimentState.UNKNOWN for s in states), (
            f"{domain}: expected all of {not_lang} to map to UNKNOWN, "
            f"got {[s.name for s in states]}"
        )


def test_amazon_rating_thresholds_match_sequence_builder():
    """
    The Amazon CSV has no string label column, so SentimentState.from_rating()
    is the ontology's entry point for that domain. It must agree with the
    star->sentiment rule that scripts/download_and_build_amazon_sequences.py
    used to write label_encoded (>=4 Positive, <=2 Negative, else Mixed).
    """
    expected = {
        1.0: SentimentState.NEGATIVE,
        2.0: SentimentState.NEGATIVE,
        3.0: SentimentState.MIXED,
        4.0: SentimentState.POSITIVE,
        5.0: SentimentState.POSITIVE,
    }
    for stars, state in expected.items():
        assert SentimentState.from_rating(stars) is state, (
            f"{stars} stars maps to "
            f"{SentimentState.from_rating(stars).name}, but the Amazon "
            f"sequence builder encoded it as {state.name}."
        )


def test_dravidian_dataset_emits_ontology_encoded_sentiments():
    """
    End-to-end: labels read by dataset.py must come out as SentimentState
    values, for every Dravidian domain.
    """
    for domain in DRAVIDIAN_DOMAINS:
        language = domain.split("_", 1)[1]
        raw_labels = list(DOMAIN_CONFIGS[domain].label_mapping)
        expected = [
            DOMAIN_CONFIGS[domain].label_mapping[lbl].value for lbl in raw_labels
        ]

        frame = pd.DataFrame({
            "text": [f"sample {i}" for i in range(len(raw_labels))],
            "label": raw_labels,
            "label_encoded": expected,
        })
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, f"{language}.csv")
            frame.to_csv(csv_path, index=False)
            ds = dataset.DravidianDataset(
                csv_path=csv_path, language=language, task="sentiment",
                split="train", train_ratio=1.0, val_ratio=0.0,
            )
        assert sorted(ds.labels) == sorted(expected), (
            f"{domain}: dataset.py emitted {sorted(ds.labels)}, "
            f"ontology expects {sorted(expected)}"
        )


def test_amazon_dataset_emits_ontology_encoded_sentiments():
    """End-to-end equivalent for the Amazon sequence dataset."""
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = [SentimentState.from_rating(r).value for r in ratings]

    frame = pd.DataFrame({
        "user_id": ["u1"] * len(ratings),
        "sequence_position": range(1, len(ratings) + 1),
        "text": [f"review {i}" for i in range(len(ratings))],
        "rating": ratings,
        "label_encoded": expected,
    })
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "amazon_beauty_sequences.csv")
        frame.to_csv(csv_path, index=False)
        ds = dataset.AmazonSequenceDataset(
            csv_path=csv_path, split="train", train_ratio=1.0, val_ratio=0.0,
        )
    assert ds[0]["sentiments"].tolist() == expected


def test_trend_and_trajectory_labels_stay_in_taxonomy_range():
    """
    dataset.py must only ever emit trend/trajectory ids the heads can predict.
    Covers every ordered pair of states, so an UNKNOWN endpoint is included.
    """
    states = list(SentimentState)
    for prev in states:
        for curr in states:
            trends = dataset.build_trend_labels([prev, curr])
            assert all(0 <= t < TransitionType.num_classes() for t in trends), (
                f"{prev.name}->{curr.name} produced trend ids {trends}, "
                f"outside TransitionType's {TransitionType.num_classes()} classes"
            )
            traj = TrajectoryType.compute([prev, curr]).value
            assert 0 <= traj < TrajectoryType.num_classes()


def test_unknown_is_excluded_from_ordinal_transition_ranking():
    """
    Regression test for the Step 1 bug: UNKNOWN must not be treated as an
    intensity below NEGATIVE, so no UNKNOWN endpoint may yield UPGRADE or
    DOWNGRADE.
    """
    for other in SentimentState:
        for pair in [(SentimentState.UNKNOWN, other),
                     (other, SentimentState.UNKNOWN)]:
            result = TransitionType.compute(*pair)
            assert result is TransitionType.STABLE, (
                f"{pair[0].name} -> {pair[1].name} returned {result.name}; "
                f"UNKNOWN must not participate in ordinal comparison."
            )


@pytest.mark.parametrize("domain", sorted(DRAVIDIAN_DOMAINS))
def test_real_preprocessed_csvs_agree_with_ontology(domain):
    """
    When the real preprocessed CSVs are present, every label string in them
    must be known to the domain's label_mapping, and the stored label_encoded
    must equal the ontology's encoding.

    Skipped when data/preprocessed is absent (it is gitignored), so this file
    stays runnable in a fresh checkout.
    """
    language = domain.split("_", 1)[1]
    prefix = LANG_FILE_PREFIX.get(language, language)
    pattern = os.path.join(PREPROCESSED_DIR, f"{prefix}_sentiment_*_preprocessed.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        pytest.skip(f"No preprocessed CSVs for {domain} (looked for {pattern})")

    mapping = DOMAIN_CONFIGS[domain].label_mapping
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8")
        if "label" not in frame.columns:
            pytest.fail(f"{path} has no 'label' column; dataset.py requires it")

        unknown_spellings = sorted(set(frame["label"].astype(str)) - set(mapping))
        assert not unknown_spellings, (
            f"{os.path.basename(path)} contains label spellings absent from "
            f"{domain}'s label_mapping: {unknown_spellings}"
        )

        if "label_encoded" in frame.columns:
            expected = [mapping[lbl].value for lbl in frame["label"].astype(str)]
            actual = frame["label_encoded"].astype(int).tolist()
            assert expected == actual, (
                f"{os.path.basename(path)}: label_encoded disagrees with the "
                f"ontology mapping of 'label'."
            )


# ═══════════════════════════════════════════════════════════════════════════
# (b) classifier.py's class counts equal the ontology's num_classes()
# ═══════════════════════════════════════════════════════════════════════════

def test_classifier_head_counts_match_ontology():
    """Default-constructed heads must size themselves from the ontology."""
    clf = MultiTaskClassifier(input_dim=32, hidden_dim=8)

    assert clf.num_sentiment_classes == SentimentState.num_classes()
    assert clf.num_trend_classes == TransitionType.num_classes()
    assert clf.num_trajectory_classes == TrajectoryType.num_classes()

    # The declared counts must also be the real output widths.
    widths = {
        "sentiment": (clf.sentiment_head.classifier[-1].out_features,
                      SentimentState.num_classes()),
        "trend": (clf.trend_head.classifier[-1].out_features,
                  TransitionType.num_classes()),
        "trajectory": (clf.trajectory_head.classifier[-1].out_features,
                       TrajectoryType.num_classes()),
    }
    for head, (actual, expected) in widths.items():
        assert actual == expected, (
            f"{head} head emits {actual} logits, ontology defines {expected}"
        )


@pytest.mark.parametrize(
    "module_name, qualname",
    [
        ("classifier", "MultiTaskClassifier"),
        ("model", "OpinionEvolutionTracker"),
    ],
)
def test_class_count_defaults_are_not_hardcoded(module_name, qualname):
    """
    Both constructors default their three class counts; those defaults must be
    the ontology's values, not literals that can silently fall out of date.
    """
    module = __import__(module_name)
    signature = inspect.signature(getattr(module, qualname).__init__)
    expected = {
        "num_sentiment_classes": SentimentState.num_classes(),
        "num_trend_classes": TransitionType.num_classes(),
        "num_trajectory_classes": TrajectoryType.num_classes(),
    }
    for param, value in expected.items():
        assert signature.parameters[param].default == value, (
            f"{module_name}.{qualname} defaults {param} to "
            f"{signature.parameters[param].default}, ontology says {value}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# (c) evaluation.py's label name lists equal the ontology's label_names()
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "attribute, taxonomy",
    [
        ("SENTIMENT_LABELS", SentimentState),
        ("TREND_LABELS", TransitionType),
        ("TRAJECTORY_LABELS", TrajectoryType),
    ],
)
def test_evaluation_label_names_come_from_ontology(attribute, taxonomy):
    """Report/confusion-matrix names must be the ontology's, not hand-typed."""
    actual = getattr(evaluation.EvaluationRunner, attribute)
    assert actual == taxonomy.label_names(), (
        f"EvaluationRunner.{attribute} is {actual}, but "
        f"{taxonomy.__name__}.label_names() is {taxonomy.label_names()}"
    )


def test_dataset_trajectory_labels_match_ontology():
    """dataset.TRAJECTORY_LABELS is consumed by demo_full_project.py."""
    assert dataset.TRAJECTORY_LABELS == {
        t.name: t.value for t in TrajectoryType
    }


def test_confusion_matrix_is_sized_by_the_ontology():
    """
    A split missing some classes must still produce a full-size, correctly
    aligned confusion matrix when ontology label names are supplied.
    """
    names = SentimentState.label_names()
    matrix = evaluation.compute_confusion_matrix(
        [SentimentState.POSITIVE.value, SentimentState.NEGATIVE.value],
        [SentimentState.POSITIVE.value, SentimentState.POSITIVE.value],
        label_names=names,
    )
    assert matrix.shape == (len(names), len(names))


def test_classification_report_covers_every_ontology_class():
    """
    classification_report must not raise when a split lacks some classes, and
    must name every ontology class.
    """
    names = SentimentState.label_names()
    report = evaluation.get_classification_report(
        [SentimentState.POSITIVE.value, SentimentState.NEGATIVE.value],
        [SentimentState.POSITIVE.value, SentimentState.POSITIVE.value],
        label_names=names,
    )
    for name in names:
        assert name in report, f"{name} missing from the classification report"


# ═══════════════════════════════════════════════════════════════════════════
# Standalone entry point (no pytest required)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
