"""Unit tests for src/aspect_sentiment.py and src/final_decision.py (pure functions; no model is loaded)."""
import json
import os
import sys

import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import aspect_sentiment as asp
import final_decision as fd
from wsd import WordSenseDisambiguator

P, N, M, U = "POSITIVE", "NEGATIVE", "MIXED", "UNKNOWN"
UP, DOWN, ST = "UPGRADE", "DOWNGRADE", "STABLE"


def project(sents, tags):
    return asp.aspect_sentiment_projection([f"m{i}" for i in range(len(sents))], sents, tags)


# ---------------------------------------------------------------- aspect-associated sentiment
def test_one_aspect_single_occurrence_has_no_trend():
    r = project([P, N], [[("bgm", "music_bgm", 1.0)], []])
    a = r["aspects"]["music_bgm"]
    assert list(r["aspects"]) == ["music_bgm"]
    assert a["occurrences"] == 1 and a["n_messages"] == 1
    assert a["counts"] == {P: 1, N: 0, M: 0, U: 0}
    assert a["tendency"] == P
    assert a["trend"] is None and a["trend_note"] == "insufficient observations for trend"
    assert a["evidence_coverage"] == 0.5
    assert r["n_messages_without_aspect"] == 1 and r["message_coverage"] == 0.5


def test_multiple_aspects_take_the_message_sentiment():
    r = project([N], [[("story", "story_screenplay", 1.0), ("songs", "music_bgm", 1.0)]])
    assert r["aspects"]["story_screenplay"]["counts"][N] == 1
    assert r["aspects"]["music_bgm"]["counts"][N] == 1           # projection, not aspect-specific
    assert r["timeline"][0]["aspects"] == ["music_bgm", "story_screenplay"]
    assert "not a sentiment expressed specifically toward that aspect" in r["limitation"]


def test_repeated_aspect_counts_and_tendency():
    tag = [("bgm", "music_bgm", 1.0)]
    r = project([P, N, N, M], [tag, tag + [("song", "music_bgm", 1.0)], tag, tag])
    a = r["aspects"]["music_bgm"]
    assert a["n_messages"] == 4 and a["occurrences"] == 5         # two words in message 2, one observation
    assert a["counts"] == {P: 1, N: 2, M: 1, U: 0}
    assert a["tendency"] == N and a["tendency_share"] == 0.5
    assert [o["message_index"] for o in a["observations"]] == [1, 2, 3, 4]


def test_tie_gives_no_tendency():
    tag = [("bgm", "music_bgm", 1.0)]
    assert project([P, N], [tag, tag])["aspects"]["music_bgm"]["tendency"] is None


@pytest.mark.parametrize("sents, trend", [
    ([P, M, N], "DECLINING"),
    ([N, P], "IMPROVING"),
    ([P, P, P], "STABLE"),
    ([P, N, P, N], "VOLATILE"),
    ([P, U], None),                                              # UNKNOWN has no ordinal position
])
def test_trends(sents, trend):
    tag = [("bgm", "music_bgm", 1.0)]
    assert project(sents, [tag] * len(sents))["aspects"]["music_bgm"]["trend"] == trend
    assert asp.associated_trend(sents) == trend


def test_missing_wsd_results_and_missing_sentiments():
    r = asp.aspect_sentiment_projection(["a", "b"], [P, N], None)
    assert r["aspects"] == {} and r["n_messages_without_aspect"] == 2
    r = asp.aspect_sentiment_projection(["a"], None, [[("bgm", "music_bgm", 1.0)]])
    a = r["aspects"]["music_bgm"]
    assert r["sentiment_available"] is False
    assert sum(a["counts"].values()) == 0 and a["tendency"] is None and a["trend"] is None


def test_unresolved_tags_do_not_create_an_aspect():
    r = project([P], [[("xyz", "unknown", 0.0), {"word": "mass", "aspect": None, "resolved": False,
                                                 "candidates": ["dialogue", "fan_stardom"]}]])
    assert r["aspects"] == {}
    assert [u["word"] for u in r["unresolved"]] == ["xyz", "mass"]


def test_tag_messages_marks_real_wsd_ambiguity():
    wsd = WordSenseDisambiguator()
    tags = asp.tag_messages(["marana mass anna", "mass bgm super", "story weak", "nothing here"], wsd)
    assert tags[0] == [{"word": "mass", "aspect": None, "resolved": False, "wsd_overlap_score": 0.5,
                        "candidates": ["dialogue", "fan_stardom", "music_bgm", "trailer_teaser"]}]
    assert ("mass", "music_bgm", True) in [(t["word"], t["aspect"], t["resolved"]) for t in tags[1]]
    assert [(t["word"], t["aspect"]) for t in tags[2]] == [("story", "story_screenplay")]
    assert tags[3] == []


def test_empty_input_and_invalid_labels():
    r = asp.aspect_sentiment_projection([], [], [])
    assert r["n_messages"] == 0 and r["aspects"] == {} and r["timeline"] == [] and r["message_coverage"] == 0.0
    with pytest.raises(ValueError, match="invalid sentiment label"):
        project(["HAPPY"], [[]])
    with pytest.raises(ValueError, match="same length"):
        asp.aspect_sentiment_projection(["a", "b"], [P], None)


# ---------------------------------------------------------------- turning points
def test_turning_point_upgrade_then_downgrade():
    t = fd.detect_turning_points([ST, UP, DOWN])
    assert t["found"] and t["position"] == 3 and t["count"] == 1
    assert t["reversals"][0]["before"] == {"transition": UP, "from_message": 2, "to_message": 3}
    assert t["reversals"][0]["after"] == {"transition": DOWN, "from_message": 3, "to_message": 4}
    assert "after message 3" in t["message"]


def test_turning_point_downgrade_then_upgrade():
    t = fd.detect_turning_points([DOWN, UP])
    assert t["found"] and t["position"] == 2
    assert t["reversals"][0]["before"]["transition"] == DOWN and t["reversals"][0]["after"]["transition"] == UP


@pytest.mark.parametrize("transitions", [[], [ST, ST], [UP, UP], [DOWN, ST, DOWN], [UP, ST, ST]])
def test_no_turning_point(transitions):
    t = fd.detect_turning_points(transitions)
    assert t == {"found": False, "position": None, "count": 0, "reversals": [], "message": "No turning point detected"}


def test_stable_is_not_a_reversal_but_does_not_hide_one():
    t = fd.detect_turning_points([UP, ST, DOWN, UP])
    assert t["count"] == 2 and [r["position"] for r in t["reversals"]] == [3, 4]
    with pytest.raises(ValueError):
        fd.detect_turning_points(["SIDEWAYS"])


# ---------------------------------------------------------------- final decision
def test_declining_with_agreeing_aspect():
    tag = [("bgm", "music_bgm", 1.0)]
    aspects = project([P, P, N, N], [tag, [], tag, tag])
    d = fd.final_decision([P, P, N, N], [ST, DOWN, ST], "DECLINING", aspects, confidence={"oet_trajectory_softmax": 0.61})
    assert d["trajectory"] == "DECLINING" and d["sentiment_distribution"] == {P: 2, N: 2, M: 0, U: 0}
    assert d["turning_point"]["found"] is False
    top = d["important_aspects"][0]
    assert (top["aspect"], top["tendency"], top["trend"], top["vs_trajectory"]) == ("music_bgm", N, "DECLINING", "agrees")
    assert d["trajectory_vs_aspect_evidence"] == "agreement"
    assert d["confidence"] == {"oet_trajectory_softmax": 0.61}
    assert d["evidence"]["n_aspect_observations"] == 3 and d["evidence"]["limited"] is False
    assert d["verdict"].startswith("Overall trajectory: Declining. No turning point detected.")
    assert "music bgm (3 of 4 messages), with predominantly negative associated message sentiment" in d["verdict"]
    json.dumps(d)                                                 # JSON-serialisable


def test_improving_with_conflicting_aspect():
    tag = [("story", "story_screenplay", 1.0)]
    aspects = project([P, N, P, P], [tag, tag, [], []])            # the aspect goes POS -> NEG
    d = fd.final_decision([P, N, P, P], [DOWN, UP, ST], "IMPROVING", aspects, confidence={"x": 0.5})
    assert d["important_aspects"][0]["vs_trajectory"] == "differs"
    assert d["trajectory_vs_aspect_evidence"] == "disagreement"
    assert d["turning_point"]["position"] == 2
    assert "differs from the overall trajectory" in d["verdict"]


def test_stable_without_aspects_or_confidence():
    d = fd.final_decision([P, P, P], [ST, ST], "STABLE")
    assert d["important_aspects"] == [] and d["trajectory_vs_aspect_evidence"] == "not assessable"
    assert d["confidence"] is None and d["confidence_note"] == "not available"
    assert d["evidence"]["limited"] is True
    assert "no aspect was detected" in " ".join(d["evidence"]["limitations"])
    assert "No aspect was identified" in d["verdict"] and "The conclusion is limited" in d["verdict"]


def test_volatile_with_reversals_and_attention():
    d = fd.final_decision([P, N, P, N], [DOWN, UP, DOWN], "VOLATILE", attention=[0.1, 0.2, 0.6, 0.1])
    assert d["turning_point"]["count"] == 2 and d["rule_derived_trajectory_from_predicted_sentiments"] == "VOLATILE"
    assert d["most_attended_message"]["position"] == 3


def test_head_disagreement_is_reported():
    d = fd.final_decision([P, P, P], [ST, ST], "DECLINING", confidence={"x": 0.4})
    assert d["rule_derived_trajectory_from_predicted_sentiments"] == "STABLE"
    assert any("ontology rule" in s for s in d["evidence"]["limitations"])


def test_insufficient_evidence_single_message_and_single_observation():
    d = fd.final_decision([P], [], "STABLE", project([P], [[("bgm", "music_bgm", 1.0)]]), confidence={"x": 0.9})
    assert d["evidence"]["limited"] is True and d["important_aspects"][0]["vs_trajectory"] == "not comparable"
    assert "insufficient observations for trend" in d["verdict"]
    assert any("only 1 message" in s for s in d["evidence"]["limitations"])


@pytest.mark.parametrize("args", [
    ([], [], "STABLE"),
    ([P, "GOOD"], [ST], "STABLE"),
    ([P, N], [DOWN], "FALLING"),
    ([P, N], [], "DECLINING"),
    ([P, N], ["DROP"], "DECLINING"),
])
def test_invalid_inputs(args):
    with pytest.raises(ValueError):
        fd.final_decision(*args)


def test_run_pipeline_without_sentiment_model_gives_no_decision():
    r = fd.run_pipeline(["mass bgm super", "story weak"], None, WordSenseDisambiguator())
    assert r["final_decision"] is None and r["aspect_analysis"]["sentiment_available"] is False
    assert set(r["aspect_analysis"]["aspects"]) == {"music_bgm", "story_screenplay"}


def test_run_pipeline_with_oet_style_output():
    oet = {"sentiments": [P, N], "sentiment_softmax": [0.8, 0.7], "transitions": [DOWN], "transition_softmax": [0.6],
           "trajectory": "DECLINING", "trajectory_softmax": 0.55, "attention": [0.4, 0.6]}
    r = fd.run_pipeline(["songs super", "songs bore"], oet, WordSenseDisambiguator())
    a = r["aspect_analysis"]["aspects"]["music_bgm"]
    assert a["trend"] == "DECLINING" and a["observations"][0]["sentiment_confidence"] == 0.8
    assert r["final_decision"]["trajectory_vs_aspect_evidence"] == "agreement"
    json.dumps(r)
