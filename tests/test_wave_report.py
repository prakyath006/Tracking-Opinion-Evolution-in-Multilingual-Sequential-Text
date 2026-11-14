"""
Tests for scripts/make_wave_files.py and scripts/wave_report.py.

SYNTHETIC FIXTURES ONLY: invented annotators and an invented batch in a temporary folder. No project annotation file is read or changed.
"""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import make_wave_files as mwf
import validate_aspect_annotations as va
import wave_report as wr
import test_adjudication_package as tap


def blank_reference(a_path):
    ref = pd.read_csv(a_path, encoding="utf-8-sig", dtype=str, keep_default_na=False).drop_duplicates("message_id").copy()
    for c in mwf.BLANK:
        ref[c] = ""
    ref["annotation_status"] = "NOT_STARTED"
    return ref


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("wave")
    a, b = tap.write(tap.rows("A"), d / "A.csv"), tap.write(tap.rows("B"), d / "B.csv")
    ref = blank_reference(a)
    ref_path = str(d / "reference.csv")
    ref.to_csv(ref_path, index=False, encoding="utf-8-sig")
    pilot = pd.DataFrame([dict(message_id="X1", entity_id="PX", aspect_category="EFFECTIVENESS", sentiment="POSITIVE", training_eligible="yes")])
    pc, pp = str(d / "pc.csv"), str(d / "pp.csv")
    pilot.drop(columns="training_eligible").to_csv(pc, index=False, encoding="utf-8-sig")
    pilot[["training_eligible"]].to_csv(pp, index=False, encoding="utf-8-sig")
    return dict(d=d, a=a, b=b, ref=ref_path, pc=pc, pp=pp)


def test_wave_files_are_identical_blank_copies_and_flag_the_small_stratum(tmp_path):
    seqs = [{"sequence_id": f"S{i}", "product": f"P{i}", "stratum": "SMALL" if i == 0 else "MEDIUM", "wave": 1, "eligible_reviews": 20, "span_days": 400.0,
             "distinctive_category": "FRAGRANCE"} for i in range(2)]
    rows = []
    for s in seqs:
        for pos in range(1, 11):
            rows.append({"sequence_id": s["sequence_id"], "sequence_type": "PRODUCT_TIMELINE", "source": "amazon_all_beauty_2023", "entity_id": s["product"],
                         "message_id": f"{s['sequence_id']}-{pos}", "position": str(pos), "timestamp_ms": str(pos), "timestamp_utc": "x", "language": "en",
                         "title": "t", "text": "some text", "parent_message_id": "", "annotation_status": "NOT_STARTED"})
    batch, man = str(tmp_path / "b.csv"), str(tmp_path / "m.json")
    pd.DataFrame(rows).reindex(columns=va.COLUMNS).fillna("").to_csv(batch, index=False, encoding="utf-8-sig")
    json.dump({"sequences": seqs + [dict(seqs[0], sequence_id="S9", wave=2)]}, open(man, "w"))
    names = mwf.main(["--wave", "1"], batch=batch, manifest=man, out_dir=str(tmp_path))
    r, one, two = (pd.read_csv(names[k], dtype=str, keep_default_na=False, encoding="utf-8-sig") for k in ("reference", "annotator_1", "annotator_2"))
    assert r.equals(one) and one.equals(two) and len(one) == 20 and (one[mwf.BLANK] == "").all().all()
    info = json.load(open(names["manifest"]))
    assert info["PENDING_TEAM_APPROVAL"]["sequence_ids"] == ["S0"] and info["PENDING_TEAM_APPROVAL"]["reviews"] == 10
    with pytest.raises(FileExistsError):
        mwf.main(["--wave", "1"], batch=batch, manifest=man, out_dir=str(tmp_path))


def test_report_runs_end_to_end_and_covers_every_requested_measure(env):
    out = str(env["d"] / "out")
    rc = wr.main(["--a", env["a"], "--b", env["b"], "--reference", env["ref"], "--wave_manifest", str(env["d"] / "none.json"),
                  "--pilot_consolidated", env["pc"], "--pilot_provenance", env["pp"], "--out_dir", out])
    assert rc == 0
    rep = json.load(open(os.path.join(out, "report.json")))
    g = next(iter(rep["agreement"]["groups"].values()))
    assert g["spans"]["span_f1"] is not None and g["category_on_matched_spans"]["n"] > 0 and g["sentiment_on_matched_spans"]["n"] > 0
    assert rep["agreement"]["span_type_on_matched_spans"]["n"] > 0
    assert rep["agreement"]["unresolved"]["NEEDS_REVIEW_rows_a"] == 1 and rep["agreement"]["unresolved"]["message_status_disagreements"] >= 1
    assert os.path.exists(os.path.join(out, "adjudication_package", "adjudication_items.xlsx")) and rep["adjudication_items"]["total"] > 0
    y = rep["yield"]
    assert y["wave"]["agreed"]["examples"] <= y["wave"]["all_spans_once"]["examples"]                 # agreed is a subset of all spans
    assert y["corpus_agreed_only"]["products"] == y["wave"]["agreed"]["products"] + y["pilot"]["conservative"]["products"]
    assert y["floors"]["agreed_only"]["hard_blockers"] and rep["recommendation"].startswith("PROCEED")   # a tiny corpus cannot clear the floors
    assert os.path.exists(os.path.join(out, "report.md"))
    with pytest.raises(FileExistsError):
        wr.main(["--a", env["a"], "--b", env["b"], "--reference", env["ref"], "--pilot_consolidated", env["pc"], "--pilot_provenance", env["pp"], "--out_dir", out])


def test_small_stratum_products_are_reported_separately(env):
    seq = pd.read_csv(env["ref"], dtype=str, keep_default_na=False, encoding="utf-8-sig").sequence_id.iloc[0]
    man = str(env["d"] / "wave_manifest.json")
    json.dump({"PENDING_TEAM_APPROVAL": {"sequence_ids": [seq]}}, open(man, "w"))
    out = str(env["d"] / "out_small")
    assert wr.main(["--a", env["a"], "--b", env["b"], "--reference", env["ref"], "--wave_manifest", man, "--pilot_consolidated", env["pc"],
                    "--pilot_provenance", env["pp"], "--out_dir", out]) == 0
    rep = json.load(open(os.path.join(out, "report.json")))
    assert rep["yield"]["wave"]["agreed_without_SMALL"]["products"] == 0 and rep["yield"]["wave"]["agreed"]["products"] == 1 and rep["pending_team_approval"]


def test_validation_errors_stop_the_report_before_agreement(env):
    bad = pd.read_csv(env["a"], encoding="utf-8-sig", dtype=str, keep_default_na=False)
    bad.loc[0, "sentiment"] = "GREAT"
    p = str(env["d"] / "bad.csv")
    bad.to_csv(p, index=False, encoding="utf-8-sig")
    out = str(env["d"] / "out_bad")
    assert wr.main(["--a", p, "--b", env["b"], "--reference", env["ref"], "--out_dir", out]) == 1
    rep = json.load(open(os.path.join(out, "report.json")))
    assert rep["validation"]["a"]["errors"] >= 1 and "agreement" not in rep and not os.path.exists(os.path.join(out, "adjudication_package"))


def test_yield_counts_series_only_from_two_or_more_ranked_reviews():
    rows = [dict(message_id=m, entity_id="P", category=c, sentiment=s) for m, c, s in
            [("r1", "FRAGRANCE", "POSITIVE"), ("r2", "FRAGRANCE", "NEGATIVE"), ("r3", "FRAGRANCE", "NEUTRAL"),        # 2 ranked: a series
             ("r1", "PRICE_VALUE", "POSITIVE"), ("r2", "PRICE_VALUE", "NEUTRAL"),                                  # 1 ranked: no series
             ("r1", "GENERAL", "POSITIVE"), ("r1", "GENERAL", "NEGATIVE"), ("r2", "GENERAL", "POSITIVE")]]          # r1 aggregates to MIXED, still ranked
    y = wr.yield_stats(rows, ["P"])
    assert (y["products"], y["reviews"], y["examples"], y["usable_series"]) == (1, 3, 7, 2) and y["class_counts"]["MIXED"] == 1


def test_the_recommendation_follows_the_floors():
    big = dict(products=60, reviews=600, examples=1500, usable_series=300, class_counts={"NEGATIVE": 300})
    small = dict(products=10, reviews=50, examples=100, usable_series=20, class_counts={"NEGATIVE": 15})
    assert "NOT needed" in wr.recommend(big, big, big)
    assert wr.recommend(small, small, wr.add(small, small)).startswith("PROCEED with Wave 2 and plan more")
    mid = dict(products=20, reviews=200, examples=500, usable_series=90, class_counts={"NEGATIVE": 100})
    assert wr.recommend(mid, mid, wr.add(mid, mid)).startswith("PROCEED with Wave 2:")
