"""
Tests for scripts/build_provisional_annotations.py.

SYNTHETIC FIXTURES ONLY: two invented annotators, an invented adjudication workbook and an invented offset workbook in a temporary folder.
No project annotation file is read or changed.
"""
import hashlib
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_provisional_annotations as bpa
import prepare_adjudication_package as pap
import validate_aspect_annotations as va
import test_adjudication_package as tap

PLACEHOLDER = "claude_adjudicator_1"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def decide(it):
    """Fill the invented decisions: (item_type, message, A span, B span) -> (span, category, sentiment, reason), or None to reject."""
    out = {}
    for i, r in it.iterrows():
        k = (r.item_type, r.message_id, r.A_span_text, r.B_span_text)
        d = None
        if r.message_id == "M1" and r.item_type == "CATEGORY_DISAGREEMENT":
            d = ("cap", "OTHER_ASPECT", "NEGATIVE", "[C2] A's category is too narrow.")           # OTHER_ASPECT with no facet note
        elif r.message_id == "M2":
            d = ("Smells", "FRAGRANCE", "NEGATIVE", "Plain negative; no pending rule.")
        elif r.message_id == "M6" and r.A_span_text == "not worth it":
            d = ("not worth it", "PRICE_VALUE", "NEGATIVE", "[S1] the evaluative phrase.")
        elif r.message_id == "M6":
            d = ("price", "PRICE_VALUE", "POSITIVE", "Aspect only.")                              # two rows, one final span
        elif r.message_id == "M7":
            d = ("Love it", "GENERAL", "POSITIVE", "Implicit overall evaluation.")                # two rows, one final span
        out[i] = d
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("prov")
    a, b = tap.write(tap.rows("A"), d / "A.csv"), tap.write(tap.rows("B"), d / "B.csv")
    a_rows = pd.read_csv(a, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    ref = a_rows.drop_duplicates("message_id").copy()
    for c in va.CONTENT[0:0] + ["aspect_status", "aspect_span_text", "span_start", "span_end", "span_type", "aspect_category", "sentiment", "annotator_id", "notes"]:
        ref[c] = ""
    ref["annotation_status"] = "NOT_STARTED"
    batch = str(d / "batch.csv")
    ref.to_csv(batch, index=False, encoding="utf-8-sig")

    pap.main(["--a", a, "--b", b, "--out_dir", str(d / "pkg")])
    it = pd.read_excel(d / "pkg" / "adjudication_items.xlsx", sheet_name="items", dtype=str, keep_default_na=False)
    review = it[it.item_type == "NEEDS_REVIEW"].index[0]
    it.loc[review, "item_id"] = "ADJ-109"                                                         # the unresolved case keeps its id
    it["adjudicator_id"] = PLACEHOLDER
    for i, dec in decide(it).items():
        if dec:
            r = it.loc[i]
            s = r.review_text.index(dec[0])
            it.loc[i, ["adjudicated_span_text", "adjudicated_span_start", "adjudicated_span_end", "adjudicated_category", "adjudicated_sentiment", "decision_reason"]] = \
                [dec[0], str(s), str(s + len(dec[0])), dec[1], dec[2], dec[3]]
    it.loc[review, "decision_reason"] = "UNRESOLVED"
    rejected = it[(it.adjudicated_span_text == "") & (it.item_id != "ADJ-109")]
    it.loc[rejected.index, "decision_reason"] = "REJECTED: [S6] test"
    items = str(d / "items.xlsx")
    with pd.ExcelWriter(items) as w:
        it.to_excel(w, sheet_name="items", index=False)

    text = {r.message_id: r.text for r in ref.itertuples()}
    cases = []
    for n, (mid, sa, sb, who) in enumerate([("M3", "bottle", "a bottle", "adjudicator_2"), ("M5", "Very strong", "strong", "adjudicator_3")], 1):
        t = text[mid]
        a0, b0 = t.index(sa), t.index(sb)
        cases.append(dict(case_no=str(n), message_id=mid, A_start=str(a0), A_end=str(a0 + len(sa)), B_start=str(b0), B_end=str(b0 + len(sb)),
                          adjudicated_span_text=sb, adjudicated_span_start=str(b0), adjudicated_span_end=str(b0 + len(sb)), adjudicator_id=who))
    offsets = str(d / "offsets.xlsx")
    with pd.ExcelWriter(offsets) as w:
        pd.DataFrame(cases).to_excel(w, sheet_name="review_27", index=False)

    audit = {r.item_id[4:]: (bpa.K, "5.4", "test", "high") for r in rejected.itertuples()}
    wife = rejected[(rejected.item_type == "UNMATCHED_SPAN_B_ONLY") & (rejected.B_span_text == "wife")]
    assert len(wife) == 1
    audit[wife.iloc[0].item_id[4:]] = (bpa.R, "6", "test: hypothetical naming an aspect", "high")
    return dict(d=d, a=a, b=b, batch=batch, items=items, offsets=offsets, audit=audit, rejected=rejected, wife=wife.iloc[0], it=it)


@pytest.fixture(scope="module")
def result(built):
    mp = pytest.MonkeyPatch()
    mp.setattr(bpa, "AUDIT", built["audit"])
    mp.setattr(bpa, "LOW_CONFIDENCE", set())
    mp.setattr(bpa, "V02_CONFLICT", set())
    mp.setattr(bpa, "V02_AMBIGUOUS", set())
    mp.setattr(bpa, "NEW_TYPE", {})
    before = {k: sha(built[k]) for k in ("a", "b", "batch", "items", "offsets")}
    out = str(built["d"] / "out")
    summ = bpa.main(["--a", built["a"], "--b", built["b"], "--batch", built["batch"], "--items", built["items"], "--offsets", built["offsets"], "--out_dir", out])
    mp.undo()
    rd = lambda f: pd.read_csv(os.path.join(out, f), encoding="utf-8-sig", dtype=str, keep_default_na=False)       # noqa: E731
    return dict(out=out, before=before, summ=summ, cons=rd("pilot_batch_01_PROVISIONAL_consolidated.csv"), prov=rd("pilot_batch_01_PROVISIONAL_provenance.csv"),
                audit=rd("rejected_span_audit_trail.csv"))


def tier_of(res, mid, span):
    p = res["prov"]
    row = p[(p.message_id == mid) & (p.span_text == span)]
    assert len(row) == 1, (mid, span, row)
    return row.iloc[0]


def test_output_is_valid_for_the_strict_schema_and_has_no_errors(result):
    cons = result["cons"]
    assert list(cons.columns) == va.COLUMNS
    res = va.validate(cons, va.load_taxonomies())
    assert [i for i in res["issues"] if i["severity"] == va.ERROR] == []


def test_two_annotators_agreeing_exactly_is_confirmed_and_training_eligible(result):
    r = tier_of(result, "M1", "scent")
    assert (r.tier, r.annotation_status, r.training_eligible, r.decision_origin) == ("CONFIRMED_AGREED", "ANNOTATED", "yes", "HUMAN_ANNOTATORS_AGREED")


def test_offset_decisions_keep_their_individual_human_ids_and_wait_for_policy(result):
    for mid, span, who in (("M3", "a bottle", "adjudicator_2"), ("M5", "strong", "adjudicator_3")):
        r = tier_of(result, mid, span)
        assert (r.tier, r.annotation_status, r.training_eligible, r.adjudicator_id, r.decision_origin) == \
            ("AGREED_OFFSET_ADJUDICATED", "ADJUDICATED", "pending_policy", who, f"HUMAN_ADJUDICATOR_{who}")
        assert "D1" in r.depends_on_decisions.split(",")
        c = result["cons"]
        row = c[(c.message_id == mid) & (c.aspect_span_text == span)].iloc[0]
        assert who in row.annotator_id and "policy-dependent" in row.notes


def test_workbook_rows_get_collective_human_attribution_and_keep_the_original_id_in_provenance(result):
    cons, prov = result["cons"], result["prov"]
    assert not cons.apply(lambda r: r.astype(str).str.contains("claude").any(), axis=1).any()
    r = tier_of(result, "M6", "not worth it")
    assert r.adjudicator_id == bpa.TEAM_ID and r.workbook_adjudicator_id_original == PLACEHOLDER and r.decision_origin == "HUMAN_TEAM_COLLECTIVE"
    assert (r.tier, r.annotation_status, r.training_eligible) == ("ADJUDICATED_POLICY_PENDING", "ADJUDICATED", "pending_policy")     # cites rule S1, so D1
    assert set(prov[prov.decision_origin == "HUMAN_TEAM_COLLECTIVE"].workbook_adjudicator_id_original) == {PLACEHOLDER}


def test_a_decision_with_no_pending_policy_and_no_flag_is_eligible(result):
    r = tier_of(result, "M2", "Smells")
    assert (r.tier, r.training_eligible, r.depends_on_decisions) == ("ADJUDICATED_HUMAN", "yes", "")


def test_other_aspect_without_a_facet_note_is_flagged_and_needs_review(result):
    r = tier_of(result, "M1", "cap")
    assert r.tier == "ADJUDICATED_FLAGGED" and "MISSING_FACET_NOTE" in r["flags"] and r.training_eligible == "no" and r.annotation_status == "NEEDS_REVIEW"


def test_the_same_final_span_from_several_items_is_counted_once(result):
    p = result["prov"]
    assert len(p[(p.message_id == "M6") & (p.span_text == "price")]) == 1
    assert len(p[(p.message_id == "M7") & (p.span_text == "Love it")]) == 1
    assert not p.duplicated(["message_id", "span_start", "span_end"]).any()
    assert "ADJ-" in tier_of(result, "M6", "price").sources and "," in tier_of(result, "M6", "price").sources


def test_unresolved_case_stays_unresolved_with_empty_sentiment(result):
    p = result["prov"]
    r = p[p.tier == "UNRESOLVED"]
    assert len(r) == 1 and r.iloc[0].sentiment == "" and r.iloc[0].annotation_status == "NEEDS_REVIEW" and r.iloc[0].training_eligible == "no"


def test_a_rejected_span_never_becomes_gold_and_only_an_audited_restore_is_provisional(result, built):
    cons, prov, audit = result["cons"], result["prov"], result["audit"]
    wife = built["wife"]
    r = tier_of(result, "M4", "wife")
    assert r.tier == "RESTORED_PROVISIONAL" and r.training_eligible == "no" and r.annotation_status == "NEEDS_REVIEW"
    restored_keys = set(zip(prov[prov.tier == "RESTORED_PROVISIONAL"].message_id, prov[prov.tier == "RESTORED_PROVISIONAL"].span_start.astype(int)))
    final_keys = set(zip(prov.message_id, prov.span_start.astype(int), prov.span_end.astype(int)))
    for a in audit[audit.v02_action == bpa.K].itertuples():
        if (a.message_id, int(a.span_start)) not in restored_keys:
            assert (a.message_id, int(a.span_start), int(a.span_end)) not in final_keys, a.item_id
    assert len(audit) == len(built["rejected"]) + 1                                                  # every rejection plus the unresolved case
    assert set(audit.workbook_decision_origin) == {"HUMAN_TEAM_COLLECTIVE"}
    assert wife.item_id in set(audit.item_id)


def test_only_agreed_or_unflagged_human_decisions_are_ever_training_eligible(result):
    p = result["prov"]
    assert set(p[p.training_eligible == "yes"].tier) <= {"CONFIRMED_AGREED", "ADJUDICATED_HUMAN"}
    assert not p[(p.training_eligible == "yes") & (p.sentiment == "")].shape[0]
    assert set(p[p.annotation_status == "ANNOTATED"].tier) == {"CONFIRMED_AGREED"}


def test_offsets_are_exact_and_inputs_are_untouched(result, built):
    cons = result["cons"]
    assert all(r.text[int(r.span_start):int(r.span_end)] == r.aspect_span_text for r in cons.itertuples())
    assert {k: sha(built[k]) for k in result["before"]} == result["before"]


def test_existing_output_is_never_overwritten_and_mismatched_workbooks_are_refused(result, built):
    with pytest.raises(FileExistsError):
        bpa.main(["--a", built["a"], "--b", built["b"], "--batch", built["batch"], "--items", built["items"], "--offsets", built["offsets"], "--out_dir", result["out"]])
    bad = pd.read_excel(built["offsets"], sheet_name="review_27", dtype=str, keep_default_na=False)
    bad.loc[0, "A_start"] = "0"
    p = str(built["d"] / "offsets_bad.xlsx")
    with pd.ExcelWriter(p) as w:
        bad.to_excel(w, sheet_name="review_27", index=False)
    with pytest.raises(KeyError):
        mp = pytest.MonkeyPatch()
        mp.setattr(bpa, "AUDIT", built["audit"])
        mp.setattr(bpa, "NEW_TYPE", {})
        try:
            bpa.main(["--a", built["a"], "--b", built["b"], "--batch", built["batch"], "--items", built["items"], "--offsets", p, "--out_dir", str(built["d"] / "out_bad")])
        finally:
            mp.undo()
