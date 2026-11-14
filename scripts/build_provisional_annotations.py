"""
Build the PROVISIONAL consolidated annotation file for pilot batch 01 from the two annotators, the 162-item adjudication workbook and the
27-case offset workbook. Read-only on every input; writes only new files into --out_dir (refuses to overwrite).

    python scripts/build_provisional_annotations.py

Provenance: the 27 offset decisions keep their individual human IDs (adjudicator_1 to adjudicator_4). The 162-item workbook was reviewed by the
same four team members collectively; its own ID field still reads claude_adjudicator_1 (kept in the provenance file), and because no row-to-member
assignment is recorded the consolidated rows say team_adjudicators_1_to_4. Nothing is final gold: the D1 to D11 guideline decisions are not approved.
training_eligible: yes = agreed by both annotators, or human-adjudicated with no pending policy and no flag; pending_policy = human-adjudicated but
depends on an unapproved decision; no = flagged, unresolved, restored or span-type-only provisional. Tiers: docs/aspect_annotation_guidelines_v0.3_DRAFT.md.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import validate_aspect_annotations as va  # noqa: E402

ADJ_DIR = os.path.join(ROOT, "outputs", "diagnostics", "annotation_adjudication", "pilot_batch_01_A_vs_B")
ANN = os.path.join(ROOT, "data", "annotations")
OUT = os.path.join(ROOT, "outputs", "annotation_consolidation", "pilot_batch_01")
TEAM_ID = "team_adjudicators_1_to_4"                 # collective attribution; no row-to-member assignment is recorded
UNRESOLVED_ID = "109"

LOW_CONFIDENCE = {"035", "036", "059", "066", "115", "135", "157", "037"}          # rules sheet: seven low-confidence groups
V02_CONFLICT = {"059"}                                                            # team decision contradicts a v0.2 rule (section 6 hearsay)
V02_AMBIGUOUS = {"066", "106"}                                                    # hypothetical/warning; v0.2 section 6 is not clear-cut
# span_type for adjudicated spans that no annotator marked with the same text, by the v0.2 section 5.3 test:
# EXPLICIT if the span contains the facet/product noun, IMPLICIT if it only evaluates a facet.
NEW_TYPE = {"005": "IMPLICIT", "009": "IMPLICIT", "021": "EXPLICIT", "032": "IMPLICIT", "037": "EXPLICIT", "040": "IMPLICIT", "042": "IMPLICIT",
            "051": "EXPLICIT", "055": "EXPLICIT", "059": "IMPLICIT", "158": "IMPLICIT", "085": "IMPLICIT", "073": "IMPLICIT", "066": "IMPLICIT", "062": "IMPLICIT", "060": "IMPLICIT", "068": "IMPLICIT", "108": "EXPLICIT", "117": "EXPLICIT", "129": "IMPLICIT",
            "132": "EXPLICIT", "135": "EXPLICIT", "136": "IMPLICIT", "142": "IMPLICIT", "143": "EXPLICIT", "153": "EXPLICIT", "154": "IMPLICIT",
            "156": "IMPLICIT", "157": "IMPLICIT", "161": "IMPLICIT"}
TYPE_FOR_DISAGREEING_AGREED_SPANS = "IMPLICIT"     # v0.2 5.3: none of those 10 same-offset spans names a facet noun

# --- audit of the 51 rejected spans against v0.2 only: (action, basis, reason, confidence)
K, R, H = "KEEP_REJECTED", "RESTORED_PROVISIONAL", "NEEDS_HUMAN"
AUDIT = {
    "002": (H, "5.4, 6", "'turbo mode' inside a clause about noise: a mention with no evaluation (v0.2 6 would make it NEUTRAL) but B labelled it POSITIVE.", "low"),
    "003": (R, "6", "Hypothetical ('If noise is a concern') that names the aspect: v0.2 6 says NEUTRAL; Annotator A's own label is NEUTRAL.", "high"),
    "004": (K, "5.2, 5.4", "'great' is the opinion word of 'works great'; the mention is already annotated.", "high"),
    "012": (K, "5.2, 5.4", "'easy' is the opinion word of the mention 'Trimming' (ADJ-011).", "high"),
    "014": (K, "5.2, 5.4", "'convenient' is the opinion word of the mention 'vacuum feature' (ADJ-013).", "high"),
    "017": (K, "5.4", "'clean' is the result clause of the 'turbo vacuum' mention, which is already annotated.", "high"),
    "020": (K, "5.2, 5.4", "'easy to clean' is the predicate of the mention 'swing door on the collector'.", "high"),
    "022": (K, "5.4", "'mess' is the object noun of the 'vacuum' mention; no separate evaluation.", "high"),
    "028": (K, "5.1, 4", "Seller name is not a facet word; the service evaluation is annotated at ADJ-029.", "high"),
    "033": (K, "5.4", "Overlaps the retained span 'continue to purchase this product' (same mention).", "high"),
    "034": (K, "5.4", "'soap' is the subject of 'smells amazing'; the facet (FRAGRANCE) is annotated.", "high"),
    "041": (K, "5.4", "Request that restates the defect already annotated at 'Falls apart'.", "high"),
    "043": (K, "5.2, 5.4", "'tone' is part of the predicate of the mention 'The scents'.", "high"),
    "047": (H, "5.5, 5.2", "v0.2 5.5 allows a pronoun span when no named mention is nearby, and prefers it to an opinion word (5.2); the retained span is the opinion word 'great' (ADJ-048).", "low"),
    "049": (K, "5.4", "Predicate of the already-annotated fragrance mention; the category question (EFFECTIVENESS or FRAGRANCE) is not settled by v0.2.", "low"),
    "050": (K, "5.2", "'adore' is the opinion verb of the mention annotated at 'fragrance'.", "high"),
    "054": (K, "5.4", "Head noun of 'super cute tail', the same mention as ADJ-052.", "high"),
    "058": (K, "5.5, 6", "Non-evaluative pronoun ('Got this for my son'); the evaluative mention is the later one.", "low"),
    "065": (K, "5.4", "'worked well' is the predicate of the mention annotated at 'pin'.", "high"),
    "071": (K, "5.4", "Elaboration of 'flexible' inside the same statement (ADJ-070).", "high"),
    "075": (H, "silent", "Evaluation of the reviewer's own workaround; v0.2 says nothing about it (proposal Q3 is unapproved).", "low"),
    "076": (K, "5.4", "Second clause of the same overall evaluation as 'Love this tail'.", "high"),
    "082": (K, "4, 6", "States the reviewer's need, not an aspect of the product.", "high"),
    "083": (H, "5.5, 5.2", "'This one' is a pronoun-like mention; the retained spans are the opinion nouns 'disappointment' and 'waste of money' (ADJ-084/085).", "low"),
    "091": (H, "6", "'quit working' is a fact after several good years; v0.2 gives no rule for events inside a long timeline.", "low"),
    "093": (K, "5.4", "Predicate of the battery-compartment mention (ADJ-092).", "high"),
    "094": (K, "5.4", "Coordinated predicate of the same component mention (ADJ-092).", "high"),
    "095": (K, "5.5, 4", "'same problem' is anaphoric with an ambiguous referent; the underlying complaints are annotated.", "low"),
    "097": (K, "5.2", "'useless' is the opinion word of the mention 'replacement rollers'.", "high"),
    "099": (K, "4", "Purpose clause; no facet is named.", "high"),
    "101": (K, "taxonomy SKIN_HAIR_REACTION", "A condition the reviewer attributes to hormones, not a reaction to the product; the product reaction is annotated.", "high"),
    "102": (K, "5.1", "Misplaced: 'Nothing cleared it up' is about other treatments, not the product.", "high"),
    "103": (H, "6", "'wanted more hold' names the facet 'hold' in a motive clause; whether it evaluates this product is an inference.", "low"),
    "104": (K, "5.4", "Use-case condition; the evaluation 'this works' (ADJ-105) is the retained mention.", "high"),
    "107": (K, "5.4", "Evidence for 'used and abused' (same aspect, same polarity).", "high"),
    "111": (K, "4", "Purpose clause ('help clean your tongue'); performance is evaluated later.", "high"),
    "118": (R, "6", "'case' is a named packaging aspect with no evaluation: v0.2 6 says NEUTRAL; Annotator B's own label is NEUTRAL.", "low"),
    "123": (K, "5.4", "Consequence of the scent complaint with a pronoun object; the facet is annotated.", "high"),
    "124": (R, "6", "Hypothetical ('If scent isn't an issue for you') naming the aspect: v0.2 6 says NEUTRAL; Annotator A's own label is NEUTRAL.", "high"),
    "125": (K, "5.4", "Restates the scent complaint with a pronoun object.", "high"),
    "127": (K, "5.4", "Purpose of the 'sunblock' feature in the same statement (ADJ-126).", "high"),
    "128": (K, "5.4", "'tube' is the subject of 'mostly empty', which is annotated; B's NEUTRAL would contradict it.", "high"),
    "137": (H, "6, taxonomy DELIVERY_SERVICE", "Availability ('out of stock') is not in the DELIVERY_SERVICE definition (proposal P5 is unapproved).", "low"),
    "138": (R, "6", "'1.7 oz size' is a named size aspect with no evaluation: v0.2 6 says NEUTRAL; Annotator A's own label is NEUTRAL.", "high"),
    "139": (K, "5.4", "Result clause of 'functions like a barrier-repair cream' (agreed span).", "high"),
    "145": (K, "4, 6", "Condition being treated (use-case), not a product facet.", "high"),
    "148": (K, "4, 6", "Pre-existing condition, not a product facet.", "high"),
    "149": (K, "taxonomy SKIN_HAIR_REACTION", "Problem that predates the product, so not a reaction to it; v0.2 does not say this explicitly (the category aliases include irritat*).", "low"),
    "150": (K, "taxonomy SKIN_HAIR_REACTION", "Problem that predates the product, so not a reaction to it; v0.2 does not say this explicitly (the category aliases include irritat*).", "low"),
    "155": (K, "4, 6", "Use-case in a recommendation condition, not a product facet.", "high"),
    "162": (K, "5.4", "Subject noun of 'the ice helps smoothen my skin'; the outcome is annotated.", "high"),
}
BLOCKING_FLAGS = ("LOW_CONFIDENCE", "V0.2_AMBIGUOUS", "V0.2_CONFLICT", "MISSING_FACET_NOTE")


def rd(p):
    return pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def spans_of(df, mid):
    g = df[(df.message_id == mid) & (df.aspect_status == "ASPECT_PRESENT")]
    return [dict(start=int(r.span_start), end=int(r.span_end), text=r.aspect_span_text, cat=r.aspect_category, sent=r.sentiment, type=r.span_type,
                 status=r.annotation_status, notes=r.notes) for r in g.itertuples() if r.span_start != ""]


def join_notes(a, b):
    parts = []
    for tag, n in (("A", a), ("B", b)):
        if n.strip():
            parts.append(f"{tag}: {n.strip()}")
    return " | ".join(parts)


def deps_from(reasons):
    d = set()
    for rule, dec in (("S1", "D1"), ("S2", "D1"), ("S4", "D2"), ("S3", "D3"), ("S5", "D3"), ("S8", "D3"), ("S7", "D4"), ("N4", "D4"),
                      ("P2", "D5"), ("P3", "D5"), ("P4", "D5"), ("P5", "D5"), ("Q3", "D5"), ("S6", "D6"), ("N5", "D6")):
        if re.search(rf"\b{rule}\b", reasons):
            d.add(dec)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=OUT)
    ap.add_argument("--a", default=f"{ANN}/pilot_batch_01_annotator_A.csv")
    ap.add_argument("--b", default=f"{ANN}/pilot_batch_01_annotator_B.csv")
    ap.add_argument("--batch", default=f"{ANN}/pilot_batch_01_product_timelines.csv", help="the unannotated batch (prepared columns)")
    ap.add_argument("--items", default=f"{ADJ_DIR}/adjudication_items_FINAL.xlsx", help="filled adjudication workbook, sheet 'items'")
    ap.add_argument("--offsets", default=f"{ADJ_DIR}/agreed_offset_review_27_cases_HUMAN_IDS.xlsx", help="filled offset workbook, sheet 'review_27'")
    a = ap.parse_args(argv)
    if os.path.exists(a.out_dir) and os.listdir(a.out_dir):
        raise FileExistsError(f"{a.out_dir} already has files; refusing to overwrite")
    os.makedirs(a.out_dir, exist_ok=True)

    A, B, REF = rd(a.a), rd(a.b), rd(a.batch)
    items = pd.read_excel(a.items, sheet_name="items", dtype=str, keep_default_na=False)
    c27 = pd.read_excel(a.offsets, sheet_name="review_27", dtype=str, keep_default_na=False)
    order = list(dict.fromkeys(REF.message_id))
    prepared = {r["message_id"]: {c: r[c] for c in va.PREPARED} for _, r in REF.drop_duplicates("message_id").iterrows()}
    text = {m: prepared[m]["text"] for m in order}

    final = {}                                    # (mid, start, end) -> dict(row fields + provenance)

    def put(mid, s, e, **kw):
        key = (mid, s, e)
        assert key not in final, f"duplicate span {key}"
        assert text[mid][s:e] == kw["span_text"], f"offset mismatch {key}"
        final[key] = dict(message_id=mid, span_start=s, span_end=e, **kw)

    # ---- 1. agreed pairs from the original files
    n_identical = n_offset = 0
    case_by_key = {(r.message_id, int(r.A_start), int(r.A_end), int(r.B_start), int(r.B_end)): r for r in c27.itertuples()}
    used_cases = set()
    for mid in order:
        sa, sb = spans_of(A, mid), spans_of(B, mid)
        for ia, ib, _ in va.match_spans([(s["start"], s["end"]) for s in sa], [(s["start"], s["end"]) for s in sb]):
            x, y = sa[ia], sb[ib]
            if not (x["cat"] == y["cat"] and x["sent"] == y["sent"] and "NEEDS_REVIEW" not in (x["status"], y["status"])):
                continue                          # disagreements are in the adjudication workbook
            notes = join_notes(x["notes"], y["notes"])
            if (x["start"], x["end"]) == (y["start"], y["end"]):
                n_identical += 1
                flags, tier, typ, tsrc = [], "CONFIRMED_AGREED", x["type"], "annotators_agree"
                if x["type"] != y["type"]:
                    tier, typ, tsrc = "AGREED_TYPE_PROVISIONAL", TYPE_FOR_DISAGREEING_AGREED_SPANS, "v0.2_5.3_test_by_assistant"
                    flags.append(f"SPAN_TYPE_CONFLICT(A={x['type']},B={y['type']})")
                if x["cat"] == "OTHER_ASPECT" and not notes:
                    flags.append("MISSING_FACET_NOTE")
                if flags and tier == "CONFIRMED_AGREED":
                    tier = "AGREED_NOTES_INCOMPLETE"
                put(mid, x["start"], x["end"], span_text=x["text"], cat=x["cat"], sent=x["sent"], type=typ, tier=tier, notes=notes,
                    sources="agreed:A+B", ref="", deps={"D10"} if tier == "AGREED_TYPE_PROVISIONAL" else set(), flags=flags,
                    origin="HUMAN_ANNOTATORS_AGREED", tsrc=tsrc, adj="", orig_adj="")
            else:
                n_offset += 1
                c = case_by_key[(mid, x["start"], x["end"], y["start"], y["end"])]
                used_cases.add(c.case_no)
                s, e = int(c.adjudicated_span_start), int(c.adjudicated_span_end)
                chosen = (s, e) in {(x["start"], x["end"]), (y["start"], y["end"])}
                typ = x["type"] if (s, e) == (x["start"], x["end"]) else y["type"] if (s, e) == (y["start"], y["end"]) else "EXPLICIT"
                flags = [f"BOUNDARY_DECIDED_BY_HUMAN_ADJUDICATOR({c.adjudicator_id})"]
                if x["type"] != y["type"]:
                    flags.append(f"SPAN_TYPE_CONFLICT(A={x['type']},B={y['type']})")
                if not chosen:
                    flags.append("NEW_SPAN_NEITHER_ANNOTATOR_MARKED")
                if c.case_no in {"7", "10", "22"}:
                    flags.append("USER_ACCEPTED_AS_PROVISIONAL")
                deps = {"D1", "D9"} | ({"D10"} if x["type"] != y["type"] else set())
                put(mid, s, e, span_text=c.adjudicated_span_text, cat=x["cat"], sent=x["sent"], type=typ, tier="AGREED_OFFSET_ADJUDICATED", notes=notes,
                    sources=f"agreed:A+B; offsets:27-case workbook case {c.case_no}", ref=f"case {c.case_no}", deps=deps, flags=flags,
                    origin=f"HUMAN_ADJUDICATOR_{c.adjudicator_id}", tsrc="inherited_from_chosen_span" if chosen else "assistant_assigned_new_span",
                    adj=c.adjudicator_id, orig_adj=c.adjudicator_id)
    assert used_cases == set(c27.case_no), "27-case workbook does not correspond to the agreed pairs"

    # ---- 2. adjudicated spans (distinct by message, start, end): human team decisions, collective attribution
    groups = defaultdict(list)
    for r in items.itertuples():
        if r.adjudicated_span_text != "":
            groups[(r.message_id, int(r.adjudicated_span_start), int(r.adjudicated_span_end))].append(r)
    for (mid, s, e), rows in groups.items():
        ids = [r.item_id[4:] for r in rows]
        first = rows[0]
        cat, sent, flags = first.adjudicated_category, first.adjudicated_sentiment, []
        types = {t for r in rows for side in "AB" for t in [getattr(r, f"{side}_span_type")]
                 if getattr(r, f"{side}_span_text") == first.adjudicated_span_text and t}
        conflict = len(types) == 2
        if conflict:
            flags.append("SPAN_TYPE_CONFLICT(A_vs_B)")
        inherited = len(types) == 1
        typ = types.pop() if inherited else NEW_TYPE[ids[0]]
        notes = join_notes("; ".join(dict.fromkeys(r.A_notes for r in rows if r.A_notes.strip())),
                           "; ".join(dict.fromkeys(r.B_notes for r in rows if r.B_notes.strip())))
        reasons = " ".join(r.decision_reason for r in rows)
        if set(ids) & LOW_CONFIDENCE:
            flags.append("LOW_CONFIDENCE")
        if set(ids) & V02_AMBIGUOUS:
            flags.append("V0.2_AMBIGUOUS_HYPOTHETICAL_OR_WARNING")
        if set(ids) & V02_CONFLICT:
            flags.append("V0.2_CONFLICT(section 6 hearsay is NEUTRAL, team decision kept)")
        if cat == "OTHER_ASPECT" and not notes:
            flags.append("MISSING_FACET_NOTE")
        if re.search(r"\b(P2|P3|P4)\b", reasons):
            flags.append("CATEGORY_PENDING_PROPOSAL")
        deps = deps_from(reasons) | ({"D10"} if conflict else set())
        if (mid, s, e) in final:
            raise AssertionError(f"adjudicated span collides with an agreed span: {(mid, s, e)} {ids}")
        tier = ("ADJUDICATED_FLAGGED" if any(f.startswith(b) for f in flags for b in BLOCKING_FLAGS)
                else "ADJUDICATED_POLICY_PENDING" if deps else "ADJUDICATED_HUMAN")
        put(mid, s, e, span_text=first.adjudicated_span_text, cat=cat, sent=sent, type=typ, tier=tier, notes=notes,
            sources="adjudication:" + ",".join("ADJ-" + i for i in ids), ref="", deps=deps, flags=flags, origin="HUMAN_TEAM_COLLECTIVE",
            tsrc="inherited_from_annotators" if inherited else "v0.2_5.3_test_by_assistant", adj=TEAM_ID, orig_adj=first.adjudicator_id)

    # ---- 3. rejected spans: v0.2 audit; restore only where v0.2 clearly requires it (a restored row overrides a team rejection, so it stays provisional)
    rej = items[(items.adjudicated_span_text == "") & (items.item_id != "ADJ-" + UNRESOLVED_ID)]
    assert set(rej.item_id.str[4:]) == set(AUDIT), "audit table does not match the rejected rows"
    audit_rows = []
    for r in rej.itertuples():
        i = r.item_id[4:]
        side = "A" if r.item_type.endswith("A_ONLY") else "B"
        s, e = int(getattr(r, f"{side}_start")), int(getattr(r, f"{side}_end"))
        action, basis, reason, conf = AUDIT[i]
        in_file = "no"
        if action == R:
            put(r.message_id, s, e, span_text=getattr(r, f"{side}_span_text"), cat=getattr(r, f"{side}_category"), sent=getattr(r, f"{side}_sentiment"),
                type=getattr(r, f"{side}_span_type"), tier="RESTORED_PROVISIONAL", notes=join_notes(r.A_notes, r.B_notes),
                sources=f"restored from team-rejected ADJ-{i} (Annotator {side} only)", ref="", deps={"D3", "D6"},
                flags=["RESTORED_UNDER_V0.2_AGAINST_TEAM_REJECTION", "SINGLE_ANNOTATOR"] + (["LOW_CONFIDENCE"] if conf == "low" else []),
                origin="ASSISTANT_V0.2_READING_OF_SINGLE_ANNOTATOR", tsrc="annotator", adj="", orig_adj=r.adjudicator_id)
            in_file = "yes (NEEDS_REVIEW, not training eligible)"
        audit_rows.append(dict(item_id=r.item_id, message_id=r.message_id, original_side=f"Annotator {side} only", span_text=getattr(r, f"{side}_span_text"),
                               span_start=s, span_end=e, original_category=getattr(r, f"{side}_category"), original_sentiment=getattr(r, f"{side}_sentiment"),
                               workbook_decision="REJECTED", workbook_decision_origin="HUMAN_TEAM_COLLECTIVE",
                               workbook_rules=",".join(re.findall(r"\[([A-Z]\d+)\]", r.decision_reason)),
                               workbook_reason=r.decision_reason, v02_action=action, v02_sections=basis, v02_reason=reason, confidence=conf,
                               in_provisional_file=in_file))
    r109 = items[items.item_id == "ADJ-" + UNRESOLVED_ID].iloc[0]
    s, e = int(r109.A_start or r109.B_start), int(r109.A_end or r109.B_end)
    put(r109.message_id, s, e, span_text=text[r109.message_id][s:e], cat=r109.A_category or r109.B_category, sent="", type=r109.A_span_type or r109.B_span_type,
        tier="UNRESOLVED", notes=f"ADJ-{UNRESOLVED_ID}: garbled text; polarity undecidable (v0.2 section 6). {r109.decision_reason[:200]}",
        sources=f"adjudication:ADJ-{UNRESOLVED_ID}", ref="", deps={"D8"}, flags=["UNRESOLVED_SENTIMENT"], origin="HUMAN_TEAM_COLLECTIVE",
        tsrc="annotator", adj=TEAM_ID, orig_adj=r109.adjudicator_id)
    audit_rows.append(dict(item_id=r109.item_id, message_id=r109.message_id, original_side="A and B", span_text=text[r109.message_id][s:e], span_start=s, span_end=e,
                           original_category=r109.A_category or r109.B_category, original_sentiment=f"A={r109.A_sentiment or '-'} B={r109.B_sentiment or '-'}",
                           workbook_decision="UNRESOLVED", workbook_decision_origin="HUMAN_TEAM_COLLECTIVE", workbook_rules="", workbook_reason=r109.decision_reason,
                           v02_action="KEEP_UNRESOLVED", v02_sections="6", v02_reason="Aspect clear, polarity undecidable: empty sentiment, NEEDS_REVIEW.",
                           confidence="n/a", in_provisional_file="yes (NEEDS_REVIEW, empty sentiment)"))

    # ---- 4. write the 22-column consolidated file (validator schema) and the provenance sidecar
    rows, prov = [], []
    for mid in order:
        for key in sorted(k for k in final if k[0] == mid):
            f = final[key]
            t, flagtxt, dep = f["tier"], ";".join(f["flags"]), ",".join(sorted(f["deps"]))
            if t == "CONFIRMED_AGREED":
                annotator, status, note, elig, blockers = "annotator_A|annotator_B", "ANNOTATED", f["notes"], "yes", ""
            elif t == "AGREED_OFFSET_ADJUDICATED":
                annotator = f"annotator_A|annotator_B (category, sentiment); offsets: {f['adj']} (human)"
                status, elig, blockers = "ADJUDICATED", "pending_policy", f"policy: {dep}"
                note = f"[HUMAN-ADJUDICATED offsets by {f['adj']}; policy-dependent on {dep}, not yet approved] {f['notes']}".strip()
            elif t in ("ADJUDICATED_HUMAN", "ADJUDICATED_POLICY_PENDING"):
                annotator, status = f"{TEAM_ID} (collective attribution)", "ADJUDICATED"
                elig, blockers = ("yes", "") if t == "ADJUDICATED_HUMAN" else ("pending_policy", f"policy: {dep}")
                note = f"[HUMAN-ADJUDICATED by team{'; policy-dependent on ' + dep + ', not yet approved' if dep else ''}] {f['notes']}".strip()
            elif t == "ADJUDICATED_FLAGGED":
                annotator, status, elig = f"{TEAM_ID} (collective attribution)", "NEEDS_REVIEW", "no"
                blockers = f"flags: {flagtxt}" + (f"; policy: {dep}" if dep else "")
                note = f"[FLAGGED team decision: {flagtxt}; deps {dep or '-'}] {f['notes']}".strip()
            elif t == "RESTORED_PROVISIONAL":
                annotator, status, elig, blockers = "single annotator (restored under v0.2 against a team rejection; not adjudicated)", "NEEDS_REVIEW", "no", f"flags: {flagtxt}"
                note = f"[PROVISIONAL {t}; deps {dep}] {f['notes']}".strip()
            elif t == "UNRESOLVED":
                annotator, status, elig, blockers = f"{TEAM_ID} (collective attribution)", "NEEDS_REVIEW", "no", "unresolved sentiment"
                note = f["notes"]
            else:                                   # AGREED_TYPE_PROVISIONAL, AGREED_NOTES_INCOMPLETE
                annotator, status, elig, blockers = "annotator_A|annotator_B (span, category, sentiment agreed; remaining fields not confirmed)", "NEEDS_REVIEW", "no", f"flags: {flagtxt}"
                note = f"[PROVISIONAL {t}; deps {dep or '-'}] {f['notes']}".strip()
            row = dict(prepared[mid])
            row.update(aspect_status="ASPECT_PRESENT", aspect_span_text=f["span_text"], span_start=str(f["span_start"]), span_end=str(f["span_end"]),
                       span_type=f["type"], aspect_category=f["cat"], sentiment=f["sent"], annotation_status=status, annotator_id=annotator, notes=note)
            rows.append(row)
            prov.append(dict(message_id=mid, span_start=f["span_start"], span_end=f["span_end"], span_text=f["span_text"], aspect_category=f["cat"],
                             sentiment=f["sent"], span_type=f["type"], span_type_source=f["tsrc"], annotation_status=status, tier=t, training_eligible=elig,
                             blockers=blockers, decision_origin=f["origin"], adjudicator_id=f["adj"], workbook_adjudicator_id_original=f["orig_adj"],
                             sources=f["sources"], case_ref=f["ref"], depends_on_decisions=dep, flags=flagtxt))
    cons = pd.DataFrame(rows, columns=va.COLUMNS)
    cons.to_csv(f"{a.out_dir}/pilot_batch_01_PROVISIONAL_consolidated.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prov).to_csv(f"{a.out_dir}/pilot_batch_01_PROVISIONAL_provenance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(f"{a.out_dir}/rejected_span_audit_trail.csv", index=False, encoding="utf-8-sig")

    summ = dict(rows=len(cons), messages=int(cons.message_id.nunique()), agreed_identical_offsets=n_identical, agreed_with_offset_decision=n_offset,
                tiers=dict(Counter(p["tier"] for p in prov)), statuses=dict(Counter(p["annotation_status"] for p in prov)),
                training_eligible=dict(Counter(p["training_eligible"] for p in prov)),
                decision_origin=dict(Counter(p["decision_origin"] for p in prov)),
                audit_actions=dict(Counter(x["v02_action"] for x in audit_rows)), restored=sorted(i for i, v in AUDIT.items() if v[0] == R),
                needs_human=sorted(i for i, v in AUDIT.items() if v[0] == H))
    with open(f"{a.out_dir}/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))
    return summ


if __name__ == "__main__":
    main()
