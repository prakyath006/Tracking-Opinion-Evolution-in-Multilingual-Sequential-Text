# Pilot batch 01: PROVISIONAL consolidated annotations

**Not final gold data.** The D1 to D11 guideline decisions are not approved, and some rows are flagged, restored or unresolved. Use `training_eligible` in the provenance file; only `yes` rows may be trained on, and nothing should be trained until the floors in `docs/annotation_requirements_and_batch_02.md` are met.

Built by `scripts/build_provisional_annotations.py` (tests: `tests/test_provisional_annotations.py`) from the Annotator A and B files, `adjudication_items_FINAL.xlsx`, `agreed_offset_review_27_cases_HUMAN_IDS.xlsx` and guidelines v0.2 (the approved baseline). Every input is unchanged. Earlier builds that marked the adjudicator identity as unverified were superseded and removed in the final cleanup.

## Who decided what (provenance)

| Decision origin | Rows | Identity in `annotator_id` / provenance |
|---|---|---|
| Two human annotators agreed | 74 | `annotator_A|annotator_B` |
| 27 offset cases, individual human adjudicators | 27 | `adjudicator_1` (7), `adjudicator_2` (7), `adjudicator_3` (7), `adjudicator_4` (6), recorded per row |
| 162-item workbook, reviewed by the four members collectively | 85 | `team_adjudicators_1_to_4`. No row-to-member assignment is recorded, so none is invented. The workbook's own field still reads `claude_adjudicator_1`; that value is kept in `workbook_adjudicator_id_original`. |
| v0.2 reading of a single annotator's span (restores a team rejection) | 4 | `single annotator (restored ...; not adjudicated)` |

Span types are not in either workbook. `span_type_source` says where each came from: both annotators agree, inherited from the annotator whose span was chosen, or assigned by the v0.2 5.3 test (assistant-derived, 26 conflicts pending D10).

## Files

| File | Content |
|---|---|
| `pilot_batch_01_PROVISIONAL_consolidated.csv` | 190 rows, 22-column validator schema, 50 reviews. |
| `pilot_batch_01_PROVISIONAL_provenance.csv` | Per span: tier, decision origin, adjudicator IDs, `span_type_source`, `training_eligible`, `blockers`, `depends_on_decisions`, flags, sources. |
| `rejected_span_audit_trail.csv` | The 51 rejected spans plus ADJ-109: team rule and reason, v0.2 re-assessment, action, confidence. |
| `validator_report.json`, `summary.json` | Validator output and counts. |

## Tiers

| Tier | Rows | Status | `training_eligible` |
|---|---|---|---|
| `CONFIRMED_AGREED` | 63 | `ANNOTATED` | **yes** |
| `AGREED_OFFSET_ADJUDICATED` (27 individual decisions, D1/D9) | 27 | `ADJUDICATED` | pending_policy |
| `ADJUDICATED_POLICY_PENDING` (team decisions that cite an unapproved rule) | 74 | `ADJUDICATED` | pending_policy |
| `ADJUDICATED_FLAGGED` (low confidence, v0.2 conflict or ambiguity, missing facet note) | 10 | `NEEDS_REVIEW` | no |
| `AGREED_TYPE_PROVISIONAL` (span, category, sentiment agreed; type conflict) | 11 | `NEEDS_REVIEW` | no |
| `RESTORED_PROVISIONAL` (ADJ-003, 118, 124, 138; `NEUTRAL`) | 4 | `NEEDS_REVIEW` | no |
| `UNRESOLVED` (ADJ-109, empty sentiment) | 1 | `NEEDS_REVIEW` | no |

No team decision is currently free of a policy dependency (`ADJUDICATED_HUMAN` is empty), because almost every one cites a rule from the unapproved set. Rows pending policy by decision (a row can wait on several): D1 88, D9 27, D3 17, D5 15, D10 14, D2 6, D6 2.
The 27 offset rows wait on D1/D9 only (plus D10 where the annotators' span types differ). Case 10 (`tail`) is a new sub-span; cases 7, 10 and 22 were accepted as provisional.

ADJ-059 ('LOVES it') keeps the team's `POSITIVE`; it contradicts v0.2 section 6 (hearsay is `NEUTRAL`), so it is flagged `V0.2_CONFLICT` and `NEEDS_REVIEW`.

## Rejected spans (51)

40 stay rejected (v0.2 supports the team rejection), 4 are restored as provisional rows (flagged as overriding a team rejection), 7 need a human decision and are not in the file: ADJ-002, 047, 075, 083, 091, 103, 137. 13 of the 51 audit calls are low confidence.

## Validation

* Normal validator: **0 errors, 36 warnings** (35 `MIXED_STATUSES`, 1 `UNFINISHED_MESSAGES`).
* Strict validator: **fails by design**: 17 of 50 messages still contain a `NEEDS_REVIEW` row.
* 190 of 190 offsets reproduce the span text; 0 duplicate spans; 0 overlapping spans; only ADJ-109 has an empty sentiment; all categories are in the current taxonomy (23 `OTHER_ASPECT` rows).
* Tests: 12 for the consolidation script (`tests/test_provisional_annotations.py`), passing together with the adjudication-package and validator tests (65).
* Rows by status: 63 `ANNOTATED`, 101 `ADJUDICATED`, 26 `NEEDS_REVIEW`. Rows by eligibility: 63 yes, 101 pending_policy, 26 no. Reviews with at least one eligible row: 39 (49 counting pending_policy).

## Remaining blockers

1. D1 to D11 approvals (above). Approving D1 releases 32 rows by itself.
2. ADJ-109 and the 7 rejected spans needing a decision; the 8 low-confidence groups, ADJ-066/106 (ambiguous) and ADJ-059 (v0.2 conflict).
3. Two `OTHER_ASPECT` rows lack the facet note ('ribbon tie', 'ribbon ties').
4. Only 61 examples in the 63 yes rows are training eligible now; the pilot cannot meet the floors by itself. Annotate batch 02 (`docs/annotation_requirements_and_batch_02.md`).
