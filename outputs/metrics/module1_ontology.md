# Ontology Evaluation Report

Standalone evaluation of `src/ontology.py` — coverage, internal consistency, and structural size. Separate from model accuracy: this measures the ontology itself, not any classifier trained against it.

## (a) Coverage

Fraction of each domain's real raw labels/ratings that map to a real sentiment (POSITIVE/NEGATIVE/MIXED) vs fall through to UNKNOWN. Domains below 80% are flagged.

| Domain | Total | Non-UNKNOWN | UNKNOWN | Coverage | Flag |
|---|---|---|---|---|---|
| amazon_beauty | 23,736 | 23,736 | 0 | 100.0% | ✅ |
| dravidian_kannada | 5,218 | 4,542 | 676 | 87.0% | ✅ |
| dravidian_malayalam | 14,514 | 9,335 | 5,179 | 64.3% | 🚩 LOW COVERAGE |
| dravidian_tamil | 33,531 | 27,988 | 5,543 | 83.5% | ✅ |

**🚩 Flagged: dravidian_malayalam — coverage below 80%.** Investigate before treating this domain's sentiment labels as reliable.

Per-domain notes:
- **amazon_beauty**: SentimentState.from_rating() has no UNKNOWN branch — every 1-5 star rating maps to POSITIVE/NEGATIVE/MIXED by construction. 100% coverage here reflects the rating scale having no 'unclear' option, not an absence of ambiguous opinions.
- **dravidian_kannada**: 'unknown_state' is a real DravidianCodeMix annotation label (annotator marked the comment's sentiment as unclear), not a mapping failure, so UNKNOWN here is expected to be nonzero.
- **dravidian_malayalam**: 'unknown_state' is a real DravidianCodeMix annotation label (annotator marked the comment's sentiment as unclear), not a mapping failure, so UNKNOWN here is expected to be nonzero.
- **dravidian_tamil**: 'unknown_state' is a real DravidianCodeMix annotation label (annotator marked the comment's sentiment as unclear), not a mapping failure, so UNKNOWN here is expected to be nonzero.

## (b) Consistency

Checked 7 distinct raw label strings pooled across all 4 domains' `label_mapping` dicts in `DOMAIN_CONFIGS`.

✅ **Consistent.** No raw label string maps to more than one SentimentState across any domain's mapping.

This mirrors `tests/test_ontology_consistency.py::test_domain_label_mapping_targets_valid_states`, which asserts each domain's mapping targets are valid SentimentState members (fails CI on violation); this report additionally checks for cross-domain label collisions, which the pass/fail test suite does not.

## (c) Structural Metrics

- **Depth**: 4 taxonomy layers (Sentiment State, Transition, Trajectory, Domain Ontology)
- **Breadth**: 15 leaf concepts (4 sentiment + 3 transition + 4 trajectory + 4 domain configs)
- **Coupling**: 15 source files import from `ontology.py`, computed by scanning `src/`, `scripts/`, `tests/` at report-generation time:
  - `scripts\convert_amazon_sequences_to_csv.py`
  - `scripts\export_ontology_output.py`
  - `scripts\sanity_check_pipeline.py`
  - `scripts\train.py`
  - `scripts\train_baselines.py`
  - `scripts\verify_guide_modules.py`
  - `src\classifier.py`
  - `src\confidence_eval.py`
  - `src\dataset.py`
  - `src\evaluation.py`
  - `src\fuzzy_domain_score.py`
  - `src\mlm_perplexity_eval.py`
  - `src\model.py`
  - `src\ontology_eval.py`
  - `tests\test_ontology_consistency.py`


## Ontology Coverage Stability Ratio

mean(coverage %) / std(coverage %) across domains with data — a single number for how evenly the ontology covers its domains.

**Ratio: 5.68** (mean=83.7%, std=14.7%)

Computed over 4 domain(s) with data (amazon_beauty, dravidian_kannada, dravidian_malayalam, dravidian_tamil).
