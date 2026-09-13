# Module 5 — Cross-Domain Generalization

## In-Domain vs. Cross-Domain F1

| Setup | Sentiment F1 | Trend F1 | Trajectory F1 | SCS |
|---|---|---|---|---|
| amazon_to_amazon | 0.5329 | 0.5749 | 0.5471 | 0.5636 |
| amazon_to_dravidian_tamil | 0.2300 | 0.3854 | 0.2382 | 0.7555 |
| amazon_to_dravidian_malayalam | 0.2038 | 0.3212 | 0.1747 | 0.4770 |
| amazon_to_dravidian_kannada | 0.2214 | 0.4037 | 0.2698 | 0.5677 |
| dravidian_tamil_to_dravidian_tamil | 0.3930 | 0.4450 | 0.2678 | 0.3062 |
| dravidian_tamil_to_amazon | 0.2505 | 0.3799 | 0.2284 | 0.4124 |
| dravidian_tamil_to_dravidian_malayalam | 0.2878 | 0.3028 | 0.2268 | 0.2693 |
| dravidian_tamil_to_dravidian_kannada | 0.2903 | 0.4645 | 0.2812 | 0.2605 |

### % Degradation (in-domain -> cross-domain, per head)

| Setup | Head | In-Domain F1 | Cross-Domain F1 | % Degradation |
|---|---|---|---|---|
| amazon_to_dravidian_tamil | sentiment | 0.5329 | 0.2300 | 56.8% |
| amazon_to_dravidian_tamil | trend | 0.5749 | 0.3854 | 33.0% |
| amazon_to_dravidian_tamil | trajectory | 0.5471 | 0.2382 | 56.5% |
| amazon_to_dravidian_malayalam | sentiment | 0.5329 | 0.2038 | 61.8% |
| amazon_to_dravidian_malayalam | trend | 0.5749 | 0.3212 | 44.1% |
| amazon_to_dravidian_malayalam | trajectory | 0.5471 | 0.1747 | 68.1% |
| amazon_to_dravidian_kannada | sentiment | 0.5329 | 0.2214 | 58.5% |
| amazon_to_dravidian_kannada | trend | 0.5749 | 0.4037 | 29.8% |
| amazon_to_dravidian_kannada | trajectory | 0.5471 | 0.2698 | 50.7% |
| dravidian_tamil_to_amazon | sentiment | 0.3930 | 0.2505 | 36.2% |
| dravidian_tamil_to_amazon | trend | 0.4450 | 0.3799 | 14.6% |
| dravidian_tamil_to_amazon | trajectory | 0.2678 | 0.2284 | 14.7% |
| dravidian_tamil_to_dravidian_malayalam | sentiment | 0.3930 | 0.2878 | 26.8% |
| dravidian_tamil_to_dravidian_malayalam | trend | 0.4450 | 0.3028 | 31.9% |
| dravidian_tamil_to_dravidian_malayalam | trajectory | 0.2678 | 0.2268 | 15.3% |
| dravidian_tamil_to_dravidian_kannada | sentiment | 0.3930 | 0.2903 | 26.1% |
| dravidian_tamil_to_dravidian_kannada | trend | 0.4450 | 0.4645 | -4.4% |
| dravidian_tamil_to_dravidian_kannada | trajectory | 0.2678 | 0.2812 | -5.0% |

## F1 Stability Ratio (per head)

mean(F1) / std(F1) across every evaluated domain/setup, for one task head.

| Head | Ratio | Mean F1 | Std F1 | Setups Used |
|---|---|---|---|---|
| sentiment | 2.72 | 0.3012 | 0.1108 | 8 |
| trend | 4.74 | 0.4097 | 0.0864 | 8 |
| trajectory | 2.46 | 0.2793 | 0.1133 | 8 |

## Fuzzy Domain-Typicality Scores

Summary of `outputs/fuzzy_domain_scores.csv`. `argmax_matches_own_domain_pct` = how often a domain's own test sequences score *highest* on that domain's own centroid (a sanity signal, not a correctness guarantee).

| Domain | Sequences | Mean Own-Domain Score | Argmax Matches Own Domain |
|---|---|---|---|
| amazon_beauty | 800 | 0.7059 | 91.9% |
| dravidian_kannada | 262 | 0.3822 | 79.4% |
| dravidian_malayalam | 727 | 0.3887 | 82.1% |
| dravidian_tamil | 1677 | 0.3784 | 79.2% |

Not yet correlated against cross-domain degradation (per task scope) — these scores are available for that analysis once both this table and the degradation table above are populated from the same real trained checkpoints.
