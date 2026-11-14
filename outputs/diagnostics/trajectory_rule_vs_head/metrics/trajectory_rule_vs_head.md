# Diagnostic: learned trajectory head vs ontology rule on predicted sentiments

800 Amazon test users, 3391 reviews; checkpoint `outputs/checkpoints/best_model_amazon.pt`; AmazonSequenceDataset(split='test'), unchanged.
Inference only; no retraining. rule applied to gold sentiments reproduces 100% of gold trajectories and gold transitions (asserted).

## Trajectory (per sequence)

| Method | Accuracy | Macro-F1 | 95% CI (sequences) |
|---|---|---|---|
| Learned trajectory head | 0.5675 | 0.5471 | 0.510 to 0.581 |
| Rule applied to predicted sentiments | 0.5400 | 0.5191 | 0.483 to 0.553 |

Head minus rule: macro-F1 +0.0280 (95% paired bootstrap -0.003 to 0.059); accuracy +0.0275 (-0.004 to 0.056).

The two agree on 74.1% of sequences. Where they disagree (n=207): head correct 88, rule correct 66, neither 53.

### Per-class F1 (trajectory)

| Class | Support | Head F1 | Rule F1 |
|---|---|---|---|
| IMPROVING | 87 | 0.512 | 0.474 |
| DECLINING | 128 | 0.513 | 0.488 |
| STABLE | 397 | 0.646 | 0.606 |
| VOLATILE | 188 | 0.518 | 0.509 |

**Learned head** (rows = gold, columns = predicted)

| gold \ pred | IMPROVING | DECLINING | STABLE | VOLATILE |
|---|---|---|---|---|
| IMPROVING | 66 | 8 | 8 | 5 |
| DECLINING | 17 | 81 | 13 | 17 |
| STABLE | 58 | 58 | 208 | 73 |
| VOLATILE | 30 | 41 | 18 | 99 |

**Rule** (rows = gold, columns = predicted)

| gold \ pred | IMPROVING | DECLINING | STABLE | VOLATILE |
|---|---|---|---|---|
| IMPROVING | 50 | 5 | 13 | 19 |
| DECLINING | 8 | 69 | 20 | 31 |
| STABLE | 53 | 45 | 197 | 102 |
| VOLATILE | 13 | 36 | 23 | 116 |

## Transitions, observed positions only (2591 evaluated; 800 placeholder positions excluded)

| Method | Accuracy | Macro-F1 |
|---|---|---|
| Learned trend head | 0.5562 | 0.5284 |
| Rule applied to predicted sentiments | 0.6137 | 0.5386 |

For reference, the learned trend head scored over all positions including the placeholder: 0.5749 macro-F1.

### Per-class F1 (transitions)

| Class | Support | Head F1 | Rule F1 |
|---|---|---|---|
| UPGRADE | 381 | 0.465 | 0.424 |
| DOWNGRADE | 441 | 0.496 | 0.482 |
| STABLE | 1769 | 0.624 | 0.710 |

**Learned trend head** (rows = gold, columns = predicted)

| gold \ pred | UPGRADE | DOWNGRADE | STABLE |
|---|---|---|---|
| UPGRADE | 262 | 74 | 45 |
| DOWNGRADE | 60 | 335 | 46 |
| STABLE | 425 | 500 | 844 |

**Rule** (rows = gold, columns = predicted)

| gold \ pred | UPGRADE | DOWNGRADE | STABLE |
|---|---|---|---|
| UPGRADE | 196 | 38 | 147 |
| DOWNGRADE | 24 | 247 | 170 |
| STABLE | 323 | 299 | 1147 |

Sentiment head for reference (3 determinate classes): accuracy 0.6700, macro-F1 0.5329.
