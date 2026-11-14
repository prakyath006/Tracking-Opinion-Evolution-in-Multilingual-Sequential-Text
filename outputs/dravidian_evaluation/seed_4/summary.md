# DravidianCodeMix comment-level sentiment (xlm-roberta-base, frozen encoder)

**Scope.** Comment-level sentiment only. DravidianCodeMix has no timestamps, authors, threads or ordering and none were invented. These results say nothing about opinion evolution over time; the Amazon temporal experiments are separate.

Encoder `xlm-roberta-base` (frozen), seeds [4], device cuda. Splits: official train/dev/test with dev/test texts already seen in an earlier split removed; the head is fitted on train, early-stopped on dev, and test is only scored.

## Data
| Language | train | dev | test | test POS / NEG / MIXED / UNKNOWN |
|---|---|---|---|---|
| tamil | 33531 | 4145 | 4112 | 2451 / 544 / 499 / 618 |
| malayalam | 14514 | 1671 | 1654 | 708 / 259 / 111 / 576 |
| kannada | 5218 | 615 | 632 | 335 / 142 / 72 / 83 |

Leakage check (exact normalised text overlap, must be 0): within-language {'tamil': {'train_dev': 0, 'train_test': 0, 'dev_test': 0}, 'malayalam': {'train_dev': 0, 'train_test': 0, 'dev_test': 0}, 'kannada': {'train_dev': 0, 'train_test': 0, 'dev_test': 0}}; cross-language train vs other test 0 texts in total.

## Results (mean over seeds)
| Setup -> test | acc (4-class) | macro-F1 (4-class) | macro-F1 (3-class POS/NEG/MIX) | majority macro-F1 (4-class) | F1 POS / NEG / MIXED / UNKNOWN |
|---|---|---|---|---|---|
| tamil->tamil | 0.448 +- 0.000 | 0.391 +- 0.000 | 0.432 +- 0.000 | 0.187 | 0.592 / 0.357 / 0.268 / 0.347 |
| malayalam->malayalam | 0.515 +- 0.000 | 0.437 +- 0.000 | 0.438 +- 0.000 | 0.150 | 0.618 / 0.317 / 0.225 / 0.589 |
| kannada->kannada | 0.483 +- 0.000 | 0.417 +- 0.000 | 0.424 +- 0.000 | 0.173 | 0.559 / 0.510 / 0.155 / 0.443 |
| joint->tamil | 0.456 +- 0.000 | 0.395 +- 0.000 | 0.438 +- 0.000 | 0.187 | 0.608 / 0.339 / 0.284 / 0.347 |
| joint->malayalam | 0.459 +- 0.000 | 0.375 +- 0.000 | 0.330 +- 0.000 | 0.150 | 0.434 / 0.363 / 0.096 / 0.607 |
| joint->kannada | 0.498 +- 0.000 | 0.399 +- 0.000 | 0.422 +- 0.000 | 0.173 | 0.621 / 0.494 / 0.100 / 0.379 |

## Confusion matrices (4-class, first seed; rows gold, columns predicted: POS, NEG, MIXED, UNKNOWN)
* `tamil->tamil`: [[1116, 473, 425, 437], [58, 283, 120, 83], [86, 133, 184, 96], [61, 151, 147, 259]]
* `malayalam->malayalam`: [[395, 59, 131, 123], [51, 77, 62, 69], [19, 18, 47, 27], [105, 73, 66, 332]]
* `kannada->kannada`: [[138, 132, 28, 37], [4, 114, 11, 13], [8, 36, 10, 18], [9, 23, 8, 43]]
* `joint->tamil`: [[1171, 435, 492, 353], [63, 257, 150, 74], [90, 128, 215, 66], [76, 153, 156, 233]]
* `joint->malayalam`: [[213, 252, 53, 190], [14, 159, 17, 69], [9, 62, 10, 30], [37, 145, 17, 377]]
* `joint->kannada`: [[169, 115, 11, 40], [10, 101, 5, 26], [11, 34, 5, 22], [19, 17, 7, 40]]

## Script subsets of each test split (first seed; derived from the text characters)
| Setup -> test | subset | n | acc (4-class) | macro-F1 (4-class) | macro-F1 (3-class) |
|---|---|---|---|---|---|
| tamil->tamil | mixed_script | 217 | 0.461 | 0.411 | 0.445 |
| tamil->tamil | native_script | 613 | 0.584 | 0.449 | 0.511 |
| tamil->tamil | romanised | 3282 | 0.422 | 0.372 | 0.408 |
| malayalam->malayalam | mixed_script | 319 | 0.608 | 0.487 | 0.449 |
| malayalam->malayalam | native_script | 443 | 0.546 | 0.454 | 0.437 |
| malayalam->malayalam | romanised | 892 | 0.465 | 0.408 | 0.436 |
| kannada->kannada | mixed_script | 93 | 0.634 | 0.388 | 0.530 |
| kannada->kannada | native_script | 128 | 0.609 | 0.352 | 0.487 |
| kannada->kannada | other (n < 30: unreliable) | 1 | 1.000 | 0.250 | n/a |
| kannada->kannada | romanised | 410 | 0.407 | 0.370 | 0.349 |
| joint->tamil | mixed_script | 217 | 0.452 | 0.414 | 0.459 |
| joint->tamil | native_script | 613 | 0.566 | 0.431 | 0.515 |
| joint->tamil | romanised | 3282 | 0.436 | 0.381 | 0.417 |
| joint->malayalam | mixed_script | 319 | 0.536 | 0.419 | 0.341 |
| joint->malayalam | native_script | 443 | 0.530 | 0.388 | 0.321 |
| joint->malayalam | romanised | 892 | 0.396 | 0.339 | 0.315 |
| joint->kannada | mixed_script | 93 | 0.581 | 0.409 | 0.487 |
| joint->kannada | native_script | 128 | 0.625 | 0.406 | 0.470 |
| joint->kannada | other (n < 30: unreliable) | 1 | 1.000 | 0.250 | n/a |
| joint->kannada | romanised | 410 | 0.439 | 0.367 | 0.378 |
