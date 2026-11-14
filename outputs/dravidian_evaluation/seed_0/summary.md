# DravidianCodeMix comment-level sentiment (xlm-roberta-base, frozen encoder)

**Scope.** Comment-level sentiment only. DravidianCodeMix has no timestamps, authors, threads or ordering and none were invented. These results say nothing about opinion evolution over time; the Amazon temporal experiments are separate.

Encoder `xlm-roberta-base` (frozen), seeds [0], device cuda. Splits: official train/dev/test with dev/test texts already seen in an earlier split removed; the head is fitted on train, early-stopped on dev, and test is only scored.

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
| tamil->tamil | 0.479 +- 0.000 | 0.397 +- 0.000 | 0.433 +- 0.000 | 0.187 | 0.639 / 0.349 / 0.236 / 0.362 |
| malayalam->malayalam | 0.524 +- 0.000 | 0.445 +- 0.000 | 0.465 +- 0.000 | 0.150 | 0.642 / 0.333 / 0.239 / 0.565 |
| kannada->kannada | 0.505 +- 0.000 | 0.454 +- 0.000 | 0.471 +- 0.000 | 0.173 | 0.607 / 0.549 / 0.211 / 0.449 |
| joint->tamil | 0.474 +- 0.000 | 0.396 +- 0.000 | 0.453 +- 0.000 | 0.187 | 0.647 / 0.328 / 0.294 / 0.314 |
| joint->malayalam | 0.458 +- 0.000 | 0.381 +- 0.000 | 0.346 +- 0.000 | 0.150 | 0.442 / 0.380 / 0.107 / 0.593 |
| joint->kannada | 0.506 +- 0.000 | 0.408 +- 0.000 | 0.434 +- 0.000 | 0.173 | 0.624 / 0.512 / 0.115 / 0.381 |

## Confusion matrices (4-class, first seed; rows gold, columns predicted: POS, NEG, MIXED, UNKNOWN)
* `tamil->tamil`: [[1269, 543, 169, 470], [68, 303, 64, 109], [105, 169, 108, 117], [79, 175, 74, 290]]
* `malayalam->malayalam`: [[450, 47, 122, 89], [62, 77, 63, 57], [30, 16, 50, 15], [151, 63, 72, 290]]
* `kannada->kannada`: [[156, 75, 59, 45], [6, 95, 24, 17], [8, 23, 20, 21], [9, 11, 15, 48]]
* `joint->tamil`: [[1299, 281, 588, 283], [83, 200, 199, 62], [96, 85, 261, 57], [89, 108, 231, 190]]
* `joint->malayalam`: [[217, 218, 74, 199], [10, 160, 20, 69], [11, 58, 13, 29], [37, 146, 25, 368]]
* `joint->kannada`: [[174, 101, 13, 47], [15, 99, 4, 24], [14, 32, 6, 20], [20, 13, 9, 41]]

## Script subsets of each test split (first seed; derived from the text characters)
| Setup -> test | subset | n | acc (4-class) | macro-F1 (4-class) | macro-F1 (3-class) |
|---|---|---|---|---|---|
| tamil->tamil | mixed_script | 217 | 0.493 | 0.439 | 0.479 |
| tamil->tamil | native_script | 613 | 0.595 | 0.446 | 0.499 |
| tamil->tamil | romanised | 3282 | 0.456 | 0.376 | 0.406 |
| malayalam->malayalam | mixed_script | 319 | 0.599 | 0.485 | 0.463 |
| malayalam->malayalam | native_script | 443 | 0.537 | 0.443 | 0.425 |
| malayalam->malayalam | romanised | 892 | 0.491 | 0.419 | 0.479 |
| kannada->kannada | mixed_script | 93 | 0.667 | 0.501 | 0.596 |
| kannada->kannada | native_script | 128 | 0.688 | 0.461 | 0.572 |
| kannada->kannada | other (n < 30: unreliable) | 1 | 1.000 | 0.250 | n/a |
| kannada->kannada | romanised | 410 | 0.410 | 0.390 | 0.380 |
| joint->tamil | mixed_script | 217 | 0.447 | 0.394 | 0.459 |
| joint->tamil | native_script | 613 | 0.581 | 0.386 | 0.487 |
| joint->tamil | romanised | 3282 | 0.456 | 0.390 | 0.438 |
| joint->malayalam | mixed_script | 319 | 0.549 | 0.414 | 0.334 |
| joint->malayalam | native_script | 443 | 0.524 | 0.403 | 0.354 |
| joint->malayalam | romanised | 892 | 0.393 | 0.347 | 0.338 |
| joint->kannada | mixed_script | 93 | 0.581 | 0.406 | 0.489 |
| joint->kannada | native_script | 128 | 0.641 | 0.419 | 0.475 |
| joint->kannada | other (n < 30: unreliable) | 1 | 1.000 | 0.250 | n/a |
| joint->kannada | romanised | 410 | 0.446 | 0.378 | 0.393 |
