# DravidianCodeMix comment-level sentiment (xlm-roberta-base, frozen encoder)

**Scope.** Comment-level sentiment only. DravidianCodeMix has no timestamps, authors, threads or ordering and none were invented. These results say nothing about opinion evolution over time; the Amazon temporal experiments are separate.

Encoder `xlm-roberta-base` (frozen), seeds [3], device cuda. Splits: official train/dev/test with dev/test texts already seen in an earlier split removed; the head is fitted on train, early-stopped on dev, and test is only scored.

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
| tamil->tamil | 0.477 +- 0.000 | 0.387 +- 0.000 | 0.424 +- 0.000 | 0.187 | 0.642 / 0.339 / 0.217 / 0.349 |
| malayalam->malayalam | 0.491 +- 0.000 | 0.424 +- 0.000 | 0.396 +- 0.000 | 0.150 | 0.520 / 0.358 / 0.200 / 0.619 |
| kannada->kannada | 0.472 +- 0.000 | 0.437 +- 0.000 | 0.451 +- 0.000 | 0.173 | 0.579 / 0.524 / 0.203 / 0.443 |
| joint->tamil | 0.486 +- 0.000 | 0.399 +- 0.000 | 0.456 +- 0.000 | 0.187 | 0.659 / 0.325 / 0.292 / 0.322 |
| joint->malayalam | 0.460 +- 0.000 | 0.368 +- 0.000 | 0.327 +- 0.000 | 0.150 | 0.442 / 0.371 / 0.067 / 0.591 |
| joint->kannada | 0.487 +- 0.000 | 0.365 +- 0.000 | 0.378 +- 0.000 | 0.173 | 0.600 / 0.488 / 0.000 / 0.373 |

## Confusion matrices (4-class, first seed; rows gold, columns predicted: POS, NEG, MIXED, UNKNOWN)
* `tamil->tamil`: [[1289, 673, 102, 387], [75, 338, 39, 92], [113, 207, 84, 95], [85, 231, 50, 252]]
* `malayalam->malayalam`: [[273, 100, 145, 190], [18, 99, 57, 85], [10, 23, 42, 36], [42, 72, 64, 398]]
* `kannada->kannada`: [[148, 60, 92, 35], [9, 82, 37, 14], [8, 20, 25, 19], [11, 9, 20, 43]]
* `joint->tamil`: [[1356, 426, 360, 309], [89, 241, 143, 71], [113, 126, 199, 61], [107, 144, 164, 203]]
* `joint->malayalam`: [[220, 274, 22, 192], [14, 174, 4, 67], [11, 67, 5, 28], [43, 164, 7, 362]]
* `joint->kannada`: [[168, 118, 3, 46], [17, 101, 2, 22], [15, 38, 0, 19], [25, 15, 4, 39]]

## Script subsets of each test split (first seed; derived from the text characters)
| Setup -> test | subset | n | acc (4-class) | macro-F1 (4-class) | macro-F1 (3-class) |
|---|---|---|---|---|---|
| tamil->tamil | mixed_script | 217 | 0.488 | 0.426 | 0.461 |
| tamil->tamil | native_script | 613 | 0.592 | 0.443 | 0.498 |
| tamil->tamil | romanised | 3282 | 0.455 | 0.362 | 0.394 |
| malayalam->malayalam | mixed_script | 319 | 0.577 | 0.482 | 0.443 |
| malayalam->malayalam | native_script | 443 | 0.526 | 0.435 | 0.406 |
| malayalam->malayalam | romanised | 892 | 0.443 | 0.394 | 0.371 |
| kannada->kannada | mixed_script | 93 | 0.613 | 0.483 | 0.568 |
| kannada->kannada | native_script | 128 | 0.609 | 0.369 | 0.512 |
| kannada->kannada | other (n < 30: unreliable) | 1 | 1.000 | 0.250 | n/a |
| kannada->kannada | romanised | 410 | 0.395 | 0.386 | 0.373 |
| joint->tamil | mixed_script | 217 | 0.461 | 0.401 | 0.446 |
| joint->tamil | native_script | 613 | 0.576 | 0.410 | 0.479 |
| joint->tamil | romanised | 3282 | 0.471 | 0.387 | 0.438 |
| joint->malayalam | mixed_script | 319 | 0.542 | 0.407 | 0.321 |
| joint->malayalam | native_script | 443 | 0.503 | 0.358 | 0.288 |
| joint->malayalam | romanised | 892 | 0.409 | 0.342 | 0.336 |
| joint->kannada | mixed_script | 93 | 0.591 | 0.374 | 0.440 |
| joint->kannada | native_script | 128 | 0.531 | 0.302 | 0.367 |
| joint->kannada | other (n < 30: unreliable) | 1 | 1.000 | 0.250 | n/a |
| joint->kannada | romanised | 410 | 0.449 | 0.352 | 0.357 |
