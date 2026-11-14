# DravidianCodeMix comment-level sentiment (xlm-roberta-base, frozen encoder)

**Scope.** Comment-level sentiment only. DravidianCodeMix has no timestamps, authors, threads or ordering and none were invented. These results say nothing about opinion evolution over time; the Amazon temporal experiments are separate.

Encoder `xlm-roberta-base` (frozen), seeds [2], device cuda. Splits: official train/dev/test with dev/test texts already seen in an earlier split removed; the head is fitted on train, early-stopped on dev, and test is only scored.

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
| tamil->tamil | 0.468 +- 0.000 | 0.403 +- 0.000 | 0.442 +- 0.000 | 0.187 | 0.617 / 0.351 / 0.276 / 0.368 |
| malayalam->malayalam | 0.474 +- 0.000 | 0.424 +- 0.000 | 0.423 +- 0.000 | 0.150 | 0.549 / 0.349 / 0.219 / 0.580 |
| kannada->kannada | 0.503 +- 0.000 | 0.422 +- 0.000 | 0.433 +- 0.000 | 0.173 | 0.596 / 0.535 / 0.127 / 0.427 |
| joint->tamil | 0.446 +- 0.000 | 0.385 +- 0.000 | 0.426 +- 0.000 | 0.187 | 0.603 / 0.311 / 0.287 / 0.339 |
| joint->malayalam | 0.449 +- 0.000 | 0.370 +- 0.000 | 0.325 +- 0.000 | 0.150 | 0.387 / 0.351 / 0.149 / 0.591 |
| joint->kannada | 0.475 +- 0.000 | 0.409 +- 0.000 | 0.430 +- 0.000 | 0.173 | 0.581 / 0.516 / 0.152 / 0.387 |

## Confusion matrices (4-class, first seed; rows gold, columns predicted: POS, NEG, MIXED, UNKNOWN)
* `tamil->tamil`: [[1198, 545, 371, 337], [69, 302, 104, 69], [93, 158, 176, 72], [71, 173, 127, 247]]
* `malayalam->malayalam`: [[312, 88, 182, 126], [31, 94, 74, 60], [13, 19, 58, 21], [73, 79, 104, 320]]
* `kannada->kannada`: [[152, 99, 23, 61], [7, 106, 5, 24], [7, 31, 7, 27], [9, 18, 3, 53]]
* `joint->tamil`: [[1147, 286, 610, 408], [55, 185, 221, 83], [87, 85, 261, 66], [63, 90, 225, 240]]
* `joint->malayalam`: [[181, 146, 90, 291], [13, 114, 29, 103], [6, 38, 21, 46], [28, 92, 30, 426]]
* `joint->kannada`: [[149, 84, 41, 61], [8, 91, 9, 34], [10, 25, 11, 26], [11, 11, 12, 49]]

## Script subsets of each test split (first seed; derived from the text characters)
| Setup -> test | subset | n | acc (4-class) | macro-F1 (4-class) | macro-F1 (3-class) |
|---|---|---|---|---|---|
| tamil->tamil | mixed_script | 217 | 0.456 | 0.432 | 0.469 |
| tamil->tamil | native_script | 613 | 0.563 | 0.438 | 0.487 |
| tamil->tamil | romanised | 3282 | 0.451 | 0.386 | 0.420 |
| malayalam->malayalam | mixed_script | 319 | 0.605 | 0.504 | 0.467 |
| malayalam->malayalam | native_script | 443 | 0.535 | 0.451 | 0.427 |
| malayalam->malayalam | romanised | 892 | 0.397 | 0.374 | 0.403 |
| kannada->kannada | mixed_script | 93 | 0.634 | 0.408 | 0.486 |
| kannada->kannada | native_script | 128 | 0.656 | 0.431 | 0.549 |
| kannada->kannada | other (n < 30: unreliable) | 1 | 1.000 | 0.250 | n/a |
| kannada->kannada | romanised | 410 | 0.424 | 0.369 | 0.353 |
| joint->tamil | mixed_script | 217 | 0.424 | 0.369 | 0.397 |
| joint->tamil | native_script | 613 | 0.569 | 0.382 | 0.453 |
| joint->tamil | romanised | 3282 | 0.424 | 0.376 | 0.409 |
| joint->malayalam | mixed_script | 319 | 0.520 | 0.411 | 0.346 |
| joint->malayalam | native_script | 443 | 0.542 | 0.440 | 0.385 |
| joint->malayalam | romanised | 892 | 0.377 | 0.327 | 0.299 |
| joint->kannada | mixed_script | 93 | 0.570 | 0.420 | 0.502 |
| joint->kannada | native_script | 128 | 0.633 | 0.415 | 0.522 |
| joint->kannada | other (n < 30: unreliable) | 1 | 1.000 | 0.250 | n/a |
| joint->kannada | romanised | 410 | 0.402 | 0.361 | 0.360 |
