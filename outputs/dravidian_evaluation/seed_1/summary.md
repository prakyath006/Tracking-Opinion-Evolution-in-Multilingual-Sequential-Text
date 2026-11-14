# DravidianCodeMix comment-level sentiment (xlm-roberta-base, frozen encoder)

**Scope.** Comment-level sentiment only. DravidianCodeMix has no timestamps, authors, threads or ordering and none were invented. These results say nothing about opinion evolution over time; the Amazon temporal experiments are separate.

Encoder `xlm-roberta-base` (frozen), seeds [1], device cuda. Splits: official train/dev/test with dev/test texts already seen in an earlier split removed; the head is fitted on train, early-stopped on dev, and test is only scored.

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
| tamil->tamil | 0.464 +- 0.000 | 0.382 +- 0.000 | 0.417 +- 0.000 | 0.187 | 0.626 / 0.335 / 0.217 / 0.349 |
| malayalam->malayalam | 0.489 +- 0.000 | 0.433 +- 0.000 | 0.454 +- 0.000 | 0.150 | 0.577 / 0.379 / 0.222 / 0.552 |
| kannada->kannada | 0.511 +- 0.000 | 0.456 +- 0.000 | 0.466 +- 0.000 | 0.173 | 0.617 / 0.532 / 0.200 / 0.474 |
| joint->tamil | 0.493 +- 0.000 | 0.409 +- 0.000 | 0.461 +- 0.000 | 0.187 | 0.662 / 0.330 / 0.305 / 0.341 |
| joint->malayalam | 0.495 +- 0.000 | 0.405 +- 0.000 | 0.371 +- 0.000 | 0.150 | 0.497 / 0.368 / 0.151 / 0.605 |
| joint->kannada | 0.522 +- 0.000 | 0.414 +- 0.000 | 0.437 +- 0.000 | 0.173 | 0.656 / 0.538 / 0.077 / 0.384 |

## Confusion matrices (4-class, first seed; rows gold, columns predicted: POS, NEG, MIXED, UNKNOWN)
* `tamil->tamil`: [[1224, 736, 107, 384], [61, 352, 50, 81], [98, 233, 87, 81], [78, 234, 60, 246]]
* `malayalam->malayalam`: [[349, 160, 100, 99], [36, 140, 39, 44], [18, 37, 38, 18], [98, 143, 54, 281]]
* `kannada->kannada`: [[162, 82, 58, 33], [10, 97, 21, 14], [8, 28, 18, 18], [10, 16, 11, 46]]
* `joint->tamil`: [[1357, 266, 396, 432], [88, 195, 166, 95], [111, 83, 223, 82], [95, 93, 178, 252]]
* `joint->malayalam`: [[259, 158, 31, 260], [25, 129, 14, 91], [12, 51, 14, 34], [39, 105, 15, 417]]
* `joint->kannada`: [[186, 73, 14, 62], [14, 91, 4, 33], [16, 24, 4, 28], [16, 8, 10, 49]]

## Script subsets of each test split (first seed; derived from the text characters)
| Setup -> test | subset | n | acc (4-class) | macro-F1 (4-class) | macro-F1 (3-class) |
|---|---|---|---|---|---|
| tamil->tamil | mixed_script | 217 | 0.493 | 0.451 | 0.485 |
| tamil->tamil | native_script | 613 | 0.594 | 0.445 | 0.510 |
| tamil->tamil | romanised | 3282 | 0.438 | 0.347 | 0.371 |
| malayalam->malayalam | mixed_script | 319 | 0.596 | 0.503 | 0.484 |
| malayalam->malayalam | native_script | 443 | 0.508 | 0.421 | 0.422 |
| malayalam->malayalam | romanised | 892 | 0.441 | 0.399 | 0.442 |
| kannada->kannada | mixed_script | 93 | 0.667 | 0.510 | 0.594 |
| kannada->kannada | native_script | 128 | 0.648 | 0.430 | 0.529 |
| kannada->kannada | other (n < 30: unreliable) | 1 | 1.000 | 0.250 | n/a |
| kannada->kannada | romanised | 410 | 0.432 | 0.404 | 0.390 |
| joint->tamil | mixed_script | 217 | 0.447 | 0.387 | 0.435 |
| joint->tamil | native_script | 613 | 0.577 | 0.395 | 0.475 |
| joint->tamil | romanised | 3282 | 0.480 | 0.403 | 0.448 |
| joint->malayalam | mixed_script | 319 | 0.583 | 0.460 | 0.394 |
| joint->malayalam | native_script | 443 | 0.551 | 0.472 | 0.441 |
| joint->malayalam | romanised | 892 | 0.436 | 0.361 | 0.342 |
| joint->kannada | mixed_script | 93 | 0.613 | 0.419 | 0.520 |
| joint->kannada | native_script | 128 | 0.656 | 0.388 | 0.492 |
| joint->kannada | other (n < 30: unreliable) | 1 | 1.000 | 0.250 | n/a |
| joint->kannada | romanised | 410 | 0.459 | 0.381 | 0.384 |
