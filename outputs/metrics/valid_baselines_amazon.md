# Valid reference scores — Amazon user histories

Produced by `scripts/compute_valid_baselines.py` from labels only (no model). Majority classes come from the training split; test labels are used only for scoring.

Users train/val/test: 3726/798/800.

| Head | Majority class (from train) | Test accuracy | Test macro-F1 | 95% CI over test users | Test n |
|---|---|---|---|---|---|
| sentiment (per review) | POSITIVE | 0.7523 | 0.2862 | 0.2817–0.2905 | 3391 |
| trend, all positions (includes the position-1 placeholder) | STABLE | 0.7576 | 0.2874 | 0.2830–0.2912 | 3391 |
| trend, observed transitions only (positions >= 2) | STABLE | 0.6827 | 0.2705 | 0.2644–0.2761 | 2591 |
| trajectory (per sequence) | STABLE | 0.4963 | 0.1658 | 0.1578–0.1729 | 800 |

Gold-sequence SCS on the 800 test users: mean 0.6699 (std 0.3574).

Class distributions per split are in the JSON file.
