# Module 4 — Sequential Model Metrics

Domain: `amazon` | Split: `test` | Encoder: `bert-base-multilingual-cased`

## Sentiment Head

- Accuracy: 0.6700
- Precision (macro): 0.5308
- Recall (macro): 0.5758
- F1 (macro): 0.5329
- Confusion matrix (POSITIVE, NEGATIVE, MIXED, UNKNOWN):
```
[[1841  186  524    0]
 [  65  276  136    0]
 [ 101  107  155    0]
 [   0    0    0    0]]
```
- Prediction entropy: mean=0.7436 nats, std=0.2908 nats (n=3391)
- Expected Calibration Error: 0.0888

## Trend Head

- Accuracy: 0.6609
- Precision (macro): 0.5556
- Recall (macro): 0.6957
- F1 (macro): 0.5749
- Confusion matrix (UPGRADE, DOWNGRADE, STABLE):
```
[[ 262   74   45]
 [  60  335   46]
 [ 425  500 1644]]
```
- Prediction entropy: mean=0.5780 nats, std=0.3633 nats (n=3391)
- Expected Calibration Error: 0.0994

## Trajectory Head

- Accuracy: 0.5675
- Precision (macro): 0.5423
- Recall (macro): 0.6105
- F1 (macro): 0.5471
- Confusion matrix (IMPROVING, DECLINING, STABLE, VOLATILE):
```
[[ 66   8   8   5]
 [ 17  81  13  17]
 [ 58  58 208  73]
 [ 30  41  18  99]]
```
- Prediction entropy: mean=0.8373 nats, std=0.2718 nats (n=800)
- Expected Calibration Error: 0.1013

## Sequence Consistency Score (SCS)

- SCS mean: 0.5636
- SCS std: 0.3383
- SCS min/max: 0.0000 / 1.0000
- Sequences scored: 800

**SCS Reliability Ratio: 1.67** (mean=0.5636, std=0.3383)
