# Module 4 — Sequential Model Metrics

Domain: `dravidian_tamil` | Split: `test` | Encoder: `bert-base-multilingual-cased`

## Sentiment Head

- Accuracy: 0.5134
- Precision (macro): 0.4095
- Recall (macro): 0.4257
- F1 (macro): 0.3930
- Confusion matrix (POSITIVE, NEGATIVE, MIXED, UNKNOWN):
```
[[3057 1152  293  438]
 [ 210  673  117   79]
 [ 256  389  158  134]
 [ 363  462  187  417]]
```
- Prediction entropy: mean=1.1981 nats, std=0.2388 nats (n=8385)
- Expected Calibration Error: 0.0588

## Trend Head

- Accuracy: 0.4766
- Precision (macro): 0.4953
- Recall (macro): 0.6164
- F1 (macro): 0.4450
- Confusion matrix (UPGRADE, DOWNGRADE, STABLE):
```
[[ 749  264   65]
 [ 190  801   58]
 [1795 2017 2446]]
```
- Prediction entropy: mean=0.7268 nats, std=0.3850 nats (n=8385)
- Expected Calibration Error: 0.1860

## Trajectory Head

- Accuracy: 0.5283
- Precision (macro): 0.2846
- Recall (macro): 0.2901
- F1 (macro): 0.2678
- Confusion matrix (IMPROVING, DECLINING, STABLE, VOLATILE):
```
[[  0   0   7  20]
 [  0   0   3  19]
 [  0   0 415 579]
 [  0   0 163 471]]
```
- Prediction entropy: mean=1.0211 nats, std=0.1222 nats (n=1677)
- Expected Calibration Error: 0.0302

## Sequence Consistency Score (SCS)

- SCS mean: 0.3062
- SCS std: 0.2307
- SCS min/max: 0.0000 / 1.0000
- Sequences scored: 1677

**SCS Reliability Ratio: 1.33** (mean=0.3062, std=0.2307)
