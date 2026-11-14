# Training run summary

Generated 2026-11-14 17:12:44.

## Stages
* **successful** (10): environment, checks, aspect_readiness, aspect_sentiment, aspect_sentiment_slim_checkpoint, dravidian_seed_0, dravidian_seed_1, dravidian_seed_2, dravidian_seed_3, dravidian_seed_4
* **failed** (0): none
* **skipped** (2): aspect_extraction, aspect_sequence
* **incomplete** (0): none
* **dry_run** (0): none

| stage | status | detail |
|---|---|---|
| environment | SUCCESS | GPU Tesla T4; internet ok |
| checks | SUCCESS | all required files exist, all code compiles, every CLI flag is accepted |
| aspect_readiness | SUCCESS | data, extraction safety and leakage checks pass; 459 sentiment examples, 30 products |
| aspect_sentiment | SUCCESS | best epoch 6; validation {'accuracy': 0.7837837837837838, 'macro_f1': 0.33242258652094714, 'macro_f1_present_classes': 0.33242258652094714, 'loss': 1.521604892369863}; test {'accuracy': 0.8507462686567164, 'macro_f1': 0.387624466571835, 'macro_f1_present_classes': 0.387624466571835, 'loss': 1.366655 |
| aspect_sentiment_slim_checkpoint | SUCCESS | 58 MB vs 1111 MB full; kept 39 tensors, dropped 165 identical pretrained tensors; rebuild verified |
| aspect_extraction | SKIPPED | no extraction evaluation is claimed. safe checks done: 35 messages BIO-encoded with the XLM-R tokenizer (83 B tags, alignment ok: True). Mentions per split {'train': 69, 'val': 6, 'test': 8} against floors (eval 100, train 500); products 18 (floor 30). Floors met: False. |
| aspect_sequence | SKIPPED | 50/20/20 usable-series floors are NOT met on any tested product split; no thresholds were lowered and --allow_small_data is not used. OOF generation blockers: none (OOF is not run because its only consumer, the sequence model, is blocked) |
| dravidian_seed_0 | SUCCESS | 4 setups trained, checkpoints and results saved |
| dravidian_seed_1 | SUCCESS | 4 setups trained, checkpoints and results saved |
| dravidian_seed_2 | SUCCESS | 4 setups trained, checkpoints and results saved |
| dravidian_seed_3 | SUCCESS | 4 setups trained, checkpoints and results saved |
| dravidian_seed_4 | SUCCESS | 4 setups trained, checkpoints and results saved |

## Aspect sentiment (agreed-only 459 examples; validation 74 and test 67 examples, below the 100 floor: noisy, a pipeline result)
best epoch 6; validation {'accuracy': 0.7837837837837838, 'macro_f1': 0.33242258652094714, 'macro_f1_present_classes': 0.33242258652094714, 'loss': 1.521604892369863}; test {'accuracy': 0.8507462686567164, 'macro_f1': 0.387624466571835, 'macro_f1_present_classes': 0.387624466571835, 'loss': 1.3666558585949797}

## DravidianCodeMix XLM-R (comment-level; mean +- std over finished seeds)
| setup -> test | seeds | acc (4-class) | macro-F1 (4-class) | macro-F1 (3-class) | majority macro-F1 |
|---|---|---|---|---|---|
| tamil_to_tamil | 5 | 0.467 | 0.392 +- 0.007 | 0.429 +- 0.008 | 0.187 |
| malayalam_to_malayalam | 5 | 0.498 | 0.433 +- 0.008 | 0.435 +- 0.024 | 0.150 |
| kannada_to_kannada | 5 | 0.495 | 0.437 +- 0.016 | 0.449 +- 0.018 | 0.173 |
| joint_to_tamil | 5 | 0.471 | 0.397 +- 0.008 | 0.447 +- 0.013 | 0.187 |
| joint_to_malayalam | 5 | 0.464 | 0.380 +- 0.014 | 0.340 +- 0.017 | 0.150 |
| joint_to_kannada | 5 | 0.498 | 0.399 +- 0.018 | 0.420 +- 0.022 | 0.173 |

Per-seed metrics, predictions and confusion matrices are in `dravidian/seed_<k>/`; summed confusion matrices are in `dravidian/aggregate_over_seeds.json`. This is a comment-level experiment; it does not show temporal opinion evolution.

## Checkpoints: all expected files present and non-empty
* OK aspect_sentiment/sentiment_xlmr_top2_agreed_only/best_checkpoint_slim.pt (57.51 MB)
* OK dravidian/seed_0/checkpoints/tamil_seed0.pt (0.80 MB)
* OK dravidian/seed_0/checkpoints/malayalam_seed0.pt (0.80 MB)
* OK dravidian/seed_0/checkpoints/kannada_seed0.pt (0.80 MB)
* OK dravidian/seed_0/checkpoints/joint_seed0.pt (0.80 MB)
* OK dravidian/seed_1/checkpoints/tamil_seed1.pt (0.80 MB)
* OK dravidian/seed_1/checkpoints/malayalam_seed1.pt (0.80 MB)
* OK dravidian/seed_1/checkpoints/kannada_seed1.pt (0.80 MB)
* OK dravidian/seed_1/checkpoints/joint_seed1.pt (0.80 MB)
* OK dravidian/seed_2/checkpoints/tamil_seed2.pt (0.80 MB)
* OK dravidian/seed_2/checkpoints/malayalam_seed2.pt (0.80 MB)
* OK dravidian/seed_2/checkpoints/kannada_seed2.pt (0.80 MB)
* OK dravidian/seed_2/checkpoints/joint_seed2.pt (0.80 MB)
* OK dravidian/seed_3/checkpoints/tamil_seed3.pt (0.80 MB)
* OK dravidian/seed_3/checkpoints/malayalam_seed3.pt (0.80 MB)
* OK dravidian/seed_3/checkpoints/kannada_seed3.pt (0.80 MB)
* OK dravidian/seed_3/checkpoints/joint_seed3.pt (0.80 MB)
* OK dravidian/seed_4/checkpoints/tamil_seed4.pt (0.80 MB)
* OK dravidian/seed_4/checkpoints/malayalam_seed4.pt (0.80 MB)
* OK dravidian/seed_4/checkpoints/kannada_seed4.pt (0.80 MB)
* OK dravidian/seed_4/checkpoints/joint_seed4.pt (0.80 MB)

## Not run
* OET E4 retraining
* OET E5
* aspect sequence / trajectory (floors not met)
* OOF generation
* Amazon + Dravidian joint multi-task training (design only)
