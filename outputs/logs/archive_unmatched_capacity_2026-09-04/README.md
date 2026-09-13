# Archived — superseded baseline runs (12 trainable encoder layers)

These are the original 2026-09-04 Kaggle results for `mbert_sentence` and
`xlmr_sentence`. They are kept for the record only and must NOT be used in
any comparison against the full model.

Both baselines trained all 12 encoder layers (~178M parameters) because
`SentenceLevelTransformer` defaulted to `freeze_encoder=False`, while the
full model ran with `freeze_encoder=True` and trained 0 encoder layers.
Their apparent wins measured training budget, not architecture.

| Run | Sentiment F1 (unmatched, 12 layers) |
|---|---|
| mbert_sentence / amazon | 0.6419 |
| xlmr_sentence / amazon | 0.6768 |
| mbert_sentence / dravidian_tamil | 0.7206 |
| xlmr_sentence / dravidian_tamil | 0.6626 |

Replaced by capacity-matched re-runs (`--encoder_finetune_layers 0 --lr 1e-3`).
See docs/capacity_matched_comparison.md.
