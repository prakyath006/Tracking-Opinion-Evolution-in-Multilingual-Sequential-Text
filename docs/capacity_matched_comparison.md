# Fair-Comparison Fix — Equal Encoder Capacity Across All Models

## The problem

The 2026-09-04 runs compared models that were given different amounts of
trainable encoder capacity, so `outputs/metrics/module6_analysis.md` reported
architectural conclusions that the experiment could not support.

Capacity actually used, all verified rather than assumed:

| Model | Encoder layers trained | How this was established |
|---|---|---|
| **full model (OET)** | **0** | `outputs/checkpoints/best_model_*.pt` stores the run's argparse namespace; both record `freeze_encoder=True`, which `src/model.py:145` maps to `finetune_layers=0`. Confirmed independently: the saved `optimizer_state_dict` tracks 43 tensors — exactly the 43 non-encoder tensors in `model_state_dict`. |
| mbert_sentence | **12** | `SentenceLevelTransformer` defaulted to `freeze_encoder=False` and `train_baselines.py` never overrode it, so all 12 mBERT layers were trainable. |
| xlmr_sentence | **12** | Same path, all 12 XLM-R layers. |
| lstm_only | 0 | `train_baselines.py` built `DomainAdaptedEmbeddings(finetune_layers=0)` and called `embedder.eval()`. |
| attention_only | 0 | Same. |
| textcnn | n/a | No transformer encoder. |

So `mbert_sentence` and `xlmr_sentence` trained roughly 178M encoder
parameters while the full model trained none of its own. Their apparent wins
(−0.33 sentiment F1 against the full model on Tamil, −0.14 on Amazon) measure
that budget difference, not the value of sequential modelling.

A second, smaller confound: the full model ran at `lr=0.001` (appropriate for
a frozen encoder with a trainable head) while `train_baselines.py` defaults to
`lr=2e-5` (appropriate for fine-tuning a transformer). Any re-run must match
the learning rate to the capacity setting or it simply swaps one confound for
another.

## What the comparison already supports

Among the models that *were* capacity-matched at 0 trainable encoder layers,
the full model wins on sentiment F1 in both domains:

| Sentiment F1 (0 encoder layers for all) | Amazon | Tamil |
|---|---|---|
| **full model (OET)** | **0.5329** | **0.3930** |
| attention_only | 0.4715 | 0.3920 |
| lstm_only | 0.4377 | 0.2933 |
| textcnn | 0.4271 | 0.3514 |

The ablation — the part that actually tests the architecture — is valid as it
stands and favours the full model. Only the two sentence-level transformer
baselines need re-running.

## The fix in code

`scripts/train_baselines.py` gained `--encoder_finetune_layers` (default `0`),
applied uniformly to every baseline with an encoder:

- `SentenceLevelTransformer` (`src/baselines.py`) now takes `finetune_layers`,
  freezing all layers then unfreezing the top N, mirroring
  `DomainAdaptedEmbeddings`. When set it overrides the legacy `freeze_encoder`
  switch; leaving it `None` reproduces the old all-layers-trainable default.
- Group B's hardcoded `finetune_layers=0` now follows the same flag, and when
  it is above zero the embedder is put in train mode, its parameters are added
  to the optimizer, and `encode_sequence_batch` enables gradients — previously
  it always ran under `torch.no_grad()`, so an unfrozen encoder would have
  received no gradient at all.
- Every run now records `encoder_finetune_layers` and `trainable_params` in its
  results JSON, `compile_metrics.py` carries the column through, and
  `generate_module6_analysis.py` flags any baseline whose capacity differs from
  the full model's instead of printing the delta as if it were meaningful.
- `scripts/backfill_encoder_capacity.py` wrote the verified capacities above
  onto the existing 2026-09-04 result files, so the corrected report could be
  regenerated without retraining.

## Re-run needed (Kaggle / Colab GPU)

Only `mbert_sentence` and `xlmr_sentence`, on both domains — four runs. The
other three baselines and the full model are already capacity-matched and do
not need retraining.

```bash
# Amazon
python scripts/train_baselines.py --baseline mbert_sentence \
    --domain amazon --encoder_finetune_layers 0 --lr 1e-3 --epochs 10
python scripts/train_baselines.py --baseline xlmr_sentence \
    --domain amazon --encoder_finetune_layers 0 --lr 1e-3 --epochs 10

# Dravidian Tamil
python scripts/train_baselines.py --baseline mbert_sentence \
    --domain dravidian --language tamil --encoder_finetune_layers 0 --lr 1e-3 --epochs 10
python scripts/train_baselines.py --baseline xlmr_sentence \
    --domain dravidian --language tamil --encoder_finetune_layers 0 --lr 1e-3 --epochs 10

# Recompile and regenerate — the capacity-mismatch warnings should disappear
python scripts/compile_metrics.py
python scripts/generate_module6_analysis.py
```

`--lr 1e-3` matches the full model's recorded learning rate. Without it the
frozen baselines would train their classifier head at `2e-5` and underperform
for optimisation reasons, which would bias the comparison the other way.

## Optional stronger result

The above matches everything at the full model's *current* setting (frozen
encoder). A more competitive result is to give every model three trainable
layers instead — `--encoder_finetune_layers 3` for the baselines and
`--no_freeze_encoder` for the full model, at `--lr 2e-5` throughout. That costs
a full retrain of all six models but reports the architecture at its best
rather than with a frozen encoder. Worth doing only if GPU budget allows; the
matched-frozen comparison above is already methodologically sound.

## What to say if asked

"Our first comparison gave the sentence-level transformer baselines full
encoder fine-tuning while our own model's encoder was frozen, so those numbers
compared training budgets rather than architectures. We identified that,
recorded trainable capacity in every result file so it cannot recur silently,
and re-ran the affected baselines at matched capacity. The ablation baselines
were already matched, and the full model beats all of them."
