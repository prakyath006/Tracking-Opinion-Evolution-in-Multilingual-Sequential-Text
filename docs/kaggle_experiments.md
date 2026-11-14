# Controlled OET experiments on Kaggle

`scripts/train_experiment.py` is a new entry point. It does not change `scripts/train.py`, `src/model.py`, the dataset split
(70/15/15 by user, seed 42, at most 20 reviews per user), the label definitions, or `outputs/checkpoints/best_model_amazon.pt`.
Every run writes its own directory under `outputs/experiments/<name>/` and refuses to reuse an existing one.

## What changed relative to the original training run
| Item | Original `train.py` | `train_experiment.py` |
|---|---|---|
| Position-0 transition (artificial STABLE) | trained on and scored | masked in the loss, the class weights and every metric |
| Class weights | full inverse frequency | `--class_weights none / sqrt / full`, from training labels only |
| Encoder | mBERT, frozen | `--encoder mbert / xlmr`; `--finetune frozen / partial / full` |
| Learning rate | one | `--lr_head` and `--lr_encoder` |
| Pooling, token length | CLS, 128 | `--pooling cls / mean`, `--max_token_length` (128, 256, ...) |
| Checkpoint choice | lowest total validation loss | `--selection_metric` on validation (default: mean of the three macro-F1) |
| Logged per epoch | loss, a few validation scores | accuracy and macro-F1 for sentiment, observed transitions, trajectory |

Test data is used once, after training, with the checkpoint chosen on validation. It never enters selection, early stopping
or the learning-rate schedule. Use `--no_test` to skip it entirely.

Metric definitions: macro-F1 is over the three determinate sentiment classes, the three transition classes (observed positions only)
and the four trajectory classes. These can differ slightly from the numbers `train.py` reported, so compare experiments with each
other, not with the old log. Run experiment E0 below to get a like-for-like reference.

## Outputs per experiment
`config.json`, `class_weights.json`, `train_log.jsonl`, `history.json`, `best_checkpoint.pt`, `best_validation.json`,
`test_metrics.json`, `predictions/{positions,sequences}.csv`, `summary.json`.
A frozen encoder is not saved in the checkpoint (about 12 MB); a trained encoder is (about 700 MB).
The prediction files can be passed straight to the trajectory diagnostic:
`python scripts/diagnose_trajectory_rule.py --phase analyze --out_dir outputs/experiments/<name>`.

## Kaggle setup (GPU on, Internet on)
The repository on GitHub must contain these new files first, so push before cloning (or upload the folder as a dataset).
```python
!git clone https://github.com/prakyath006/Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text.git
%cd Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text
!pip -q install -r requirements.txt
!mkdir -p data && cp -r /kaggle/input/datasets/prakyathnandigam/opinion-tracker-data/data/* data/
!ls data/preprocessed | head
!python -m pytest tests/test_experiment_utils.py -q
```
Only `data/preprocessed/amazon_beauty_sequences.csv` is needed. No existing checkpoint is needed.

## Recommended experiments
Same seed (0) first, one change at a time against E0. Epoch time is unmeasured on Kaggle for the new script: the original run took about
170 s per epoch for a frozen mBERT at 128 tokens, so budget roughly 2x for 256 tokens and 3 to 5x for fine-tuning, and time E0 first.

| ID | Question | Settings (changes from E0 in bold) |
|---|---|---|
| E0 | Reference under the new protocol | mBERT, frozen, CLS, 128, full weights, constant LR |
| E1 | Do softer weights help? | E0 with **`--class_weights sqrt`**, then **`none`** |
| E2 | Does the encoder matter? | **XLM-R**, frozen, CLS, 128, sqrt weights |
| E3 | Pooling and length | XLM-R frozen with **`--pooling mean`**; then **`--max_token_length 256`**; then both |
| E4 | Partial fine-tuning | XLM-R, **`--finetune partial --finetune_layers 2`**, `--lr_encoder 2e-5 --lr_head 1e-3`, `--schedule linear`, batch 8 x accumulation 2, 8 epochs |
| E6 | Seeds | repeat the best two configurations with seeds 1 and 2 |

Memory: a step encodes every review of every user in the batch with gradients when the encoder trains. Not tested on a GPU. If E4 runs
out of memory, halve `--batch_size` and double `--grad_accumulation`.

## Commands
Run one cell per block. Set `-u` to see progress as it happens.
```python
# E0 reference
!python -u scripts/train_experiment.py --name E0_mbert_frozen_cls128_wfull --encoder mbert --finetune frozen --pooling cls \
    --max_token_length 128 --class_weights full --epochs 10 --batch_size 16 --seed 0

# E1 class weights
!python -u scripts/train_experiment.py --name E1a_mbert_frozen_wsqrt --encoder mbert --class_weights sqrt --epochs 10 --seed 0
!python -u scripts/train_experiment.py --name E1b_mbert_frozen_wnone --encoder mbert --class_weights none --epochs 10 --seed 0

# E2 encoder (use the better weight mode from E1; sqrt shown)
!python -u scripts/train_experiment.py --name E2_xlmr_frozen_cls128_wsqrt --encoder xlmr --class_weights sqrt --epochs 10 --seed 0

# E3 pooling and length
!python -u scripts/train_experiment.py --name E3a_xlmr_frozen_mean128 --encoder xlmr --pooling mean --class_weights sqrt --epochs 10 --seed 0
!python -u scripts/train_experiment.py --name E3b_xlmr_frozen_cls256 --encoder xlmr --max_token_length 256 --class_weights sqrt --epochs 10 --seed 0
!python -u scripts/train_experiment.py --name E3c_xlmr_frozen_mean256 --encoder xlmr --pooling mean --max_token_length 256 --class_weights sqrt --epochs 10 --seed 0

# E4 partial fine-tuning (use the best pooling and length from E3)
!python -u scripts/train_experiment.py --name E4_xlmr_top2_mean256 --encoder xlmr --finetune partial --finetune_layers 2 \
    --pooling mean --max_token_length 256 --class_weights sqrt --lr_encoder 2e-5 --lr_head 1e-3 --schedule linear \
    --batch_size 8 --grad_accumulation 2 --epochs 8 --seed 0

# E6 seeds (example for E4)
for s in 1 2:
    !python -u scripts/train_experiment.py --name E6_E4config_seed{s} --encoder xlmr --finetune partial --finetune_layers 2 \
        --pooling mean --max_token_length 256 --class_weights sqrt --lr_encoder 2e-5 --lr_head 1e-3 --schedule linear \
        --batch_size 8 --grad_accumulation 2 --epochs 8 --seed {s}
```
After the runs:
```python
!python scripts/summarize_experiments.py
!python scripts/diagnose_trajectory_rule.py --phase analyze --out_dir outputs/experiments/E4_xlmr_top2_mean256
# download everything except the checkpoints (the fine-tuned ones are about 700 MB each)
!zip -r experiments.zip outputs/experiments -x "*best_checkpoint.pt"
```
Keep a checkpoint only for the configuration you finally choose, and download it from the notebook's Output panel.
Choose the final configuration on the validation columns of `summary_table.csv`; the test columns are for reporting the chosen one.
