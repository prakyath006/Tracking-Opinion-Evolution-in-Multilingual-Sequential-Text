# Panel 1 checkpoints: technical compatibility report

Read-only inspection of `outputs/checkpoints/best_model_amazon.pt` and `best_model_dravidian_tamil.pt` (each 748 MB; `hf_space/outputs/checkpoints/` holds files of identical size, not hash-compared). Both were opened with `mmap` and only keys, shapes, saved arguments and weights were read. Nothing was merged, overwritten, retrained, deleted or evaluated; no experiment result was modified.

## 1. What the checkpoints are

| | `best_model_amazon.pt` | `best_model_dravidian_tamil.pt` |
|---|---|---|
| Saved fields | `epoch` 6, `val_loss` 2.235, `model_state_dict`, `optimizer_state_dict`, `args` | `epoch` 8, `val_loss` 2.981, same fields |
| Architecture (identical) | `OpinionEvolutionTracker`: mBERT encoder, input projection 768 to 256, 2-layer bidirectional LSTM (hidden 256), single-head attention (dim 128), three heads | same |
| Encoder | `bert-base-multilingual-cased` (vocab 119,547), **frozen**: all 199 encoder tensors are bit-identical to the pretrained model, so about 710 MB of each file is a redundant copy of public weights | same |
| Trainable part | about 3.0 M parameters (BiLSTM 2.63 M, attention 66 k, heads about 200 k); optimizer has one group, lr 1e-3 | same |
| Heads (label classes) | sentiment 4 (POSITIVE, NEGATIVE, MIXED, UNKNOWN); trend 3 (UPGRADE, DOWNGRADE, STABLE); trajectory 4 (IMPROVING, DECLINING, STABLE, VOLATILE) | same |
| Saved config | CLS pooling, 128 tokens, sequence length 20, batch 16, dropout 0.3, class weights from the training data, weight decay 1e-4, patience 5, 10 epochs | same |
| Data (from `src/dataset.py`, `scripts/train.py`) | `AmazonSequenceDataset`: real user review histories, split 70/15/15 by user, seed 42 | `DravidianSequenceDataset`: **sliding windows of 5 consecutive comments (stride 3) cut from the preprocessed TRAIN file, then split at random 70/15/15** |
| Original test result (`outputs/logs/test_results_*.json`) | sentiment macro-F1 0.533, trend 0.575, trajectory 0.547, SCS 0.564 (800 test sequences) | sentiment macro-F1 0.393, trend 0.445, trajectory 0.268 (two classes with F1 0), SCS 0.306 (1,677 test windows) |

## 2. Validity of the old training data

* **Amazon:** a real temporal sequence per user. This is the setting OET is designed for.
* **Tamil:** the dataset has no timestamps, authors or threads, so the "sequences" are consecutive-row windows (code comment: "pseudo-threads"). The trajectory and trend labels derived from them describe the file's row order, not time. Consecutive windows overlap by 2 comments and were assigned to splits at random, so a test window shares comments with training windows by construction; `scripts/audit_tamil_checkpoint_leakage.py` exists to measure this (only its aborted CPU log is stored locally, no result). The official dev/test files were never used. These results must not be presented as multilingual temporal evolution.

## 3. Compatibility with the current pipelines

| Pipeline | Result |
|---|---|
| Current OET (`src/model.py` as used by `scripts/train_experiment.py`; mBERT, frozen, CLS, 128) | **Both load with `strict=True`** ("All keys matched", checked on CPU, no forward pass). Existing scripts (`demo_end_to_end.py`, `cross_domain_eval.py`, `diagnose_trajectory_rule.py`) already load them. The Amazon file is equivalent to the E0 reference configuration |
| Current OET with XLM-R (E2 to E4) | **Not loadable as a whole**: XLM-R's vocabulary is 250,002 against 119,547, and the two encoders have different embedding spaces. The BiLSTM, attention and heads have the same shapes, so they could be copied, but they were fitted to mBERT CLS vectors and are not a meaningful initialisation |
| DravidianCodeMix comment-level evaluation (`scripts/dravidian_evaluation.py`) | **Not compatible**: that pipeline classifies one comment with a head on [CLS] (768 to 256 to 4); the checkpoint's sentiment head sits on the attention output of a BiLSTM over a sequence (512 to 128 to 4). It could only score a comment as a sequence of length 1, outside the training distribution (windows of 5) |

## 4. Reuse

* **Amazon checkpoint, inference:** yes. It is a valid frozen-mBERT baseline and demo model, comparable only to E0 (see section 5).
* **Tamil checkpoint, inference:** not as a DravidianCodeMix baseline. If the team wants it as a documented extra, score it on the cleaned official test split with comments fed as length-1 sequences (its training used only the official train file), labelled "pseudo-sequence model, out-of-distribution input". Run it on a GPU machine, not the laptop.
* **Transfer learning:** not recommended. Heads from either checkpoint cannot initialise an XLM-R model usefully, and the Tamil heads were trained on artificial sequences. If a lightweight artifact is wanted, extract the roughly 12 MB of non-encoder weights into a new file (the originals stay untouched).
* **Compression:** because the encoder is frozen and identical to the public model, new checkpoints of frozen runs need not store it (the new pipeline already records `includes_encoder`).

## 5. Comparison with E3c and E4

The E3c and E4 results are **not in the repository** (no `outputs/experiments/` directory; `docs/kaggle_experiments.md` holds only the commands), so no metric comparison is possible here. Design differences only:

| | Old Amazon checkpoint | E0 (new protocol) | E3c | E4 |
|---|---|---|---|---|
| Encoder | mBERT frozen | mBERT frozen | XLM-R frozen | XLM-R, top 2 layers trained |
| Pooling / length | CLS / 128 | CLS / 128 | mean / 256 | mean / 256 |
| Class weights | full inverse frequency | full | sqrt | sqrt |
| Selection / schedule | validation loss, constant lr | validation macro-F1, same lr | same, 10 epochs | linear schedule, lr 2e-5 encoder and 1e-3 head, 8 epochs |
| Other | transition position 0 scored | position 0 masked as an artificial placeholder | same | same |

The old checkpoint's numbers are therefore only comparable with E0 after re-scoring under the new protocol (masking and metric choice differ). To compare, paste the E0, E3c and E4 `summary.json` files.

## 6. Shared XLM-R encoder with task-specific heads for Amazon plus Dravidian

**Feasible, in two stages.**

1. **Frozen shared encoder (low risk, can be done now).** One XLM-R, one embedding cache, an OET sequence branch for Amazon and a comment-level head for Dravidian. This is effectively what `train_experiment.py` (E3c) and `dravidian_evaluation.py` already do separately, so the work is organisation, not new modelling. No interference is possible.
2. **Jointly fine-tuned top layers (needs new code).** Alternate batches: Amazon user sequences through BiLSTM, attention and the three OET heads; Dravidian comments through a sentence head. Loss `L = a * L_OET + b * L_comment`. Requirements and risks:
   * a multi-task trainer and a mixed data loader do not exist yet;
   * the sentiment label space is shared (ontology 0 to 3; Amazon never contains UNKNOWN, so mask it) but each language needs its own head or a language-aware head, and Dravidian classes are far more imbalanced;
   * data sizes differ greatly (Dravidian 53 k training comments against Amazon user sequences), so use temperature sampling and per-task early stopping;
   * negative transfer: compare against single-task E4 and keep the single-task result as the control;
   * memory: E4 already encodes up to 20 reviews x 8 users per step at 256 tokens; adding a comment batch needs a smaller batch or gradient accumulation;
   * **never feed Dravidian pseudo-windows into the trend or trajectory heads.** Dravidian supervises only the comment-level sentiment head. The Amazon trajectory results and the Dravidian sentiment results stay separate claims.

## 7. Recommendations

1. Keep both checkpoints untouched; treat the Amazon one as the E0-equivalent baseline and the Tamil one as a historical pseudo-sequence artefact that is excluded from the multilingual claims.
2. Report DravidianCodeMix only through the new comment-level protocol and describe it as a comment-level (non-temporal) experiment.
3. Re-score the Amazon checkpoint under the new protocol before comparing it with E0 to E4, and add the E3c/E4 summaries to the repository outputs for a real comparison.
4. Do stage 1 of section 6 first; build stage 2 only if the frozen shared-encoder results justify fine-tuning.
5. Run the optional Tamil-checkpoint inference and any GPU work on Kaggle, not locally.
