# Design: shared XLM-R encoder with separate Amazon and Dravidian task heads

**Status: design only. Nothing is implemented or trained.** It starts only after the separate baselines are finished: E0, E3c, E4 on Amazon, and the DravidianCodeMix comment-level run (`docs/dravidian_evaluation.md`). The existing checkpoints, results and the OET model files are not touched; the combined model would be a new module and a new entry point.

## 1. Architecture

```
                         +--> Amazon branch:  [reviews of one user, up to 20] -> pooled review vectors (768)
 text -> XLM-R encoder --+        -> input projection -> 2-layer BiLSTM -> attention -> sentiment head (per review, 4)
 (shared; frozen, then              (reused OET)                                     -> trend head (per transition, 3)
  top 2 layers trainable)                                                            -> trajectory head (per user, 4)
                         +--> Dravidian branch: [one comment] -> pooled vector (768)
                                  -> language-aware sentence head (768-256-4), one head per language (ta, ml, kn)
```
* **Shared:** the XLM-R weights and the pooling (mean pooling, 256 tokens, as in E3c/E4). Nothing else is shared in version 1.
* **Amazon branch:** exactly the current OET code (`OpinionEvolutionTracker` without its own encoder, fed by the shared encoder), so single-task Amazon results stay comparable.
* **Dravidian branch:** the head of `scripts/eval_comment_level.py` (Dropout, Linear 256, LayerNorm, ReLU, Dropout, Linear 4), one per language. A single shared Dravidian head is an ablation, not the default.
* **No sequence model on Dravidian.** Comments have no order, so the BiLSTM, attention, trend and trajectory heads never see them.
* **Encoder state:** the combined checkpoint stores only the changed layers when the encoder is partially trained (`includes_encoder` flag, as `train_experiment.py` already does); a frozen encoder is not stored.

## 2. Dataset-to-task mapping

| Task | Data | Unit and splits | Heads trained | Supervision |
|---|---|---|---|---|
| A: opinion evolution | `data/preprocessed/amazon_beauty_sequences.csv`, existing `AmazonSequenceDataset` split by user (train 3,726 / val 798 / test 800 sequences; 15,619 / 3,512 / 3,391 reviews) | a user's chronological reviews | sentiment, trend (position 0 masked), trajectory | labels derived from ratings by the ontology, as now |
| D: comment sentiment | `data/comment_level/{tamil,malayalam,kannada}_{train,dev,test}.csv` (53,263 / 6,431 / 6,398 comments) | one comment | the language's sentence head | human sentiment labels, official splits with dev/test overlap removed |
| (future) aspect sentiment | the annotated Amazon batches | review plus aspect | not in this design | separate third task, later |

A and D never share an example, and D is never turned into sequences, windows or pseudo-threads.

## 3. Label harmonization

* **Shared sentiment space:** ontology `SentimentState` 0 POSITIVE, 1 NEGATIVE, 2 MIXED, 3 UNKNOWN for both tasks; the same four output units.
* **Amazon:** UNKNOWN never occurs (ratings of 4 or more are POSITIVE, 2 or less NEGATIVE, 3 is MIXED). Its UNKNOWN unit is never a target; a predicted UNKNOWN counts as an error at evaluation, as in the existing protocol.
* **Dravidian:** `Mixed_feelings` is MIXED and `unknown_state` is UNKNOWN (a real annotated label, 13% to 36% of training comments); `not-<language>` rows are already dropped.
* **The labels mean different things.** A rating-3 review is a proxy for MIXED, a Dravidian MIXED is an annotator's judgement. For that reason the two sentiment heads are separate and the classes are not pooled in any metric. The comparable view is the existing 3-class POSITIVE/NEGATIVE/MIXED macro-F1, reported per task.
* **Trend and trajectory:** Amazon only. Dravidian rows get no target (ignore index `-1`/`-100`), so they contribute zero gradient to those heads.

## 4. Multi-task loss

`L = w_A * (1.0 * L_sent_A + 0.5 * L_trend_A + 1.0 * L_traj_A) + w_D * L_sent_D`

* The Amazon inner weights are the current `MultiTaskLoss` defaults, with the E3c/E4 square-root class weights and the position-0 transition mask.
* `L_sent_D` is cross-entropy with square-root inverse-frequency weights from each language's training split.
* Start with `w_A = w_D = 1`. Each batch holds one task, so the loss of a step is that task's loss only (task-alternating batches, not mixed batches). Gradients are accumulated over one A step and one D step before the optimizer updates (the E4 accumulation machinery already exists).
* Treat `w_D` as a tuned hyper-parameter on validation data only (grid 0.25, 0.5, 1); do not use uncertainty weighting in version 1 (more moving parts for a first result).
* Learning rates as in E4: 2e-5 for the encoder layers, 1e-3 for heads.

## 5. Balanced sampling

* **Budget per step in texts encoded, not examples:** an Amazon step is 8 users (up to 160 reviews); a Dravidian step is about 128 comments of at most 256 tokens. This keeps memory and the encoder's gradient exposure comparable.
* **Task schedule:** alternate A and D steps 1:1 by default; run an A-heavy 2:1 variant as an ablation. One epoch is defined as one pass over the Amazon train users, so Amazon training matches the single-task runs.
* **Inside D, temperature sampling over languages:** `p_l proportional to n_l^0.5` gives about 49% Tamil, 32% Malayalam and 19% Kannada (against 63%, 27% and 10% by size), so Kannada is not drowned out. Within a language sample uniformly, with class weights in the loss rather than oversampling.
* Seeded and logged: the sampler, seed and per-task step counts go into `config.json`.

## 6. Leakage prevention

* Keep each task's own splits: users for Amazon, the official train/dev/test files for Dravidian. Never re-split or pool.
* Early stopping and the `w_D` and schedule choices use **validation/dev only**; test is scored once per final configuration.
* Re-run the exact text overlap check (train versus dev versus test, within and across languages) at the start of every run, as `dravidian_evaluation.py` does, and stop on any overlap.
* Check cross-task overlap too: Amazon review texts against Dravidian comments (expected none; the check proves it).
* The old Tamil checkpoint and its sliding-window data are not used, because its test windows share comments with training windows.
* A shared encoder must not see test texts during training (no masked-language-model or domain-adaptation pass over test or dev data).
* The Amazon annotation files and their gold labels are not inputs to this model.

## 7. Single-task baselines (control for negative transfer)

| Control | Purpose |
|---|---|
| Amazon only: E3c (frozen) and E4 (top 2 layers), same seeds and budget | the Amazon reference |
| Dravidian only, per language: frozen XLM-R (done by `dravidian_evaluation.py`) and top-2-layer fine-tuned XLM-R with the same sentence head | the Dravidian reference |
| Joint frozen encoder (stage 1: one shared embedding cache, separately trained heads) | shows the cost of nothing; must equal the single-task numbers |
| Joint fine-tuned (stage 2, this design) | the experiment |
| Joint fine-tuned with `w_D = 0` | isolates the effect of Dravidian data from the effect of the training recipe |

Negative transfer is declared when the joint model is worse than its single-task control on the **same** validation metric by more than the seed spread (at least 3 seeds each; E6 supplies this for Amazon) on a task, using paired bootstrap over products or users, never one seed.

## 8. Evaluation protocol

* **Amazon (task A):** the existing metrics on the unchanged test split: per-head accuracy and macro-F1, observed-position transition F1, trajectory F1, SCS, confusion matrices, plus the diagnostic against the ontology rule baseline. Comparison table via `scripts/compare_experiments.py`.
* **Dravidian (task D):** per language, accuracy, 4-class and 3-class macro-F1, per-class F1, confusion matrices, majority baseline, bootstrap interval, script subsets (native, romanised, mixed), exactly as `dravidian_evaluation.py`.
* **Reporting rules:** each task is reported on its own; no pooled score; no claim that Dravidian results demonstrate temporal opinion evolution, since Dravidian has no time axis. A shared-encoder benefit is claimed only for what each task's own test split shows.
* **Seeds:** 3 or more for every configuration in the final table; one seed only for exploration.
* **Cross-lingual probes (optional, labelled as such):** train on two Dravidian languages, test on the third; Amazon-trained encoder features on Dravidian are not evaluated in version 1.
* **Resources:** run on a Kaggle GPU; E4 already encodes up to 20 reviews x 8 users per step at 256 tokens, so the joint run uses batch 4 x accumulation 4 for A first and reduces the comment batch if memory is short.

## 9. Order of work and decision gates

1. Finish the separate baselines: Amazon E0, E3c, E4 with `compare_experiments.py`; the Dravidian XLM-R run on Kaggle.
2. Gate: proceed only if XLM-R partial fine-tuning (E4) clearly beats frozen (E3c) on Amazon, or the Dravidian run shows fine-tuning headroom. If not, stage 1 (frozen shared encoder) already answers the question and stage 2 is dropped.
3. Implement the multi-task trainer as a new script with unit tests on synthetic data (task masking, ignore indices, sampler shares, leakage checks), smoke-test on a few batches, then run the single-task controls and the joint model with identical seeds.
