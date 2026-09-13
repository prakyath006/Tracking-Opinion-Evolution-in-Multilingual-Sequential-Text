# Panel Review-2 Comments — Status

No file tracking this table existed anywhere in the repository before this
one (checked: no match for "Panel Review", "panel comment", or any of the
row labels below across `README.md`, `docs/`, or `outputs/`). Created fresh
on 2026-09-13 rather than located, since the panel comment items had already
been worked through this project's own `docs/defect_register.md` (O1-O5) and
two prior sessions' commits without a table matching this exact rubric.
Every status below is checked against real files, not asserted.

| Panel Comment | Status | Evidence |
|---|---|---|
| Baselines / ablation | **Completed** | 5 architectural baselines (`mbert_sentence`, `xlmr_sentence`, `lstm_only`, `attention_only`, `textcnn`) trained and evaluated at capacity matched to the full model (0 trainable encoder layers throughout). D1 (unequal trainable capacity) found, fixed, and closed by a real re-run — see `docs/defect_register.md` D1/O1, verified against `outputs/logs/baseline_*.json` and `outputs/metrics/results_table.md`. The ablation itself (`lstm_only` vs `attention_only` vs full model) was valid from the start. |
| One well-defined task | **Completed** | Sentiment (`SentimentState`, 4 classes) + trend (`TransitionType`, 3 classes) + trajectory (`TrajectoryType`, 4 classes) classification over per-user review sequences, defined once in `src/ontology.py` and shared across every domain and every model that supports it — see `outputs/metrics/module1_ontology.md` (0 label conflicts across 7 pooled raw label sources). |
| Multilingual data, two domains | **Completed** | Amazon Beauty (English, e-commerce) and Dravidian Tamil (code-mixed, social media) both have real trained full-model checkpoints, in-domain results, and cross-domain transfer results in both directions — `outputs/metrics/results_table.md`, `outputs/metrics/module5_cross_domain.md`. Malayalam and Kannada have cross-domain transfer results (as transfer *targets*) but no dedicated full-model training run of their own — this does not block the "two domains" requirement, which Amazon + Tamil satisfy on their own. |
| LLM classification + cross-domain eval | **In Progress — Colab run pending** | LLM-based baseline (`llm_prompt`, `google/flan-t5-base`) implemented in `src/baselines.py` (commit `b9a3d3c`), wired into the same training/eval loop as every other baseline, and verified correct via a real local smoke test on real Amazon review text (using the smaller `flan-t5-small`, since `flan-t5-base`'s download alone timed out in this CPU-only environment). **Not yet run on the real Amazon/Tamil test sets** — no accuracy/F1 number exists for it anywhere. `notebooks/colab_training.ipynb` section 8c has the exact ready-to-run GPU cell; running it is the only remaining step. Cross-domain evaluation itself (for the full model) is already Completed — see the row above — this row is held back specifically by the LLM baseline half not having real numbers yet. |
| 100+ examples per class (balanced comparison) | **Partial — out of scope for this pass** | Explicitly assigned to a teammate; not touched here. Prior status stands: no such balanced evaluation exists in the codebase yet (confirmed absent as of the last audit that checked for it). |

## What would move the one "In Progress" row to Completed

Run `notebooks/colab_training.ipynb` section 8c on a GPU (Colab or
equivalent) for both `--domain amazon` and `--domain dravidian --language
tamil`. That writes `outputs/logs/baseline_llm_prompt_<run_id>.json` for
each domain, exactly like every other baseline already does. Re-running
`python scripts/compile_metrics.py` and `python scripts/generate_module6_analysis.py`
afterward folds the real numbers into `outputs/metrics/results_table.md` and
`module6_analysis.md` automatically — no new code or report format is
needed, only the run itself.

## Keeping this file honest going forward

Update this table only against real files (`outputs/metrics/`,
`outputs/logs/`) the same way `docs/defect_register.md` now asks of itself —
this table drifting out of sync with actual status is exactly the failure
mode that document documented and fixed twice (O1, O4) in prior sessions.
