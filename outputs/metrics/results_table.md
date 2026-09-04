| model | setting | source | target | encoder_finetune_layers | sentiment_accuracy | sentiment_f1_macro | trend_f1_macro | trajectory_accuracy | trajectory_f1_macro | scs_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| full_model (OpinionEvolutionTracker) | in-domain | amazon | amazon | 0 | 0.6700 | 0.5329 | 0.5749 | 0.5675 | 0.5471 | 0.5636 |
| full_model (OpinionEvolutionTracker) | in-domain | dravidian_tamil | dravidian_tamil | 0 | 0.5134 | 0.3930 | 0.4450 | 0.5283 | 0.2678 | 0.3062 |
| attention_only | in-domain | amazon | amazon | 0 | 0.5756 | 0.4715 | - | 0.4375 | 0.3307 | 0.5198 |
| attention_only | in-domain | dravidian_tamil | dravidian_tamil | 0 | 0.4860 | 0.3920 | - | 0.5850 | 0.2347 | 0.3087 |
| lstm_only | in-domain | amazon | amazon | 0 | 0.5665 | 0.4377 | - | 0.4600 | 0.4250 | 0.8925 |
| lstm_only | in-domain | dravidian_tamil | dravidian_tamil | 0 | 0.3863 | 0.2933 | - | 0.5176 | 0.2624 | 0.6585 |
| mbert_sentence | in-domain | amazon | amazon | 12 | 0.8186 | 0.6419 | - | - | - | - |
| mbert_sentence | in-domain | dravidian_tamil | dravidian_tamil | 12 | 0.7906 | 0.7206 | - | - | - | - |
| textcnn | in-domain | amazon | amazon | - | 0.5391 | 0.4271 | - | - | - | - |
| textcnn | in-domain | dravidian_tamil | dravidian_tamil | - | 0.4032 | 0.3514 | - | - | - | - |
| xlmr_sentence | in-domain | amazon | amazon | 12 | 0.8275 | 0.6768 | - | - | - | - |
| xlmr_sentence | in-domain | dravidian_tamil | dravidian_tamil | 12 | 0.7256 | 0.6626 | - | - | - | - |
| full_model (OpinionEvolutionTracker) | in-domain | amazon | amazon | - | 0.6700 | 0.5329 | - | 0.5675 | 0.5471 | 0.5636 |
| full_model (OpinionEvolutionTracker) | cross-domain | amazon | dravidian_tamil | - | 0.5429 | 0.2300 | - | 0.5086 | 0.2382 | 0.7555 |
| full_model (OpinionEvolutionTracker) | cross-domain | amazon | dravidian_malayalam | - | 0.3125 | 0.2038 | - | 0.2902 | 0.1747 | 0.4770 |
| full_model (OpinionEvolutionTracker) | cross-domain | amazon | dravidian_kannada | - | 0.4328 | 0.2214 | - | 0.4237 | 0.2698 | 0.5677 |
| full_model (OpinionEvolutionTracker) | in-domain | dravidian_tamil | dravidian_tamil | - | 0.5134 | 0.3930 | - | 0.5283 | 0.2678 | 0.3062 |
| full_model (OpinionEvolutionTracker) | cross-domain | dravidian_tamil | amazon | - | 0.3751 | 0.2505 | - | 0.4275 | 0.2284 | 0.4124 |
| full_model (OpinionEvolutionTracker) | cross-domain | dravidian_tamil | dravidian_malayalam | - | 0.3488 | 0.2878 | - | 0.4732 | 0.2268 | 0.2693 |
| full_model (OpinionEvolutionTracker) | cross-domain | dravidian_tamil | dravidian_kannada | - | 0.3985 | 0.2903 | - | 0.5611 | 0.2812 | 0.2605 |
