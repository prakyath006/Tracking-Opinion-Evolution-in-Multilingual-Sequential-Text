# Module 3 — BERT Encoder Comparison (MLM Perplexity)

Masked-language-model perplexity per domain, for each encoder considered as a standalone language model (via `AutoModelForMaskedLM` -- a separate model instance from the frozen encoder the classification pipeline actually uses; this does not touch or affect that pipeline).

Capped at 500 test examples per domain.

| Encoder | Domain | Perplexity | Avg Loss | Texts Used | Texts Skipped | Masked Tokens |
|---|---|---|---|---|---|---|
| bert-base-multilingual-cased | amazon_beauty | 19.35 | 2.9626 | 451 | 49 | 4,332 |
| xlm-roberta-base | amazon_beauty | 6.38 | 1.8533 | 444 | 56 | 4,287 |
| bert-base-multilingual-cased | dravidian_kannada | 107.67 | 4.6790 | 399 | 101 | 1,559 |
| xlm-roberta-base | dravidian_kannada | 216.79 | 5.3789 | 378 | 122 | 1,220 |
| bert-base-multilingual-cased | dravidian_malayalam | 78.06 | 4.3575 | 467 | 33 | 2,172 |
| xlm-roberta-base | dravidian_malayalam | 111.51 | 4.7141 | 465 | 35 | 1,756 |
| bert-base-multilingual-cased | dravidian_tamil | 151.97 | 5.0237 | 450 | 50 | 1,724 |
| xlm-roberta-base | dravidian_tamil | 118.11 | 4.7716 | 448 | 52 | 1,558 |
