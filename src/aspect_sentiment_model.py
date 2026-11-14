"""
Aspect-conditioned sentiment classification.

Component 2 of 3 of the aspect-level architecture (independent of the extractor and sequence models; the existing OET model is not used).

The input is a sentence PAIR (aspect text, review text) encoded as in src/aspect_data.py (`encode_sentiment_examples`). The encoder reads the
aspect first and the review second, so the same review gives different outputs for different aspects. The output is the sentiment toward THAT
aspect, one of POSITIVE, NEGATIVE, NEUTRAL, MIXED (`aspect_data.AspectSentiment`; not the ontology's SentimentState). It is not a review-level
sentiment: a review with two aspects is scored twice, once per aspect.

    (aspect, review) --mBERT / XLM-R--> token vectors --pool (cls or masked mean)--> head --> 4 logits

NOT TRAINED: nothing here has been trained, and no performance is claimed.
"""
import os
import sys
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aspect_data import AspectSentiment, conditioning_text, category_phrase               # noqa: E402,F401
from aspect_encoders import POOLING, apply_encoder_mode, count_parameters, pool, resolve_encoder   # noqa: E402
from classifier import ClassificationHead                                                    # noqa: E402  (existing head, reused)

LABELS = AspectSentiment.label_names()                                                      # index = label id


class AspectSentimentClassifier(nn.Module):
    """
    encoder   : 'mbert', 'xlmr', a Hugging Face model name, or an encoder module.
    pooling   : 'cls' (first token) or 'mean' (average over real tokens of both segments).
    finetune  : 'frozen', 'partial' (top finetune_layers) or 'full' encoder training.
    """

    def __init__(self, encoder: Union[str, nn.Module] = "mbert", *, pooling: str = "cls", hidden_dim: int = 256, dropout: float = 0.1,
                 finetune: str = "full", finetune_layers: int = 2, pretrained: bool = True, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        if pooling not in POOLING:
            raise ValueError(f"pooling must be one of {POOLING}")
        self.pooling = pooling
        self.encoder = resolve_encoder(encoder, pretrained=pretrained)
        apply_encoder_mode(self.encoder, finetune, finetune_layers)
        self.hidden_size = self.encoder.config.hidden_size
        self.head = ClassificationHead(self.hidden_size, hidden_dim, len(LABELS), dropout)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, return_embedding: bool = False) -> Dict[str, torch.Tensor]:
        """
        input_ids, attention_mask (and token_type_ids for BERT-style tokenizers): [batch, tokens], one row per (aspect, review) pair.
        labels: [batch] ids of AspectSentiment (optional). Returns {"logits": [batch, 4], "loss"?, "embedding"?: [batch, hidden]}.
        """
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        emb = pool(self.encoder(**kw).last_hidden_state, attention_mask, self.pooling)
        out = {"logits": self.head(emb)}
        if labels is not None:
            out["loss"] = F.cross_entropy(out["logits"], labels, weight=self.class_weights)
        if return_embedding:
            out["embedding"] = emb
        return out

    def parameter_counts(self) -> Dict[str, int]:
        return count_parameters(self)

    @torch.no_grad()
    def predict_for_aspects(self, review: str, aspect_texts: Sequence[str], tokenizer, *, max_length: int = 128) -> torch.Tensor:
        """
        Class probabilities [n_aspects, 4] for each aspect text of ONE review (rows follow the order given; columns follow LABELS).
        No aspects (an empty list) gives an empty [0, 4] tensor, so a review with no detected aspect produces no sentiment at all.
        """
        if not aspect_texts:
            return torch.zeros((0, len(LABELS)))
        if not getattr(tokenizer, "is_fast", False):
            raise TypeError("a fast tokenizer is required")
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        enc = tokenizer(list(aspect_texts), [review] * len(aspect_texts), truncation="only_second", max_length=max_length, padding=True,
                        return_tensors="pt")
        kw = {k: enc[k].to(device) for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
        probs = F.softmax(self(**kw)["logits"], dim=-1).cpu()
        self.train(was_training)
        return probs


def sentiment_batch(encoded: Sequence[dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pad the dicts of aspect_data.encode_sentiment_examples into tensors (padding: attention 0, token type 0)."""
    n = max(len(e["input_ids"]) for e in encoded)
    ids = torch.full((len(encoded), n), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(encoded), n), dtype=torch.long)
    has_types = all("token_type_ids" in e for e in encoded)
    types = torch.zeros((len(encoded), n), dtype=torch.long)
    for i, e in enumerate(encoded):
        k = len(e["input_ids"])
        ids[i, :k] = torch.tensor(e["input_ids"])
        mask[i, :k] = torch.tensor(e["attention_mask"])
        if has_types:
            types[i, :k] = torch.tensor(e["token_type_ids"])
    batch = {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor([e["label"] for e in encoded], dtype=torch.long)}
    if has_types:
        batch["token_type_ids"] = types
    return batch
