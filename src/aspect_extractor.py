"""
Aspect extraction: multilingual transformer token classification with BIO tags.

Component 1 of 3 of the aspect-level architecture (independent of the sentiment and sequence models; the existing OET model is not used).

    text --fast tokenizer--> tokens --mBERT / XLM-R--> one hidden vector per token --dropout, linear--> a BIO tag per token
    BIO tags + token character offsets --bio_to_spans--> aspect spans in the original text

The tag set is the one of src/aspect_data.py: generic `O, B-ASP, I-ASP`, or category-specific `O, B-<category>, I-<category>, ...`.
Labels use -100 for special tokens, padding and masked messages (aspect_data.IGNORE_INDEX); the loss ignores them.

NOT TRAINED: nothing here has been trained, and no performance is claimed.
"""
import os
import sys
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aspect_data import IGNORE_INDEX, ExtractionExample, bio_tagset              # noqa: E402
from aspect_encoders import apply_encoder_mode, count_parameters, resolve_encoder   # noqa: E402


class AspectExtractor(nn.Module):
    """
    Token classifier over a pretrained multilingual encoder.

    encoder   : 'mbert', 'xlmr', a Hugging Face model name, or an encoder module.
    tagset    : list of tags; the position of a tag is its label id (use aspect_data.bio_tagset(categories)).
    finetune  : 'frozen', 'partial' (top finetune_layers) or 'full' encoder training. Default 'full', the usual choice for token classification.
    """

    def __init__(self, encoder: Union[str, nn.Module] = "mbert", tagset: Optional[Sequence[str]] = None, *, dropout: float = 0.1,
                 finetune: str = "full", finetune_layers: int = 2, pretrained: bool = True, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.tagset = list(tagset) if tagset else bio_tagset()
        if self.tagset[0] != "O":
            raise ValueError("tag 0 must be 'O'")
        self.encoder = resolve_encoder(encoder, pretrained=pretrained)
        apply_encoder_mode(self.encoder, finetune, finetune_layers)
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, len(self.tagset))
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        input_ids, attention_mask, labels : [batch, tokens] (labels optional; -100 is ignored).
        Returns {"logits": [batch, tokens, num_tags], "loss": scalar if labels were given}.
        """
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kw).last_hidden_state
        logits = self.classifier(self.dropout(hidden))
        out = {"logits": logits}
        if labels is not None:
            valid = labels != IGNORE_INDEX
            # cross_entropy with no valid target would be NaN; a batch of fully masked messages contributes zero loss instead.
            out["loss"] = (F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), weight=self.class_weights,
                                           ignore_index=IGNORE_INDEX) if bool(valid.any()) else logits.sum() * 0.0)
        return out

    def parameter_counts(self) -> Dict[str, int]:
        return count_parameters(self)

    @torch.no_grad()
    def predict_spans(self, texts: Sequence[str], tokenizer, *, max_length: int = 128, batch_size: int = 16) -> List[List[dict]]:
        """Aspect spans in each text (possibly none): [{start, end, text, category, n_tokens}] in text order."""
        if not getattr(tokenizer, "is_fast", False):
            raise TypeError("a fast tokenizer is required (character offsets)")
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        out: List[List[dict]] = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            enc = tokenizer(chunk, return_offsets_mapping=True, return_special_tokens_mask=True, truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt")
            kw = {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device)}
            if "token_type_ids" in enc:
                kw["token_type_ids"] = enc["token_type_ids"].to(device)
            pred = self(**kw)["logits"].argmax(-1).cpu().tolist()
            for b, text in enumerate(chunk):
                real = enc["attention_mask"][b].bool().tolist()
                offs = [tuple(o) for o, r in zip(enc["offset_mapping"][b].tolist(), real) if r]
                spec = [s for s, r in zip(enc["special_tokens_mask"][b].tolist(), real) if r]
                tags = [t for t, r in zip(pred[b], real) if r]
                out.append(bio_to_spans(tags, offs, spec, self.tagset, text))
        self.train(was_training)
        return out


def bio_to_spans(tag_ids: Sequence[int], offsets: Sequence[Sequence[int]], special_tokens_mask: Sequence[int], tagset: Sequence[str],
                 text: str) -> List[dict]:
    """
    Predicted (or gold) BIO tag ids to character spans.
    A span starts at a B tag and continues over following I tags of the same category. An I tag with no open span starts one (a lenient
    reading of an ill-formed prediction). Special tokens end any open span and are never part of one. Generic tags give category None.
    """
    spans, cur = [], None

    def close():
        nonlocal cur
        if cur is not None:
            s, e = cur["start"], cur["end"]
            spans.append({"start": s, "end": e, "text": text[s:e], "category": cur["category"], "n_tokens": cur["n"]})
            cur = None

    for tid, (s, e), sp in zip(tag_ids, offsets, special_tokens_mask):
        if sp or e <= s:
            close()
            continue
        tag = tagset[tid]
        if tag == "O":
            close()
            continue
        prefix, _, cat = tag.partition("-")
        cat = None if cat == "ASP" else cat
        if prefix == "B" or cur is None or cur["category"] != cat:
            close()
            cur = {"start": s, "end": e, "category": cat, "n": 1}
        else:
            cur["end"], cur["n"] = e, cur["n"] + 1
    close()
    return spans


def extraction_batch(examples: Sequence[ExtractionExample], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pad aspect_data.ExtractionExample encodings into tensors: input_ids, attention_mask, labels (-100 on padding)."""
    n = max(len(e.encoding.input_ids) for e in examples)
    ids = torch.full((len(examples), n), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(examples), n), dtype=torch.long)
    lab = torch.full((len(examples), n), IGNORE_INDEX, dtype=torch.long)
    for i, e in enumerate(examples):
        k = len(e.encoding.input_ids)
        ids[i, :k] = torch.tensor(e.encoding.input_ids)
        mask[i, :k] = torch.tensor(e.encoding.attention_mask)
        lab[i, :k] = torch.tensor(e.encoding.labels)
    return {"input_ids": ids, "attention_mask": mask, "labels": lab}
