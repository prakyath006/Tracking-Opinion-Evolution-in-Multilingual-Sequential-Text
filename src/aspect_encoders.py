"""
Shared helpers for the aspect-level models (aspect_extractor.py, aspect_sentiment_model.py): loading a pretrained mBERT or XLM-R encoder,
choosing how much of it trains, pooling, and counting parameters. The three aspect models stay independent of each other; this module
holds only what is not specific to one of them. It does not touch the existing OET model (src/model.py).
"""
import os
import sys
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_utils import ENCODERS, FINETUNE_MODES, set_encoder_trainable          # noqa: E402  (reused, not modified)

POOLING = ("cls", "mean")


def resolve_encoder(spec: Union[str, nn.Module] = "mbert", *, pretrained: bool = True) -> nn.Module:
    """
    spec: an already built encoder module (used as is, e.g. a tiny one in tests), 'mbert', 'xlmr', or any Hugging Face model name.
    pretrained=False builds the same architecture with random weights from the model's config (no weight download; used for parameter
    counts). The pooler is left out: the aspect models never use it.
    """
    if isinstance(spec, nn.Module):
        return spec
    from transformers import AutoConfig, AutoModel
    name = ENCODERS.get(spec, spec)
    if pretrained:
        return AutoModel.from_pretrained(name, add_pooling_layer=False)
    return AutoModel.from_config(AutoConfig.from_pretrained(name), add_pooling_layer=False)


def apply_encoder_mode(encoder: nn.Module, mode: str, n_layers: int = 2) -> int:
    """frozen / partial (top n_layers) / full. Returns the number of trainable encoder parameters."""
    if mode not in FINETUNE_MODES:
        raise ValueError(f"finetune must be one of {FINETUNE_MODES}, got {mode!r}")
    return set_encoder_trainable(encoder, mode, n_layers)


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, strategy: str = "cls") -> torch.Tensor:
    """[batch, tokens, dim] -> [batch, dim]: first token, or the mean over real (unmasked) tokens only."""
    if strategy == "cls":
        return hidden[:, 0]
    if strategy == "mean":
        m = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
    raise ValueError(f"pooling must be one of {POOLING}, got {strategy!r}")


def count_parameters(model: nn.Module, encoder: Optional[nn.Module] = None) -> Dict[str, int]:
    """Total and trainable parameters, split into the pretrained encoder and everything else (the new layers)."""
    enc = encoder if encoder is not None else getattr(model, "encoder", None)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_total = sum(p.numel() for p in enc.parameters()) if enc is not None else 0
    enc_train = sum(p.numel() for p in enc.parameters() if p.requires_grad) if enc is not None else 0
    return {"total": total, "trainable": trainable, "encoder_total": enc_total, "encoder_trainable": enc_train,
            "new_layers_total": total - enc_total, "new_layers_trainable": trainable - enc_train}
