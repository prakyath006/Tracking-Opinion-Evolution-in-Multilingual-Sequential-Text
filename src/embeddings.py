"""
=============================================================================
Module 1b — Domain-Adapted Embedding Generation (Own Embeddings)
=============================================================================
This module provides "own embeddings" by fine-tuning pre-trained multilingual
transformers (mBERT or XLM-RoBERTa) on our specific Tamil/Telugu code-mixed
data. Unlike using frozen pre-trained embeddings, this module:

  1. Selectively unfreezes the last N layers of the transformer
  2. Adds a domain-adaptive projection layer for our specific tasks
  3. Supports fine-tuning on domain-specific data to create "own" embeddings

This transforms generic mBERT embeddings into domain-adapted representations
that are specialized for Tamil and Telugu code-mixed opinion text.

Reference: Guide Module 1 (Structural Ontology & Own Embeddings)
=============================================================================
"""

import logging
from typing import Dict, Union, Optional
import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)


class DomainAdaptedEmbeddings(nn.Module):
    """
    Domain-Adapted Embedding Generator — "Own Embeddings".

    Unlike standard frozen embeddings, this module:
    1. Loads mBERT/XLM-R as the base
    2. Unfreezes the last `finetune_layers` transformer layers
    3. Adds a trainable domain-adaptive projection layer
    4. Produces domain-specific 768-dim embeddings

    This makes the embeddings "our own" — adapted specifically for
    Tamil/Telugu code-mixed opinion text.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        use_cuda: bool = True,
        finetune_layers: int = 3,
        projection_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model name ('bert-base-multilingual-cased' or 'xlm-roberta-base').
        use_cuda : bool
            Whether to use GPU if available.
        finetune_layers : int
            Number of transformer layers to unfreeze from the top.
            Default 3 means the last 3 layers are trainable (layers 10, 11, 12).
            Set to 0 to keep all layers frozen (not recommended for "own embeddings").
        projection_dim : int or None
            If provided, adds a domain-adaptive projection layer that maps
            768-dim embeddings to projection_dim. If None, no projection is applied.
        dropout : float
            Dropout rate for the projection layer.
        """
        super().__init__()
        self.model_name = model_name
        self.finetune_layers = finetune_layers

        # Set device
        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )

        # Load pre-trained transformer
        logger.info(f"Loading base transformer: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)

        # ── Freeze/Unfreeze Strategy ──
        # First, freeze ALL parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Then, unfreeze the last N layers to make them trainable ("own")
        if finetune_layers > 0:
            # Get the encoder layers (works for both BERT and XLM-R)
            if hasattr(self.encoder, "encoder"):
                encoder_layers = self.encoder.encoder.layer
            else:
                encoder_layers = self.encoder.layers

            total_layers = len(encoder_layers)
            unfreeze_from = max(0, total_layers - finetune_layers)

            for i in range(unfreeze_from, total_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True

            # Also unfreeze the pooler if it exists
            if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
                for param in self.encoder.pooler.parameters():
                    param.requires_grad = True

            logger.info(
                f"Fine-tuning last {finetune_layers} of {total_layers} layers "
                f"(layers {unfreeze_from}-{total_layers - 1} are trainable)"
            )
        else:
            logger.info("All encoder layers frozen (no fine-tuning)")

        # ── Domain-Adaptive Projection Layer ──
        self.hidden_size = self.encoder.config.hidden_size  # 768
        self.projection = None

        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.output_dim = projection_dim
            logger.info(
                f"Domain-adaptive projection: {self.hidden_size} → {projection_dim}"
            )
        else:
            self.output_dim = self.hidden_size

        self.to(self.device)

        # Log parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params
        logger.info(
            f"Embedding params: {total_params:,} total, "
            f"{trainable_params:,} trainable, {frozen_params:,} frozen"
        )

    def forward(self, tokenized_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to generate domain-adapted embeddings.

        Parameters
        ----------
        tokenized_inputs : Dict[str, torch.Tensor]
            Dictionary with 'input_ids', 'attention_mask', etc.

        Returns
        -------
        torch.Tensor
            Last hidden state: [batch_size, seq_len, output_dim]
        """
        inputs_on_device = {
            key: val.to(self.device) for key, val in tokenized_inputs.items()
        }

        outputs = self.encoder(**inputs_on_device)
        hidden_states = outputs.last_hidden_state

        # Apply domain-adaptive projection if configured
        if self.projection is not None:
            hidden_states = self.projection(hidden_states)

        return hidden_states

    def generate_embeddings(
        self,
        tokenized_inputs: Dict[str, torch.Tensor],
        strategy: str = "cls",
    ) -> torch.Tensor:
        """
        Generate domain-adapted embeddings using the specified pooling strategy.

        Parameters
        ----------
        tokenized_inputs : Dict[str, torch.Tensor]
            Tensors from the tokenizer.
        strategy : str
            'cls': CLS token representation [batch_size, output_dim]
            'mean': Mean pooling over non-pad tokens [batch_size, output_dim]
            'none': Raw token states [batch_size, seq_len, output_dim]

        Returns
        -------
        torch.Tensor
            Domain-adapted contextual embeddings.
        """
        last_hidden_state = self.forward(tokenized_inputs)

        if strategy == "none":
            return last_hidden_state

        elif strategy == "cls":
            return last_hidden_state[:, 0, :]

        elif strategy == "mean":
            attention_mask = tokenized_inputs["attention_mask"].to(self.device)
            mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(last_hidden_state.size())
                .float()
            )
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Choose 'cls', 'mean', or 'none'."
            )

    def get_embedding_dim(self) -> int:
        """Returns the dimension of the generated embeddings."""
        return self.output_dim
