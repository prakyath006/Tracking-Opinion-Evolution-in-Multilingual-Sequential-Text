"""
=============================================================================
Baseline Models for Comparison
=============================================================================
Implements 6 baseline models to compare against the full Opinion Evolution
Tracker (mBERT + Bi-LSTM + Attention + Multi-task):

  Baseline 1: mBERT Sentence-Level     — Fine-tuned mBERT, each review independently
  Baseline 2: XLM-R Sentence-Level     — Fine-tuned XLM-R, each review independently
  Baseline 3: LSTM-only (no attention)  — mBERT + LSTM without attention mechanism
  Baseline 4: Attention-only (no LSTM)  — mBERT + attention without recurrence
  Baseline 5: TextCNN                   — CNN-based text classifier, embeddings learned from scratch
  Baseline 6: LLM Prompt (flan-t5-base) — instruction-tuned LLM prompted to generate a sentiment label,
                                           no added classification head (see LLMPromptClassifier)

These baselines demonstrate the value of each component:
  - Baselines 1,2 show why sequential modeling (LSTM) matters
  - Baseline 3 shows why attention matters
  - Baseline 5 shows why pretrained transformers matter
  - Baseline 6 shows how a general-purpose instruction-tuned LLM, prompted
    rather than trained with a task-specific head, compares to models built
    specifically for this task

Author : Opinion Evolution Tracking Project
Date   : 2026
=============================================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from transformers import AutoModel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 1 & 2: Sentence-Level Transformer Classifier
# ──────────────────────────────────────────────────────────────────────────────

class SentenceLevelTransformer(nn.Module):
    """
    Baselines 1 & 2: Fine-tuned mBERT or XLM-R for sentence-level
    classification (no sequential modeling).
    
    Each review is classified independently — there is no LSTM or attention
    over the sequence. This demonstrates what you lose by ignoring sequential
    opinion evolution.
    
    Architecture:
        Text -> Tokenize -> mBERT/XLM-R -> CLS token -> Linear -> Prediction
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        num_classes: int = 4,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        finetune_layers: Optional[int] = None,
        use_cuda: bool = True,
    ):
        """
        Parameters
        ----------
        freeze_encoder : bool
            Legacy all-or-nothing switch. Only consulted when
            `finetune_layers` is None.
        finetune_layers : int or None
            Number of transformer layers to unfreeze from the top, matching
            DomainAdaptedEmbeddings' parameter of the same name (see
            src/embeddings.py). When set, this is authoritative and
            `freeze_encoder` is ignored: 0 freezes the encoder entirely, 3
            trains the top 3 layers, and so on.

            This exists so a baseline can be given the SAME trainable encoder
            capacity as OpinionEvolutionTracker. Leaving it None reproduces
            the original default, which fine-tunes ALL encoder layers and so
            gives this baseline far more trainable capacity than the full
            model receives -- not a valid architectural comparison.
        """
        super().__init__()
        
        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768
        
        if finetune_layers is not None:
            # Capacity-matched mode: freeze all, then unfreeze the top N.
            # Mirrors DomainAdaptedEmbeddings so the two are comparable.
            for param in self.encoder.parameters():
                param.requires_grad = False

            if finetune_layers > 0:
                if hasattr(self.encoder, "encoder"):
                    encoder_layers = self.encoder.encoder.layer
                else:
                    encoder_layers = self.encoder.layers

                total_layers = len(encoder_layers)
                unfreeze_from = max(0, total_layers - finetune_layers)

                for i in range(unfreeze_from, total_layers):
                    for param in encoder_layers[i].parameters():
                        param.requires_grad = True

                if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
                    for param in self.encoder.pooler.parameters():
                        param.requires_grad = True

                logger.info(
                    f"Fine-tuning last {finetune_layers} of {total_layers} layers "
                    f"(layers {unfreeze_from}-{total_layers - 1} are trainable)"
                )
            else:
                logger.info("All encoder layers frozen (no fine-tuning)")
            self.finetune_layers = finetune_layers
        elif freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.finetune_layers = 0
        else:
            # Legacy default: every encoder layer is trainable.
            self.finetune_layers = len(
                self.encoder.encoder.layer
                if hasattr(self.encoder, "encoder")
                else self.encoder.layers
            )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        
        self.to(self.device)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"SentenceLevelTransformer ({model_name}): "
            f"{trainable:,} trainable params"
        )
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Forward pass for a batch of individual texts.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Shape [batch, seq_len].
        attention_mask : torch.Tensor
            Shape [batch, seq_len].
            
        Returns
        -------
        torch.Tensor
            Logits of shape [batch, num_classes].
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_output)
        
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 3: LSTM-only (No Attention)
# ──────────────────────────────────────────────────────────────────────────────

class LSTMOnlyModel(nn.Module):
    """
    Baseline 3: mBERT embeddings + LSTM, but NO attention mechanism.
    
    Uses the final hidden state of the LSTM directly for classification
    instead of the attention-weighted context vector. This shows the
    value of the attention layer in isolating important reviews.
    
    Architecture:
        Text Sequence -> mBERT (frozen) -> Bi-LSTM -> Final Hidden -> Classification
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        use_cuda: bool = True,
    ):
        super().__init__()
        
        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        output_dim = hidden_dim * 2  # bidirectional
        
        # Per-review sentiment head
        self.sentiment_head = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        
        # Trajectory head (uses final hidden state)
        self.trajectory_head = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )
        
        self.to(self.device)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"LSTMOnlyModel: {trainable:,} trainable params")
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        seq_lens: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Shape [batch, seq_len, 768].
        seq_lens : List[int], optional
            Actual sequence lengths.
            
        Returns
        -------
        Dict with sentiment_logits and trajectory_logits.
        """
        embeddings = embeddings.to(self.device)
        
        if seq_lens is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            max_len = embeddings.size(1)
            seq_lens_clamped = [min(s, max_len) for s in seq_lens]
            packed = pack_padded_sequence(
                embeddings, seq_lens_clamped, batch_first=True, enforce_sorted=False
            )
            output, (h_n, _) = self.lstm(packed)
            hidden_states, _ = pad_packed_sequence(output, batch_first=True, total_length=max_len)
        else:
            hidden_states, (h_n, _) = self.lstm(embeddings)
        
        # Per-review sentiment
        sentiment_logits = self.sentiment_head(hidden_states)
        
        # Trajectory from final hidden state (concat forward + backward)
        forward_final = h_n[-2]
        backward_final = h_n[-1]
        final_hidden = torch.cat([forward_final, backward_final], dim=-1)
        trajectory_logits = self.trajectory_head(final_hidden)
        
        return {
            "sentiment_logits": sentiment_logits,
            "trajectory_logits": trajectory_logits,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 4: TextCNN
# ──────────────────────────────────────────────────────────────────────────────

class TextCNN(nn.Module):
    """
    Baseline 4: CNN-based text classifier.
    
    Uses 1D convolutions with multiple kernel sizes to capture n-gram
    features. This is a classical baseline that doesn't use pretrained
    transformers or sequential modeling.
    
    Architecture:
        Text -> Embedding -> Conv1D (multiple kernels) -> MaxPool -> FC -> Prediction
    """
    
    def __init__(
        self,
        vocab_size: int = 120000,
        embedding_dim: int = 300,
        num_classes: int = 4,
        num_filters: int = 128,
        kernel_sizes: list = None,
        dropout: float = 0.5,
        use_cuda: bool = True,
    ):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]
        
        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,
            )
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        
        self.to(self.device)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"TextCNN: {trainable:,} trainable params")
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape [batch, seq_len].
            
        Returns
        -------
        torch.Tensor
            Logits of shape [batch, num_classes].
        """
        input_ids = input_ids.to(self.device)
        
        # Embedding: [batch, seq_len, embedding_dim]
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # Transpose for Conv1d: [batch, embedding_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # Apply each conv filter + max-pool
        conv_outputs = []
        for conv in self.convs:
            x = F.relu(conv(embedded))       # [batch, num_filters, seq_len]
            x = F.max_pool1d(x, x.size(2))  # [batch, num_filters, 1]
            x = x.squeeze(2)                 # [batch, num_filters]
            conv_outputs.append(x)
        
        # Concatenate filter outputs
        combined = torch.cat(conv_outputs, dim=1)  # [batch, num_filters * len(kernels)]
        combined = self.dropout(combined)
        
        logits = self.classifier(combined)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Ablation: Attention-Only Model (No Bi-LSTM)
# ──────────────────────────────────────────────────────────────────────────────

class AttentionOnlyModel(nn.Module):
    """
    Ablation variant: Attention directly over embeddings (no Bi-LSTM).
    
    Demonstrates the value of the Bi-LSTM component by removing it.
    Attention is applied directly to the 768-dim embeddings without
    any sequential encoding.
    
    Architecture:
        Embeddings -> Attention -> Classification (no LSTM in between)
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        attention_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
        use_cuda: bool = True,
    ):
        super().__init__()
        
        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        
        # Attention directly on embeddings (no LSTM)
        self.W = nn.Linear(embedding_dim, attention_dim)
        self.u = nn.Linear(attention_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        
        # Per-review sentiment head
        self.sentiment_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        
        # Trajectory head (from attended context)
        self.trajectory_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.to(self.device)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"AttentionOnlyModel: {trainable:,} trainable params")
    
    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        embeddings = embeddings.to(self.device)
        
        # Attention directly on embeddings
        energy = self.tanh(self.W(embeddings))
        scores = self.u(energy).squeeze(-1)
        
        if mask is not None:
            mask = mask.to(self.device)
            scores = scores.masked_fill(~mask, float("-inf"))
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        context = torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)
        
        # Predictions
        sentiment_logits = self.sentiment_head(embeddings)
        trajectory_logits = self.trajectory_head(context)
        
        return {
            "sentiment_logits": sentiment_logits,
            "trajectory_logits": trajectory_logits,
            "attention_weights": weights,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 6: LLM-based prompted text-to-text classification
# ──────────────────────────────────────────────────────────────────────────────

class LLMPromptClassifier(nn.Module):
    """
    Baseline 6: an instruction-tuned encoder-decoder LLM (flan-t5-base by
    default), prompted to generate a sentiment label from the ontology's
    vocabulary, rather than trained with an added nn.Linear classification
    head the way every other baseline in this file is.

    Two prediction paths, both real, used for different purposes:
      - `forward()` scores each of the 4 candidate label strings by their
        teacher-forced conditional log-likelihood under the decoder (a
        standard rank-classification technique for text-to-text models --
        see e.g. how T0/FLAN evaluation harnesses score multiple-choice
        answers) and returns a [batch, num_classes] tensor of those scores
        as "logits". This is what train_baselines.py's shared train/eval
        loop calls, because it needs a fixed-size, differentiable,
        argmax-able tensor to compute cross-entropy loss and predictions
        exactly like every other baseline. Argmax over these scores is what
        greedy `.generate()` would produce for a vocabulary this
        well-separated (each label is a distinct first generated token).
      - `generate_and_parse()` runs real, unconstrained `.generate()` and
        parses the produced text back into a class id by substring match
        against LABEL_VOCAB, falling back to UNKNOWN if nothing matches.
        This is the literal "generate then parse" path; it is not used by
        the shared training loop (which needs a differentiable tensor,
        and `.generate()` is not differentiable) but is the reported
        prediction path for qualitative spot checks.

    Capacity discipline (mirrors SentenceLevelTransformer / D1's fix, see
    docs/defect_register.md): every encoder and decoder transformer block is
    frozen by default, with only the top `finetune_layers` blocks of each
    unfrozen. Unlike every other baseline, this model has no separate
    classification head at all -- at `finetune_layers=0` (the matched-capacity
    default every other baseline is evaluated at) there would be nothing
    trainable and no gradient path at all, so a small per-class calibration
    bias (`class_bias`, exactly `num_classes` parameters) is always
    trainable, added to the scores before they're returned. This is a real,
    known technique (contextual/prior calibration for prompted
    classifiers), not a disguised classification head -- it cannot represent
    anything input-dependent, only a fixed per-class offset. At
    `finetune_layers=0` this baseline is therefore effectively a zero-shot
    evaluation of the base LLM plus a 4-parameter calibration term, which is
    the honest characterization to report, not "fine-tuned" like the other
    baselines at higher finetune_layers settings.
    """

    LABEL_VOCAB = ["POSITIVE", "NEGATIVE", "MIXED", "UNKNOWN"]  # index i must equal SentimentState(i).name

    PROMPT_TEMPLATE = (
        "Classify the sentiment of this review or comment. "
        "Answer with exactly one word: POSITIVE, NEGATIVE, MIXED, or UNKNOWN.\n"
        "Text: {text}\nSentiment:"
    )

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        num_classes: int = 4,
        finetune_layers: int = 0,
        use_cuda: bool = True,
    ):
        super().__init__()
        if num_classes != len(self.LABEL_VOCAB):
            raise ValueError(
                f"LLMPromptClassifier's LABEL_VOCAB has {len(self.LABEL_VOCAB)} "
                f"entries; num_classes={num_classes} must match SentimentState.num_classes()"
            )
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze everything, then unfreeze the top `finetune_layers` encoder
        # AND decoder blocks -- mirrors SentenceLevelTransformer.finetune_layers.
        for param in self.model.parameters():
            param.requires_grad = False
        self.finetune_layers = finetune_layers
        if finetune_layers > 0:
            for blocks in (self.model.encoder.block, self.model.decoder.block):
                total = len(blocks)
                unfreeze_from = max(0, total - finetune_layers)
                for i in range(unfreeze_from, total):
                    for p in blocks[i].parameters():
                        p.requires_grad = True
            logger.info(
                f"LLMPromptClassifier: fine-tuning top {finetune_layers} "
                f"encoder+decoder blocks"
            )
        else:
            logger.info("LLMPromptClassifier: encoder and decoder fully frozen (zero-shot + calibration only)")

        # Always-trainable per-class calibration bias -- see class docstring.
        # This is what makes training possible at finetune_layers=0 without
        # a disguised classification head.
        self.class_bias = nn.Parameter(torch.zeros(len(self.LABEL_VOCAB)))

        # Pre-tokenize each label's target token sequence once (fixed, tiny).
        self._label_token_ids = [
            self.tokenizer(label, return_tensors="pt").input_ids
            for label in self.LABEL_VOCAB
        ]

        self.to(self.device)
        self._label_token_ids = [t.to(self.device) for t in self._label_token_ids]
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"LLMPromptClassifier ({model_name}): {trainable:,} trainable params "
            f"(finetune_layers={finetune_layers})"
        )

    def _prompts(self, texts: List[str]) -> List[str]:
        return [self.PROMPT_TEMPLATE.format(text=t) for t in texts]

    def forward(self, texts: List[str], **kwargs) -> torch.Tensor:
        """
        Returns [batch, num_classes] label-likelihood scores (see class
        docstring for why this is the reported "logits" tensor rather than
        raw .generate() output).
        """
        prompts = self._prompts(texts)
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self.device)

        batch_size = len(texts)
        scores = torch.zeros(batch_size, len(self.LABEL_VOCAB), device=self.device)
        for c, label_ids in enumerate(self._label_token_ids):
            decoder_labels = label_ids.repeat(batch_size, 1)
            out = self.model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                labels=decoder_labels,
            )
            log_probs = torch.log_softmax(out.logits, dim=-1)  # [batch, tgt_len, vocab]
            token_ll = log_probs.gather(-1, decoder_labels.unsqueeze(-1)).squeeze(-1)  # [batch, tgt_len]
            scores[:, c] = token_ll.sum(dim=-1)  # higher (less negative) = more likely

        return scores + self.class_bias

    @torch.no_grad()
    def generate_and_parse(self, texts: List[str], max_new_tokens: int = 8):
        """Real generation path: .generate() then parse text back to a class id."""
        was_training = self.training
        self.eval()
        prompts = self._prompts(texts)
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self.device)
        out_ids = self.model.generate(**enc, max_new_tokens=max_new_tokens)
        generated = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        preds = []
        for text in generated:
            text_upper = text.strip().upper()
            match = next((i for i, lbl in enumerate(self.LABEL_VOCAB) if lbl in text_upper), None)
            preds.append(match if match is not None else self.LABEL_VOCAB.index("UNKNOWN"))
        self.train(was_training)
        return preds, generated


# ──────────────────────────────────────────────────────────────────────────────
# Baseline Comparison Summary
# ──────────────────────────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "mbert_sentence": {
        "class": SentenceLevelTransformer,
        "default_args": {"model_name": "bert-base-multilingual-cased"},
        "description": "mBERT fine-tuned per-sentence (no sequential modeling)",
    },
    "xlmr_sentence": {
        "class": SentenceLevelTransformer,
        "default_args": {"model_name": "xlm-roberta-base"},
        "description": "XLM-R fine-tuned per-sentence (no sequential modeling)",
    },
    "lstm_only": {
        "class": LSTMOnlyModel,
        "default_args": {},
        "description": "mBERT + Bi-LSTM (no attention mechanism)",
    },
    "attention_only": {
        "class": AttentionOnlyModel,
        "default_args": {},
        "description": "Attention over embeddings (no Bi-LSTM)",
    },
    "textcnn": {
        "class": TextCNN,
        "default_args": {},
        "description": "TextCNN with word embeddings (no transformers, no LSTM)",
    },
    "llm_prompt": {
        "class": LLMPromptClassifier,
        "default_args": {"model_name": "google/flan-t5-base"},
        "description": "flan-t5-base prompted to generate a sentiment label (text-to-text, no added classification head)",
    },
}


def get_baseline_model(name: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate a baseline model by name.
    
    Parameters
    ----------
    name : str
        One of: 'mbert_sentence', 'xlmr_sentence', 'lstm_only',
                'attention_only', 'textcnn'
    **kwargs
        Override default arguments.
    
    Returns
    -------
    nn.Module
        The instantiated baseline model.
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline: {name}. "
            f"Available: {list(BASELINE_REGISTRY.keys())}"
        )
    
    entry = BASELINE_REGISTRY[name]
    args = {**entry["default_args"], **kwargs}
    
    logger.info(f"Creating baseline: {name} -- {entry['description']}")
    return entry["class"](**args)

