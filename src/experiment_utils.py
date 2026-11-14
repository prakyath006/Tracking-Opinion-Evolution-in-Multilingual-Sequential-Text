"""
Building blocks for controlled OET experiments (scripts/train_experiment.py).

Everything here is new and self-contained: it does not change src/model.py, scripts/train.py, the dataset
classes or the label definitions. The existing training script and checkpoints keep working exactly as before.

Contents
    * placeholder masking   : position 0 of every sequence carries an artificial STABLE transition label
                              (src/dataset.py::build_trend_labels). It is masked out of the loss and of every metric.
    * class weights         : none / sqrt / full inverse frequency, from TRAINING labels only
    * encoder training mode : frozen / partial (top N layers) / full, with separate learning rates
    * schedulers            : constant / linear / cosine (warm-up) / plateau
    * evaluation            : accuracy and macro-F1 for sentiment, observed transitions and trajectory
    * checkpoint selection  : on a validation metric, never on test
"""
import json
import math
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_utils as eu                                                     # noqa: E402
from classifier import compute_class_weights                                # noqa: E402
from evaluation import sequence_consistency_score                          # noqa: E402
from ontology import SentimentState, TrajectoryType, TransitionType        # noqa: E402

ENCODERS = {"mbert": "bert-base-multilingual-cased", "xlmr": "xlm-roberta-base"}
WEIGHT_MODES = ("none", "sqrt", "full")
POOLING = ("cls", "mean")
FINETUNE_MODES = ("frozen", "partial", "full")
SCHEDULES = ("constant", "linear", "cosine", "plateau")
SELECTION_METRICS = ("mean_macro_f1", "sentiment_macro_f1", "transition_macro_f1", "trajectory_macro_f1", "val_loss")

SENT_LABELS, SENT_NAMES = [0, 1, 2], SentimentState.label_names()[:3]      # Amazon has no UNKNOWN label
TRANS_LABELS, TRANS_NAMES = [0, 1, 2], TransitionType.label_names()
TRAJ_LABELS, TRAJ_NAMES = [0, 1, 2, 3], TrajectoryType.label_names()


# ───────────────────────────── placeholder masking ──────────────────────────────
def mask_placeholder_trends(trends: torch.Tensor) -> torch.Tensor:
    """
    Copy of the [batch, seq_len] trend targets with position 0 set to -1 (the loss' ignore_index).
    Padding is already -1. The dataset tensors themselves are not modified, so gold files stay unchanged.
    """
    out = trends.clone()
    out[:, 0] = -1
    return out


# ───────────────────────────── class weights ───────────────────────────────────
def class_weight_vector(labels: Sequence[int], num_classes: int, mode: str) -> Optional[torch.Tensor]:
    """
    none : None (uniform loss)
    full : total / (C * count_c), the project's existing inverse-frequency rule
    sqrt : the square root of the full weight (a softer correction of the same imbalance)
    Classes absent from `labels` keep weight 0, as in classifier.compute_class_weights.
    """
    if mode not in WEIGHT_MODES:
        raise ValueError(f"class weight mode must be one of {WEIGHT_MODES}, got {mode!r}")
    if mode == "none":
        return None
    w = compute_class_weights(list(labels), num_classes)
    return w.sqrt() if mode == "sqrt" else w


def training_class_weights(train_dataset, mode: str) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict]:
    """
    Weights for the three heads, computed from the TRAINING dataset only. The transition counts exclude the
    artificial position-0 placeholder, so the weights describe observed transitions.
    Returns (weights per head or None, the label counts used).
    """
    sent, trend, traj = [], [], []
    for s in train_dataset.sequences:
        sent += list(s["sentiments"])
        trend += list(s["trends"][1:])
        traj.append(s["trajectory"])
    counts = {
        "sentiment": np.bincount(sent, minlength=SentimentState.num_classes()).tolist(),
        "transition_observed": np.bincount(trend, minlength=TransitionType.num_classes()).tolist(),
        "trajectory": np.bincount(traj, minlength=TrajectoryType.num_classes()).tolist(),
    }
    if mode == "none":
        return None, counts
    return {
        "sentiment": class_weight_vector(sent, SentimentState.num_classes(), mode),
        "trend": class_weight_vector(trend, TransitionType.num_classes(), mode),
        "trajectory": class_weight_vector(traj, TrajectoryType.num_classes(), mode),
    }, counts


# ───────────────────────────── encoder training mode ───────────────────────────
def set_encoder_trainable(hf_encoder: torch.nn.Module, mode: str, n_layers: int = 0) -> int:
    """
    frozen  : nothing trains
    partial : the top `n_layers` transformer layers train
    full    : every transformer layer and the embeddings train
    The unused pooler never trains. Works for BERT and XLM-R (both expose .encoder.layer).
    Returns the number of trainable encoder parameters.
    """
    if mode not in FINETUNE_MODES:
        raise ValueError(f"encoder mode must be one of {FINETUNE_MODES}, got {mode!r}")
    for p in hf_encoder.parameters():
        p.requires_grad = False
    layers = hf_encoder.encoder.layer
    if mode == "partial":
        if not 1 <= n_layers <= len(layers):
            raise ValueError(f"partial fine-tuning needs 1..{len(layers)} layers, got {n_layers}")
        chosen = layers[len(layers) - n_layers:]
    elif mode == "full":
        chosen = layers
        for p in hf_encoder.embeddings.parameters():
            p.requires_grad = True
    else:
        chosen = []
    for layer in chosen:
        for p in layer.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in hf_encoder.parameters() if p.requires_grad)


def build_param_groups(model: torch.nn.Module, lr_encoder: float, lr_head: float, weight_decay: float) -> List[Dict]:
    """Encoder parameters (if any are trainable) get lr_encoder; every other trainable parameter gets lr_head."""
    enc = [p for p in model.embedding_generator.parameters() if p.requires_grad]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("embedding_generator.")]
    groups = [{"params": head, "lr": lr_head, "weight_decay": weight_decay, "name": "head"}]
    if enc:
        groups.append({"params": enc, "lr": lr_encoder, "weight_decay": weight_decay, "name": "encoder"})
    return groups


# ───────────────────────────── scheduling ──────────────────────────────────────
def updates_per_epoch(num_batches: int, accumulation: int) -> int:
    return math.ceil(num_batches / accumulation)


def make_scheduler(optimizer, kind: str, total_updates: int, warmup_ratio: float = 0.1, plateau_mode: str = "max"):
    """
    Returns (scheduler, per_update). constant/linear/cosine step once per optimizer update; plateau steps once
    per epoch on the selection metric. linear and cosine warm up from 0 for warmup_ratio of the updates.
    """
    if kind not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {kind!r}")
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=plateau_mode, factor=0.5, patience=1), False
    warm = int(total_updates * warmup_ratio) if kind != "constant" else 0

    def factor(step: int) -> float:
        if kind == "constant":
            return 1.0
        if warm and step < warm:
            return step / warm
        progress = (step - warm) / max(1, total_updates - warm)
        return max(0.0, 1.0 - progress) if kind == "linear" else 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor), True


# ───────────────────────────── training and evaluation ─────────────────────────
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scheduler_per_update: bool, accumulation: int,
                    grad_clip: float, device) -> Dict[str, float]:
    """One epoch with gradient accumulation and a masked position-0 transition target."""
    encoder_trainable = any(p.requires_grad for p in model.embedding_generator.parameters())
    model.train()
    if not encoder_trainable:
        model.embedding_generator.eval()                  # frozen encoder: no dropout noise in its features
    params = [p for g in optimizer.param_groups for p in g["params"]]
    totals = {"total": 0.0, "sentiment": 0.0, "trend": 0.0, "trajectory": 0.0}
    n_batches, updates = len(loader), 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        pred = model(batch["texts"], seq_lens=batch["seq_lens"], padding_mask=batch["padding_mask"].to(device))
        targets = {"sentiments": batch["sentiments"].to(device),
                   "trends": mask_placeholder_trends(batch["trends"]).to(device),
                   "trajectories": batch["trajectories"].to(device)}
        losses = loss_fn(pred, targets)
        (losses["total_loss"] / accumulation).backward()
        if (i + 1) % accumulation == 0 or i + 1 == n_batches:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            updates += 1
            if scheduler is not None and scheduler_per_update:
                scheduler.step()
        totals["total"] += losses["total_loss"].item()
        totals["sentiment"] += losses["sentiment_loss"].item()
        totals["trend"] += losses["trend_loss"].item()
        totals["trajectory"] += losses["trajectory_loss"].item()
    out = {k: v / max(1, n_batches) for k, v in totals.items()}
    out["optimizer_updates"] = updates
    return out


def metrics_from_arrays(a: Dict) -> Dict:
    """
    a: sent_true/sent_pred (all real positions), trend_true/trend_pred and position (same length),
       traj_true/traj_pred (per sequence), pred_sequences, seq_lens.
    Transition metrics use observed positions only (position >= 1).
    """
    m = eu.observed_transition_mask(a["position"])
    out = {
        "sentiment": eu.classification_metrics(a["sent_true"], a["sent_pred"], SENT_LABELS, SENT_NAMES),
        "transition_observed": eu.classification_metrics(np.asarray(a["trend_true"])[m], np.asarray(a["trend_pred"])[m],
                                                         TRANS_LABELS, TRANS_NAMES),
        "trajectory": eu.classification_metrics(a["traj_true"], a["traj_pred"], TRAJ_LABELS, TRAJ_NAMES),
        "scs": sequence_consistency_score(a["pred_sequences"], a["seq_lens"]),
    }
    out["mean_macro_f1"] = float(np.mean([out["sentiment"]["macro_f1"], out["transition_observed"]["macro_f1"],
                                          out["trajectory"]["macro_f1"]]))
    return out


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> Tuple[Dict, Dict]:
    """Returns (metrics, raw arrays). The loss also ignores the position-0 placeholder."""
    model.eval()
    a = {k: [] for k in ("sent_true", "sent_pred", "trend_true", "trend_pred", "position", "user_row",
                         "traj_true", "traj_pred", "pred_sequences", "seq_lens")}
    losses = {"total": 0.0, "sentiment": 0.0, "trend": 0.0, "trajectory": 0.0}
    seen = 0
    for batch in loader:
        pred = model(batch["texts"], seq_lens=batch["seq_lens"], padding_mask=batch["padding_mask"].to(device))
        targets = {"sentiments": batch["sentiments"].to(device),
                   "trends": mask_placeholder_trends(batch["trends"]).to(device),
                   "trajectories": batch["trajectories"].to(device)}
        l = loss_fn(pred, targets)
        losses["total"] += l["total_loss"].item(); losses["sentiment"] += l["sentiment_loss"].item()
        losses["trend"] += l["trend_loss"].item(); losses["trajectory"] += l["trajectory_loss"].item()
        sp = pred["sentiment_logits"].argmax(-1).cpu(); tp = pred["trend_logits"].argmax(-1).cpu()
        jp = pred["trajectory_logits"].argmax(-1).cpu()
        for b, L in enumerate(batch["seq_lens"]):
            a["sent_true"] += batch["sentiments"][b, :L].tolist(); a["sent_pred"] += sp[b, :L].tolist()
            a["trend_true"] += batch["trends"][b, :L].tolist(); a["trend_pred"] += tp[b, :L].tolist()
            a["position"] += list(range(L)); a["user_row"] += [seen + b] * L
            a["pred_sequences"].append(sp[b, :L].tolist()); a["seq_lens"].append(L)
            a["traj_true"].append(int(batch["trajectories"][b])); a["traj_pred"].append(int(jp[b]))
        seen += len(batch["seq_lens"])
    n = max(1, len(loader))
    metrics = metrics_from_arrays(a)
    metrics["loss"] = {k: v / n for k, v in losses.items()}
    return metrics, a


def flat_summary(metrics: Dict) -> Dict[str, float]:
    """The numbers logged every epoch."""
    return {
        "val_loss": metrics["loss"]["total"],
        "sentiment_accuracy": metrics["sentiment"]["accuracy"], "sentiment_macro_f1": metrics["sentiment"]["macro_f1"],
        "transition_accuracy": metrics["transition_observed"]["accuracy"],
        "transition_macro_f1": metrics["transition_observed"]["macro_f1"],
        "trajectory_accuracy": metrics["trajectory"]["accuracy"], "trajectory_macro_f1": metrics["trajectory"]["macro_f1"],
        "mean_macro_f1": metrics["mean_macro_f1"], "scs_mean": metrics["scs"]["scs_mean"],
    }


# ───────────────────────────── checkpoint selection ────────────────────────────
def selection_value(summary: Dict[str, float], name: str) -> Tuple[float, bool]:
    """(value, higher_is_better). Only validation summaries are ever passed in."""
    if name not in SELECTION_METRICS:
        raise ValueError(f"selection metric must be one of {SELECTION_METRICS}, got {name!r}")
    return (summary["val_loss"], False) if name == "val_loss" else (summary[name], True)


def is_improvement(new: float, best: Optional[float], higher_is_better: bool, min_delta: float = 1e-4) -> bool:
    if best is None:
        return True
    return new > best + min_delta if higher_is_better else new < best - min_delta


def checkpoint_state(model: torch.nn.Module) -> Tuple[Dict, bool]:
    """
    State dict to save. A fully frozen encoder is reproduced from its public pretrained weights, so it is left
    out (about 700 MB smaller); any trained encoder weight forces the whole encoder to be saved.
    """
    encoder_trained = any(p.requires_grad for p in model.embedding_generator.parameters())
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if encoder_trained or not k.startswith("embedding_generator.")}
    return state, encoder_trained


def load_checkpoint_state(model: torch.nn.Module, state: Dict, includes_encoder: bool) -> None:
    """Strict in effect: the only keys allowed to be missing are the frozen encoder's, and nothing may be unexpected."""
    res = model.load_state_dict(state, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {res.unexpected_keys[:5]}")
    bad = [k for k in res.missing_keys if includes_encoder or not k.startswith("embedding_generator.")]
    if bad:
        raise RuntimeError(f"missing checkpoint keys: {bad[:5]}")


# ───────────────────────────── configuration and bookkeeping ───────────────────
def validate_config(c: Dict) -> None:
    if c["encoder"] not in ENCODERS:
        raise ValueError(f"encoder must be one of {sorted(ENCODERS)}")
    if c["pooling"] not in POOLING:
        raise ValueError(f"pooling must be one of {POOLING}")
    if c["class_weights"] not in WEIGHT_MODES:
        raise ValueError(f"class_weights must be one of {WEIGHT_MODES}")
    if c["finetune"] not in FINETUNE_MODES:
        raise ValueError(f"finetune must be one of {FINETUNE_MODES}")
    if c["schedule"] not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if c["selection_metric"] not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
    if not 8 <= c["max_token_length"] <= 512:
        raise ValueError("max_token_length must be between 8 and 512")
    if c["grad_accumulation"] < 1 or c["batch_size"] < 1 or c["epochs"] < 1:
        raise ValueError("batch_size, grad_accumulation and epochs must be at least 1")
    if c["finetune"] == "partial" and c["finetune_layers"] < 1:
        raise ValueError("partial fine-tuning needs finetune_layers >= 1")
    if c["finetune"] == "frozen" and c.get("lr_encoder_explicit"):
        raise ValueError("lr_encoder was given but the encoder is frozen")


def default_experiment_name(c: Dict) -> str:
    ft = {"frozen": "frozen", "partial": f"top{c['finetune_layers']}", "full": "full"}[c["finetune"]]
    return f"{c['encoder']}_{ft}_{c['pooling']}_len{c['max_token_length']}_w{c['class_weights']}_{c['selection_metric']}_s{c['seed']}"


def prepare_experiment_dir(root: str, name: str) -> str:
    """Creates root/name; refuses to touch an existing experiment, so history is never overwritten."""
    path = os.path.join(root, name)
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists; choose another --name or --output_root")
    os.makedirs(os.path.join(path, "predictions"))
    return path


def write_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def environment_info() -> Dict:
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        info["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                                            text=True, timeout=10).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info
