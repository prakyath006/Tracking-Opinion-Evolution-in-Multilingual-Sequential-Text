"""
End-to-End Smoke Test — Verifies all modules connect correctly.

Tests:
  1. Dataset loading (both domains)
  2. BiLSTM forward pass (tensor shapes)
  3. Attention forward pass (tensor shapes)
  4. Multi-task classifier (tensor shapes)
  5. Full model (small batch, end-to-end)
  6. Evaluation metrics (SCS computation)
"""

import os
import sys
import torch
import logging

# Setup path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def test_datasets():
    """Test 1: Dataset classes load correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Dataset Loading")
    print("=" * 60)
    
    from dataset import AmazonSequenceDataset, DravidianDataset
    
    # Amazon dataset
    amazon = AmazonSequenceDataset(split="train")
    print(f"  Amazon train: {len(amazon)} sequences")
    sample = amazon[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Texts: {len(sample['texts'])} reviews")
    print(f"  Sentiments shape: {sample['sentiments'].shape}")
    print(f"  Trajectory: {sample['trajectory'].item()}")
    
    # Dravidian dataset
    dravidian = DravidianDataset(language="tamil", task="sentiment", split="train")
    print(f"  Dravidian Tamil train: {len(dravidian)} samples")
    sample = dravidian[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Text: {sample['text'][:60]}...")
    print(f"  Label: {sample['label'].item()}")
    
    print("  [PASS] Datasets loaded successfully!")
    return True


def test_bilstm():
    """Test 2: Bi-LSTM forward pass."""
    print("\n" + "=" * 60)
    print("TEST 2: Bi-LSTM Forward Pass")
    print("=" * 60)
    
    from bilstm import BiLSTMEncoder
    
    bilstm = BiLSTMEncoder(input_dim=768, hidden_dim=256, num_layers=2)
    
    # Simulate: batch of 4 users, max 5 reviews each, 768-dim embeddings
    batch = torch.randn(4, 5, 768)
    seq_lens = [5, 3, 4, 2]
    
    hidden_states, (h_n, c_n) = bilstm(batch, seq_lens)
    
    print(f"  Input shape:  {batch.shape}")
    print(f"  Output shape: {hidden_states.shape}")
    print(f"  Expected:     [4, 5, 512] (hidden*2)")
    print(f"  h_n shape:    {h_n.shape}")
    
    assert hidden_states.shape == (4, 5, 512), f"Wrong shape: {hidden_states.shape}"
    
    # Test final hidden extraction
    final = bilstm.get_final_hidden(h_n)
    print(f"  Final hidden: {final.shape}")
    assert final.shape == (4, 512)
    
    print("  [PASS] Bi-LSTM works correctly!")
    return True


def test_attention():
    """Test 3: Attention layer forward pass."""
    print("\n" + "=" * 60)
    print("TEST 3: Attention Layer")
    print("=" * 60)
    
    from attention import SelfAttention, MultiHeadSequenceAttention
    
    hidden_states = torch.randn(4, 5, 512)
    mask = torch.tensor([
        [True, True, True, True, True],
        [True, True, True, False, False],
        [True, True, True, True, False],
        [True, True, False, False, False],
    ])
    
    # Single-head attention
    attn = SelfAttention(input_dim=512, attention_dim=128)
    context, weights = attn(hidden_states, mask)
    
    print(f"  Single-head:")
    print(f"    Context shape: {context.shape} (expected [4, 512])")
    print(f"    Weights shape: {weights.shape} (expected [4, 5])")
    print(f"    Weights sum:   {weights[0].sum().item():.4f} (expected ~1.0)")
    
    assert context.shape == (4, 512)
    assert weights.shape == (4, 5)
    
    # Multi-head attention
    multi_attn = MultiHeadSequenceAttention(input_dim=512, num_heads=4, attention_dim=128)
    context_m, weights_m = multi_attn(hidden_states, mask)
    
    print(f"  Multi-head (4 heads):")
    print(f"    Context shape: {context_m.shape} (expected [4, 512])")
    print(f"    Weights shape: {weights_m.shape} (expected [4, 5])")
    
    assert context_m.shape == (4, 512)
    
    print("  [PASS] Attention works correctly!")
    return True


def test_classifier():
    """Test 4: Multi-task classifier."""
    print("\n" + "=" * 60)
    print("TEST 4: Multi-Task Classifier")
    print("=" * 60)
    
    from classifier import MultiTaskClassifier, MultiTaskLoss
    
    classifier = MultiTaskClassifier(
        input_dim=512, hidden_dim=128,
        num_sentiment_classes=4, num_trend_classes=3, num_trajectory_classes=4,
    )
    
    hidden_states = torch.randn(4, 5, 512)
    context_vector = torch.randn(4, 512)
    
    outputs = classifier(hidden_states, context_vector)
    
    print(f"  Sentiment logits: {outputs['sentiment_logits'].shape} (expected [4, 5, 4])")
    print(f"  Trend logits:     {outputs['trend_logits'].shape} (expected [4, 5, 3])")
    print(f"  Trajectory logits:{outputs['trajectory_logits'].shape} (expected [4, 4])")
    
    assert outputs['sentiment_logits'].shape == (4, 5, 4)
    assert outputs['trend_logits'].shape == (4, 5, 3)
    assert outputs['trajectory_logits'].shape == (4, 4)
    
    # Test loss computation
    loss_fn = MultiTaskLoss()
    
    targets = {
        "sentiments": torch.randint(0, 4, (4, 5)),
        "trends": torch.randint(0, 3, (4, 5)),
        "trajectories": torch.randint(0, 4, (4,)),
    }
    
    losses = loss_fn(outputs, targets)
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print(f"  Sentiment loss: {losses['sentiment_loss'].item():.4f}")
    print(f"  Trend loss: {losses['trend_loss'].item():.4f}")
    print(f"  Trajectory loss: {losses['trajectory_loss'].item():.4f}")
    
    print("  [PASS] Multi-task classifier works correctly!")
    return True


def test_evaluation():
    """Test 5: Evaluation metrics."""
    print("\n" + "=" * 60)
    print("TEST 5: Evaluation Metrics + SCS")
    print("=" * 60)
    
    from evaluation import (
        compute_classification_metrics,
        sequence_consistency_score,
        sequence_consistency_score_with_ground_truth,
    )
    
    # Classification metrics
    y_true = [0, 1, 2, 0, 1, 0, 2, 1, 0, 0]
    y_pred = [0, 1, 2, 0, 2, 0, 2, 1, 1, 0]
    
    metrics = compute_classification_metrics(y_true, y_pred)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 macro: {metrics['f1_macro']:.4f}")
    
    # SCS
    sequences = [
        [0, 0, 1, 1],       # 1 flip / 3 = SCS 0.67
        [0, 1, 0, 1],       # 3 flips / 3 = SCS 0.00
        [0, 0, 0, 0],       # 0 flips / 3 = SCS 1.00
        [1, 1, 0, 0, 0],    # 1 flip / 4 = SCS 0.75
    ]
    
    scs = sequence_consistency_score(sequences)
    print(f"  SCS mean: {scs['scs_mean']:.4f} (expected ~0.605)")
    print(f"  SCS min:  {scs['scs_min']:.4f} (expected 0.000)")
    print(f"  SCS max:  {scs['scs_max']:.4f} (expected 1.000)")
    
    assert scs['scs_min'] == 0.0
    assert scs['scs_max'] == 1.0
    
    # Comparative SCS
    gt_sequences = [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ]
    comp = sequence_consistency_score_with_ground_truth(sequences, gt_sequences)
    print(f"  Transition accuracy: {comp['transition_accuracy']:.4f}")
    print(f"  SCS delta: {comp['scs_delta']:.4f}")
    
    print("  [PASS] Evaluation metrics work correctly!")
    return True


def test_baselines():
    """Test 6: Baseline model creation."""
    print("\n" + "=" * 60)
    print("TEST 6: Baseline Models")
    print("=" * 60)
    
    from baselines import get_baseline_model, TextCNN, LSTMOnlyModel
    
    # TextCNN
    textcnn = get_baseline_model("textcnn", vocab_size=1000, num_classes=4)
    dummy_input = torch.randint(0, 1000, (4, 50))
    output = textcnn(dummy_input)
    print(f"  TextCNN output: {output.shape} (expected [4, 4])")
    assert output.shape == (4, 4)
    
    # LSTM-only
    lstm_model = get_baseline_model("lstm_only", num_classes=4)
    embeddings = torch.randn(4, 5, 768)
    output = lstm_model(embeddings, seq_lens=[5, 3, 4, 2])
    print(f"  LSTM-only sentiment: {output['sentiment_logits'].shape} (expected [4, 5, 4])")
    print(f"  LSTM-only trajectory: {output['trajectory_logits'].shape} (expected [4, 4])")
    
    print("  [PASS] Baseline models work correctly!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  END-TO-END SMOKE TEST")
    print("=" * 60)
    
    tests = [
        ("Datasets", test_datasets),
        ("Bi-LSTM", test_bilstm),
        ("Attention", test_attention),
        ("Classifier", test_classifier),
        ("Evaluation", test_evaluation),
        ("Baselines", test_baselines),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS"))
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            results.append((name, f"FAIL: {e}"))
    
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, status in results:
        print(f"  {name:<20} {status}")
    
    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 60)
