"""
Parameter counts of the three aspect-level models for the real mBERT and XLM-R architectures.
Builds each model with RANDOM weights from the model configuration (no weight download, nothing is trained or evaluated) and prints how many
parameters would be trainable under each encoder mode. Counts only; no performance is implied.

    python scripts/aspect_model_summary.py            # needs the configs in the Hugging Face cache (or Internet for the first run)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.dont_write_bytecode = True
import aspect_data as ad                                           # noqa: E402
import aspect_extractor as ax                                      # noqa: E402
import aspect_sentiment_model as sm                                # noqa: E402
import aspect_sequence_model as sq                                 # noqa: E402

CATS = ad.category_ids("product")
MODES = (("frozen", 0), ("partial", 2), ("full", 0))


def row(name, c):
    print(f"{name:<58}{c['total']:>14,}{c['trainable']:>14,}{c['encoder_trainable']:>14,}{c['new_layers_total']:>12,}")


def main():
    print(f"{'model':<58}{'total':>14}{'trainable':>14}{'encoder train.':>14}{'new layers':>12}")
    for enc in ("mbert", "xlmr"):
        for mode, n in MODES:
            label = f"{mode}" + (f" (top {n})" if n else "")
            row(f"extractor {enc}, {label}, generic tags", ax.AspectExtractor(enc, pretrained=False, finetune=mode, finetune_layers=n or 2).parameter_counts())
            row(f"extractor {enc}, {label}, {1 + 2 * len(CATS)} category tags",
                ax.AspectExtractor(enc, ad.bio_tagset(CATS), pretrained=False, finetune=mode, finetune_layers=n or 2).parameter_counts())
            row(f"sentiment {enc}, {label}, cls pooling",
                sm.AspectSentimentClassifier(enc, pretrained=False, finetune=mode, finetune_layers=n or 2).parameter_counts())
    row("sequence model (default sizes, no text vectors)", sq.AspectSequenceModel(len(CATS)).parameter_counts())
    row("sequence model (default sizes, 768-d text vectors)", sq.AspectSequenceModel(len(CATS), text_dim=768).parameter_counts())


if __name__ == "__main__":
    main()
