"""
End-to-end sample for the panel demonstration: OET (Amazon checkpoint) -> WSD aspect tags ->
aspect-associated sentiment projection -> final decision. One short sequence, CPU, a few seconds
after the model has loaded. Writes outputs/demo/sample_end_to_end.json.

Run: python scripts/demo_end_to_end.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch

from final_decision import oet_outputs, run_pipeline
from model import OpinionEvolutionTracker
from ontology import SentimentState, TrajectoryType, TransitionType
from wsd import WordSenseDisambiguator

REVIEWS = [
    "The trailer looked amazing and the music is wonderful",
    "Watched it today, the story is gripping and the songs are lovely",
    "Second half drags, the story falls apart and the music gets loud and annoying",
    "Terrible ending, the story made no sense, a complete waste of money",
]

if __name__ == "__main__":
    ckpt = torch.load(os.path.join(ROOT, "outputs", "checkpoints", "best_model_amazon.pt"), map_location="cpu", weights_only=False)
    saved = vars(ckpt["args"]) if hasattr(ckpt.get("args"), "__dict__") else (ckpt.get("args") or {})
    model = OpinionEvolutionTracker(model_name=saved.get("model_name", "bert-base-multilingual-cased"), use_cuda=False,
                                    freeze_encoder=saved.get("freeze_encoder", True))
    model.load_state_dict(ckpt["model_state_dict"])          # strict: stops on any mismatch
    oet = oet_outputs(model, REVIEWS)

    # The integration must not change OET's predictions: compare with the original predict().
    ref = model.predict(REVIEWS)
    n = len(REVIEWS)
    assert oet["sentiments"] == [SentimentState.label_names()[i] for i in ref["sentiments"].tolist()[:n]]
    assert oet["transitions"] == [TransitionType.label_names()[i] for i in ref["trends"].tolist()[1:n]]
    assert oet["trajectory"] == TrajectoryType.label_names()[ref["trajectory"]]

    result = run_pipeline(REVIEWS, oet, WordSenseDisambiguator())
    result["note"] = ("Sample output of the integrated pipeline on one typed sequence. English film reviews are outside "
                      "the product category OET was trained on (Amazon Beauty); this illustrates the pipeline and is "
                      "not an evaluation result.")
    out = os.path.join(ROOT, "outputs", "demo", "sample_end_to_end.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(result["final_decision"]["verdict"])
    print("wrote", out)
