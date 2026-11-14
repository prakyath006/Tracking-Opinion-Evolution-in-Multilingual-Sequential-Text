"""
Split one wave of annotation batch 02 into two identical, blank annotation files (one per independent annotator) plus a reference copy.

    python scripts/make_wave_files.py --wave 1

Reads data/annotations/pilot_batch_02_product_timelines.csv and pilot_batch_02_manifest.json (never modified); writes, next to them,
    batch_02_wave{N}_reference.csv     the unannotated wave (validator --reference; nobody edits it)
    batch_02_wave{N}_annotator_1.csv   identical blank copy for the first annotator
    batch_02_wave{N}_annotator_2.csv   identical blank copy for the second annotator
    batch_02_wave{N}_manifest.json     products, strata, hashes and the SMALL-stratum approval flag
Refuses to overwrite. Annotators fill the annotation columns (guidelines v0.2) and their own annotator_id; the two ids must differ.
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, "data", "annotations")
BATCH = os.path.join(ANN, "pilot_batch_02_product_timelines.csv")
MANIFEST = os.path.join(ANN, "pilot_batch_02_manifest.json")
BLANK = ["aspect_status", "aspect_span_text", "span_start", "span_end", "span_type", "aspect_category", "sentiment", "annotator_id", "notes"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rel(p):
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:                                  # another drive (temporary test folders)
        return p


def main(argv=None, batch=BATCH, manifest=MANIFEST, out_dir=ANN):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave", type=int, default=1)
    a = ap.parse_args(argv)
    man = json.load(open(manifest, encoding="utf-8"))
    df = pd.read_csv(batch, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    seqs = [s for s in man["sequences"] if s["wave"] == a.wave]
    if not seqs:
        raise ValueError(f"no products in wave {a.wave}")
    w = df[df.sequence_id.isin({s["sequence_id"] for s in seqs})].copy()
    assert w.sequence_id.nunique() == len(seqs) and len(w) == 10 * len(seqs)
    assert (w[BLANK] == "").all().all() and set(w.annotation_status) == {"NOT_STARTED"}, "the batch already holds annotations"
    names = {k: os.path.join(out_dir, f"batch_02_wave{a.wave}_{k}.csv") for k in ("reference", "annotator_1", "annotator_2")}
    names["manifest"] = os.path.join(out_dir, f"batch_02_wave{a.wave}_manifest.json")
    clash = [p for p in names.values() if os.path.exists(p)]
    if clash:
        raise FileExistsError(f"refusing to overwrite {clash}")
    for k in ("reference", "annotator_1", "annotator_2"):
        w.to_csv(names[k], index=False, encoding="utf-8-sig")
    small = [s["sequence_id"] for s in seqs if s["stratum"] == "SMALL"]
    info = {"wave": a.wave, "products": len(seqs), "reviews": int(len(w)), "source_batch_sha256": sha(batch),
            "files": {k: {"path": rel(p), "sha256": sha(p)} for k, p in names.items() if k != "manifest"},
            "strata": pd.Series([s["stratum"] for s in seqs]).value_counts().to_dict(),
            "PENDING_TEAM_APPROVAL": {"what": "SMALL-stratum sampling deviation: >= 12 reviews over >= 180 days instead of >= 20 over >= 365",
                                      "sequence_ids": small, "reviews": 10 * len(small),
                                      "if_rejected": "drop these sequences from every count and re-draw SMALL products; the other products are unaffected"},
            "sequences": [{k: s[k] for k in ("sequence_id", "product", "stratum", "eligible_reviews", "span_days", "distinctive_category")} for s in seqs]}
    json.dump(info, open(names["manifest"], "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: info[k] for k in ("wave", "products", "reviews", "strata")}), "| SMALL pending approval:", len(small), "products")
    for k, p in names.items():
        print(f"  {k}: {p}")
    return names


if __name__ == "__main__":
    main()
