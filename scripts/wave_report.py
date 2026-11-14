"""
After two annotators finish a wave: validate both files, report agreement, build the adjudication package and measure the real yield.

    python scripts/wave_report.py --a DONE_1.csv --b DONE_2.csv --out_dir outputs/annotation_waves/batch_02_wave1

Steps (stops after step 1 if either file has validation errors):
  1. validate both files against the wave reference (scripts/validate_aspect_annotations.py rules);
  2. agreement on messages finished and valid in both: span F1 (IoU >= 0.5), message aspect_status, category, sentiment and span_type
     agreement (raw and Cohen's kappa) and every unresolved case (NEEDS_REVIEW, UNCLEAR_ASPECT, status disagreements, unfinished messages);
  3. adjudication package (scripts/prepare_adjudication_package.py) in OUT/adjudication_package;
  4. measured yield: products, reviews, examples (review, category) and usable series (two or more ranked events) from the AGREED mentions
     (conservative: identical category and sentiment on matched spans) and from ALL annotated spans counted once (optimistic upper bound, counts
     only), with and without the SMALL-stratum products (their sampling rule is pending team approval); the pilot's numbers are added from its
     provenance file; both corpora are checked against the pipeline floors (src/aspect_training.py) and a Wave 2 recommendation is printed.
Read-only on every input; writes only into the new OUT directory. Nothing is trained and no floor is changed.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src")):
    sys.path.insert(0, p)
sys.dont_write_bytecode = True
import aspect_data as ad  # noqa: E402
import aspect_training as at  # noqa: E402
import prepare_adjudication_package as pap  # noqa: E402
import validate_aspect_annotations as va  # noqa: E402

ANN = os.path.join(ROOT, "data", "annotations")
PILOT = os.path.join(ROOT, "outputs", "annotation_consolidation", "pilot_batch_01")
RANKED = {"POSITIVE", "NEGATIVE", "MIXED"}
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}


def yield_stats(rows, products):
    """rows: dicts with message_id, entity_id, category, sentiment (one per annotated span). products: entity ids with a finished review."""
    by = defaultdict(list)
    for r in rows:
        if r["sentiment"]:
            by[(r["message_id"], r["entity_id"], r["category"])].append(r["sentiment"])
    ex = {k: ad.aggregate_polarity(v) for k, v in by.items()}
    ser = Counter((e, c) for (m, e, c), lab in ex.items() if lab in RANKED)
    return {"products": len(set(products)), "reviews": len({k[0] for k in ex}), "examples": len(ex),
            "usable_series": sum(n >= 2 for n in ser.values()), "class_counts": dict(Counter(ex.values()))}


def add(a, b):
    return {"products": a["products"] + b["products"], "reviews": a["reviews"] + b["reviews"], "examples": a["examples"] + b["examples"],
            "usable_series": a["usable_series"] + b["usable_series"],
            "class_counts": {k: a["class_counts"].get(k, 0) + b["class_counts"].get(k, 0) for k in set(a["class_counts"]) | set(b["class_counts"])}}


def check_floors(s):
    """Hard floors refuse (at.cv_blockers / series_blockers). Soft floors warn. The 70/15/15 product split is applied by proportion, so this is an estimate."""
    hard = at.cv_blockers(s["products"], s["examples"], 5)
    hard += at.series_blockers({k: int(s["usable_series"] * r) for k, r in SPLIT.items()})
    soft = []
    if s["products"] < at.MIN_TOTAL_PRODUCTS:
        soft.append(f"{s['products']} products (< {at.MIN_TOTAL_PRODUCTS})")
    if int(s["products"] * SPLIT["val"]) < at.MIN_EVAL_PRODUCTS:
        soft.append(f"about {int(s['products'] * SPLIT['val'])} products per evaluation split (< {at.MIN_EVAL_PRODUCTS})")
    if s["examples"] * SPLIT["train"] < at.MIN_TRAIN_ITEMS:
        soft.append(f"about {int(s['examples'] * SPLIT['train'])} training examples (< {at.MIN_TRAIN_ITEMS})")
    if s["examples"] * SPLIT["val"] < at.MIN_EVAL_ITEMS:
        soft.append(f"about {int(s['examples'] * SPLIT['val'])} examples per evaluation split (< {at.MIN_EVAL_ITEMS})")
    if s["class_counts"].get("NEGATIVE", 0) * SPLIT["val"] < at.MIN_EVAL_CLASS_EXAMPLES:
        soft.append(f"about {int(s['class_counts'].get('NEGATIVE', 0) * SPLIT['val'])} NEGATIVE examples per evaluation split (< {at.MIN_EVAL_CLASS_EXAMPLES})")
    return {"hard_blockers": hard, "warnings": soft}


def pilot_stats(consolidated, provenance):
    c = pd.read_csv(consolidated, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    p = pd.read_csv(provenance, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    c["elig"] = p.training_eligible.to_numpy()
    out = {}
    for name, keep in (("conservative", {"yes"}), ("optimistic", {"yes", "pending_policy"})):
        d = c[c.elig.isin(keep)]
        out[name] = yield_stats([dict(message_id=r.message_id, entity_id=r.entity_id, category=r.aspect_category, sentiment=r.sentiment) for r in d.itertuples()],
                                d.entity_id)
    return out


def mention_rows(df_a, df_b, ids):
    """Agreed mentions and all annotated spans counted once, over the messages in `ids`."""
    ent = df_a.drop_duplicates("message_id").set_index("message_id").entity_id
    agreed, union = [], []
    for m in ids:
        sa, sb = va._message_spans(df_a[df_a.message_id == m]), va._message_spans(df_b[df_b.message_id == m])
        pairs = va.match_spans(sa, sb)
        ma, mb = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
        mk = lambda s: dict(message_id=m, entity_id=ent[m], category=s[2], sentiment=s[3])           # noqa: E731
        for i, j, _ in pairs:
            union.append(mk(sa[i]))
            if sa[i][2] == sb[j][2] and sa[i][3] == sb[j][3] and sa[i][3]:
                agreed.append(mk(sa[i]))
        union += [mk(s) for i, s in enumerate(sa) if i not in ma] + [mk(s) for j, s in enumerate(sb) if j not in mb]
    return agreed, union


def agreement_extras(df_a, df_b, ids):
    types_a, types_b = [], []
    for m in ids:
        ga, gb = df_a[df_a.message_id == m], df_b[df_b.message_id == m]
        ta = {(int(r.span_start), int(r.span_end)): r.span_type for r in ga.itertuples() if r.aspect_status == "ASPECT_PRESENT"}
        tb = {(int(r.span_start), int(r.span_end)): r.span_type for r in gb.itertuples() if r.aspect_status == "ASPECT_PRESENT"}
        sa, sb = list(ta), list(tb)
        for i, j, _ in va.match_spans(sa, sb):
            types_a.append(ta[sa[i]]); types_b.append(tb[sb[j]])
    return va._agreement_block(types_a, types_b)


def unresolved(df_a, ra, df_b, rb, ids):
    st = lambda df, m: va._message_status(df[df.message_id == m])                                   # noqa: E731
    return {"NEEDS_REVIEW_rows_a": int((df_a.annotation_status == "NEEDS_REVIEW").sum()), "NEEDS_REVIEW_rows_b": int((df_b.annotation_status == "NEEDS_REVIEW").sum()),
            "UNCLEAR_ASPECT_messages_a": sum(st(df_a, m) == "UNCLEAR_ASPECT" for m in ids), "UNCLEAR_ASPECT_messages_b": sum(st(df_b, m) == "UNCLEAR_ASPECT" for m in ids),
            "message_status_disagreements": sum(st(df_a, m) != st(df_b, m) for m in ids),
            "messages_unfinished_or_invalid_in_a": len(ra["unfinished_messages"]), "messages_unfinished_or_invalid_in_b": len(rb["unfinished_messages"])}


def recommend(cons, opt, cons_w2, name="Wave 2"):
    c, o, w2 = check_floors(cons), check_floors(opt), check_floors(cons_w2)
    if not c["hard_blockers"] and not c["warnings"]:
        return f"{name} is NOT needed for the pipeline floors: the agreed-only corpus (pilot + this wave) already clears every hard floor and warning."
    if not w2["hard_blockers"] and not w2["warnings"]:
        return f"PROCEED with {name}: this wave leaves {len(c['hard_blockers'])} hard blocker(s) and {len(c['warnings'])} warning(s) on agreed-only data, and a second wave of the same size at the measured yield clears all of them."
    return (f"PROCEED with {name} and plan more: even with a second wave of the same size at the measured yield, {len(w2['hard_blockers'])} hard blocker(s) and "
            f"{len(w2['warnings'])} warning(s) remain ({'; '.join(w2['hard_blockers'] + w2['warnings'])}).")


def fmt(s):
    return f"{s['products']} products, {s['reviews']} reviews, {s['examples']} examples, {s['usable_series']} usable series"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--reference", default=os.path.join(ANN, "batch_02_wave1_reference.csv"))
    ap.add_argument("--wave_manifest", default=os.path.join(ANN, "batch_02_wave1_manifest.json"))
    ap.add_argument("--pilot_consolidated", default=os.path.join(PILOT, "pilot_batch_01_PROVISIONAL_consolidated.csv"))
    ap.add_argument("--pilot_provenance", default=os.path.join(PILOT, "pilot_batch_01_PROVISIONAL_provenance.csv"))
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out_dir) and os.listdir(a.out_dir):
        raise FileExistsError(f"{a.out_dir} already has files; refusing to overwrite")
    os.makedirs(a.out_dir, exist_ok=True)
    tax = va.load_taxonomies()
    (df_a, ra, _), (df_b, rb, _) = va.run(a.a, tax, a.reference), va.run(a.b, tax, a.reference)
    errs = {n: [i for i in r["issues"] if i["severity"] == va.ERROR] for n, r in (("a", ra), ("b", rb))}
    report = {"inputs": {"a": a.a, "b": a.b, "reference": a.reference}, "validation": {n: {"errors": len(e), "first_errors": e[:10]} for n, e in errs.items()}}
    if errs["a"] or errs["b"]:
        json.dump(report, open(f"{a.out_dir}/report.json", "w", encoding="utf-8"), indent=1, default=str)
        print("STOPPED: validation errors (a: %d, b: %d). Fix them, then rerun. See %s/report.json" % (len(errs["a"]), len(errs["b"]), a.out_dir))
        return 1
    ag = va.compare(df_a, ra, df_b, rb)
    ids = sorted((ra["complete_messages"] & rb["complete_messages"]) - set(ag["prepared_columns_differ_between_files"]))
    g = next(iter(ag["groups"].values()), {})
    report["agreement"] = {"messages_compared": len(ids), "warnings": ag["warnings"], "groups": ag["groups"], "span_type_on_matched_spans": agreement_extras(df_a, df_b, ids),
                           "unresolved": unresolved(df_a, ra, df_b, rb, ids)}
    pap.main(["--a", a.a, "--b", a.b, "--reference", a.reference, "--out_dir", f"{a.out_dir}/adjudication_package"])
    package = pd.read_csv(f"{a.out_dir}/adjudication_package/adjudication_items.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    report["adjudication_items"] = {"total": len(package), "by_type": package.item_type.value_counts().to_dict()}

    agreed, union = mention_rows(df_a, df_b, ids)
    ent = df_a.drop_duplicates("message_id").set_index("message_id")
    prod_of = {m: ent.at[m, "entity_id"] for m in ids}
    small = set()
    if os.path.exists(a.wave_manifest):
        small = set(json.load(open(a.wave_manifest, encoding="utf-8"))["PENDING_TEAM_APPROVAL"]["sequence_ids"])
    seq_of = {m: ent.at[m, "sequence_id"] for m in ids}
    nosmall = lambda rows: [r for r in rows if seq_of[r["message_id"]] not in small]                                       # noqa: E731
    w = {"agreed": yield_stats(agreed, prod_of.values()), "all_spans_once": yield_stats(union, prod_of.values()),
         "agreed_without_SMALL": yield_stats(nosmall(agreed), [p for m, p in prod_of.items() if seq_of[m] not in small])}
    pilot = pilot_stats(a.pilot_consolidated, a.pilot_provenance)
    cons, opt = add(pilot["conservative"], w["agreed"]), add(pilot["optimistic"], w["all_spans_once"])
    cons_w2 = add(cons, w["agreed"])
    report["yield"] = {"wave": w, "pilot": pilot, "corpus_agreed_only": cons, "corpus_optimistic": opt, "corpus_agreed_only_plus_second_wave": cons_w2,
                       "per_review": {"examples": w["agreed"]["examples"] / max(w["agreed"]["reviews"], 1), "series_per_product": w["agreed"]["usable_series"] / max(w["agreed"]["products"], 1)},
                       "estimate_to_beat": {"examples_per_review": 2.6, "series_per_product_at_10_reviews": 5.4},
                       "floors": {"agreed_only": check_floors(cons), "optimistic": check_floors(opt), "agreed_only_plus_second_wave": check_floors(cons_w2)}}
    report["recommendation"] = recommend(cons, opt, cons_w2)
    report["pending_team_approval"] = f"SMALL-stratum products ({len(small)} sequences) are included in the main counts; 'agreed_without_SMALL' shows the yield without them." if small else None
    json.dump(report, open(f"{a.out_dir}/report.json", "w", encoding="utf-8"), indent=1, default=str)

    sp = g.get("spans", {})
    L = ["# Wave report", "", f"Messages compared (finished and valid in both files): {len(ids)}.", ""]
    L += ["## Agreement", f"* span F1 (IoU >= 0.5): {sp.get('span_f1')}  (in A {sp.get('in_a')}, in B {sp.get('in_b')}, matched {sp.get('matched_iou_ge_0.5')})"]
    for k, t in (("aspect_status_per_message", "message aspect_status"), ("category_on_matched_spans", "category"), ("sentiment_on_matched_spans", "sentiment")):
        L.append(f"* {t}: raw {g.get(k, {}).get('raw_agreement')}, kappa {g.get(k, {}).get('cohens_kappa')}, n {g.get(k, {}).get('n')}")
    st = report["agreement"]["span_type_on_matched_spans"]
    L += [f"* span_type: raw {st['raw_agreement']}, kappa {st['cohens_kappa']}, n {st['n']}", f"* unresolved: {report['agreement']['unresolved']}", "",
          f"## Adjudication package: {len(package)} items ({report['adjudication_items']['by_type']}) in adjudication_package/", "", "## Measured yield (this wave)"]
    L += [f"* agreed mentions only: {fmt(w['agreed'])}", f"* all annotated spans counted once (upper bound): {fmt(w['all_spans_once'])}",
          f"* agreed, without SMALL products (sampling rule pending approval): {fmt(w['agreed_without_SMALL'])}", "",
          "## Corpus with the pilot", f"* agreed-only: {fmt(cons)}", f"* optimistic: {fmt(opt)}", f"* agreed-only plus a second wave like this one: {fmt(cons_w2)}"]
    for k, f in report["yield"]["floors"].items():
        L.append(f"* floors, {k}: hard blockers {f['hard_blockers'] or 'none'}; warnings {f['warnings'] or 'none'}")
    L += ["", "## Recommendation", report["recommendation"]]
    open(f"{a.out_dir}/report.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
