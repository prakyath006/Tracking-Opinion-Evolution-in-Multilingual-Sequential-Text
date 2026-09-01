"""
=============================================================================
Ontology Evaluation — Module 1a Standalone Metrics
=============================================================================
Evaluates the ontology (src/ontology.py) on its own terms — coverage,
internal consistency, and structural size — separate from any downstream
model's accuracy. A model's F1 score tells you how well a classifier fits
the labels the ontology produces; it says nothing about whether the ontology
itself maps real data cleanly or is internally coherent. This module answers
that second question.

Three kinds of metric:

  (a) Coverage    — of the real labels/ratings in each domain's data, what
                     fraction land on a real sentiment (POSITIVE/NEGATIVE/
                     MIXED) vs fall through to UNKNOWN?
  (b) Consistency — does any raw label string map to more than one
                     SentimentState anywhere in DOMAIN_CONFIGS? A raw label
                     must be deterministic, or "the same input" would silently
                     mean different things depending on which domain read it.
  (c) Structure   — depth (taxonomy layers), breadth (leaf concept count),
                     coupling (how many source files import the ontology,
                     as a rough measure of how load-bearing it is).

This mirrors the checks in tests/test_ontology_consistency.py (which guard
the same invariants as pass/fail assertions in CI) but reports them as
numbers for a human-readable report instead of raising on the first failure
— the two are complementary, not duplicates: the test suite is a gate, this
module is a report.

Usage:
    python src/ontology_eval.py
    -> outputs/ontology_evaluation_report.md
=============================================================================
"""

import os
import sys
import glob
import logging
import statistics
from typing import Dict, List, Optional

import pandas as pd

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ontology import (
    DOMAIN_CONFIGS,
    SentimentState,
    TransitionType,
    TrajectoryType,
    map_labels_to_ontology,
)

logger = logging.getLogger(__name__)

PREPROCESSED_DIR = os.path.join(WORKSPACE_ROOT, "data", "preprocessed")
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "outputs")
REPORT_PATH = os.path.join(OUTPUT_DIR, "ontology_evaluation_report.md")
METRICS_DIR = os.path.join(WORKSPACE_ROOT, "outputs", "metrics")
MODULE1_REPORT_PATH = os.path.join(METRICS_DIR, "module1_ontology.md")

# Same language -> preprocessed-CSV-prefix mapping used throughout the
# pipeline (src/dataset.py, tests/test_ontology_consistency.py).
LANG_FILE_PREFIX = {"tamil": "tamil", "malayalam": "mal", "kannada": "kannada"}

# Coverage below this is flagged prominently in the report rather than
# quietly listed. Chosen, not derived: DravidianCodeMix's "unknown_state"
# label is a legitimate annotator-uncertainty category (not a bug), and
# earlier work in this project observed it at 13-36% of real Dravidian data,
# so a strict 95%+ bar would flag every domain as "broken" when it isn't.
# 80% catches domains where UNKNOWN dominates enough to be a real, distinct
# problem (e.g. a broken label_mapping) without flagging expected messiness.
LOW_COVERAGE_THRESHOLD_PCT = 80.0

REAL_SENTIMENTS = (SentimentState.POSITIVE, SentimentState.NEGATIVE, SentimentState.MIXED)


# ──────────────────────────────────────────────────────────────────────────────
# (a) Coverage
# ──────────────────────────────────────────────────────────────────────────────

def compute_domain_coverage(domain: str) -> Dict:
    """
    Compute the fraction of a domain's real raw labels/ratings that map to
    POSITIVE/NEGATIVE/MIXED vs fall through to UNKNOWN.

    amazon_beauty has no raw string label column — its ontology entry point
    is SentimentState.from_rating() on the numeric 'rating' column (see
    src/dataset.py's AmazonSequenceDataset), and from_rating() has no
    UNKNOWN branch at all, so its coverage is 100% by construction: every
    star rating 1-5 lands on POSITIVE, NEGATIVE or MIXED. This is reported
    explicitly rather than silently treated as "the domain looks perfect",
    since it reflects an absence of an UNKNOWN case in the rating scale, not
    evidence about the underlying opinions' clarity.

    The three Dravidian domains have a real raw string label column, and
    'unknown_state' is one of the actual label values annotators used, so
    coverage there is a genuine measurement of how much of the raw data
    resolves to a sentiment vs. how much was marked/falls through as unclear.

    Returns
    -------
    Dict with: domain, total, non_unknown, unknown, coverage_pct, source,
    low_coverage (bool), note.
    """
    if domain == "amazon_beauty":
        csv_path = os.path.join(PREPROCESSED_DIR, "amazon_beauty_sequences.csv")
        if not os.path.exists(csv_path):
            return _missing_data_result(domain, csv_path)

        df = pd.read_csv(csv_path, encoding="utf-8")
        states = [SentimentState.from_rating(r) for r in df["rating"]]
        total = len(states)
        non_unknown = sum(1 for s in states if s in REAL_SENTIMENTS)
        unknown = total - non_unknown
        return {
            "domain": domain,
            "total": total,
            "non_unknown": non_unknown,
            "unknown": unknown,
            "coverage_pct": (non_unknown / total * 100) if total else 0.0,
            "source": os.path.relpath(csv_path, WORKSPACE_ROOT),
            "low_coverage": False,
            "note": (
                "SentimentState.from_rating() has no UNKNOWN branch — every "
                "1-5 star rating maps to POSITIVE/NEGATIVE/MIXED by "
                "construction. 100% coverage here reflects the rating scale "
                "having no 'unclear' option, not an absence of ambiguous "
                "opinions."
            ),
        }

    if not domain.startswith("dravidian_"):
        raise ValueError(f"Unhandled domain for coverage: {domain}")

    language = domain.split("_", 1)[1]
    prefix = LANG_FILE_PREFIX.get(language, language)
    pattern = os.path.join(PREPROCESSED_DIR, f"{prefix}_sentiment_*_preprocessed.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return _missing_data_result(domain, pattern)

    total = non_unknown = 0
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8")
        states = map_labels_to_ontology(df["label"].astype(str).tolist(), domain)
        total += len(states)
        non_unknown += sum(1 for s in states if s in REAL_SENTIMENTS)
    unknown = total - non_unknown
    coverage_pct = (non_unknown / total * 100) if total else 0.0

    return {
        "domain": domain,
        "total": total,
        "non_unknown": non_unknown,
        "unknown": unknown,
        "coverage_pct": coverage_pct,
        "source": ", ".join(os.path.relpath(p, WORKSPACE_ROOT) for p in paths),
        "low_coverage": coverage_pct < LOW_COVERAGE_THRESHOLD_PCT,
        "note": (
            "'unknown_state' is a real DravidianCodeMix annotation label "
            "(annotator marked the comment's sentiment as unclear), not a "
            "mapping failure, so UNKNOWN here is expected to be nonzero."
        ),
    }


def _missing_data_result(domain: str, looked_for: str) -> Dict:
    return {
        "domain": domain, "total": 0, "non_unknown": 0, "unknown": 0,
        "coverage_pct": None, "source": None, "low_coverage": False,
        "note": f"SKIPPED — no data found (looked for {looked_for}). "
                f"data/ is gitignored; this is expected in a fresh checkout.",
    }


def compute_all_domain_coverage() -> List[Dict]:
    return [compute_domain_coverage(domain) for domain in sorted(DOMAIN_CONFIGS)]


def compute_coverage_stability_ratio(coverage_rows: Optional[List[Dict]] = None) -> Dict:
    """
    Ontology Coverage Stability Ratio = mean(coverage_pct) / std(coverage_pct)
    across domains with data available.

    A single summary number for "how evenly does the ontology cover its
    domains" — high when coverage is uniformly high (or uniformly low) across
    domains, low when domains disagree sharply (e.g. amazon_beauty at 100%
    next to dravidian_malayalam at 64.3%, as observed on real data). This is
    a dispersion ratio in the same family as a Sharpe-ratio-style
    mean/std construction, but named for what it actually measures here —
    coverage stability, not a risk-adjusted financial return — since that is
    the concrete, unambiguous metric asked for.

    Requires at least 2 domains with data (std of one value is undefined).
    Uses sample standard deviation (ddof=1, statistics.stdev), matching the
    conventional definition for a ratio computed over a small population of
    domains rather than an infinite one.

    Returns
    -------
    Dict with: ratio (float or None), mean_coverage, std_coverage,
    domains_used (list), domains_skipped (list, no data), note.
    """
    if coverage_rows is None:
        coverage_rows = compute_all_domain_coverage()

    available = [r for r in coverage_rows if r["coverage_pct"] is not None]
    skipped = [r["domain"] for r in coverage_rows if r["coverage_pct"] is None]

    if len(available) < 2:
        return {
            "ratio": None,
            "mean_coverage": available[0]["coverage_pct"] if available else None,
            "std_coverage": None,
            "domains_used": [r["domain"] for r in available],
            "domains_skipped": skipped,
            "note": (
                f"Need >=2 domains with data to compute a standard deviation; "
                f"only {len(available)} available. Ratio not computed."
            ),
        }

    values = [r["coverage_pct"] for r in available]
    mean_cov = statistics.mean(values)
    std_cov = statistics.stdev(values)

    if std_cov == 0:
        return {
            "ratio": None,
            "mean_coverage": mean_cov,
            "std_coverage": 0.0,
            "domains_used": [r["domain"] for r in available],
            "domains_skipped": skipped,
            "note": "All domains have identical coverage (std=0); ratio is undefined (division by zero).",
        }

    return {
        "ratio": mean_cov / std_cov,
        "mean_coverage": mean_cov,
        "std_coverage": std_cov,
        "domains_used": [r["domain"] for r in available],
        "domains_skipped": skipped,
        "note": (
            f"Computed over {len(available)} domain(s) with data "
            f"({', '.join(r['domain'] for r in available)})."
            + (f" Skipped (no data): {', '.join(skipped)}." if skipped else "")
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# (b) Consistency
# ──────────────────────────────────────────────────────────────────────────────

def check_label_mapping_consistency() -> Dict:
    """
    Verify no raw label string maps to more than one SentimentState anywhere
    across DOMAIN_CONFIGS' label_mapping dicts, pooled together.

    Each domain's label_mapping is its own namespace in practice (dataset.py
    always passes a domain key alongside the label), so a collision here
    wouldn't actually corrupt a single domain's mapping today. It matters
    anyway: if the SAME raw string is meant to mean different sentiments in
    different domains, that is worth knowing explicitly rather than
    discovering by accident, and if it's an accidental typo (e.g. two
    domains meant to agree but one has a stale value), this catches it.

    Returns
    -------
    Dict with: consistent (bool), label_count (distinct labels checked),
    conflicts (list of {label, domain_states: [(domain, state), ...]}).
    """
    label_to_domain_states: Dict[str, List] = {}
    for domain, config in DOMAIN_CONFIGS.items():
        for raw_label, state in config.label_mapping.items():
            label_to_domain_states.setdefault(raw_label, []).append((domain, state))

    conflicts = []
    for raw_label, domain_states in label_to_domain_states.items():
        distinct_states = {state for _, state in domain_states}
        if len(distinct_states) > 1:
            conflicts.append({"label": raw_label, "domain_states": domain_states})

    return {
        "consistent": len(conflicts) == 0,
        "label_count": len(label_to_domain_states),
        "conflicts": conflicts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# (c) Structural metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_structural_metrics() -> Dict:
    """
    depth    — number of taxonomy layers (Sentiment, Transition, Trajectory,
               Domain), matching ontology.py's own section numbering.
    breadth  — total leaf concepts: sentiment states + transition types +
               trajectory types + domain configs.
    coupling — number of source files (src/, scripts/, tests/) that import
               from ontology.py, as a rough proxy for how load-bearing it is.
               Computed by scanning the actual files at call time rather than
               hardcoded, so it can't silently drift out of date.
    """
    depth = 4  # Sentiment State / Transition / Trajectory / Domain Ontology

    breadth = (
        SentimentState.num_classes()
        + TransitionType.num_classes()
        + TrajectoryType.num_classes()
        + len(DOMAIN_CONFIGS)
    )

    coupling_files = []
    for subdir in ("src", "scripts", "tests"):
        for path in glob.glob(os.path.join(WORKSPACE_ROOT, subdir, "*.py")):
            if os.path.basename(path) == "ontology.py":
                continue
            with open(path, encoding="utf-8") as f:
                content = f.read()
            if "from ontology import" in content or "import ontology" in content:
                coupling_files.append(os.path.relpath(path, WORKSPACE_ROOT))

    return {
        "depth": depth,
        "breadth": breadth,
        "breadth_detail": {
            "sentiment_states": SentimentState.num_classes(),
            "transition_types": TransitionType.num_classes(),
            "trajectory_types": TrajectoryType.num_classes(),
            "domain_configs": len(DOMAIN_CONFIGS),
        },
        "coupling": len(coupling_files),
        "coupling_files": sorted(coupling_files),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_ontology_eval_report(write: bool = True) -> str:
    """
    Assemble the full ontology evaluation as markdown.

    Parameters
    ----------
    write : bool
        If True (default), also write the report to
        outputs/ontology_evaluation_report.md.

    Returns
    -------
    str
        The markdown report text.
    """
    coverage_rows = compute_all_domain_coverage()
    consistency = check_label_mapping_consistency()
    structure = compute_structural_metrics()

    lines = []
    lines.append("# Ontology Evaluation Report")
    lines.append("")
    lines.append(
        "Standalone evaluation of `src/ontology.py` — coverage, internal "
        "consistency, and structural size. Separate from model accuracy: "
        "this measures the ontology itself, not any classifier trained "
        "against it."
    )
    lines.append("")

    # ── Coverage ──
    lines.append("## (a) Coverage")
    lines.append("")
    lines.append(
        "Fraction of each domain's real raw labels/ratings that map to a "
        "real sentiment (POSITIVE/NEGATIVE/MIXED) vs fall through to "
        f"UNKNOWN. Domains below {LOW_COVERAGE_THRESHOLD_PCT:.0f}% are "
        "flagged."
    )
    lines.append("")
    lines.append("| Domain | Total | Non-UNKNOWN | UNKNOWN | Coverage | Flag |")
    lines.append("|---|---|---|---|---|---|")

    flagged_domains = []
    skipped_domains = []
    for row in coverage_rows:
        if row["coverage_pct"] is None:
            skipped_domains.append(row["domain"])
            lines.append(f"| {row['domain']} | - | - | - | - | ⏭️ SKIPPED (no data) |")
            continue
        flag = "🚩 LOW COVERAGE" if row["low_coverage"] else "✅"
        if row["low_coverage"]:
            flagged_domains.append(row["domain"])
        lines.append(
            f"| {row['domain']} | {row['total']:,} | {row['non_unknown']:,} | "
            f"{row['unknown']:,} | {row['coverage_pct']:.1f}% | {flag} |"
        )
    lines.append("")

    if flagged_domains:
        lines.append(
            f"**🚩 Flagged: {', '.join(flagged_domains)} — coverage below "
            f"{LOW_COVERAGE_THRESHOLD_PCT:.0f}%.** Investigate before "
            f"treating this domain's sentiment labels as reliable."
        )
        lines.append("")
    if skipped_domains:
        lines.append(
            f"*Skipped (no local data): {', '.join(skipped_domains)}. "
            f"`data/` is gitignored, so this is expected outside a "
            f"checkout that has pulled the datasets.*"
        )
        lines.append("")

    lines.append("Per-domain notes:")
    for row in coverage_rows:
        lines.append(f"- **{row['domain']}**: {row['note']}")
    lines.append("")

    # ── Consistency ──
    lines.append("## (b) Consistency")
    lines.append("")
    lines.append(
        f"Checked {consistency['label_count']} distinct raw label strings "
        f"pooled across all {len(DOMAIN_CONFIGS)} domains' `label_mapping` "
        f"dicts in `DOMAIN_CONFIGS`."
    )
    lines.append("")
    if consistency["consistent"]:
        lines.append(
            "✅ **Consistent.** No raw label string maps to more than one "
            "SentimentState across any domain's mapping."
        )
    else:
        lines.append(
            f"🚩 **{len(consistency['conflicts'])} conflict(s) found.** The "
            f"same raw label maps to different SentimentStates in different "
            f"domains:"
        )
        lines.append("")
        for c in consistency["conflicts"]:
            states_str = ", ".join(f"{d}→{s.name}" for d, s in c["domain_states"])
            lines.append(f"- `{c['label']}`: {states_str}")
    lines.append("")
    lines.append(
        "This mirrors `tests/test_ontology_consistency.py::"
        "test_domain_label_mapping_targets_valid_states`, which asserts "
        "each domain's mapping targets are valid SentimentState members "
        "(fails CI on violation); this report additionally checks for "
        "cross-domain label collisions, which the pass/fail test suite "
        "does not."
    )
    lines.append("")

    # ── Structure ──
    lines.append("## (c) Structural Metrics")
    lines.append("")
    lines.append(f"- **Depth**: {structure['depth']} taxonomy layers "
                  f"(Sentiment State, Transition, Trajectory, Domain Ontology)")
    lines.append(f"- **Breadth**: {structure['breadth']} leaf concepts "
                  f"({structure['breadth_detail']['sentiment_states']} sentiment "
                  f"+ {structure['breadth_detail']['transition_types']} transition "
                  f"+ {structure['breadth_detail']['trajectory_types']} trajectory "
                  f"+ {structure['breadth_detail']['domain_configs']} domain configs)")
    lines.append(f"- **Coupling**: {structure['coupling']} source files import "
                  f"from `ontology.py`, computed by scanning `src/`, `scripts/`, "
                  f"`tests/` at report-generation time:")
    for f in structure["coupling_files"]:
        lines.append(f"  - `{f}`")
    lines.append("")

    report = "\n".join(lines)

    if write:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Saved: {REPORT_PATH}")

    return report


def generate_module1_report(write: bool = True) -> str:
    """
    Module 1 (Ontology) metrics report for the per-module metrics framework
    -- reuses generate_ontology_eval_report()'s coverage/consistency/structure
    sections verbatim (no duplicated logic) and appends the Coverage
    Stability Ratio.

    Parameters
    ----------
    write : bool
        If True (default), write to outputs/metrics/module1_ontology.md.

    Returns
    -------
    str
        The markdown report text.
    """
    base_report = generate_ontology_eval_report(write=False)
    stability = compute_coverage_stability_ratio()

    lines = [base_report, "", "## Ontology Coverage Stability Ratio", ""]
    lines.append(
        "mean(coverage %) / std(coverage %) across domains with data — a "
        "single number for how evenly the ontology covers its domains."
    )
    lines.append("")
    if stability["ratio"] is None:
        lines.append(f"**Ratio: not computed.** {stability['note']}")
    else:
        lines.append(f"**Ratio: {stability['ratio']:.2f}** "
                      f"(mean={stability['mean_coverage']:.1f}%, "
                      f"std={stability['std_coverage']:.1f}%)")
        lines.append("")
        lines.append(stability["note"])
    lines.append("")

    report = "\n".join(lines)

    if write:
        os.makedirs(METRICS_DIR, exist_ok=True)
        with open(MODULE1_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Saved: {MODULE1_REPORT_PATH}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    print(generate_module1_report())
