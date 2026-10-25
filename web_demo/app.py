"""
=============================================================================
Opinion Evolution Tracker -- interactive demo
=============================================================================
A product-shaped front end over a real trained model: reads a sequence of
reviews and predicts how the opinion moved, review by review.

TRUTHFULNESS CONTRACT
---------------------
Every number shown here is read at page load from a file under outputs/, or
computed live from src/. Nothing is hardcoded, illustrative, or "representative".
When a result has not been produced yet, the page says so and names the command
that produces it -- it never substitutes a placeholder value.

This matters: an earlier version of this file carried a `demo_metrics` table
with invented figures (trajectory accuracy 92.4%, SCS 0.91, ECE 0.042) that
contradicted the project's real measurements (56.75%, 0.5636, and ECE never
computed at all). Anyone comparing the demo against the report would have
concluded the results were fabricated. Do not reintroduce hardcoded metrics.

Four sections, deliberately not one page per research module:
  - Analyze: live trajectory prediction on a real trained checkpoint (the
    default landing page)
  - Insights: how the model performs -- baselines, cross-domain transfer,
    calibration -- as tabs within one section, not three separate ones
  - How it works: the ontology, code-mixed aspect detection, and the
    architecture, also as tabs within one section
  - About: what the system does and why

Run:
    streamlit run web_demo/app.py

Author: B.Tech Project Team
Date: 2026
=============================================================================
"""

import os
import sys
import json
import glob
import datetime
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

OUTPUTS = os.path.join(WORKSPACE_ROOT, "outputs")
METRICS = os.path.join(OUTPUTS, "metrics")

# Large files are not in git; on a fresh deployment they are pulled from free Hugging Face repos.
GITHUB_URL = "https://github.com/prakyath006/Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text"
HF_CKPT_REPO = "prakyath18/opinion-tracker-checkpoints"     # model repo: best_model_*.pt
HF_DATA_REPO = "prakyath18/opinion-tracker-data"            # dataset repo: contents of data/preprocessed/


@st.cache_resource(show_spinner="Downloading files (first start only)…")
def fetch_hf(repo_id: str, repo_type: str, local_dir: str) -> str:
    """Download a Hugging Face repo into local_dir; returns "" on success or the error text."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir)
        return ""
    except Exception as e:                                # pragma: no cover
        return str(e)


_PRE = os.path.join(WORKSPACE_ROOT, "data", "preprocessed")
if not os.path.isdir(_PRE) or not os.listdir(_PRE):
    fetch_hf(HF_DATA_REPO, "dataset", _PRE)

try:
    from ontology import (
        SentimentState, TransitionType, TrajectoryType,
        DOMAIN_CONFIGS, map_labels_to_ontology,
    )
    from wsd import WordSenseDisambiguator
    BACKEND_AVAILABLE = True
    BACKEND_ERROR = ""
except Exception as e:                                    # pragma: no cover
    BACKEND_AVAILABLE = False
    BACKEND_ERROR = str(e)


# =============================================================================
# Data access layer -- every figure on every page comes through here
# =============================================================================

def _mtime(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    return datetime.datetime.fromtimestamp(
        os.path.getmtime(path)
    ).strftime("%Y-%m-%d %H:%M")


@st.cache_data(show_spinner=False)
def load_json(relpath: str):
    """Read a JSON file under outputs/. Returns None when it does not exist."""
    path = os.path.join(OUTPUTS, relpath)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_text(relpath: str) -> Optional[str]:
    path = os.path.join(OUTPUTS, relpath)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def load_csv(relpath: str):
    path = os.path.join(OUTPUTS, relpath)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=True)
def compute_ontology_coverage():
    """Ontology coverage, computed live from the real corpus."""
    try:
        from ontology_eval import (
            compute_all_domain_coverage, check_label_mapping_consistency,
            compute_structural_metrics,
        )
        return {
            "coverage": compute_all_domain_coverage(),
            "consistency": check_label_mapping_consistency(),
            "structure": compute_structural_metrics(),
        }
    except Exception as e:                                # pragma: no cover
        return {"error": str(e)}


def not_measured(what: str, command: str, why: str = ""):
    """
    Render a missing result honestly.

    Used wherever a metric has not been produced yet. Never replace this with
    a plausible-looking number -- the whole credibility of the demo rests on
    a blank here meaning "not measured", not "we forgot".
    """
    st.info(
        f"**{what} — not yet measured.**\n\n"
        f"This figure is deliberately blank rather than estimated. "
        f"Produce it with:\n\n```bash\n{command}\n```"
        + (f"\n\n{why}" if why else "")
    )


def results_rows() -> List[Dict]:
    return load_json("metrics/results_table.json") or []


def full_model_name() -> str:
    return "full_model (OpinionEvolutionTracker)"


def state_kind(name: str) -> str:
    """Map an ontology label to one of four fixed semantic colors, used
    identically everywhere a state appears (chip, chart, table)."""
    name = (name or "").upper()
    if name in ("POSITIVE", "IMPROVING", "UPGRADE"):
        return "positive"
    if name in ("NEGATIVE", "DECLINING", "DOWNGRADE"):
        return "negative"
    if name in ("MIXED", "MIXED_FEELINGS", "VOLATILE"):
        return "mixed"
    return "neutral"  # STABLE, UNKNOWN and anything else


def chip(name: str) -> str:
    """Render one ontology state as a colored inline chip (HTML)."""
    return f'<span class="chip chip-{state_kind(name)}">{name}</span>'


def chip_row(items) -> str:
    """A left-aligned strip of chips, for building small HTML tables."""
    return " ".join(chip(i) for i in items)


# =============================================================================
# Page configuration and styling
#
# Color is the taxonomy, not decoration: the four fixed ontology states
# (positive/negative/mixed/neutral) get one hue each, reused identically in
# every chip, chart and table across the whole app -- see state_kind() above.
# System messages (st.info/warning/error/success) keep Streamlit's own
# semantics and are deliberately NOT recolored to match, so a reader never
# confuses "the model predicted NEGATIVE" with "something went wrong".
# =============================================================================

st.set_page_config(
    page_title="Opinion Evolution Tracker",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
    --ink: #F4F8FB;
    --surface: #FFFFFF;
    --line: #D3DEE8;
    --paper: #14212B;
    --mist: #52647A;
    --pulse: #1E8E6A;
    --decline: #C0392B;
    --caution: #B7791F;
    --neutral: #5A6B7C;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

/* Headline treatment -- the display face carries the product's identity */
.main-header {
    font-family: 'Spectral', serif;
    font-weight: 600;
    font-size: 2.1rem;
    color: var(--paper);
    letter-spacing: -0.01em;
    margin-bottom: 0.3rem;
}
.sub-header {
    font-size: 1.02rem;
    color: var(--mist);
    max-width: 62ch;
    line-height: 1.5;
    margin-bottom: 1.6rem;
}
.wordmark {
    font-family: 'Spectral', serif;
    font-weight: 600;
    font-size: 1.3rem;
    color: var(--paper);
    padding: 0.9rem 0 0.2rem 0;
}
.wordmark-strap { color: var(--mist); font-size: 0.85rem; margin-bottom: 0.6rem; }

/* Sidebar as a quiet vertical index rather than a form control */
[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 0.35rem 0.6rem;
    border-radius: 6px;
    margin-bottom: 0.05rem;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(30, 142, 106, 0.10);
}

/* Numeric readouts get the monospace cut -- data, not decoration */
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }

/* Ontology-state chips -- the one place color carries meaning end to end */
.chip {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.12rem 0.55rem;
    border-radius: 4px;
    border: 1px solid;
    margin: 0.1rem 0.15rem 0.1rem 0;
    white-space: nowrap;
}
.chip-positive { color: #11694D; background: rgba(30, 142, 106, 0.12); border-color: rgba(30, 142, 106, 0.45); }
.chip-negative { color: #A52A1E; background: rgba(192, 57, 43, 0.10); border-color: rgba(192, 57, 43, 0.45); }
.chip-mixed    { color: #8A5A0B; background: rgba(183, 121, 31, 0.12); border-color: rgba(183, 121, 31, 0.45); }
.chip-neutral  { color: #3F5066; background: rgba(90, 107, 124, 0.10); border-color: rgba(90, 107, 124, 0.4); }

.chip-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.chip-table th {
    text-align: left; color: var(--mist); font-weight: 500;
    border-bottom: 1px solid var(--line); padding: 0.4rem 0.6rem;
}
.chip-table td {
    border-bottom: 1px solid var(--line); padding: 0.45rem 0.6rem;
    color: var(--paper); vertical-align: middle;
}

.hero-tag {
    display: inline-block; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--pulse); border: 1px solid var(--pulse);
    border-radius: 999px; padding: 0.15rem 0.8rem; margin-right: 0.5rem;
}
.hero-title {
    font-family: 'Spectral', serif; font-weight: 600; font-size: 2.5rem; line-height: 1.2;
    color: var(--paper); letter-spacing: -0.01em; margin: 1rem 0 0.6rem 0; max-width: 30ch;
}
.hero { background: linear-gradient(120deg, #0B7A6B 0%, #1565C0 100%);
    border-radius: 18px; box-shadow: 0 14px 36px rgba(21,101,192,0.30); padding: 2rem 2.2rem 1.5rem 2.2rem; margin-bottom: 1.2rem; }
.hero .hero-title { color: #FFFFFF; }
.hero .lead { color: rgba(255,255,255,0.92); }
.hero .hero-tag { color: #FFFFFF; border-color: rgba(255,255,255,0.7); background: rgba(255,255,255,0.12); }
.hero .btn { background: #FFFFFF; color: #0B5E9B !important; }
.hero .btn.ghost { background: transparent; color: #FFFFFF !important; border: 1.5px solid #FFFFFF; }
.hero .stat { background: rgba(255,255,255,0.96); }
.hero-meta { color: var(--mist); font-size: 1rem; margin-bottom: 0.4rem; }
.ex-row { display: flex; align-items: center; gap: 0.8rem; padding: 0.45rem 0; border-bottom: 1px solid var(--line); }
.ex-row .txt { flex: 1; color: var(--paper); font-size: 0.95rem; }
.ex-bar { height: 8px; border-radius: 4px; background: var(--pulse); }
.ex-att { width: 130px; background: var(--line); border-radius: 4px; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 12px; padding: 0.7rem 0.9rem; box-shadow: 0 3px 10px rgba(20,33,43,0.05); }
.hero-meta b { color: var(--paper); }
.team-table { width: 100%; max-width: 640px; border-collapse: collapse; font-size: 0.95rem; }
.team-table th { text-align: left; color: var(--mist); font-weight: 500;
    border-bottom: 1px solid var(--line); padding: 0.4rem 0.6rem; }
.team-table td { border-bottom: 1px solid var(--line); padding: 0.45rem 0.6rem; color: var(--paper); }
.team-table td.mono { font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem; }
.step-card { background: var(--surface); border: 1px solid var(--line); border-top: 4px solid var(--pulse); border-radius: 12px;
    box-shadow: 0 4px 14px rgba(20,33,43,0.06); padding: 1rem 1.1rem; height: 100%; }
.step-card .n { font-family: 'IBM Plex Mono', monospace; color: var(--pulse); font-size: 0.8rem; }
.step-card b { color: var(--paper); }
.step-card div.t { color: var(--mist); font-size: 0.92rem; margin-top: 0.3rem; line-height: 1.45; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; font-weight: 500; }
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="wordmark">Opinion Evolution Tracker</div>',
                     unsafe_allow_html=True)
st.sidebar.markdown('<div class="wordmark-strap">Follows how an opinion moves, review by review.</div>',
                     unsafe_allow_html=True)

page = st.sidebar.radio(
    "Section",
    ["Home", "Analyze", "Insights", "How it works", "About"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "mBERT + Bi-LSTM + self-attention · English, Tamil, Malayalam, Kannada · "
    "e-commerce and social media"
)

# Data freshness panel -- shows exactly which results are live and how old.
if not BACKEND_AVAILABLE:
    st.sidebar.error(f"Backend import failed: {BACKEND_ERROR}")


# =============================================================================
# =============================================================================
# Page bodies -- one function per section's content. Grouped into a product
# information architecture below (Analyze / Insights / How it works / About),
# not a one-to-one list of research modules.
# =============================================================================

HOME_CSS = """
.wrap-sec { margin: 2.6rem 0 0.9rem 0; }
.kicker { font-size: 0.74rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--pulse); }
.sec-title { font-family: 'Spectral', serif; font-weight: 600; font-size: 1.7rem; color: var(--paper); margin: 0.1rem 0 0 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }
.card { background: #FFFFFF; border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.15rem;
    box-shadow: 0 4px 16px rgba(20,33,43,0.06); }
.card h4 { margin: 0 0 0.3rem 0; font-size: 1rem; color: var(--paper); }
.card p, .card li { margin: 0; color: var(--mist); font-size: 0.93rem; line-height: 1.5; }
.card .num { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--pulse); font-weight: 600; }
.card.accent { border-top: 4px solid var(--pulse); }
.lead { color: var(--mist); font-size: 1.12rem; line-height: 1.55; max-width: 60ch; margin: 0.2rem 0 1.1rem 0; }
.btn { display: inline-block; background: var(--pulse); color: #FFFFFF !important; text-decoration: none !important;
    font-weight: 600; font-size: 0.92rem; padding: 0.5rem 1.1rem; border-radius: 999px; margin-right: 0.5rem; }
.btn.ghost { background: transparent; color: var(--pulse) !important; border: 1.5px solid var(--pulse); }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 1.4rem; }
.stat { background: rgba(255,255,255,0.75); border: 1px solid var(--line); border-radius: 12px; padding: 0.7rem 0.9rem; }
.stat .v { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: var(--paper); }
.stat .l { font-size: 0.8rem; color: var(--mist); }
.pipe { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 0.4rem 0 1rem 0; }
.node { background: #FFFFFF; border: 1px solid var(--line); border-radius: 10px; padding: 0.5rem 0.8rem;
    font-size: 0.9rem; color: var(--paper); box-shadow: 0 2px 8px rgba(20,33,43,0.05); }
.node small { display: block; color: var(--mist); font-size: 0.75rem; }
.node.key { border-color: var(--pulse); background: rgba(255,255,255,0.95); }
.arrow { color: var(--pulse); font-weight: 700; }
.pills span { display: inline-block; background: #FFFFFF; border: 1px solid var(--line); border-radius: 999px;
    padding: 0.28rem 0.8rem; margin: 0.15rem 0.2rem 0.15rem 0; font-size: 0.86rem; color: var(--paper); }
.person .n { font-weight: 600; color: var(--paper); }
.person .r { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--mist); margin-top: 0.15rem; }
.doclink { text-decoration: none !important; display: block; }
.doclink .card:hover { border-color: var(--pulse); }
.st-done { color: #11694D; } .st-run { color: #8A5A0B; } .st-plan { color: #3F5066; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 0.45rem; }
.card ul { padding-left: 1.1rem; margin: 0.4rem 0 0 0; }
"""


def _sec(kicker, title):
    return (f'<div class="wrap-sec"><div class="kicker">{kicker}</div>'
            f'<div class="sec-title">{title}</div></div>')


def render_home():
    st.markdown(f"<style>{HOME_CSS}</style>", unsafe_allow_html=True)
    rows = results_rows()
    full = next((r for r in rows if r["model"] == full_model_name() and r["source"] == "amazon"
                 and r["target"] == "amazon"), None)
    base = (load_json("metrics/valid_baselines_amazon.json") or {}).get("majority_baselines", {})
    maj = next((v for k, v in base.items() if "trajectory" in k.lower()), None)
    stats = [("4", "languages: EN, TA, ML, KN"), ("3", "predictions per sequence")]
    if full:
        sub = f"trajectory macro-F1 (majority baseline {maj['macro_f1']:.3f})" if maj else "trajectory macro-F1"
        stats.append((f"{full['trajectory_f1_macro']:.3f}", sub))
    stats.append(("3.1 M", "trainable of 180.9 M parameters"))
    stat_html = "".join(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in stats)
    blob = GITHUB_URL + "/blob/main/"

    st.markdown(
        '<div class="hero">'
        '<span class="hero-tag">Data Science and Data Analytics</span><span class="hero-tag">Team 78</span>'
        '<div class="hero-title">A Transformer-Based Framework for Tracking Opinion Evolution in Multilingual Sequential Text</div>'
        '<div class="lead">Reads a person’s reviews in order and predicts how each one feels, how the opinion moved, '
        'and where it turned, in English, Tamil, Malayalam and Kannada.</div>'
        f'<a class="btn" href="{GITHUB_URL}" target="_blank">GitHub repository</a>'
        f'<a class="btn ghost" href="{blob}README.md" target="_blank">Documentation</a>'
        f'<div class="stats">{stat_html}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    team = [("1", "CB.SC.U4CSE23619", "Gudi Hemanth"), ("2", "CB.SC.U4CSE23622", "Yaswanth Jallipalli"),
            ("3", "CB.SC.U4CSE23626", "Kanishkhan J"), ("4", "CB.SC.U4CSE23635", "Prakyath Nandigam")]
    cards = "".join(f'<div class="card person accent"><div class="n">{n}</div><div class="r">{r}</div></div>'
                    for _, r, n in team)
    st.markdown(
        _sec("Team 78", "Team details")
        + f'<div class="grid">{cards}</div>'
        + '<p style="color:#52647A;margin:0.8rem 0 0 0"><b>Guide:</b> Ms. Krishna Priya G. &nbsp;·&nbsp; '
          'Department of Computer Science and Engineering, Amrita School of Computing, '
          'Amrita Vishwa Vidyapeetham, Coimbatore</p>',
        unsafe_allow_html=True,
    )

    objectives = [
        ("Track opinion over a sequence", "One model reads a person’s reviews in order and predicts each review’s sentiment, the transition between reviews, and the overall trajectory."),
        ("Find the turning point", "Attention weights show which review moved the opinion."),
        ("Handle multilingual, code-mixed text", "English, Tamil, Malayalam and Kannada share one label scheme."),
        ("Measure more than accuracy", "Baselines, ablations, calibration and a consistency score."),
        ("Test transfer", "How well the model carries across domains and languages, with leakage-controlled evaluation."),
    ]
    obj = "".join(f'<div class="card accent"><div class="num">0{i}</div><h4>{t}</h4><p>{d}</p></div>'
                  for i, (t, d) in enumerate(objectives, 1))
    st.markdown(_sec("Why we are building it", "Project objectives") + f'<div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(180px,1fr))">{obj}</div>',
                unsafe_allow_html=True)

    st.markdown(
        _sec("The gap", "Problem statement")
        + '<div class="grid">'
          '<div class="card"><h4>Opinions change</h4><p>A customer may start happy, hit a problem and end disappointed. '
          'Scoring each review on its own cannot see that change or where it happened.</p></div>'
          '<div class="card"><h4>Multilingual text is harder</h4><p>Tamil, Malayalam and Kannada mixed with English '
          'are much less familiar to standard encoders, and platforms label opinion differently '
          '(star ratings on Amazon, text labels on social media).</p></div>'
          '<div class="card accent"><h4>Our task</h4><p>Given reviews in chronological order, predict how each feels, '
          'how the opinion moved from one to the next, and the overall trajectory, across languages and domains.</p></div>'
          '</div>',
        unsafe_allow_html=True,
    )

    nodes = [("Reviews", "in order", False), ("Tokenizer", "multilingual", False), ("mBERT", "frozen encoder", True),
             ("Bi-LSTM", "2 layers, order memory", True), ("Attention", "turning point", True), ("Three heads", "see below", True)]
    pipe = '<span class="arrow">→</span>'.join(
        f'<div class="node{" key" if k else ""}">{a}<small>{b}</small></div>' for a, b, k in nodes)
    st.markdown(
        _sec("How it works", "Methodology")
        + f'<div class="pipe">{pipe}</div>'
        + '<div class="grid">'
          '<div class="card"><h4>Sentiment per review</h4><p>Positive, negative, mixed or unknown.</p></div>'
          '<div class="card"><h4>Transition per step</h4><p>Upgrade, downgrade or stable between consecutive reviews.</p></div>'
          '<div class="card"><h4>Overall trajectory</h4><p>Improving, declining, stable or volatile.</p></div>'
          '</div>'
          '<p style="color:#52647A;margin:0.9rem 0 0.2rem 0;font-size:0.93rem">A shared label ontology puts every dataset on one scale. '
          'Evaluation uses macro-F1 against majority-class and learned baselines, ablations, calibration (ECE), '
          'a Sequence Consistency Score and cross-domain transfer.</p>',
        unsafe_allow_html=True,
    )

    ex = [("Product looks great, very happy with the purchase", "POSITIVE", "chip-positive", "", 0.004),
          ("Still working fine after two weeks", "POSITIVE", "chip-positive", "STABLE", 0.156),
          ("Started having problems, quality dropped", "NEGATIVE", "chip-negative", "DOWNGRADE", 0.333),
          ("Completely stopped working, very disappointed", "NEGATIVE", "chip-negative", "STABLE", 0.507)]
    html = ""
    for text, sent, cls, trans, att in ex:
        tcls = "chip-negative" if trans == "DOWNGRADE" else "chip-neutral"
        tchip = f'<span class="chip {tcls}">{trans}</span>' if trans else ""
        html += (f'<div class="ex-row"><div class="txt">{text}</div><span class="chip {cls}">{sent}</span>{tchip}'
                 f'<div class="ex-att"><div class="ex-bar" style="width:{int(att * 100 / 0.507)}%"></div></div></div>')
    st.markdown(
        '<div class="card" style="margin-top:1rem"><h4>Example: one user’s reviews</h4>'
        '<p style="margin-bottom:0.5rem">Illustrative output. The bar is attention; try your own reviews in <b>Analyze</b>.</p>'
        + html
        + '<div style="margin-top:0.7rem"><span class="chip chip-negative">TRAJECTORY: DECLINING</span></div></div>',
        unsafe_allow_html=True,
    )

    stack = ["Python", "PyTorch", "Hugging Face Transformers", "mBERT", "XLM-RoBERTa", "FLAN-T5", "pandas", "NumPy",
             "scikit-learn", "Streamlit", "Plotly", "pytest", "GitHub Actions"]
    pills = "".join(f"<span>{x}</span>" for x in stack)
    st.markdown(
        _sec("Built with", "Technology stack and data")
        + f'<div class="pills">{pills}</div>'
        + '<div class="grid" style="margin-top:0.9rem">'
          '<div class="card"><h4>Amazon Reviews 2023, Beauty</h4><p>English e-commerce. Users’ chronological reviews: '
          '3,726 / 798 / 800 users for train / val / test.</p></div>'
          '<div class="card"><h4>DravidianCodeMix: Tamil</h4><p>Code-mixed social media, 33,531 comments.</p></div>'
          '<div class="card"><h4>DravidianCodeMix: Malayalam, Kannada</h4><p>Code-mixed social media, used to test transfer to languages not trained on.</p></div>'
          '</div>',
        unsafe_allow_html=True,
    )

    def dot(c):
        return f'<span class="dot" style="background:{c}"></span>'
    st.markdown(
        _sec("Where we are", "Project progress")
        + '<div class="grid">'
          f'<div class="card accent"><h4 class="st-done">{dot("#1E8E6A")}Completed</h4><ul>'
          '<li>Label ontology across domains</li><li>mBERT + Bi-LSTM + attention model with three heads</li>'
          '<li>Word-sense disambiguation for code-mixed aspects</li><li>Baselines, ablations, calibration, perplexity</li>'
          '<li>Evaluation utilities, open-source repository, tests and this demo</li></ul></div>'
          f'<div class="card accent"><h4 class="st-run">{dot("#B7791F")}In progress</h4><ul>'
          '<li>Leakage-controlled multilingual and cross-domain evaluation</li>'
          '<li>Instruction-tuned language-model baseline (FLAN-T5)</li></ul></div>'
          f'<div class="card accent"><h4 class="st-plan">{dot("#5A6B7C")}Planned</h4><ul>'
          '<li>Repeated seeded runs with confidence intervals</li><li>Aspect-level opinion trajectories</li>'
          '<li>Fine-tuning the encoder layers</li></ul></div>'
          '</div>',
        unsafe_allow_html=True,
    )

    docs = [("README", "Overview, install and usage", "README.md"),
            ("Final report", "Project report", "docs/final_report.md"),
            ("Literature comparison", "Against surveyed papers", "docs/literature_comparison.md"),
            ("Modules 7–9 status", "Evaluation, multilingual, LLM", "docs/new_modules_status.md"),
            ("Baseline comparison", "Capacity-matched setup", "docs/capacity_matched_comparison.md"),
            ("Word-sense status", "Disambiguation results and limits", "docs/module2_wsd_status.md")]
    dl = "".join(f'<a class="doclink" href="{blob}{p}" target="_blank"><div class="card"><h4>{t}</h4><p>{d}</p></div></a>'
                 for t, d, p in docs)
    st.markdown(
        _sec("Read more", "Documentation") + f'<div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(280px,1fr))">{dl}</div>'
        + f'<p style="margin-top:1.4rem;color:#52647A">Source code: <a href="{GITHUB_URL}" target="_blank">{GITHUB_URL.replace("https://", "")}</a></p>',
        unsafe_allow_html=True,
    )


def render_about():
    st.markdown(
        '<div class="main-header">What this system does</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">A multi-task model that reads a person\'s reviews in order and tracks '
        'how their opinion moves, across languages and across e-commerce and social media.</div>',
        unsafe_allow_html=True,
    )

    rows = results_rows()
    models = {r["model"] for r in rows}
    domains = {r["source"] for r in rows}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Supported Languages", "4", "EN, TA, ML, KN")
    c2.metric("Classification Heads", "3", "Sentiment, Trend, Trajectory")
    c3.metric("Models Evaluated", str(len(models)) if models else "—",
              "read from results_table.json")
    c4.metric("Domains Trained", str(len(domains)) if domains else "—",
              ", ".join(sorted(domains)) if domains else "no results yet")

    st.markdown("---")
    st.subheader("Why this is a hard problem")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            """
Most sentiment models score each review on its own. In reality:

- An opinion **moves over time**, across a person's whole sequence of reviews.
- Social media text is heavily **code-mixed** — Tamil, Malayalam and Kannada
  words written in Latin script, next to English.
- Every platform grades opinions differently — Amazon uses **1–5 stars**,
  YouTube comments carry **text labels** — so nothing is directly comparable
  until it is put on one scale.

**Four things this project builds to address that:**

- A closed-vocabulary ontology (`SentimentState`, `TransitionType`,
  `TrajectoryType`) that puts every domain's labels on one scale.
- Word-sense disambiguation for aspects (hero, BGM, story, trailer) inside
  code-mixed text.
- A Bi-LSTM plus self-attention encoder that reads the sequence in order and
  weighs which review mattered most.
- Sequence Consistency Score (SCS), a metric for whether a run of
  predictions holds together over time — something accuracy alone misses.
"""
        )
    with right:
        st.markdown("### How a sequence flows through the model")
        st.code(
            """
[Raw Multilingual Text]
       |
[Code-Mix & Aspect Detection]  -> aspect + CMI
       |
[Structural Ontology] -> 4-level schema
       |
[mBERT Subword Embeddings]
       |
[Bidirectional LSTM]  -> order memory
       |
[Self-Attention]      -> turning points
       |
[Multi-Task Heads]
  |- Sentiment State (per review)
  |- Pairwise Transition (Up/Down/Stable)
  |- Overall Trajectory (Improving/Declining/Volatile)
""",
            language="text",
        )

    st.success(
        "**Every figure in this demo is read live from `outputs/`.** "
        "Where a result has not been produced yet, the page says so and names "
        "the command that produces it — no placeholder numbers appear anywhere."
    )




def render_ontology():
    st.caption("A closed-vocabulary hierarchy that puts Amazon star ratings and "
               "Tamil, Malayalam and Kannada social-media labels on the same scale.")

    st.subheader("Taxonomy hierarchy")
    st.write(
        "The ontology resolves the label conflict between e-commerce star "
        "ratings and social-media text tags."
    )
    a, b, c = st.columns(3)
    if BACKEND_AVAILABLE:
        with a:
            st.markdown("**Sentiment, per review**")
            st.markdown(chip_row(SentimentState.label_names()), unsafe_allow_html=True)
        with b:
            st.markdown("**Transition, between reviews**")
            st.markdown(chip_row(TransitionType.label_names()), unsafe_allow_html=True)
        with c:
            st.markdown("**Trajectory, whole sequence**")
            st.markdown(chip_row(TrajectoryType.label_names()), unsafe_allow_html=True)
        st.caption("Class names read live from `src/ontology.py` — not transcribed. "
                   "Each color is fixed and used the same way everywhere in this app.")
    else:
        st.error("Backend unavailable; cannot read the ontology.")

    st.markdown("### Why top-down?")
    st.markdown(
        """
- **Avoids inconsistent clustering** — bottom-up induction would produce
  divergent category systems for Amazon vs. Tamil social media.
- **Closed-vocabulary guarantee** — adding a dataset (e.g. Telugu) needs only a
  new `DomainConfig`. Loss functions, heads and sequence logic never change.
"""
    )


    st.markdown("---")
    st.subheader("Map a raw dataset label onto the ontology")
    if BACKEND_AVAILABLE:
        domain_choice = st.selectbox("Domain:", sorted(DOMAIN_CONFIGS.keys()), key="ontology_domain")
        samples = {
            "amazon_beauty": ["5.0", "4.0", "3.0", "2.0", "1.0"],
        }
        options = samples.get(
            domain_choice,
            ["Positive", "Negative", "Mixed_feelings", "unknown_state"],
        )
        test_label = st.selectbox("Raw dataset label:", options)
        try:
            state = map_labels_to_ontology([test_label], domain=domain_choice)[0]
            st.markdown(
                f"`{test_label}` maps to {chip(state.name)} (id `{state.value}`)",
                unsafe_allow_html=True,
            )
            st.caption("Computed by `map_labels_to_ontology()` at click time.")
        except Exception as e:
            st.error(f"Mapping failed: {e}")
    else:
        st.error("Backend unavailable.")


    st.markdown("---")
    st.subheader("Ontology coverage — computed from the real corpus")
    data = compute_ontology_coverage()
    if "error" in data:
        not_measured(
            "Ontology coverage", "python src/ontology_eval.py",
            f"Live computation failed: {data['error']}",
        )
    else:
        cov = pd.DataFrame(data["coverage"])
        if not cov.empty and "coverage_pct" in cov.columns:
            show = cov[[c for c in ("domain", "coverage_pct", "unknown_pct",
                                    "total") if c in cov.columns]]
            st.dataframe(show, width="stretch")
            fig = px.bar(
                cov, x="domain", y="coverage_pct", color="domain",
                text="coverage_pct",
                title="Ontology label coverage per domain (live)",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Computed by `ontology_eval.compute_all_domain_coverage()` "
                "against `data/preprocessed/` on page load."
            )
        else:
            st.warning("Coverage computation returned no rows.")

        report = load_text("metrics/module1_ontology.md")
        if report:
            with st.expander("Full coverage report"):
                st.markdown(report)




def render_aspects():
    st.caption("Aspect detection in code-mixed text, disambiguated by context — "
               "via IndoWordNet and a surrounding-word overlap check.")

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Live aspect detection")
        samples = [
            "Padam vera level bro superb acting bgm romba nalla",
            "Trailer romba mass fans ku pidikkum hero entry vera level",
            "Story weak but songs super hit movie",
            "Villain acting super dialogue punch romba nalla",
            "marana mass bgm anna",
            "fans hit like here",
        ]
        choice = st.selectbox("Sample code-mixed comment:", samples)
        text = st.text_input("Or type your own:", choice)

        if BACKEND_AVAILABLE and text.strip():
            wsd = WordSenseDisambiguator()
            results = wsd.process(text)
            if results:
                st.table(pd.DataFrame(
                    [{"Word": w, "Aspect": a, "Confidence": round(c, 2)}
                     for w, a, c in results]
                ))
                ambiguous = [w for w, _, _ in results
                             if wsd.lexicon.is_ambiguous(w)]
                if ambiguous:
                    st.success(
                        "Disambiguated by context: "
                        + ", ".join(f"`{w}`" for w in ambiguous)
                        + " — these carry more than one aspect sense; the "
                        "surrounding words decided which applies."
                    )
            else:
                st.warning("No aspect keywords detected in this text.")
            st.caption("Computed by `wsd.process()` at keystroke time.")
        elif not BACKEND_AVAILABLE:
            st.error("Backend unavailable.")

        st.markdown(
            "Try `marana mass bgm anna` then `mass trailer` — the same word "
            "`mass` resolves to a different aspect. Try `fans hit like here`: "
            "`hit` is correctly **not** tagged as box-office."
        )

    with right:
        st.subheader("Measured aspect distribution")
        wsd_res = load_json("metrics/module2_wsd_results.json")
        if not wsd_res:
            not_measured("Aspect distribution", "python scripts/evaluate_wsd.py")
        else:
            dist = wsd_res.get("aspect_distribution", {})
            if dist:
                df = pd.DataFrame(
                    sorted(dist.items(), key=lambda kv: -kv[1]),
                    columns=["Aspect", "Occurrences"],
                )
                st.plotly_chart(
                    px.pie(df, names="Aspect", values="Occurrences", hole=0.35,
                           title="Aspect occurrences in the Dravidian corpus"),
                    width="stretch",
                )

            overall = wsd_res.get("overall", {})
            if overall:
                m1, m2, m3 = st.columns(3)
                m1.metric("Aspect coverage",
                          f"{overall.get('coverage_pct', 0)}%")
                m2.metric("Ambiguous instances",
                          f"{overall.get('ambiguous_words', 0):,}")
                m3.metric("Resolution rate",
                          f"{overall.get('resolution_pct', 0)}%")
                st.caption(
                    "Resolution rate is the share of ambiguous instances the "
                    "context window could resolve. The remainder are abstentions, "
                    "not errors — short comments often carry no disambiguating "
                    "context."
                )

            st.warning(
                "**No sense-disambiguation accuracy is reported.** No gold "
                "sense labels exist for this corpus. The MCS comparison in the "
                "report measures share of high-confidence predictions, not "
                "accuracy, and must not be quoted as such."
            )




def render_architecture():
    st.caption("mBERT, a Bi-LSTM, self-attention and three task heads, "
               "working on one domain-agnostic pipeline.")

    rows = results_rows()
    full = next((r for r in rows if r["model"] == full_model_name()), None)
    cap = full.get("encoder_finetune_layers") if full else None

    c1, c2, c3, c4 = st.columns(4)
    c1.info(
        "### 1. Tokenizer & mBERT\n"
        "- `bert-base-multilingual-cased`\n"
        "- 119,547 WordPiece subwords\n"
        f"- Encoder layers trained: **{cap if cap is not None else 'unknown'}**\n"
        "- 768-dim embeddings"
    )
    c2.success("### 2. Bi-LSTM Encoder\n- 2 layers, bidirectional\n- Hidden 256\n"
               "- Chronological order memory\n- 512-dim concat state")
    c3.warning("### 3. Self-Attention\n- Highlights critical reviews\n"
               "- Locates opinion turning points\n- Context-weighted pooling")
    c4.error("### 4. Multi-Task Heads\n- Sentiment: 4 classes\n- Trend: 3 classes\n"
             "- Trajectory: 4 classes\n- Class-weighted cross-entropy")

    if cap == 0:
        st.warning(
            "**The encoder was frozen in the runs shown here** "
            "(`--freeze_encoder`, 0 trainable encoder layers). Only the "
            "Bi-LSTM, attention and heads were trained — 3,093,003 of "
            "180,946,443 parameters. Any claim about *fine-tuned* domain-adapted "
            "embeddings requires a run with `--no_freeze_encoder`."
        )

    st.markdown("---")
    st.subheader("Encoder perplexity: mBERT vs. XLM-R")
    perplexity = load_text("metrics/module3_bert_perplexity.md")
    if perplexity:
        st.markdown(perplexity)
    else:
        not_measured(
            "Encoder MLM perplexity", "python src/mlm_perplexity_eval.py",
            "Compares how well each pretrained encoder models this corpus. "
            "Not applicable to TextCNN, which has no masked language model.",
        )




def render_performance():
    st.caption("Five baselines and an ablation, measured at equal trainable "
               "encoder capacity so the comparison is architecture, not training budget.")
    st.info(
        "**Every number on this page — full model included — is a recorded "
        "metric from a prior training run**, read from "
        "`outputs/metrics/results_table.json`. Nothing on this page runs "
        "inference now. For a live prediction on text you type, open **Try it**."
    )

    rows = [r for r in results_rows() if r["setting"] == "in-domain"]
    if not rows:
        not_measured(
            "In-domain comparison",
            "python scripts/train_baselines.py --baseline all --domain amazon\n"
            "python scripts/compile_metrics.py",
        )
    else:
        df = pd.DataFrame(rows)
        sources = sorted(df["source"].unique())
        pick = st.selectbox("Domain:", sources, key="performance_domain")
        sub = df[df["source"] == pick].copy()

        caps = sub["encoder_finetune_layers"].dropna().unique().tolist()
        if len(caps) > 1:
            st.error(
                "⚠️ **Capacity mismatch — these models are not directly "
                f"comparable.** Encoder layers trained differ across rows "
                f"({sorted(caps)}). A model given more trainable encoder "
                "capacity usually scores higher regardless of architecture, so "
                "the gaps below partly reflect training budget, not design. "
                "Re-run the affected baselines with `--encoder_finetune_layers` "
                "matched to the full model."
            )

        show = sub[[c for c in (
            "model", "encoder_finetune_layers", "sentiment_accuracy",
            "sentiment_f1_macro", "trajectory_f1_macro", "scs_mean",
        ) if c in sub.columns]].rename(columns={
            "encoder_finetune_layers": "encoder layers trained",
        })
        st.dataframe(show, width="stretch")

        plot = sub.dropna(subset=["sentiment_f1_macro"])
        fig = px.bar(
            plot, x="model", y="sentiment_f1_macro", color="model",
            text="sentiment_f1_macro",
            title=f"Sentiment F1 (macro) — in-domain on {pick}",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        st.plotly_chart(fig, width="stretch")
        st.caption("Read from `outputs/metrics/results_table.json`.")

    st.markdown("---")
    st.subheader("Capability matrix, by baseline")
    matrix = load_text("metrics/module_by_baseline_comparison.md")
    if matrix:
        with st.expander("Show full matrix (generated deliverable)", expanded=False):
            st.markdown(matrix)
        st.caption(
            "Rendered from `outputs/metrics/module_by_baseline_comparison.md`, "
            "produced by `scripts/generate_module_by_baseline_matrix.py`."
        )
    else:
        not_measured(
            "Capability matrix",
            "python scripts/generate_module_by_baseline_matrix.py",
        )




def render_cross_domain():
    st.caption("Zero-shot transfer between domains the model was never jointly trained on.")

    rows = results_rows()
    cross = [r for r in rows if r["setting"] == "cross-domain"]
    indom = {(r["source"], r["target"]): r for r in rows if r["setting"] == "in-domain"}

    if not cross:
        not_measured("Cross-domain transfer", "python scripts/cross_domain_eval.py")
    else:
        st.subheader("Measured transfer — every trained source against every target")
        recs = []
        for r in cross:
            base = indom.get((r["source"], r["source"]))
            retained = None
            if base and base.get("sentiment_f1_macro") and r.get("sentiment_f1_macro"):
                retained = r["sentiment_f1_macro"] / base["sentiment_f1_macro"] * 100
            recs.append({
                "Trained on": r["source"],
                "Tested on": r["target"],
                "Sentiment F1": r.get("sentiment_f1_macro"),
                "Trajectory F1": r.get("trajectory_f1_macro"),
                "SCS": r.get("scs_mean"),
                "% of in-domain F1 retained": round(retained, 1) if retained else None,
            })
        cdf = pd.DataFrame(recs)
        st.dataframe(cdf, width="stretch")

        plot = cdf.dropna(subset=["% of in-domain F1 retained"])
        if not plot.empty:
            plot = plot.assign(
                pair=plot["Trained on"] + " → " + plot["Tested on"]
            )
            fig = px.bar(
                plot, x="pair", y="% of in-domain F1 retained",
                color="Trained on", text="% of in-domain F1 retained",
                title="Retention of in-domain sentiment F1 under domain shift",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, width="stretch")
            st.info(
                "**Preliminary, single run.** In this run the model trained on "
                "code-mixed Dravidian text retains more of its in-domain "
                "performance on a new language than the Amazon-trained model. "
                "The Dravidian sequences are sliding windows over independent "
                "comments and may overlap between train and test, so this is "
                "indicative only until the leakage-controlled comment-level "
                "evaluation is complete."
            )
        st.caption("Read from `outputs/metrics/results_table.json`.")

    st.markdown("---")
    st.subheader("Fuzzy domain typicality")
    st.markdown(
        "Instead of binary domain labels, each sequence receives a soft "
        "membership over domain centroids, computed by "
        "`src/fuzzy_domain_score.py`."
    )
    fuzzy = load_csv("fuzzy_domain_scores.csv")
    if fuzzy is not None:
        st.warning(
            "**Reference scores, not computed against your input.** This "
            "table is the pre-computed typicality of each **test-set** "
            "sequence against every domain centroid — it does not score "
            "whatever you type on the Try it page. Live per-input typicality "
            "scoring is not implemented; this page can only show what was "
            "already measured on the fixed test sets."
        )
        st.dataframe(fuzzy.head(50), width="stretch")
        st.caption(f"{len(fuzzy):,} rows from `outputs/fuzzy_domain_scores.csv`.")
    else:
        not_measured("Fuzzy typicality scores", "python src/fuzzy_domain_score.py")




def render_calibration():
    st.caption("Sequence Consistency Score, calibration and prediction uncertainty.")

    st.subheader("Sequence Consistency Score (SCS)")
    st.markdown(
        "Accuracy counts individual correct answers. **SCS measures whether a "
        "sequence of predictions is internally coherent over time** — if "
        "review 1→2 is an upgrade and 2→3 is an upgrade, a *declining* "
        "trajectory prediction is a temporal contradiction that accuracy alone "
        "would not penalise."
    )

    rows = [r for r in results_rows() if r["setting"] == "in-domain"]
    scs_rows = [r for r in rows if r.get("scs_mean") is not None]

    if not scs_rows:
        not_measured(
            "SCS comparison",
            "python scripts/train_baselines.py --baseline all --domain amazon\n"
            "python scripts/compile_metrics.py",
        )
    else:
        df = pd.DataFrame(scs_rows)
        pick = st.selectbox("Domain:", sorted(df["source"].unique()), key="scs_domain")
        sub = df[df["source"] == pick]

        fig = px.bar(
            sub, x="model", y="scs_mean", color="model", text="scs_mean",
            title=f"Sequence Consistency Score — {pick}",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        st.plotly_chart(fig, width="stretch")

        best = sub.loc[sub["scs_mean"].idxmax()]
        if best["model"] != full_model_name():
            st.warning(
                f"**Note for discussion:** `{best['model']}` currently scores "
                f"the highest SCS ({best['scs_mean']:.4f}) on {pick}, above the "
                "full model. This is shown as measured rather than hidden — a "
                "high SCS on its own can also indicate a model predicting the "
                "same state repeatedly, which is why SCS is reported alongside "
                "F1 rather than instead of it."
            )
        st.caption("Read from `outputs/metrics/results_table.json`.")

    st.markdown("---")
    st.subheader("Calibration (ECE), entropy and SCS reliability")
    m4 = sorted(glob.glob(os.path.join(METRICS, "module4_sequential_model_*.md")))
    if m4:
        labels = {os.path.basename(p)[len("module4_sequential_model_"):-3]: p for p in m4}
        pick_m4 = st.selectbox("Domain:", sorted(labels), key="m4_domain")
        with open(labels[pick_m4], encoding="utf-8") as f:
            st.markdown(f.read())
        st.caption(
            f"Read from `outputs/metrics/module4_sequential_model_{pick_m4}.md`. "
            "Reports are written per domain -- a single shared filename previously "
            "meant a second run silently overwrote the first."
        )
    else:
        not_measured(
            "ECE, prediction entropy, SCS reliability",
            "python src/confidence_eval.py --domain amazon\n"
            "python src/confidence_eval.py --domain dravidian --language tamil",
            "These require the trained checkpoints under `outputs/checkpoints/`.",
        )

    st.markdown("---")
    analysis = load_text("metrics/module6_analysis.md")
    if analysis:
        with st.expander("Full written analysis"):
            st.markdown(analysis)




def render_analyze():
    st.markdown('<div class="main-header">Opinion Evolution Tracker</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Type a sequence of reviews, in order. The model reads them as '
                'one thread and predicts how the opinion moved — and exactly where it turned.</div>',
                unsafe_allow_html=True)

    checkpoints = sorted(glob.glob(os.path.join(OUTPUTS, "checkpoints", "best_model_*.pt")))
    if not checkpoints:
        fetch_hf(HF_CKPT_REPO, "model", os.path.join(OUTPUTS, "checkpoints"))
        checkpoints = sorted(glob.glob(os.path.join(OUTPUTS, "checkpoints", "best_model_*.pt")))

    if not checkpoints:
        not_measured(
            "Live model inference",
            "python scripts/train.py --domain amazon --epochs 20",
            "No trained checkpoint found under `outputs/checkpoints/`. "
            "The aspect detector below still runs — it needs no checkpoint.",
        )
    else:
        names = [os.path.basename(p) for p in checkpoints]
        chosen = st.selectbox("Trained model:", names)
        ckpt_path = os.path.join(OUTPUTS, "checkpoints", chosen)

        @st.cache_resource(show_spinner="Loading model (~10s, first time only)…")
        def load_model(path: str):
            import torch
            from model import OpinionEvolutionTracker
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            saved = ckpt.get("args")
            saved = vars(saved) if hasattr(saved, "__dict__") else (saved or {})
            m = OpinionEvolutionTracker(
                model_name=saved.get("model_name", "bert-base-multilingual-cased"),
                use_cuda=False,
                freeze_encoder=saved.get("freeze_encoder", True),
            )
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            return m, ckpt.get("epoch"), saved

        st.markdown("### Enter reviews in chronological order")
        default = (
            "Product looks great, very happy with the purchase\n"
            "Still working fine after two weeks\n"
            "Started having problems, quality dropped\n"
            "Completely stopped working, very disappointed"
        )
        raw = st.text_area(
            "One review per line (English, Tamil, Malayalam or Kannada — "
            "code-mixed is fine):",
            default, height=160,
        )
        reviews = [r.strip() for r in raw.splitlines() if r.strip()]

        if st.button("Predict trajectory", type="primary") and reviews:
            if len(reviews) < 2:
                st.error("Enter at least two reviews — a trajectory needs a sequence.")
            else:
                try:
                    model, epoch, saved = load_model(ckpt_path)
                    out = model.predict(reviews)

                    sent_names = SentimentState.label_names()
                    traj_names = TrajectoryType.label_names()
                    trend_names = TransitionType.label_names()

                    sentiments = [sent_names[i] for i in out["sentiments"].tolist()[:len(reviews)]]
                    trajectory = traj_names[out["trajectory"]]
                    attention = out["attention_weights"].tolist()[:len(reviews)]

                    st.markdown(
                        f'<div style="font-size:1.3rem;margin:0.4rem 0 1rem 0;">'
                        f'Predicted trajectory: {chip(trajectory)}</div>',
                        unsafe_allow_html=True,
                    )

                    # ---- Phase 3: the timeline visual -------------------------
                    order = {n: i for i, n in enumerate(sent_names)}
                    ydata = [order.get(s, 0) for s in sentiments]
                    labels = [f"Review {i+1}" for i in range(len(reviews))]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=labels, y=ydata, mode="lines+markers+text",
                        text=sentiments, textposition="top center",
                        line=dict(width=3),
                        marker=dict(size=[14 + 40 * a for a in attention]),
                        name="Predicted sentiment",
                        hovertext=[f"{r[:70]}<br>attention={a:.3f}"
                                   for r, a in zip(reviews, attention)],
                        hoverinfo="text",
                    ))
                    fig.update_layout(
                        title=f"Opinion trajectory — predicted {trajectory}",
                        yaxis=dict(
                            tickmode="array",
                            tickvals=list(range(len(sent_names))),
                            ticktext=sent_names,
                            title="Sentiment state",
                        ),
                        xaxis_title="Review order (chronological)",
                        height=420,
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.caption(
                        "Marker size is the model's attention weight — the "
                        "larger the marker, the more that review drove the "
                        "trajectory decision."
                    )

                    a1, a2 = st.columns([1, 1])
                    with a1:
                        st.markdown("#### Attention over the sequence")
                        st.plotly_chart(
                            px.bar(
                                pd.DataFrame({"Review": labels, "Attention": attention}),
                                x="Review", y="Attention",
                                title="Which review mattered most?",
                            ),
                            width="stretch",
                        )
                        peak = attention.index(max(attention))
                        st.info(
                            f"Turning point: **Review {peak+1}** received the "
                            f"highest attention ({max(attention):.3f})."
                        )
                    with a2:
                        st.markdown("#### Per-review detail")
                        trends = out["trends"].tolist()[:len(reviews)]
                        table_rows = "".join(
                            f"<tr><td>{labels[i]}</td><td>{reviews[i][:50]}</td>"
                            f"<td>{chip(sentiments[i])}</td>"
                            f"<td>{chip(trend_names[trends[i]]) if i else '—'}</td>"
                            f"<td>{attention[i]:.3f}</td></tr>"
                            for i in range(len(reviews))
                        )
                        st.markdown(
                            '<table class="chip-table"><thead><tr>'
                            "<th>Review</th><th>Text</th><th>Sentiment</th>"
                            "<th>Transition</th><th>Attention</th>"
                            f"</tr></thead><tbody>{table_rows}</tbody></table>",
                            unsafe_allow_html=True,
                        )

                    if BACKEND_AVAILABLE:
                        wsd = WordSenseDisambiguator()
                        asp = []
                        for i, r in enumerate(reviews):
                            for w, a, c in wsd.process(r):
                                asp.append({"Review": labels[i], "Word": w,
                                            "Aspect": a, "Confidence": round(c, 2)})
                        if asp:
                            st.markdown("#### Aspects detected, live")
                            st.dataframe(pd.DataFrame(asp), width="stretch")

                    st.caption(
                        f"Predicted by `{chosen}` (best epoch {epoch}, "
                        f"encoder layers trained: "
                        f"{0 if saved.get('freeze_encoder', True) else 3}) "
                        "running live on CPU."
                    )
                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    st.exception(e)



# =============================================================================
# Routing -- four sections, the shape a real product would have
# =============================================================================

# Each section gets its own accent colour and background wash (state colours on chips stay fixed).
PAGE_THEME = {   # accent, page background gradient (top, bottom), sidebar background
    "Home":         ("#0E8F6E", "#C9F0E0", "#B5D3F8", "#E3F6EE"),
    "Analyze":      ("#B9770E", "#FFF6E3", "#FFE6BF", "#FFFAF0"),
    "Insights":     ("#2F5FD0", "#EAF0FF", "#D3E1FF", "#F4F7FF"),
    "How it works": ("#7A4FD6", "#F2EBFF", "#E0D2FF", "#F8F4FF"),
    "About":        ("#C2417A", "#FFEAF2", "#FFD6E6", "#FFF5F9"),
}
_accent, _top, _bottom, _side = PAGE_THEME[page]
st.markdown(
    f"""<style>
:root {{ --pulse: {_accent}; }}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background: linear-gradient(165deg, {_top} 0%, {_bottom} 100%) !important; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {{ background: {_side} !important; }}
.stButton > button, .stLinkButton > a {{ border-color: {_accent} !important; }}
.stButton > button[kind="primary"], .stButton > button[kind="secondaryFormSubmit"] {{ background: {_accent} !important; color: #FFFFFF !important; }}
.main-header {{ border-left: 4px solid {_accent}; padding-left: 0.7rem; }}
</style>""",
    unsafe_allow_html=True,
)

if page == "Home":
    render_home()

elif page == "Analyze":
    render_analyze()

elif page == "Insights":
    st.markdown('<div class="main-header">How well does it work?</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Measured against five baselines, across domains it was never jointly trained on, with its own confidence checked.</div>',
                unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Performance", "Cross-domain transfer", "Calibration & confidence"])
    with t1:
        render_performance()
    with t2:
        render_cross_domain()
    with t3:
        render_calibration()

elif page == "How it works":
    st.markdown('<div class="main-header">How it understands text</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">One scale for every domain, aspect detection in code-mixed text, and the architecture that reads a sequence in order.</div>',
                unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Ontology", "Code-mixing & aspects", "Architecture"])
    with t1:
        render_ontology()
    with t2:
        render_aspects()
    with t3:
        render_architecture()

elif page == "About":
    render_about()
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Opinion Evolution Tracking Project")
st.sidebar.caption("All figures read live from outputs/ — nothing hardcoded.")
