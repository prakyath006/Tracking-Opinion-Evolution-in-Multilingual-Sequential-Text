# Demo decision — `web_demo/app.py` (Streamlit) is the primary panel demo

**`web_demo_gradio/`, the alternative this decision was meant to weigh against,
does not exist.** Checked exhaustively before writing this: the working tree,
`git log --all -- web_demo_gradio/` (zero commits on any branch ever touched
it), `origin/main`, `origin/ontology-integration` (the only other remote
branch), the local `backup-before-trailer-strip` branch, and a filesystem
search of the user's Downloads and home directories. Confirmed with the user
directly, who pointed at `web_demo/app.py` as the intended target. There is no
comparison to make — `web_demo/app.py` is primary because it is the only demo
that exists.

## What it is

A Streamlit dashboard (`streamlit run web_demo/app.py`) with 8 pages: an
overview, one page per module (1 through 6), and a **Live Interactive
Playground** that loads a real trained checkpoint
(`outputs/checkpoints/best_model_{amazon,dravidian_tamil}.pt`) and runs actual
forward-pass inference on whatever review sequence the panel types in —
sentiment per review, transition, trajectory, and attention-weighted turning
point, rendered as a live chart. This satisfies "panel types text, model runs
inference, output appears in seconds," verified in this session (see
Step 6 results below).

## Verification performed in this session (not assumed)

- Both checkpoints load into `OpinionEvolutionTracker` with 0 missing / 0
  unexpected state_dict keys.
- Server starts and answers HTTP 200 (`streamlit run ... && curl`).
- All 8 pages driven through Streamlit's own `AppTest` harness (not a
  screenshot): each set as the active sidebar page, each asserted to render
  with no exception and non-zero output elements. 8/8 passed.
- The Playground's "Predict trajectory" button was actually clicked with a
  real 4-review English sequence and a real 3-review Tamil sequence, through
  both checkpoints. All four produced a correct, non-degenerate prediction
  (e.g. the English declining-quality sequence predicted **DECLINING**, not a
  constant or None).
- Latency measured, not estimated: cold load (first click in a fresh process)
  14.65s (Amazon) / 5.54s (Tamil); warm (cached model, second click onward)
  0.61s (Amazon) / 0.55s (Tamil). Well under the 10-second bar.

## Two honesty fixes made during this session

Both pages already read every number from `outputs/` rather than hardcoding
it (see the file's own "TRUTHFULNESS CONTRACT" docstring), but two labels
were not explicit enough for a panel setting and were tightened:

1. **Module 4 (baselines) page** — added a banner stating plainly that every
   number on that page, full model included, is a recorded metric from a
   prior training run, not a live prediction, and pointing to the Playground
   for the live path.
2. **Module 5 (cross-domain) page, fuzzy typicality table** — added a banner
   stating these are pre-computed reference scores for the fixed test sets,
   not scored against whatever the panel types. There is no live per-input
   typicality scoring implemented.

## What is honestly not "live" in this demo

Only the Playground page runs inference on arbitrary input. Every other page
is a dashboard over pre-computed results. This is not a defect — the task
only required *one* genuinely live surface, and the Playground is exactly
that — but the demo does not pretend the other seven pages are live, and
after the fixes above it says so explicitly rather than leaving it
ambiguous.
