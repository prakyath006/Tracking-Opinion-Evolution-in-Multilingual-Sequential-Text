# Panel Demo — `web_demo/app.py`

Live-inference Streamlit dashboard for the Opinion Evolution Tracker. See
[`docs/demo_decision.md`](../docs/demo_decision.md) for why this is the
primary demo and what was verified before calling it done.

## A. Local run (verified in this session)

```bash
streamlit run web_demo/app.py
```

Opens on `http://localhost:8501` by default. Verified in this session on
`--server.port 8910` (any free port works) with `--server.headless true`;
starting it normally without those flags opens your browser automatically.

Requires: the two checkpoints in `outputs/checkpoints/` and the preprocessed
corpus in `data/preprocessed/` (see the Panel Day Runbook below for exactly
what to check).

**Silent health check**, safe to run without disturbing anything on screen:

```bash
curl -s localhost:8501 > /dev/null && echo OK || echo DEAD
```

## B. Shareable URL

Streamlit has no Gradio-style built-in `--share` flag. Two real options,
**neither deployed or tunnel-tested in this session** — documented, not
verified end-to-end, per the same constraint the task set for HF Spaces:

**Option B1 — Streamlit Community Cloud (recommended for anything beyond a
single sitting).** Free, persists beyond 72 hours, needs a GitHub connection.
1. Push this repo to GitHub (already done).
2. At [share.streamlit.io](https://share.streamlit.io), "New app" → point at
   this repo, branch `main`, file path `web_demo/app.py`.
3. **The two checkpoints (714 MB each) will not fit in the repo or the free
   tier's typical storage.** Either add a startup step that downloads them
   from the Kaggle dataset (`opinion-tracker-checkpoints`) before the app
   imports `torch`, or skip Community Cloud and use B2 for panel day, since
   panel day only needs a same-day link.

**Option B2 — a local tunnel, same-day only.** Run the app locally as in (A),
then in a second terminal:
```bash
# either tool, pick one that's installed
npx localtunnel --port 8501
# or
cloudflared tunnel --url http://localhost:8501
```
Gives a temporary public URL pointing at your own running process — closest
equivalent to Gradio's `--share`. Your laptop must stay on and connected for
the link to keep working, which is the same constraint Gradio's `--share`
has.

**For most panel settings, plain local (A) on the presenter's laptop is the
right choice** — no network dependency, no tunnel to fail mid-demo.

## Panel Day Runbook

- **T-30 min** — `git pull`. Verify `outputs/checkpoints/best_model_amazon.pt`
  and `outputs/checkpoints/best_model_dravidian_tamil.pt` both exist (`ls -la
  outputs/checkpoints/`, should be ~715 MB each). Verify `data/preprocessed/`
  is present (needed for Modules 1, 2, 5's live/reference computations; the
  Playground itself only needs the checkpoints).
- **T-15 min** — `streamlit run web_demo/app.py`. Open
  `http://localhost:8501`.
- **T-10 min** — Click through all 8 sidebar pages once. On the Playground
  page, run one preset per domain (an English Amazon-style sequence and a
  Tamil sequence) to warm `st.cache_resource` — this is what takes the first
  click from ~15s down to ~0.6s for every click after.
- **T-5 min** — Click one more preset and time it with a watch or your phone.
  It should be well under a second; if it's not, the cache did not warm
  correctly — re-click the same model once more before going live.
- **If the live app dies mid-demo:**
  - First, check the terminal for a traceback and try `curl -s localhost:8501
    > /dev/null && echo OK || echo DEAD` to confirm it's actually down and not
    just a slow page.
  - If genuinely down and no time to debug: every other page (Modules 1–6,
    Overview) is a static dashboard reading pre-generated files under
    `outputs/` — restart with `streamlit run web_demo/app.py` again; the
    static pages come back even if the Playground model fails to reload.
  - If Streamlit itself won't start at all: there is no second demo to fall
    back to in this repo (see `docs/demo_decision.md` — `web_demo_gradio/`
    does not exist). The fallback is the static reports directly:
    `outputs/metrics/results_table.md`, `outputs/metrics/module6_analysis.md`,
    and the other `outputs/metrics/*.md` files, which contain the same
    numbers the dashboard renders.
- **Silent health check for the presenter to run without narrating it:**
  `curl -s localhost:8501 > /dev/null && echo OK || echo DEAD`

## What was actually verified in this session, not assumed

See the full report in the commit message and chat — summarized:

- Both checkpoints load cleanly: 0 missing / 0 unexpected state_dict keys.
- Environment: Python 3.14.6, torch 2.13.0+cpu, transformers 5.13.1,
  streamlit 1.63.0.
- All 8 pages driven through Streamlit's `AppTest` harness, each asserted to
  render with no exception and non-empty output: **8/8 passed**.
- Playground's predict button actually clicked with real text through both
  checkpoints; both produced correct, non-degenerate predictions.
- Latency, measured not estimated: cold 14.65s (Amazon) / 5.54s (Tamil),
  warm 0.61s (Amazon) / 0.55s (Tamil).

## Known issues to be aware of on panel day

- **No live per-input domain-typicality scoring.** The Module 5 fuzzy table
  shows pre-computed scores for the fixed test sets, not a score for whatever
  the panel types — now labeled explicitly in the app.
- **Module 4's page is entirely static**, full model included — it shows
  recorded training-run metrics, not a live re-prediction — now labeled
  explicitly in the app.
- **The LLM baseline is not implemented** (open item O3 in
  `docs/defect_register.md`) — do not present it as done if asked.
- **Cold-load latency (~5–15s) happens once per checkpoint per process.**
  This is why the T-10 warm-up step exists; skipping it means the panel's
  first click during the actual demo eats that delay instead of your warm-up
  click.
