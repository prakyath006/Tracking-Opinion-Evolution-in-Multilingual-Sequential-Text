"""
Assemble hf_space/, a self-contained folder for a Hugging Face Docker Space that runs the Streamlit demo.

    python scripts/build_hf_space.py
    hf upload <user>/<space-name> hf_space . --repo-type space      # needs `hf auth login`

Includes src/, web_demo/, ontology/, outputs/ (metrics, logs and the two checkpoints) and data/preprocessed/
(the demo reads them at page load). hf_space/ is git-ignored.
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "hf_space")
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")

README = """---
title: Opinion Evolution Tracker
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

Streamlit demo for "Tracking Opinion Evolution in Multilingual Sequential Text".
Code: https://github.com/prakyath006/Tracking-Opinion-Evolution-in-Multilingual-Sequential-Text
"""

DOCKERFILE = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "-m", "streamlit", "run", "web_demo/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
"""

if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)
for d in ("src", "web_demo", "ontology", "outputs", ".streamlit"):
    shutil.copytree(os.path.join(ROOT, d), os.path.join(OUT, d), ignore=SKIP)
os.makedirs(os.path.join(OUT, "data"))
shutil.copytree(os.path.join(ROOT, "data", "preprocessed"), os.path.join(OUT, "data", "preprocessed"), ignore=SKIP)
shutil.copy(os.path.join(ROOT, "requirements.txt"), OUT)
for name, text in (("README.md", README), ("Dockerfile", DOCKERFILE)):
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
total = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(OUT) for f in fs)
print(f"built {OUT} ({total / 1e6:.0f} MB)")
