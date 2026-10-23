"""HTTP service: POST /classify -> intent + route (auto | human).

The LLM band is OFF by default: on this task the measured cascade did not beat classifier + human
(see README). Thresholds come from reports/results.json (tuned on validation).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from cascade.policy import AUTO, route

ROOT = Path(__file__).resolve().parents[1]
POLICY = os.getenv("CASCADE_POLICY", "classifier+human@5%")


class Request(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class Response(BaseModel):
    intent: str | None          # None when the message goes to a human
    route: str
    confidence: float
    top3: list[dict]
    latency_ms: float


class Classifier:
    def __init__(self, path=ROOT / "artifacts/classifier.joblib"):
        import joblib
        from sentence_transformers import SentenceTransformer
        art = joblib.load(path)
        self.clf, self.labels = art["clf"], art["labels"]
        self.encoder = SentenceTransformer(art["embed_model"], device="cpu")

    def predict_proba(self, text: str) -> np.ndarray:
        v = self.encoder.encode([text], normalize_embeddings=True)
        return self.clf.predict_proba(v)[0]


def load_thresholds(policy: str = POLICY) -> tuple[float, float]:
    pol = json.loads((ROOT / "reports/results.json").read_text())["policies_test"][policy]["tuned_on_val"]
    return pol["t_auto"], pol["t_llm"]


def create_app(model=None, thresholds=None) -> FastAPI:
    app = FastAPI(title="bank-intent-cascade", version="1.0.0")
    state = {"model": model, "thresholds": thresholds, "counts": {"auto": 0, "human": 0}}

    def get_model():
        if state["model"] is None:
            state["model"] = Classifier()
            state["thresholds"] = state["thresholds"] or load_thresholds()
        return state["model"]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bank Intent Cascade Demo</title>
  <style>
    :root { --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #f8fafc; --muted: #94a3b8; --accent: #38bdf8; --green: #22c55e; --amber: #f59e0b; }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); display: flex; justify-content: center; padding: 40px 20px; }
    .container { max-width: 680px; width: 100%; }
    .header { margin-bottom: 24px; }
    h1 { font-size: 24px; font-weight: 700; color: #fff; margin-bottom: 6px; }
    p.sub { color: var(--muted); font-size: 14px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
    label { display: block; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 8px; }
    textarea { width: 100%; height: 90px; background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 12px; color: #fff; font-size: 15px; resize: vertical; outline: none; }
    textarea:focus { border-color: var(--accent); }
    .samples { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 16px; }
    .chip { background: #334155; color: #cbd5e1; font-size: 12px; padding: 4px 10px; border-radius: 999px; cursor: pointer; border: none; transition: background 0.15s; }
    .chip:hover { background: #475569; color: #fff; }
    button.submit { background: var(--accent); color: #0f172a; font-weight: 600; font-size: 15px; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; width: 100%; }
    button.submit:hover { opacity: 0.9; }
    .result { display: none; margin-top: 20px; border-top: 1px solid var(--border); padding-top: 20px; }
    .badge { display: inline-block; font-size: 12px; font-weight: 700; text-transform: uppercase; padding: 4px 10px; border-radius: 6px; margin-bottom: 12px; }
    .badge-auto { background: rgba(34, 197, 94, 0.2); color: var(--green); border: 1px solid var(--green); }
    .badge-human { background: rgba(245, 158, 11, 0.2); color: var(--amber); border: 1px solid var(--amber); }
    .metric-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 14px; }
    .metric-val { font-size: 18px; font-weight: 600; }
    .bar-bg { background: #0f172a; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 4px; }
    .bar-fill { height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.3s; }
    .top-item { margin-top: 10px; }
    .top-header { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 2px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Bank Intent Cascade Router</h1>
      <p class="sub">Routing banking customer queries between calibrated ML classifier and human agents.</p>
    </div>
    <div class="card">
      <label>Customer Message</label>
      <textarea id="text" placeholder="Type customer message or select a sample..."></textarea>
      <div class="samples">
        <button class="chip" onclick="setMsg('Why was my transfer cancelled and where is my refund?')">Transfer cancelled</button>
        <button class="chip" onclick="setMsg('I forgot my PIN code for my card, can you reset it?')">PIN reset</button>
        <button class="chip" onclick="setMsg('Can you send a replacement card to my new home address?')">Card replacement</button>
        <button class="chip" onclick="setMsg('Hello, can somebody help me please?')">Vague greeting (Human)</button>
      </div>
      <button class="submit" id="btn" onclick="classify()">Route Query</button>
      <div class="result" id="res">
        <div id="badge"></div>
        <div class="metric-row">
          <div><span style="color:var(--muted); font-size:13px">Assigned Intent: </span><b id="intent" style="font-size:16px">-</b></div>
          <div><span style="color:var(--muted); font-size:13px">Latency: </span><b id="lat">-</b></div>
        </div>
        <div style="margin-bottom:16px">
          <div style="display:flex; justify-content:space-between; font-size:13px">
            <span style="color:var(--muted)">Top Confidence</span>
            <span id="conf-val" style="font-weight:600">-</span>
          </div>
          <div class="bar-bg"><div class="bar-fill" id="conf-bar" style="width:0%"></div></div>
        </div>
        <label>Top 3 Candidate Intents</label>
        <div id="top3-list"></div>
      </div>
    </div>
  </div>
  <script>
    function setMsg(m) { document.getElementById('text').value = m; classify(); }
    async function classify() {
      const text = document.getElementById('text').value.trim();
      if(!text) return;
      const btn = document.getElementById('btn');
      btn.textContent = 'Routing...'; btn.disabled = true;
      try {
        const res = await fetch('/classify', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text}) });
        const data = await res.json();
        document.getElementById('res').style.display = 'block';
        const badge = document.getElementById('badge');
        if(data.route === 'auto') {
          badge.className = 'badge badge-auto'; badge.textContent = '⚡ Answered Automatically';
          document.getElementById('intent').textContent = data.intent;
        } else {
          badge.className = 'badge badge-human'; badge.textContent = '👤 Hand-off to Human Agent';
          document.getElementById('intent').textContent = 'Escalated to agent';
        }
        document.getElementById('lat').textContent = data.latency_ms + ' ms';
        const pct = (data.confidence * 100).toFixed(1) + '%';
        document.getElementById('conf-val').textContent = pct;
        document.getElementById('conf-bar').style.width = pct;
        let html = '';
        data.top3.forEach(item => {
          const p = (item.p * 100).toFixed(1);
          html += `<div class="top-item"><div class="top-header"><span>${item.intent}</span><span>${p}%</span></div><div class="bar-bg"><div class="bar-fill" style="width:${p}%; background:#64748b"></div></div></div>`;
        });
        document.getElementById('top3-list').innerHTML = html;
      } catch(e) { alert('Error: ' + e); }
      finally { btn.textContent = 'Route Query'; btn.disabled = false; }
    }
  </script>
</body>
</html>"""

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        n = sum(state["counts"].values())
        return {**state["counts"], "human_rate": round(state["counts"]["human"] / n, 4) if n else None}

    @app.post("/classify", response_model=Response)
    def classify(req: Request):
        t = time.perf_counter()
        m = get_model()
        p = m.predict_proba(req.text)
        t_auto, _ = state["thresholds"]
        conf = float(p.max())
        r = route(conf, t_auto, t_auto)            # LLM band disabled: auto or human
        state["counts"]["auto" if r == AUTO else "human"] += 1
        top = np.argsort(-p)[:3]
        return Response(
            intent=m.labels[int(top[0])] if r == AUTO else None, route=r, confidence=round(conf, 4),
            top3=[{"intent": m.labels[int(i)], "p": round(float(p[i]), 4)} for i in top],
            latency_ms=round((time.perf_counter() - t) * 1000, 2))

    return app


app = create_app()
