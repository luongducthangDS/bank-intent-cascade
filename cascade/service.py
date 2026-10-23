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
