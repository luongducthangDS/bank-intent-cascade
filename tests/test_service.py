"""API contract with a stub model (no weights needed)."""
import numpy as np
from fastapi.testclient import TestClient

from cascade.service import create_app


class Stub:
    labels = ("card_arrival", "lost_or_stolen_card", "top_up_failed")

    def predict_proba(self, text):
        return np.array([0.9, 0.05, 0.05]) if "card" in text else np.array([0.4, 0.35, 0.25])


client = TestClient(create_app(model=Stub(), thresholds=(0.6, 0.6)))


def test_confident_message_is_answered():
    r = client.post("/classify", json={"text": "where is my card"}).json()
    assert r["route"] == "auto" and r["intent"] == "card_arrival"
    assert next(x["intent"] for x in r["top3"]) == "card_arrival"


def test_unsure_message_goes_to_human_without_an_intent():
    r = client.post("/classify", json={"text": "hmm help"}).json()
    assert r["route"] == "human" and r["intent"] is None


def test_input_validation():
    assert client.post("/classify", json={"text": ""}).status_code == 422
    assert client.post("/classify", json={"text": "x" * 1001}).status_code == 422


def test_metrics_count_routes():
    m = client.get("/metrics").json()
    assert m["auto"] >= 1 and m["human"] >= 1
