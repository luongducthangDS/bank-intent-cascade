"""Cheap first-stage classifiers. Both return calibrated-ish probabilities (multinomial LR)."""
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

CACHE = Path(__file__).resolve().parents[1] / ".cache"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 33M params, 384-dim, runs on CPU


class TfidfLR:
    name = "tfidf_lr"

    def fit(self, texts, y):
        self.pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                                  LogisticRegression(C=10, max_iter=3000))
        self.pipe.fit(texts, y)
        return self

    def predict_proba(self, texts):
        return self.pipe.predict_proba(texts)


class EmbeddingLR:
    name = "embedding_lr"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(EMBED_MODEL, device="cpu")

    def embed(self, texts, cache_key: str | None = None):
        if cache_key:
            path = CACHE / f"{cache_key}.npy"
            if path.exists():
                return np.load(path)
        vecs = self.encoder.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        if cache_key:
            CACHE.mkdir(exist_ok=True)
            np.save(path, vecs)
        return vecs

    def fit(self, texts, y, cache_key=None):
        self.clf = LogisticRegression(C=20, max_iter=3000)
        self.clf.fit(self.embed(texts, cache_key), y)
        return self

    def predict_proba(self, texts, cache_key=None):
        return self.clf.predict_proba(self.embed(texts, cache_key))


def timed_single(model, texts, n=200):
    """Per-request latency: one text at a time, the way a live service sees traffic."""
    out = []
    for t in texts[:n]:
        s = time.perf_counter()
        model.predict_proba([t])
        out.append((time.perf_counter() - s) * 1000)
    return np.array(out)
