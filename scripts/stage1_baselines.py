"""Stage 1: cheap classifiers on Banking77 test. Writes reports/stage1.json."""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.data import load
from cascade.models import EmbeddingLR, TfidfLR, timed_single

ROOT = Path(__file__).resolve().parents[1]


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def main():
    (ROOT / ".cache").mkdir(exist_ok=True)
    d = load()
    print("train/val/test", len(d.train.texts), len(d.val.texts), len(d.test.texts), flush=True)
    res = {}
    for model in (TfidfLR(), EmbeddingLR()):
        t = time.perf_counter()
        kw = lambda split, m=model: {"cache_key": f"bge_{split}"} if m.name == "embedding_lr" else {}
        model.fit(d.train.texts, d.train.labels, **kw("train"))
        fit_s = time.perf_counter() - t
        for split in ("val", "test"):
            s = getattr(d, split)
            p = model.predict_proba(s.texts, **kw(split))
            np.save(ROOT / ".cache" / f"{model.name}_{split}_proba.npy", p)
            res.setdefault(model.name, {})[split] = {
                "accuracy": round(accuracy_score(s.labels, p.argmax(1)), 4),
                "macro_f1": round(f1_score(s.labels, p.argmax(1), average="macro"), 4),
                "ece": round(ece(p, s.labels), 4),
                "top5_recall": round(float(np.mean([y in np.argsort(-row)[:5] for row, y in zip(p, s.labels)])), 4),
            }
        lat = timed_single(model, d.test.texts)
        res[model.name]["latency_ms"] = {"p50": round(float(np.percentile(lat, 50)), 2),
                                         "p95": round(float(np.percentile(lat, 95)), 2)}
        res[model.name]["fit_seconds"] = round(fit_s, 1)
        print(model.name, res[model.name], flush=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "stage1.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
