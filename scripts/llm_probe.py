"""Quick speed/quality probe of a local model before the full run."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.data import load
from cascade.llm import MODELS, IntentLLM

ROOT = Path(__file__).resolve().parents[1]
model = MODELS / sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
d = load()
p = np.load(ROOT / ".cache" / "embedding_lr_val_proba.npy")
llm = IntentLLM(d.label_names, model)
shots = int(sys.argv[3]) if len(sys.argv) > 3 else 0
tr_emb = np.load(ROOT / ".cache" / "bge_train.npy")
va_emb = np.load(ROOT / ".cache" / "bge_val.npy")


def nearest_examples(i, cands):
    """For each candidate intent, the `shots` training messages most similar to this message."""
    out = {}
    for c in cands:
        m = np.where(d.train.labels == c)[0]
        sims = tr_emb[m] @ va_emb[i]
        out[c] = [d.train.texts[j] for j in m[np.argsort(-sims)[:shots]]]
    return out
idx = np.argsort(p.max(1))[:n]          # the hardest (least confident) validation examples
top5_ok = llm_ok = clf_ok = 0
lat = []
for i in idx:
    cands = list(np.argsort(-p[i])[:5])
    r = llm.choose(d.val.texts[i], cands, nearest_examples(i, cands) if shots else None)
    lat.append(r.latency_ms)
    llm_ok += r.label == d.val.labels[i]
    clf_ok += cands[0] == d.val.labels[i]
    top5_ok += d.val.labels[i] in cands
print(f"{model.name} shots={shots}: n={n} hardest val | classifier {clf_ok/n:.2%} | llm top5-rerank {llm_ok/n:.2%} | "
      f"gold in top5 {top5_ok/n:.2%} | p50 {np.percentile(lat,50):.0f} ms p95 {np.percentile(lat,95):.0f} ms")
zs = [llm.zero_shot(d.val.texts[i]) for i in idx[:10]]
print("zero-shot 77 labels: acc", np.mean([z.label == d.val.labels[i] for z, i in zip(zs, idx[:10])]),
      "latencies ms", [round(z.latency_ms) for z in zs], "prompt tokens", zs[-1].prompt_tokens)
