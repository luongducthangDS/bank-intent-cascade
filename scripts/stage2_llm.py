"""Stage 2: run the local LLM on every message the classifier is unsure about (confidence < CAP),
for both validation and test, and cache predictions + latency + tokens.

    python scripts/stage2_llm.py <model.gguf> <shots> [k]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.data import load
from cascade.fewshot import ExampleBank
from cascade.llm import MODELS, IntentLLM

ROOT = Path(__file__).resolve().parents[1]
CAP = 0.50   # the LLM band never reaches above this confidence: LLM cost is bounded by design


def main():
    model_file, shots = sys.argv[1], int(sys.argv[2])
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    d = load()
    bank = ExampleBank(np.load(ROOT / ".cache/bge_train.npy"), d.train.labels, d.train.texts)
    llm = IntentLLM(d.label_names, MODELS / model_file)
    tag = f"{Path(model_file).stem}_s{shots}_k{k}"
    for split in ("val", "test"):
        out_path = ROOT / ".cache" / f"llm_{tag}_{split}.json"
        if out_path.exists():
            continue
        s = getattr(d, split)
        p = np.load(ROOT / f".cache/embedding_lr_{split}_proba.npy")
        emb = np.load(ROOT / f".cache/bge_{split}.npy")
        todo = np.where(p.max(1) < CAP)[0]
        rows, t0 = [], time.perf_counter()
        for n, i in enumerate(todo):
            cands = [int(c) for c in np.argsort(-p[i])[:k]]
            r = llm.choose(s.texts[i], cands, bank.nearest(emb[i], cands, shots) if shots else None)
            rows.append({"i": int(i), "pred": int(r.label), "ms": round(r.latency_ms, 1),
                         "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens})
            if n % 50 == 0:
                print(f"{split} {n}/{len(todo)} {time.perf_counter() - t0:.0f}s", flush=True)
        out_path.write_text(json.dumps(rows))
        print(split, "done", len(rows), flush=True)


if __name__ == "__main__":
    main()
