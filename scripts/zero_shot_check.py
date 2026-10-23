"""Sanity check of 77-label zero-shot on RANDOM (not hardest) validation messages."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.data import load  # noqa: E402
from cascade.llm import MODELS, IntentLLM  # noqa: E402

d = load()
llm = IntentLLM(d.label_names, MODELS / sys.argv[1])
rng = np.random.default_rng(0)
idx = rng.choice(len(d.val.texts), int(sys.argv[2]), replace=False)
ok = 0
for n, i in enumerate(idx):
    r = llm.zero_shot(d.val.texts[i])
    ok += r.label == d.val.labels[i]
    if n < 8:
        print(repr(d.val.texts[i][:70]), "| gold:", d.readable(d.val.labels[i]), "| llm:", d.readable(r.label), flush=True)
print(f"zero-shot accuracy on {len(idx)} random val: {ok / len(idx):.2%}")
