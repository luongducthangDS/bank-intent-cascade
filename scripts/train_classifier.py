"""Train the production classifier (bge-small + LR) on train+val and save it with the label names."""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.data import load
from cascade.models import EMBED_MODEL

ROOT = Path(__file__).resolve().parents[1]
d = load()
x = np.vstack([np.load(ROOT / ".cache/bge_train.npy"), np.load(ROOT / ".cache/bge_val.npy")])
y = np.concatenate([d.train.labels, d.val.labels])
clf = LogisticRegression(C=20, max_iter=3000).fit(x, y)
(ROOT / "artifacts").mkdir(exist_ok=True)
joblib.dump({"clf": clf, "labels": d.label_names, "embed_model": EMBED_MODEL}, ROOT / "artifacts/classifier.joblib")
print("saved artifacts/classifier.joblib")
