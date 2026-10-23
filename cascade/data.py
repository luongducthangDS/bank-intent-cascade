"""Banking77: 13,083 real customer messages to a bank, 77 intents (PolyAI, CC-BY-4.0).

Official split: 10,003 train / 3,080 test. We carve a stratified 15 % validation set out of train;
every threshold in this project is tuned on validation and reported once on test.
"""
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
# Original release by PolyAI (the HF dataset is a loader script pointing to these same files).
BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
RAW = Path(__file__).resolve().parents[1] / "data"


@dataclass
class Split:
    texts: list[str]
    labels: np.ndarray


@dataclass
class Banking77:
    train: Split
    val: Split
    test: Split
    label_names: list[str]

    def readable(self, i: int) -> str:
        return self.label_names[i].replace("_", " ")


def _fetch(name: str) -> Path:
    RAW.mkdir(exist_ok=True)
    path = RAW / name
    if not path.exists():
        urllib.request.urlretrieve(BASE + name, path)
    return path


def load() -> Banking77:
    names = json.loads(_fetch("categories.json").read_text())
    idx = {n: i for i, n in enumerate(names)}
    train = pd.read_csv(_fetch("train.csv"))
    test = pd.read_csv(_fetch("test.csv"))
    ty = train.category.map(idx).to_numpy()
    tr_x, va_x, tr_y, va_y = train_test_split(train.text.tolist(), ty, test_size=0.15, stratify=ty, random_state=SEED)
    return Banking77(
        train=Split(tr_x, np.asarray(tr_y)),
        val=Split(va_x, np.asarray(va_y)),
        test=Split(test.text.tolist(), test.category.map(idx).to_numpy()),
        label_names=names,
    )
