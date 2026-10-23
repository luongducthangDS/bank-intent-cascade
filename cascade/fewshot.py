"""Pick real training messages to show the LLM for each candidate intent."""
import numpy as np


class ExampleBank:
    def __init__(self, train_emb: np.ndarray, train_labels: np.ndarray, train_texts: list[str]):
        self.emb, self.labels, self.texts = train_emb, train_labels, train_texts
        self.by_label = {c: np.where(train_labels == c)[0] for c in np.unique(train_labels)}

    def nearest(self, query_emb: np.ndarray, candidates, shots: int) -> dict[int, list[str]]:
        """For each candidate intent: its `shots` training messages most similar to the query (cosine)."""
        out = {}
        for c in candidates:
            idx = self.by_label[int(c)]
            top = idx[np.argsort(-(self.emb[idx] @ query_emb))[:shots]]
            out[int(c)] = [self.texts[j] for j in top]
        return out
