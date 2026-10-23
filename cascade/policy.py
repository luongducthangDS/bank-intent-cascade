"""Routing policy: classifier -> (optional) LLM -> human, decided by classifier confidence.

    confidence >= t_auto           -> answer with the classifier            (cheap, ~30 ms)
    t_llm <= confidence < t_auto   -> ask the LLM to pick among the top-k   (expensive, seconds)
    confidence <  t_llm            -> hand over to a human agent

t_llm == t_auto means "no LLM band": classifier + human only.

Two ways to use the LLM inside its band:
    mode="rerank"  the LLM's choice is the answer
    mode="verify"  the LLM is a second opinion: if it agrees with the classifier the answer is sent,
                   otherwise the message goes to a human
"""
from dataclasses import asdict, dataclass

import numpy as np

AUTO, LLM, HUMAN = "auto", "llm", "human"


def route(conf: float, t_auto: float, t_llm: float) -> str:
    if not 0.0 <= t_llm <= t_auto <= 1.0:
        raise ValueError("need 0 <= t_llm <= t_auto <= 1")
    if conf >= t_auto:
        return AUTO
    if conf >= t_llm:
        return LLM
    return HUMAN


@dataclass
class PolicyResult:
    t_auto: float
    t_llm: float
    auto_rate: float
    llm_rate: float
    human_rate: float
    automated_accuracy: float   # accuracy on everything NOT sent to a human
    end_to_end_accuracy: float  # assumes the human resolves their share correctly
    errors_per_1k: float        # wrong answers sent to customers per 1,000 messages

    def as_dict(self):
        return {k: round(v, 4) for k, v in asdict(self).items()}


def evaluate(y, clf_pred, conf, llm_pred, t_auto, t_llm, mode="rerank") -> PolicyResult:
    """llm_pred[i] must be filled (>= 0) for every i routed to the LLM."""
    routes = np.array([route(c, t_auto, t_llm) for c in conf])
    auto, llm, human = routes == AUTO, routes == LLM, routes == HUMAN
    if (llm_pred[llm] < 0).any():
        raise ValueError("LLM prediction missing for a message routed to the LLM")
    if mode == "verify":
        disagree = llm & (llm_pred != clf_pred)
        human = human | disagree
        llm_answer = clf_pred
    elif mode == "rerank":
        llm_answer = llm_pred
    else:
        raise ValueError(mode)
    llm_rate = llm.mean()
    llm = llm & ~human
    final = np.where(auto, clf_pred, np.where(llm, llm_answer, y))   # human -> correct by assumption
    correct = final == y
    n = len(y)
    served = ~human
    return PolicyResult(
        t_auto=t_auto, t_llm=t_llm,
        auto_rate=auto.mean(), llm_rate=llm_rate, human_rate=human.mean(),
        automated_accuracy=float(correct[served].mean()) if served.any() else 1.0,
        end_to_end_accuracy=float(correct.mean()),
        errors_per_1k=float((~correct).sum() / n * 1000),
    )


def search(y, clf_pred, conf, llm_pred, max_human: float, use_llm: bool, grid=None,
           mode="rerank", llm_cap: float = 1.0) -> PolicyResult:
    """Best policy on a validation set: fewest customer-facing errors with human_rate <= max_human.
    Ties -> less LLM traffic (cheaper). The LLM band may not reach above `llm_cap`."""
    grid = grid if grid is not None else np.round(np.arange(0.0, 1.0001, 0.02), 2)
    best = None
    for t_auto in grid:
        # a non-empty LLM band [t_llm, t_auto) must sit below llm_cap; t_llm == t_auto means no band
        llm_options = grid[grid <= t_auto] if use_llm and t_auto <= llm_cap else []
        for t_llm in [t_auto, *[t for t in llm_options if t < t_auto]]:
            r = evaluate(y, clf_pred, conf, llm_pred, float(t_auto), float(t_llm), mode)
            if r.human_rate > max_human:
                continue
            key = (r.errors_per_1k, r.llm_rate)
            if best is None or key < (best.errors_per_1k, best.llm_rate):
                best = r
    return best
