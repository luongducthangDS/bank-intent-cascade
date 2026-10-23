import numpy as np
import pytest

from cascade.policy import AUTO, HUMAN, LLM, evaluate, route, search


@pytest.mark.parametrize("conf, expected", [(0.95, AUTO), (0.90, AUTO), (0.89, LLM), (0.50, LLM), (0.49, HUMAN)])
def test_route_boundaries(conf, expected):
    assert route(conf, t_auto=0.90, t_llm=0.50) == expected


def test_route_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        route(0.5, t_auto=0.4, t_llm=0.6)


def test_evaluate_counts_and_accuracy():
    y = np.array([0, 1, 2, 3])
    clf = np.array([0, 9, 9, 9])           # only the confident one is right
    conf = np.array([0.99, 0.7, 0.6, 0.1])
    llm = np.array([-1, 1, 9, -1])         # LLM fixes one of its two
    r = evaluate(y, clf, conf, llm, t_auto=0.9, t_llm=0.5)
    assert (r.auto_rate, r.llm_rate, r.human_rate) == (0.25, 0.5, 0.25)
    assert r.automated_accuracy == pytest.approx(2 / 3)
    assert r.end_to_end_accuracy == 0.75     # human case counted as resolved
    assert r.errors_per_1k == 250


def test_missing_llm_prediction_is_an_error_not_a_silent_miss():
    with pytest.raises(ValueError):
        evaluate(np.array([0]), np.array([0]), np.array([0.6]), np.array([-1]), 0.9, 0.5)


def test_search_respects_human_budget_and_prefers_less_llm():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    conf = rng.random(500)
    clf = np.where(conf > 0.5, y, (y + 1) % 5)   # confident -> right
    llm = y.copy()                              # perfect LLM
    best = search(y, clf, conf, llm, max_human=0.0, use_llm=True)
    assert best.human_rate == 0.0
    assert best.errors_per_1k == 0.0
    no_llm = search(y, clf, conf, llm, max_human=0.0, use_llm=False)
    assert no_llm.llm_rate == 0.0 and no_llm.errors_per_1k > 0


def test_verify_mode_sends_disagreements_to_a_human():
    y = np.array([0, 1, 2])
    clf = np.array([0, 1, 9])
    conf = np.array([0.99, 0.6, 0.6])
    llm = np.array([-1, 1, 2])          # agrees on #1, disagrees on #2
    r = evaluate(y, clf, conf, llm, t_auto=0.9, t_llm=0.5, mode="verify")
    assert r.human_rate == pytest.approx(1 / 3)     # the disagreement
    assert r.llm_rate == pytest.approx(2 / 3)       # LLM was consulted twice
    assert r.errors_per_1k == 0                     # wrong classifier answer never reached the customer


def test_search_never_puts_llm_band_above_cap():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 300)
    conf = rng.random(300)
    clf = np.where(conf > 0.8, y, (y + 1) % 5)
    best = search(y, clf, conf, y.copy(), max_human=0.0, use_llm=True, llm_cap=0.5)
    assert best.t_llm == best.t_auto or best.t_auto <= 0.5   # any LLM band stays under the cap
