"""Stage 3: tune routing policies on validation, report them ONCE on test, with bootstrap CIs.

    python scripts/stage3_report.py <llm tag, e.g. qwen2.5-3b-instruct-q4_k_m_s3_k5>
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.data import load  # noqa: E402
from cascade.policy import evaluate, search  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAP = 0.50
BUDGETS = (0.05, 0.10)          # share of messages we can afford to hand to humans
B = 2000                         # bootstrap resamples


def load_split(split, tag, n):
    p = np.load(ROOT / f".cache/embedding_lr_{split}_proba.npy")
    llm = np.full(n, -1)
    rows = json.loads((ROOT / f".cache/llm_{tag}_{split}.json").read_text())
    for r in rows:
        llm[r["i"]] = r["pred"]
    return p, llm, rows


def errors_ci(y, final_fn, idx_sets):
    """95 % bootstrap CI of errors per 1k for a fixed policy (thresholds already chosen)."""
    vals = [final_fn(ix) for ix in idx_sets]
    return [round(float(np.percentile(vals, 2.5)), 1), round(float(np.percentile(vals, 97.5)), 1)]


def main():
    tag = sys.argv[1]
    d = load()
    yv, yt = d.val.labels, d.test.labels
    pv, lv, _ = load_split("val", tag, len(yv))
    pt, lt, rows_t = load_split("test", tag, len(yt))
    cv, ct = pv.max(1), pt.max(1)
    clf_v, clf_t = pv.argmax(1), pt.argmax(1)
    tfidf_t = np.load(ROOT / ".cache/tfidf_lr_test_proba.npy").argmax(1)
    stage1 = json.loads((ROOT / "reports/stage1.json").read_text())

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(yt), len(yt)) for _ in range(B)]
    policies = {}

    def add(name, r_val, mode="rerank"):
        r = evaluate(yt, clf_t, ct, lt, r_val.t_auto, r_val.t_llm, mode)
        ci = errors_ci(yt, lambda ix: evaluate(yt[ix], clf_t[ix], ct[ix], lt[ix], r_val.t_auto, r_val.t_llm,
                                               mode).errors_per_1k, boots[:300])
        policies[name] = {**r.as_dict(), "errors_per_1k_ci95": ci, "tuned_on_val": r_val.as_dict()}

    # classifier only = every message answered automatically
    for name, pred in (("tfidf_only", tfidf_t), ("embedding_only", clf_t)):
        err = (pred != yt)
        policies[name] = {"auto_rate": 1.0, "human_rate": 0.0, "llm_rate": 0.0,
                          "automated_accuracy": round(float(1 - err.mean()), 4),
                          "errors_per_1k": round(float(err.mean() * 1000), 1),
                          "errors_per_1k_ci95": [round(float(np.percentile([err[ix].mean() * 1000 for ix in boots], q)), 1)
                                                 for q in (2.5, 97.5)]}
    for budget in BUDGETS:
        b = int(budget * 100)
        add(f"classifier+human@{b}%", search(yv, clf_v, cv, lv, budget, use_llm=False))
        add(f"cascade_rerank@{b}%", search(yv, clf_v, cv, lv, budget, use_llm=True, llm_cap=CAP), "rerank")
        add(f"cascade_verify@{b}%", search(yv, clf_v, cv, lv, budget, use_llm=True, llm_cap=CAP, mode="verify"),
            "verify")

    # LLM cost and latency, measured on every test call
    ms = np.array([r["ms"] for r in rows_t])
    ptok = np.array([r["prompt_tokens"] for r in rows_t])
    llm_stats = {"calls_measured": len(rows_t), "latency_ms_p50": round(float(np.percentile(ms, 50))),
                 "latency_ms_p95": round(float(np.percentile(ms, 95))),
                 "prompt_tokens_mean": round(float(ptok.mean())),
                 "accuracy_on_its_band": round(float(np.mean([r["pred"] == yt[r["i"]] for r in rows_t])), 4),
                 "classifier_accuracy_on_same_band": round(float(np.mean([clf_t[r["i"]] == yt[r["i"]] for r in rows_t])), 4)}
    clf_ms = stage1["embedding_lr"]["latency_ms"]["p50"]
    for name, pol in policies.items():
        pol["mean_latency_ms"] = round(clf_ms + pol.get("llm_rate", 0) * float(ms.mean()), 1)
        pol["llm_prompt_tokens_per_1k_msgs"] = round(pol.get("llm_rate", 0) * 1000 * float(ptok.mean()))

    # error analysis: most confused intent pairs for the plain classifier
    conf_pairs = Counter((d.readable(a), d.readable(b)) for a, b in zip(yt, clf_t) if a != b).most_common(8)

    # risk-coverage curve: answer only when confidence >= t
    ts = np.linspace(0, 0.99, 100)
    cov = [float((ct >= t).mean()) for t in ts]
    acc = [float((clf_t[ct >= t] == yt[ct >= t]).mean()) if (ct >= t).any() else 1.0 for t in ts]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(np.array(cov) * 100, np.array(acc) * 100, color="#2563EB", lw=2, label="embedding classifier, threshold sweep")
    for name, col, mk in (("classifier+human@5%", "#2563EB", "o"), ("cascade_rerank@5%", "#DC2626", "s"),
                          ("cascade_verify@5%", "#059669", "^")):
        pol = policies[name]
        ax.scatter((1 - pol["human_rate"]) * 100, pol["automated_accuracy"] * 100, color=col, marker=mk, s=60,
                   zorder=3, label=f"{name}: {pol['errors_per_1k']:.0f} errors/1k")
    ax.set_xlabel("messages answered automatically (%)")
    ax.set_ylabel("accuracy on answered messages (%)")
    ax.set_title("Banking77 test: automation vs accuracy", loc="left", fontweight="bold")
    ax.grid(color="#E5E7EB")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/risk_coverage.png", dpi=140)

    out = {"llm": tag, "llm_cap": CAP, "llm_stats_test": llm_stats, "policies_test": policies,
           "top_confusions": [{"gold": a, "predicted": b, "count": n} for (a, b), n in conf_pairs]}
    (ROOT / "reports/results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
