# LLM probes (raw outputs, recorded 2026-10-23 on a laptop CPU: i7-1365U, 32 GB, no GPU)

Hardest 60 validation messages (lowest classifier confidence), top-5 candidates from the classifier.
Command: `python scripts/llm_probe.py <model> 60 <shots>`

| Model (GGUF q4_k_m) | Shots per option | Classifier acc | LLM rerank acc | Gold in top-5 | LLM p50 / p95 |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 0 | 45.00% | 26.67% | 93.33% | 1,278 / 2,080 ms |
| Qwen2.5-1.5B-Instruct | 3 (nearest train examples) | 45.00% | 36.67% | 93.33% | 4,949 / 11,050 ms |
| Qwen2.5-3B-Instruct | 0 | 45.00% | 25.00% | 93.33% | 1,902 / 2,939 ms |
| Qwen2.5-3B-Instruct | 3 (nearest train examples) | 45.00% | 45.00% | 93.33% | 10,256 / 19,723 ms |

Zero-shot over all 77 intents (no classifier), 40 random validation messages, Qwen2.5-3B:
**32.50%** (`python scripts/zero_shot_check.py qwen2.5-3b-instruct-q4_k_m.gguf 40`).
Prompt = 614 tokens; the 77-label system prompt is reused from llama.cpp's KV cache after the first call
(first call ~19-22 s, later calls ~1.1-3.8 s).

Chosen for the full run: Qwen2.5-3B, 3 shots, top-5 (the only configuration that matched the classifier).
