"""Second stage: a small local LLM (llama.cpp, CPU) picks the intent.

Output is constrained by a GBNF grammar to one of the allowed option numbers, so the model can never
return free text or an invalid label (0 parse failures by construction).
"""
import time
from dataclasses import dataclass
from pathlib import Path

from llama_cpp import Llama, LlamaGrammar

MODELS = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL = MODELS / "qwen2.5-3b-instruct-q4_k_m.gguf"


@dataclass
class LLMResult:
    label: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


def _grammar(n: int) -> LlamaGrammar:
    options = " | ".join(f'"{i}"' for i in range(1, n + 1))
    return LlamaGrammar.from_string(f"root ::= {options}", verbose=False)


class IntentLLM:
    def __init__(self, label_names: list[str], model_path: Path = DEFAULT_MODEL, threads: int = 8):
        self.names = [n.replace("_", " ") for n in label_names]
        self.llm = Llama(model_path=str(model_path), n_ctx=4096, n_threads=threads, verbose=False, seed=0)
        self._grammars: dict[int, LlamaGrammar] = {}

    def _ask(self, system: str, user: str, n_options: int) -> tuple[int, float, dict]:
        g = self._grammars.setdefault(n_options, _grammar(n_options))
        t = time.perf_counter()
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            grammar=g, temperature=0.0, max_tokens=3)
        ms = (time.perf_counter() - t) * 1000
        return int(out["choices"][0]["message"]["content"].strip()), ms, out["usage"]

    def choose(self, text: str, candidates: list[int], examples: dict[int, list[str]] | None = None) -> LLMResult:
        """Rerank the classifier's top-k candidates.

        `examples` maps a candidate intent to real training messages of that intent (the ones nearest to this
        message). Intent names alone are ambiguous ("card payment fee charged" vs "transfer fee charged");
        2-3 real examples per option tell the model what each label actually means.
        """
        def option(i, c):
            line = f"{i + 1}. {self.names[c]}"
            if examples and examples.get(c):
                line += "".join(f'\n   e.g. "{e}"' for e in examples[c])
            return line
        options = "\n".join(option(i, c) for i, c in enumerate(candidates))
        system = ("You route customer messages for a bank's support team. "
                  "Pick the intent that best matches the customer's message. Answer with the option number only.")
        user = f"Customer message: {text}\n\nIntents:\n{options}\n\nBest intent number:"
        k, ms, usage = self._ask(system, user, len(candidates))
        return LLMResult(candidates[k - 1], ms, usage["prompt_tokens"], usage["completion_tokens"])

    def zero_shot(self, text: str) -> LLMResult:
        """All 77 intents in the prompt. The fixed label list is the system prompt, so llama.cpp reuses its
        KV cache across calls and only the customer message is processed per request."""
        options = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(self.names))
        system = ("You route customer messages for a bank's support team. Answer with the number of the intent "
                  f"that best matches the message.\n\nIntents:\n{options}")
        k, ms, usage = self._ask(system, f"Customer message: {text}\nBest intent number:", len(self.names))
        return LLMResult(k - 1, ms, usage["prompt_tokens"], usage["completion_tokens"])
