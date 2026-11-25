"""
LLM client for the coordination layer.

Talks to any OpenAI-compatible chat endpoint via the lightweight native ``openai``
SDK -- no PaperQA/litellm dependency. Configuration is injected through
``BATTAGENT_LLM_*`` environment variables so **no API key, endpoint, or model name
is ever hard-coded** in this package (the defaults below are empty).

Design point (reproducibility): the LLM is used ONLY for the
coordination layer's *reasoning* (justifying tool selection, attributing
failures, translating cross-scale fields). It never fabricates physical values,
and it never makes the go/no-go decision -- the validation gates run on registry
/ published-case numbers. When no key is configured (or the request fails), a
``DeterministicStub`` returns an empty rationale and the module falls back to its
own deterministic rationale, so the no-key gate decisions are reproducible.

Configure (copy llm_config.example to a local, git-ignored file)::

    set BATTAGENT_LLM_API_KEY=...               # required to go "live"; never committed
    set BATTAGENT_LLM_BASE_URL=<your OpenAI-compatible endpoint>
    set BATTAGENT_LLM_MODEL=<your model>
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# No default endpoint or model: set BATTAGENT_LLM_BASE_URL and BATTAGENT_LLM_MODEL
# (any OpenAI-compatible chat endpoint) together with BATTAGENT_LLM_API_KEY.
DEFAULT_BASE_URL = ""
DEFAULT_MODEL = ""


@dataclass
class LLMResult:
    text: str
    backend: str          # "openai" | "stub" | "error"
    model: str = ""

    @property
    def is_live(self) -> bool:
        return self.backend == "openai" and bool(self.text.strip())


@dataclass
class LLMClient:
    base_url: str = field(default_factory=lambda: os.environ.get("BATTAGENT_LLM_BASE_URL", DEFAULT_BASE_URL))
    api_key: str = field(default_factory=lambda: os.environ.get("BATTAGENT_LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("BATTAGENT_LLM_MODEL", DEFAULT_MODEL))
    temperature: float = field(default_factory=lambda: float(os.environ.get("BATTAGENT_LLM_TEMPERATURE", "1.0")))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("BATTAGENT_LLM_MAX_TOKENS", "4096")))

    def __post_init__(self) -> None:
        self._client = None
        self.calls: list[dict] = []
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def is_live(self) -> bool:
        return self._client is not None

    @property
    def backend(self) -> str:
        return "openai" if self.is_live else "stub"

    def complete(self, system: str, user: str) -> LLMResult:
        """Return reasoning text. Logs every call to ``self.calls`` for traceability."""
        if self._client is not None:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                result = LLMResult(text=resp.choices[0].message.content or "", backend="openai", model=self.model)
            except Exception as exc:  # noqa: BLE001
                result = LLMResult(text=f"[llm-error: {exc}]", backend="error", model=self.model)
        else:
            result = LLMResult(text="", backend="stub", model="deterministic")

        self.calls.append(
            {
                "backend": result.backend,
                "model": result.model,
                "system": system[:160],
                "user": user[:400],
                "response": result.text[:400],
            }
        )
        return result

    def last_call(self) -> dict:
        return self.calls[-1] if self.calls else {}


# Shared default client (lazy singleton) so all agents log to one trail.
_DEFAULT: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LLMClient()
    return _DEFAULT


if __name__ == "__main__":
    c = get_llm_client()
    print(f"backend = {c.backend} (live={c.is_live}); base_url={c.base_url}; model={c.model}")
    r = c.complete("You are a terse assistant.", "Reply with the single word OK.")
    print(f"result.backend={r.backend} text={r.text!r}")
