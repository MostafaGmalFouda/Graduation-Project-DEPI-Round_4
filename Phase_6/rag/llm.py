import os
import requests


class LocalLLM:
    """
    Free, runs on your machine via Ollama. Weaker reasoning, especially
    for Arabic and for numeric/logical nuances (e.g. "0 means none").
    """

    def __init__(self, model: str = None):
        self.url = "http://localhost:11434/api/generate"
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                # lower temperature = fewer contradictory/confident-sounding mistakes
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]


class ClaudeLLM:
    """
    Uses Anthropic's Claude API — much better reasoning quality and Arabic
    support than a small local model. Requires an API key.

    Set:
        ANTHROPIC_API_KEY=<your key>
        CHATBOT_LLM=claude
    to activate it (see get_llm() below).
    """

    def __init__(self, model: str = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.url = "https://api.anthropic.com/v1/messages"

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Either set it, or unset "
                "CHATBOT_LLM to fall back to the local Ollama model."
            )

        response = requests.post(
            self.url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return "".join(block.get("text", "") for block in data.get("content", []))


def get_llm():
    """
    Picks which LLM backend the chatbot uses.

    Default: LocalLLM (free, local Ollama, no setup needed).
    To switch to Claude for noticeably better answers:
        export CHATBOT_LLM=claude
        export ANTHROPIC_API_KEY=sk-ant-...
    """
    backend = os.environ.get("CHATBOT_LLM", "local").lower()
    if backend == "claude":
        return ClaudeLLM()
    return LocalLLM()