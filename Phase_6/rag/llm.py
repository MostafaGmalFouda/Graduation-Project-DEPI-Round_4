import os
import requests
import google.generativeai as genai

LLM_TEMPERATURE = 0.1  # low = grounded/repeatable, not creative. Same value for every backend.


class LocalLLM:
    """
    Free, runs on your machine via Ollama. Weaker reasoning, especially
    for Arabic and for numeric/logical nuances (e.g. "0 means none").
    """

    def __init__(self, model: str = None):
        self.url = "http://localhost:11434/api/generate"
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")

    def generate(self, system: str, user: str) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "system": system,
                "prompt": user,
                "stream": False,
                "options": {"temperature": LLM_TEMPERATURE},
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

    def generate(self, system: str, user: str) -> str:
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
                "temperature": LLM_TEMPERATURE,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return "".join(block.get("text", "") for block in data.get("content", []))

class GeminiLLM:

    def __init__(self, model=None):
        self.api_key = os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        genai.configure(api_key=self.api_key)

        self.model = genai.GenerativeModel(
        model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")   
                  )

    def generate(self, system: str, user: str) -> str:

        prompt = f"""
{system}

{user}
"""

        try:
             response = self.model.generate_content(
        prompt,
        generation_config={
            "temperature": LLM_TEMPERATURE
        }
    )
             return response.text

        except Exception as e:
                raise RuntimeError(f"Gemini API Error: {e}")

def get_llm():
    """
   Picks which LLM backend the chatbot uses.

    Available backends:

    - local   -> Ollama
    - claude  -> Anthropic Claude
    - gemini  -> Google Gemini

    Set:

    CHATBOT_LLM=local
    CHATBOT_LLM=claude
    CHATBOT_LLM=gemini

    """

    backend = os.environ.get("CHATBOT_LLM", "local").lower()
    print(f"Using LLM backend: {backend}")

    if backend == "claude":
        return ClaudeLLM()

    elif backend == "gemini":
        return GeminiLLM()

    elif backend == "local":
        return LocalLLM()

    else:
        raise RuntimeError(
            f"Unknown LLM backend: {backend}"
        )