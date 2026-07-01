import re


def detect_language(text: str) -> str:
    """Very small heuristic: any Arabic letter present → treat as Arabic."""
    return "Arabic" if re.search(r"[\u0600-\u06FF]", text or "") else "English"


def build_prompt(context: str, question: str, history: str = "") -> str:
    lang = detect_language(question)

    if lang == "Arabic":
        language_rule = (
            "The user's question is written in ARABIC. You MUST answer ONLY "
            "in Arabic (Egyptian/Modern Standard Arabic is fine). "
            "Never answer in English when the question is in Arabic."
        )
    else:
        language_rule = (
            "The user's question is written in ENGLISH. You MUST answer ONLY "
            "in English. Never answer in Arabic when the question is in English."
        )

    history_block = (
        f"\nPrevious conversation in this session (use it to resolve "
        f"follow-up questions like \"how many?\" or \"why?\" that refer back "
        f"to what was just discussed):\n{history}\n"
        if history else ""
    )

    return f"""
You are a data analysis assistant. Answer the user's question using ONLY the
Context and, if relevant, the Previous conversation below.

Rules:
- {language_rule}
- If the Context shows a count of 0 for something (e.g. "0 missing values",
  "0 duplicate rows"), that means it does NOT exist. Never say something
  "exists" or "is missing" when its count is 0 — 0 always means NONE.
- If the answer is not in the Context or Previous conversation, say clearly
  that you couldn't find that information in the dataset (in the required
  answer language), instead of guessing.
- Keep the answer short and directly address the question.
{history_block}
Context:
{context}

Question:
{question}

Answer:
"""