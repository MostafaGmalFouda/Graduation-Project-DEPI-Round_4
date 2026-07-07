import re


def detect_language(text: str) -> str:
    """Very small heuristic: any Arabic letter present → treat as Arabic."""
    return "Arabic" if re.search(r"[\u0600-\u06FF]", text or "") else "English"


def build_system_prompt(lang: str) -> str:
    """
    Fixed behavioral rules that don't change per-question (aside from the
    language directive, which depends on the detected question language).
    These belong in the `system` role/field, not mixed into the user
    message — models weight system instructions more heavily and it keeps
    them from competing with the retrieved Context for attention.
    """
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

    return f"""You are a data analysis assistant. Answer the user's question using
ONLY the Guaranteed Facts, Context, and Previous conversation provided in
the user message. Never use outside knowledge about datasets in general.

Rules:
- {language_rule}
- The dataset has TWO stages: "raw" (before cleaning) and "clean" (after
  cleaning). They can have different row/column counts and different
  missing/duplicate numbers. Always match the stage the question is asking
  about — if the question doesn't specify, prefer the "clean" stage and say
  which stage your answer refers to.
- If the Context or Guaranteed Facts show a count of 0 for something (e.g.
  "0 missing values", "0 duplicate rows"), that means it does NOT exist.
  Never say something "exists" or "is missing" when its count is 0 — 0
  always means NONE.
- NEVER invent a number, column name, or statistic that isn't explicitly
  present in the Guaranteed Facts or Context. If you're not 100% sure a
  detail is in there, don't state it as fact.
- If the answer is not in the Guaranteed Facts, Context, or Previous
  conversation, say clearly that you couldn't find that information in the
  dataset (in the required answer language), instead of guessing.
- Keep the answer short and directly address the question."""


def build_user_prompt(context: str, question: str, history: str = "", guaranteed_facts: str = "") -> str:
    """
    The variable, per-question payload: guaranteed facts, retrieved
    context, conversation history, and the question itself. The fixed
    behavioral rules live in build_system_prompt() instead.
    """
    history_block = (
        f"\nPrevious conversation in this session (use it to resolve "
        f"follow-up questions like \"how many?\" or \"why?\" that refer back "
        f"to what was just discussed):\n{history}\n"
        if history else ""
    )

    facts_block = (
        f"Guaranteed Facts (always accurate, use these for any counting "
        f"question — rows, columns, missing values, duplicates):\n{guaranteed_facts}\n"
        if guaranteed_facts else ""
    )

    return f"""{facts_block}
Context:
{context}
{history_block}
Question:
{question}

Answer:"""


# ── Backward-compatible single-string builder ────────────────────────────
# Kept in case any call site still wants one combined string instead of a
# separate system/user pair.
def build_prompt(context: str, question: str, history: str = "", guaranteed_facts: str = "") -> str:
    lang = detect_language(question)
    system = build_system_prompt(lang)
    user = build_user_prompt(context, question, history, guaranteed_facts)
    return f"{system}\n\n{user}"