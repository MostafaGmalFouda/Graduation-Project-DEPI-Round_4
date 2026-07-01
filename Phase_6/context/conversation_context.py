class ConversationContext:
    """
    Keeps a short rolling history of (question, answer) turns for ONE
    session, so follow-up questions like "how many?" or "عددهم إيه؟"
    right after a previous question can be resolved using what was just
    discussed.

    Only the last MAX_TURNS are kept, to keep the prompt sent to the LLM
    bounded in size.
    """

    MAX_TURNS = 8

    def __init__(self):
        self.turns = []

    def add_turn(self, question: str, answer: str):
        self.turns.append({"question": question, "answer": answer})
        if len(self.turns) > self.MAX_TURNS:
            self.turns = self.turns[-self.MAX_TURNS:]

    def get_context(self):
        return self.turns

    def as_text(self) -> str:
        """Render the history as plain text for prompt injection."""
        if not self.turns:
            return ""
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)
    