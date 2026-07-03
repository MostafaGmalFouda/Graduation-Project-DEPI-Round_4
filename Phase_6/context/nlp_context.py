class NLPContext:
    """Mirrors EDAContext, but for NLP-phase results (keywords, sentiment,
    text statistics) so the RAG chatbot can answer questions about them.

    Also holds a capped SAMPLE of the actual raw text (see raw_texts below).
    Without this, the chatbot only ever sees a one-line statistical summary
    and can't answer questions about what the text actually says (e.g.
    "what are people complaining about?").
    """

    def __init__(self):
        self.analyses = []
        self.raw_texts = []

    def add_analysis(self, text_column, description):
        self.analyses.append({
            "text_column": text_column,
            "description": description,
        })

    def add_raw_texts(self, text_column, texts):
        """`texts` should already be capped by the caller (see
        NLPAnalyzer.raw_text_samples) — kept as a plain list here so this
        class doesn't need to know about pandas/analyzer internals."""
        self.raw_texts.append({
            "text_column": text_column,
            "texts": list(texts),
        })

    def get_context(self):
        return self.analyses

    def get_raw_texts(self):
        return self.raw_texts