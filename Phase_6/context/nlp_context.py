class NLPContext:
    """Mirrors EDAContext, but for NLP-phase results (keywords, sentiment,
    text statistics) so the RAG chatbot can answer questions about them."""

    def __init__(self):
        self.analyses = []

    def add_analysis(self, text_column, description):
        self.analyses.append({
            "text_column": text_column,
            "description": description,
        })

    def get_context(self):
        return self.analyses
