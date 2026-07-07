from langchain_community.vectorstores import FAISS


class VectorStore:

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.db = None

    def build(self, documents):
        self.db = FAISS.from_documents(
            documents,
            self.embeddings
        )

    def search(self, question, k=3, filter=None):
        return self.db.similarity_search(
            question,
            k=k,
            filter=filter,
        )

    def search_relevant(self, question, k=5, score_threshold=0.5, filter=None):
        """
        Uses LangChain's normalized relevance score (0..1, HIGHER is more
        relevant) instead of raw FAISS L2 distance, which is unbounded and
        hard to reason about without first profiling this exact embedding
        model's score distribution. Still: treat score_threshold as a
        starting point, not gospel — check it against real queries and
        adjust.

        `filter`: optional dict of exact-match metadata, e.g.
        {"source": "dataset"}, to restrict the search to one section of
        the index instead of the whole thing.
        """
        try:
            results = self.db.similarity_search_with_relevance_scores(
                question, k=k, filter=filter
            )
        except Exception:
            # Some FAISS distance strategies can't produce a normalized
            # relevance score — fall back to plain search rather than error.
            return self.db.similarity_search(question, k=k, filter=filter)
        return [doc for doc, score in results if score >= score_threshold]