from langchain_huggingface import HuggingFaceEmbeddings

_embeddings = None


class E5Embeddings(HuggingFaceEmbeddings):
    """
    E5 models are trained with "query: " / "passage: " prefixes baked into
    the training data itself — this isn't a style choice, skipping it
    measurably hurts retrieval quality for this model family. Documents
    (dataset stats, chart descriptions, ...) get "passage: ", the user's
    question gets "query: ".
    """

    def embed_documents(self, texts):
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


def load_embeddings():
    global _embeddings

    if _embeddings is None:
        print("Loading Embedding Model...")
        # E5 is purpose-built for retrieval (trained with a contrastive
        # objective specifically for query-vs-passage matching), unlike
        # general-purpose paraphrase models. multilingual-e5-small also
        # covers Arabic well, which matters since the documents are
        # written in English but users frequently ask in Arabic.
        _embeddings = E5Embeddings(
            model_name="intfloat/multilingual-e5-small"
        )

    return _embeddings