from langchain_huggingface import HuggingFaceEmbeddings

_embeddings = None

def load_embeddings():
    global _embeddings

    if _embeddings is None:
        print("Loading Embedding Model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embeddings