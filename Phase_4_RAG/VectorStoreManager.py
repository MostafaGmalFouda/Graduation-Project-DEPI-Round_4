import uuid
import numpy as np


class VectorStoreManager:
    """
    VectorStoreManager Class

    Responsible for the embedding + storage + retrieval layer of the RAG
    (Retrieval-Augmented Generation) pipeline:

        text chunks -> embeddings -> vector database -> similarity search

    Backed by:
        - sentence-transformers : open-source, free, runs locally (no API
          key / no per-call cost) to turn text into dense vector
          embeddings. Default model "all-MiniLM-L6-v2" is small, fast,
          and a very common production default for RAG prototypes.
        - ChromaDB              : lightweight, embedded vector database.
          Runs locally with zero external services, perfect for a
          graduation project / prototype scale RAG system.

    Design notes:
        - The embedding model is loaded lazily (on first use) since
          loading it is somewhat expensive (~100MB download + load time)
          and not every pipeline run necessarily needs RAG.
        - initialize_vector_db() must be called before insert_vectors()
          or similarity_search() — this mirrors the class diagram's
          intended call order and keeps the "create the collection" step
          explicit and separate from "fill the collection with data".

    Attributes:
        db_connection      : The active ChromaDB client (None until
                              initialize_vector_db() is called).
        collection          : The active ChromaDB collection (None until
                              initialize_vector_db() is called).
        _embedding_model    : Lazily-loaded SentenceTransformer instance.
        _embedding_model_name (str): Name of the currently loaded model.

    Methods:
        generate_embeddings(chunks, embedding_model) : Text chunks -> vectors.
        initialize_vector_db(db_type, collection_name, persist_directory) :
            Create/connect to the vector database collection.
        insert_vectors(vectors, metadata)             : Store embeddings + metadata.
        similarity_search(query, k)                    : Retrieve the k most
                                                            relevant chunks for a query.
    """

    def __init__(self):
        self.db_connection = None
        self.collection = None
        self._embedding_model = None
        self._embedding_model_name = None

    # ── Embedding generation ─────────────────────────────────────────────
    def generate_embeddings(
        self, chunks: list, embedding_model: str = "all-MiniLM-L6-v2"
    ) -> np.ndarray:
        """
        Convert a list of text chunks into dense vector embeddings using
        a local sentence-transformers model.

        Args:
            chunks (list[str]): Text chunks to embed (typically the
                "chunk_text" column produced by TextChunker).
            embedding_model (str): Name of the sentence-transformers
                model to use. "all-MiniLM-L6-v2" is a strong, lightweight
                default (384-dim vectors, fast on CPU).

        Returns:
            np.ndarray: Array of shape (n_chunks, embedding_dim).
        """
        if not chunks:
            raise ValueError("chunks list cannot be empty")

        if self._embedding_model is None or self._embedding_model_name != embedding_model:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "The 'sentence-transformers' package is not installed. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

            try:
                self._embedding_model = SentenceTransformer(embedding_model)
            except Exception as exc:
                # The most common cause here is no internet access on the
                # FIRST run (the model has to download from Hugging Face
                # before it can be used locally). Surface that plainly
                # instead of letting a network error look like a generic
                # crash with no actionable next step.
                raise RuntimeError(
                    f"Could not load the embedding model '{embedding_model}'. "
                    "This usually means the model needs to download from "
                    "Hugging Face on first use, which requires internet "
                    "access on this machine. Check your connection and try "
                    f"again. (Original error: {exc})"
                ) from exc

            self._embedding_model_name = embedding_model

        embeddings = self._embedding_model.encode(
            chunks, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings

    # ── Vector database initialization ───────────────────────────────────
    def initialize_vector_db(
        self,
        db_type: str = "chroma",
        collection_name: str = "rag_collection",
        persist_directory: str = "./vector_store",
    ) -> None:
        """
        Create (or connect to) a persistent vector database collection.

        Args:
            db_type (str): Vector DB backend to use. Currently supports
                "chroma" (ChromaDB, embedded/local). Kept as a parameter
                rather than hard-coded so the system can later support
                other backends (e.g. FAISS, Pinecone) without changing
                the public method signature.
            collection_name (str): Name of the collection to create or
                load — acts like a "table name" for vectors.
            persist_directory (str): Local folder where ChromaDB will
                persist its data to disk, so the index survives between
                runs of the application.

        Returns:
            None
        """
        if db_type != "chroma":
            raise NotImplementedError(
                f"db_type='{db_type}' is not supported yet. Use 'chroma'."
            )

        import chromadb

        self.db_connection = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.db_connection.get_or_create_collection(
            name=collection_name
        )

    # ── Insert vectors + metadata into the store ─────────────────────────
    def insert_vectors(self, vectors: np.ndarray, metadata: list, documents: list = None) -> None:
        """
        Insert embeddings, their source metadata, and (optionally) the
        original chunk text into the active vector database collection.

        Args:
            vectors (np.ndarray): Array of shape (n_chunks, embedding_dim),
                typically the output of generate_embeddings().
            metadata (list[dict]): One metadata dict per vector — e.g.
                {"source_row": 12, "chunk_id": 0}. Used to trace a
                retrieved chunk back to where it came from.
            documents (list[str], optional): The raw chunk text
                corresponding to each vector. Strongly recommended —
                ChromaDB can return this directly on search, avoiding a
                separate lookup back to the original DataFrame.

        Returns:
            None
        """
        if self.collection is None:
            raise RuntimeError(
                "Vector DB not initialized. Call initialize_vector_db() first."
            )
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have the same length")
        if documents is not None and len(documents) != len(vectors):
            raise ValueError("documents must have the same length as vectors")

        ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        self.collection.add(
            ids=ids,
            embeddings=np.asarray(vectors).tolist(),
            metadatas=metadata,
            documents=documents if documents is not None else [""] * len(vectors),
        )

    # ── Similarity search ─────────────────────────────────────────────────
    def similarity_search(self, query: str, k: int = 5) -> list:
        """
        Embed a natural-language query and retrieve the k most similar
        chunks from the vector database.

        This is the "Retrieval" half of Retrieval-Augmented Generation —
        the chunks returned here become the context that gets handed to
        the LLM in RAGOrchestrator.build_prompt().

        Args:
            query (str): The user's natural-language question.
            k (int): Number of top matching chunks to return.

        Returns:
            list[dict]: Up to k results, each:
                {
                    "text": str        -> the chunk's text,
                    "metadata": dict   -> the chunk's stored metadata,
                    "distance": float  -> similarity distance (lower = more similar)
                }
        """
        if self.collection is None:
            raise RuntimeError(
                "Vector DB not initialized. Call initialize_vector_db() first."
            )

        query_embedding = self.generate_embeddings(
            [query], embedding_model=self._embedding_model_name or "all-MiniLM-L6-v2"
        )

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]
