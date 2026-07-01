import pandas as pd


class TextChunker:
    """
    TextChunker Class

    Responsible for splitting long text into smaller, overlapping chunks
    suitable for embedding and storage in a vector database. This is a
    core RAG (Retrieval-Augmented Generation) preprocessing step: an
    embedding model has a limited context window and similarity search
    works far better on small, focused chunks than on huge documents.

    Why overlap matters:
        Splitting text into non-overlapping blocks can cut a sentence —
        or an important idea — exactly in half, so the chunk that gets
        embedded loses meaning. Adding an overlap (e.g. the last 50
        characters/tokens of chunk N are repeated at the start of chunk
        N+1) preserves context across chunk boundaries and improves
        retrieval quality at the cost of slightly more storage.

    Attributes:
        data (pd.DataFrame): Dataset containing the text column to chunk.

    Methods:
        split_by_character(text_col, chunk_size, chunk_overlap) :
            Chunk text using raw character counts.
        split_by_token(text_col, chunk_size, chunk_overlap) :
            Chunk text using whitespace-token counts (closer to how LLMs
            actually measure context length than character counts).
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    # ── Internal helper: the actual sliding-window chunking logic ──────
    @staticmethod
    def _sliding_window_chunks(units: list, chunk_size: int, chunk_overlap: int) -> list:
        """
        Generic sliding-window chunker shared by both character-based and
        token-based splitting.

        Args:
            units (list): Sequence of atomic units to chunk — either
                individual characters (as a string) or individual tokens
                (as a list of words).
            chunk_size (int): Maximum number of units per chunk.
            chunk_overlap (int): Number of units repeated between
                consecutive chunks.

        Returns:
            list[str | list]: List of chunks (same type as `units`).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        chunks = []
        step = chunk_size - chunk_overlap
        total_len = len(units)

        start = 0
        while start < total_len:
            end = min(start + chunk_size, total_len)
            chunks.append(units[start:end])
            if end == total_len:
                break
            start += step

        return chunks

    # ── Character-based chunking ─────────────────────────────────────────
    def split_by_character(
        self, text_col: str, chunk_size: int = 500, chunk_overlap: int = 50
    ) -> pd.DataFrame:
        """
        Split each document into overlapping chunks based on raw
        character count.

        Best for: quick, language-agnostic chunking where exact token
        counting isn't critical (e.g. short product descriptions, logs).

        Args:
            text_col (str): Name of the column containing clean text.
            chunk_size (int): Maximum characters per chunk.
            chunk_overlap (int): Characters of overlap between consecutive
                chunks (must be smaller than chunk_size).

        Returns:
            pd.DataFrame: A NEW "long" DataFrame — one row PER CHUNK —
            with columns:
                "source_row"  : index of the original document this
                                 chunk came from (so chunks can be traced
                                 back to their source row / metadata),
                "chunk_id"     : sequential id of the chunk within its
                                 source document,
                "chunk_text"   : the chunk's text content.
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        records = []
        for row_idx, text in self.data[text_col].astype(str).items():
            chunks = self._sliding_window_chunks(text, chunk_size, chunk_overlap)
            for chunk_id, chunk_text in enumerate(chunks):
                records.append(
                    {
                        "source_row": row_idx,
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text,
                    }
                )

        return pd.DataFrame(records)

    # ── Token-based chunking ──────────────────────────────────────────────
    def split_by_token(
        self, text_col: str, chunk_size: int = 100, chunk_overlap: int = 20
    ) -> pd.DataFrame:
        """
        Split each document into overlapping chunks based on whitespace
        token (word) count.

        Best for: text intended for an LLM/embedding model, since model
        context limits are measured in tokens, not characters — this
        gives a much more accurate approximation of how much "context
        budget" each chunk will consume.

        Args:
            text_col (str): Name of the column containing clean text.
            chunk_size (int): Maximum tokens (words) per chunk.
            chunk_overlap (int): Tokens of overlap between consecutive
                chunks (must be smaller than chunk_size).

        Returns:
            pd.DataFrame: A NEW "long" DataFrame — one row PER CHUNK —
            with columns:
                "source_row"  : index of the original document,
                "chunk_id"     : sequential id of the chunk within its
                                 source document,
                "chunk_text"   : the chunk's text content (tokens rejoined
                                 with single spaces),
                "token_count"  : number of tokens in this specific chunk.
        """
        if text_col not in self.data.columns:
            raise ValueError(f"Column '{text_col}' not found in data")

        records = []
        for row_idx, text in self.data[text_col].astype(str).items():
            tokens = text.split()
            chunks = self._sliding_window_chunks(tokens, chunk_size, chunk_overlap)
            for chunk_id, chunk_tokens in enumerate(chunks):
                records.append(
                    {
                        "source_row": row_idx,
                        "chunk_id": chunk_id,
                        "chunk_text": " ".join(chunk_tokens),
                        "token_count": len(chunk_tokens),
                    }
                )

        return pd.DataFrame(records)
