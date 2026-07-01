import os


class RAGOrchestrator:
    """
    RAGOrchestrator Class

    The final stage of the RAG (Retrieval-Augmented Generation) pipeline.
    Takes retrieved context chunks (from VectorStoreManager) plus the
    user's question, builds a grounded prompt, and calls an LLM
    (Anthropic's Claude API) to generate a natural-language answer.

    Why an LLM provider abstraction matters:
        This class is intentionally the ONLY place in the whole RAG
        pipeline that talks to an external LLM API. Every other module
        (TextChunker, VectorStoreManager, NLPAnalyzer...) is pure
        local/offline logic. Keeping the LLM call isolated here means:
            - API keys are only needed/read in one place,
            - swapping providers later only requires changing this file,
            - the rest of the pipeline can be unit-tested without
              network calls or API costs.

    Setup:
        Requires the ANTHROPIC_API_KEY environment variable to be set,
        e.g.:
            export ANTHROPIC_API_KEY="sk-ant-..."
        Get a key at https://console.anthropic.com/

    Attributes:
        model_name (str)  : Claude model to use for generation
                             (default: "claude-sonnet-4-6").
        temperature (float): Sampling temperature — lower = more
                              deterministic/grounded answers, which is
                              usually preferred for RAG (we want answers
                              based on retrieved facts, not creativity).
        vector_store       : Reference to a VectorStoreManager instance,
                              used by query_pipeline() to retrieve context
                              automatically. Optional — can be set later
                              via set_vector_store(), or context can be
                              passed manually to build_prompt().
        _client             : Lazily-initialized Anthropic API client.

    Methods:
        build_prompt(query, context_chunks)   : Assemble a grounded prompt.
        generate_answer(prompt, model_name)   : Call Claude and get a text answer.
        query_pipeline(user_query)            : Full end-to-end RAG call —
                                                  retrieve -> build prompt -> generate.
        set_vector_store(vector_store)        : Attach a VectorStoreManager
                                                  for automatic retrieval.
    """

    SYSTEM_PROMPT = (
        "You are a helpful, precise assistant answering questions strictly "
        "based on the provided context. If the answer cannot be found in "
        "the context, say so clearly instead of guessing or using outside "
        "knowledge."
    )

    def __init__(self, model_name: str = "claude-sonnet-4-6", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        self.vector_store = None
        self._client = None

    # ── Lazy client initialization ───────────────────────────────────────
    def _get_client(self):
        """
        Lazily create the Anthropic API client on first use, so importing
        this class doesn't require an API key to already be set (useful
        for testing the rest of the pipeline without LLM calls).
        """
        if self._client is None:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Get a key from https://console.anthropic.com/ and set it with:\n"
                    "    export ANTHROPIC_API_KEY='sk-ant-...'"
                )
            self._client = anthropic.Anthropic(api_key=api_key)

        return self._client

    # ── Attach a vector store for automatic retrieval ────────────────────
    def set_vector_store(self, vector_store) -> None:
        """
        Attach a VectorStoreManager instance so query_pipeline() can
        automatically retrieve context for incoming questions.

        Args:
            vector_store (VectorStoreManager): An already-initialized
                vector store (initialize_vector_db() must have been
                called, and data must have been inserted).

        Returns:
            None
        """
        self.vector_store = vector_store

    # ── Prompt construction ──────────────────────────────────────────────
    def build_prompt(self, query: str, context_chunks: list) -> str:
        """
        Assemble a grounded RAG prompt from the user's question and the
        retrieved context chunks.

        The prompt explicitly separates CONTEXT from QUESTION and
        instructs the model to answer using only the given context —
        this is the core technique that reduces hallucination in RAG
        systems compared to asking the LLM "cold" with no grounding.

        Args:
            query (str): The user's natural-language question.
            context_chunks (list[str] | list[dict]): Retrieved chunks —
                either plain strings, or dicts shaped like
                VectorStoreManager.similarity_search() output
                (each containing a "text" key).

        Returns:
            str: The fully assembled prompt, ready to send to the LLM.
        """
        if not context_chunks:
            context_text = "(no relevant context was found)"
        else:
            texts = [
                chunk["text"] if isinstance(chunk, dict) else str(chunk)
                for chunk in context_chunks
            ]
            context_text = "\n\n".join(
                f"[Context {i + 1}]\n{text}" for i, text in enumerate(texts)
            )

        prompt = (
            f"Use the following context to answer the question. "
            f"If the context does not contain the answer, say you don't "
            f"have enough information.\n\n"
            f"--- CONTEXT ---\n{context_text}\n\n"
            f"--- QUESTION ---\n{query}\n\n"
            f"--- ANSWER ---"
        )
        return prompt

    # ── LLM call ──────────────────────────────────────────────────────────
    def generate_answer(self, prompt: str, model_name: str = None, max_tokens: int = 1024) -> str:
        """
        Send a prompt to Claude and return the generated text answer.

        Args:
            prompt (str): The fully built RAG prompt (output of
                build_prompt()).
            model_name (str, optional): Override the instance's default
                model for this single call.
            max_tokens (int): Maximum tokens to generate in the response.

        Returns:
            str: The model's text answer.
        """
        client = self._get_client()

        response = client.messages.create(
            model=model_name or self.model_name,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        # response.content is a list of content blocks; concatenate any
        # text blocks to build the final answer string.
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    # ── Full end-to-end RAG pipeline ─────────────────────────────────────
    def query_pipeline(self, user_query: str, k: int = 5) -> str:
        """
        Run the complete RAG flow for a user question in one call:

            1. Retrieve the top-k most relevant chunks from the attached
               vector store (similarity_search).
            2. Build a grounded prompt from those chunks + the question.
            3. Generate and return the final answer from Claude.

        Args:
            user_query (str): The user's natural-language question.
            k (int): Number of context chunks to retrieve.

        Returns:
            str: The final, generated answer.
        """
        if self.vector_store is None:
            raise RuntimeError(
                "No vector store attached. Call set_vector_store() first, "
                "or build the prompt manually with build_prompt()."
            )

        retrieved_chunks = self.vector_store.similarity_search(user_query, k=k)
        prompt = self.build_prompt(user_query, retrieved_chunks)
        return self.generate_answer(prompt)
