from Phase_6.rag.vector_store import VectorStore
from Phase_6.rag.embeddings import load_embeddings
from Phase_6.rag.document_builder import build_documents
from Phase_6.rag.prompt import build_prompt
from Phase_6.rag.llm import get_llm


NO_DATA_MESSAGE_AR = "لا يوجد بيانات محمّلة بعد. من فضلك ارفع ملف البيانات وشغّل التحليل أولاً."
NO_DATA_MESSAGE_EN = "No data has been loaded yet. Please upload a data file and run the analysis first."


class Chatbot:
    """
    One Chatbot instance is shared by the whole Flask app (it holds the
    heavy embeddings model + LLM client). Internally it keeps a SEPARATE
    FAISS index per session_id, so one user's data/questions never leak
    into another user's answers.

    It also gives each session short-term conversational memory: every
    (question, answer) pair is appended to chat_context.conversation_context
    so follow-up questions ("how many?", "why?") can be resolved.
    """

    def __init__(self):
        self.embeddings = load_embeddings()
        self.llm = get_llm()
        # session_id -> (doc_count, VectorStore)
        self._vector_cache = {}

    def forget(self, session_id: str):
        """Drop any cached index for this session (e.g. on reset / new upload)."""
        self._vector_cache.pop(session_id, None)

    def _get_vector_store(self, session_id: str, chat_context):
        docs = build_documents(chat_context)

        if not docs:
            self._vector_cache.pop(session_id, None)
            return None

        cached = self._vector_cache.get(session_id)
        if cached is not None and cached[0] == len(docs):
            return cached[1]

        vector = VectorStore(self.embeddings)
        vector.build(docs)
        self._vector_cache[session_id] = (len(docs), vector)
        return vector

    def ask(self, question: str, chat_context, session_id: str = "default") -> str:
        from Phase_6.rag.prompt import detect_language

        vector = self._get_vector_store(session_id, chat_context)

        if vector is None:
            no_data_msg = NO_DATA_MESSAGE_AR if detect_language(question) == "Arabic" else NO_DATA_MESSAGE_EN
            return no_data_msg

        results = vector.search(question, k=5)
        rag_context = "\n\n".join(doc.page_content for doc in results)
        history = chat_context.conversation_context.as_text()

        prompt = build_prompt(context=rag_context, question=question, history=history)
        answer = self.llm.generate(prompt)

        # Remember this turn so follow-up questions can refer back to it.
        chat_context.add_conversation_turn(question, answer)

        return answer