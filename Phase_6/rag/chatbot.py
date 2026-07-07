import hashlib
import json
import re

from Phase_6.rag.vector_store import VectorStore
from Phase_6.rag.embeddings import load_embeddings
from Phase_6.rag.document_builder import build_documents, build_guaranteed_facts
from Phase_6.rag.prompt import detect_language, build_system_prompt, build_user_prompt
from Phase_6.rag.llm import get_llm


NO_DATA_MESSAGE_AR = "لا يوجد بيانات محمّلة بعد. من فضلك ارفع ملف البيانات وشغّل التحليل أولاً."
NO_DATA_MESSAGE_EN = "No data has been loaded yet. Please upload a data file and run the analysis first."

# Questions that are purely about dataset structure/shape are best answered
# from the "dataset" document type alone — restricting retrieval to it
# avoids pulling in unrelated NLP/visualization noise for what is really a
# lookup, not a semantic search. This is intentionally a small, cheap
# heuristic, not a full query classifier — it only needs to catch the
# common "how many / missing / duplicate / shape" phrasing.
_STRUCTURAL_KEYWORDS = re.compile(
    r"rows?|columns?|missing|duplicate|shape|dtype|data type|"
    r"صف|صفوف|عمود|أعمدة|مفقود|مكرر|تكرار|شكل\s*الداتا",
    re.IGNORECASE,
)


def _docs_fingerprint(docs) -> str:
    """
    Content+metadata hash of all documents, used to decide whether the
    cached FAISS index for this session is still valid. Hashing content
    alone would miss changes that only touch metadata (e.g. a re-tagged
    section); hashing count alone (the original bug) misses edits that
    don't change the document COUNT at all. sha256 over both is the
    standard, unambiguous choice here.
    """
    payload = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


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
        # session_id -> (docs_fingerprint, VectorStore)
        self._vector_cache = {}

    def forget(self, session_id: str):
        """Drop any cached index for this session (e.g. on reset / new upload)."""
        self._vector_cache.pop(session_id, None)

    def _get_vector_store(self, session_id: str, chat_context):
        docs = build_documents(chat_context)

        if not docs:
            self._vector_cache.pop(session_id, None)
            return None

        # Rebuilding/embedding is naturally skipped whenever the fingerprint
        # is unchanged (e.g. two chat messages in a row with no new upload/
        # viz/NLP/model step in between) — so this already only pays the
        # real (embedding) cost when the data actually changed, without
        # needing separate rebuild-trigger wiring at every call site.
        fingerprint = _docs_fingerprint(docs)
        cached = self._vector_cache.get(session_id)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]

        vector = VectorStore(self.embeddings)
        vector.build(docs)
        self._vector_cache[session_id] = (fingerprint, vector)
        return vector

    def ask(self, question: str, chat_context, session_id: str = "default") -> str:
        vector = self._get_vector_store(session_id, chat_context)

        if vector is None:
            no_data_msg = NO_DATA_MESSAGE_AR if detect_language(question) == "Arabic" else NO_DATA_MESSAGE_EN
            return no_data_msg

        # Structural questions ("how many rows/columns", "missing values?")
        # only need the "dataset" document type — narrows the search space
        # instead of competing against NLP/visualization/model documents.
        metadata_filter = {"type": "dataset"} if _STRUCTURAL_KEYWORDS.search(question) else None

        # Relevance-filtered: don't force-feed the model the 5 nearest
        # neighbors if none of them are actually close to the question.
        results = vector.search_relevant(question, k=5, score_threshold=0.5, filter=metadata_filter)
        if not results:
            # Fall back to unfiltered best-effort search rather than
            # returning nothing, e.g. if the metadata filter over-narrowed.
            results = vector.search(question, k=5)
        rag_context = "\n\n".join(doc.page_content for doc in results)

        # Exact counting facts (rows/columns/missing/duplicates) go in
        # directly — they must never depend on similarity search finding
        # the right document among k results.
        guaranteed_facts = build_guaranteed_facts(chat_context)

        history = chat_context.conversation_context.as_text()
        lang = detect_language(question)

        system_prompt = build_system_prompt(lang)
        user_prompt = build_user_prompt(
            context=rag_context,
            question=question,
            history=history,
            guaranteed_facts=guaranteed_facts,
        )
        answer = self.llm.generate(system_prompt, user_prompt)

        # Remember this turn so follow-up questions can refer back to it.
        chat_context.add_conversation_turn(question, answer)

        return answer