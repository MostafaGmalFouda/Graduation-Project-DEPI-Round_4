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

    def search(self, question, k=3):
        return self.db.similarity_search(
            question,
            k=k
        )