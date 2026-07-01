class Retriever:

    def __init__(self, vector_store):

        self.vector_store = vector_store

    def retrieve(self, question):

        docs = self.vector_store.search(question)

        return "\n".join(

            doc.page_content

            for doc in docs

        )