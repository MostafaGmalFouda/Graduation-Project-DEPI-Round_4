# 🔷 Phase 4 — RAG & LLM Engine

Adds Retrieval-Augmented Generation on top of Phase 3's cleaned text.
Lets users **ask natural-language questions about their dataset** and
get grounded, context-aware answers from an LLM (Claude), instead of
hallucinated guesses.

## 📦 Modules

| Module | Responsibility |
|---|---|
| `TextChunker` | Splits long text into overlapping chunks (by character or by token) sized for embedding |
| `VectorStoreManager` | Generates embeddings (sentence-transformers) and stores/searches them in a local ChromaDB vector database |
| `RAGOrchestrator` | Builds grounded prompts from retrieved chunks and calls Claude to generate the final answer |

## 🔄 Data Flow

```
[ Clean Text (Phase 3) ] → [ TextChunker ] → chunks
   → [ VectorStoreManager.generate_embeddings() ] → vectors
   → [ VectorStoreManager.insert_vectors() ] → ChromaDB

[ User Question ] → [ VectorStoreManager.similarity_search() ] → top-k chunks
   → [ RAGOrchestrator.build_prompt() ] → grounded prompt
   → [ RAGOrchestrator.generate_answer() ] → Claude → final answer
```

## 🚀 Quick Start

```python
import pandas as pd
from Phase_4_RAG.TextChunker import TextChunker
from Phase_4_RAG.VectorStoreManager import VectorStoreManager
from Phase_4_RAG.RAGOrchestrator import RAGOrchestrator

df = pd.DataFrame({"doc": ["... long cleaned document text ..."]})

# 1. Chunk the text
chunker = TextChunker(df)
chunks_df = chunker.split_by_token("doc", chunk_size=200, chunk_overlap=30)

# 2. Embed + store
store = VectorStoreManager()
store.initialize_vector_db(persist_directory="./vector_store")
vectors = store.generate_embeddings(chunks_df["chunk_text"].tolist())
metadata = chunks_df[["source_row", "chunk_id"]].to_dict("records")
store.insert_vectors(vectors, metadata, documents=chunks_df["chunk_text"].tolist())

# 3. Ask a question
rag = RAGOrchestrator(model_name="claude-sonnet-4-6")
rag.set_vector_store(store)
answer = rag.query_pipeline("What does this dataset say about X?")
print(answer)
```

## ⚙️ Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # https://console.anthropic.com/
```

## 🧩 Integration Notes

- `VectorStoreManager` uses **sentence-transformers** (`all-MiniLM-L6-v2`
  by default) for embeddings — fully local, free, no API key required.
- ChromaDB is **persistent**: data written via `insert_vectors()`
  survives across app restarts as long as the same `persist_directory`
  is used.
- `RAGOrchestrator` is the ONLY module in this phase that makes an
  external network call (to Claude). Every other module runs fully
  offline — keep it that way when extending this phase.
- For larger or production-scale corpora, increase `chunk_size` /
  tune `k` in `similarity_search()` based on retrieval quality testing.
