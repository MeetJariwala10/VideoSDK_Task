"""RAG vector store setup utilities.

This module builds or loads a persistent Chroma vector store using Mistral
embeddings. It ingests `.txt` and `.pdf` files from the `docs/` directory and
persists the index to `rag_store/`.
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


# Load .env so API key is available
load_dotenv()


def build_vector_store(docs_path: str = "docs", store_path: str = "rag_store"):
    """Build or load a Chroma vector store and return (store, embedder).

    If a persisted store exists in `store_path`, it will be loaded and reused.
    Otherwise, we will load documents from `docs_path`, chunk them with
    `semanticChunker`, and persist a new index.

    Args:
        docs_path: Directory containing `.txt` and `.pdf` files to ingest.
        store_path: Directory path where the Chroma index will be persisted.

    Returns:
        tuple[Chroma, MistralAIEmbeddings]: (vector_store, embedder)
    """
    embedder = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRALAI_API_KEY")
    )

    # Reuse existing store if present
    if os.path.exists(store_path) and os.listdir(store_path):
        print(f"Loading existing vector store from {store_path}")
        vector_store = Chroma(persist_directory=store_path, embedding_function=embedder)
        return vector_store, embedder

    # Build new store from docs
    print(f"Building new vector store from documents in {docs_path}")
    documents = []
    for filename in os.listdir(docs_path):
        file_path = os.path.join(docs_path, filename)
        if filename.lower().endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        elif filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    # Semantic chunking typically yields higher-quality chunks than naive splitting
    splitter = SemanticChunker(embeddings=embedder)
    texts = splitter.split_documents(documents)

    # Persist vector store
    vector_store = Chroma.from_documents(
        texts,
        embedder,
        persist_directory=store_path
    )
    print(f"Vector store built and persisted to {store_path}")
    return vector_store, embedder


if __name__ == "__main__":
    vs, emb = build_vector_store()
    print("Vector store created")