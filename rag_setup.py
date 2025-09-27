import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


# Load .env so API key is available
load_dotenv()

def build_vector_store(docs_path="docs", store_path="rag_store"):
    """
    Build or load a Chroma vector store.
    If store_path exists, load the store. Otherwise, build it from documents.
    """
    embedder = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRALAI_API_KEY")
    )

    if os.path.exists(store_path) and os.listdir(store_path):
        print(f"Loading existing vector store from {store_path}")
        vector_store = Chroma(persist_directory=store_path, embedding_function=embedder)
        return vector_store, embedder

    print(f"Building new vector store from documents in {docs_path}")
    documents = []
    for filename in os.listdir(docs_path):
        file_path = os.path.join(docs_path, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    
    splitter = SemanticChunker(embeddings=embedder)
    texts = splitter.split_documents(documents)

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
    